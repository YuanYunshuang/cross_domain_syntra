from __future__ import annotations
import math
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence

import torch
from torch.utils.data import Sampler


def get_sampler(dataset, batch_size, sampler_name='uniform'):
    prob_dict = {'hameln': 0.5, 'bayern': 0.5, 'railway': 0.01, 'vineyard': 0.01, 'oem': 1.0, 'medical': 1.0}
    if sampler_name == 'prob':
        probs = []
        for filename in dataset.file_list:
            res = filename.split('.')
            dataset_name = res[1] if 'siegfried' in res else res[0]
            if dataset_name in prob_dict:
                probs.append(prob_dict[dataset_name])
        else:
            probs.append(1.0)
        sampler = ProbabilitySampler(probs)
        batch_sampler = None
    elif sampler_name == 'uniform':
        return None, None
    elif sampler_name == 'grouped_prob':
        labels = [dataset.subdataset_id(i) for i in range(len(dataset))]
        groups = group_indices_by_labels(labels)
        group_probs = [prob_dict[k] for k in dataset._loc2id.keys()]  # adjust as needed
        assert batch_size >= len(groups), "batch_size must be >= number of subdatasets"

        batch_sampler = ProbGroupedBatchSampler(
            groups=groups,
            batch_size=batch_size,
            group_probs=group_probs,
            replacement=True,          # set False to avoid repeats across the epoch
            generator=torch.Generator().manual_seed(42),
            # num_batches=None -> inferred
        )
        sampler = None
    elif sampler_name.startswith('group_fixed_m'):
        # pattern: "group_fixed_m:<m>" e.g. "group_fixed_m:3"
        parts = sampler_name.split(':')
        labels = [dataset.subdataset_id(i) for i in range(len(dataset))]
        groups = group_indices_by_labels(labels)
        print(f"Sampler group names: {dataset._loc2id.keys()}")
        if len(parts) == 2:
            samples_per_group = int(parts[1])
        else:
            samples_per_group = math.ceil(batch_size / len(groups))
        batch_sampler = FixedPerGroupEpochBatchSampler(
            groups=groups,
            samples_per_group=samples_per_group,
            batch_size=batch_size,
            replacement=False,
            shuffle_epoch=True,
            generator=torch.Generator().manual_seed(42),
        )
        sampler = None
    else:
        sampler = None
        batch_sampler = None
    return sampler, batch_sampler


def group_indices_by_labels(labels: Sequence[int]) -> List[List[int]]:
    groups: Dict[int, List[int]] = {}
    for idx, g in enumerate(labels):
        groups.setdefault(int(g), []).append(idx)
    # sort keys to have stable group order [0,1,2,3,...]
    return [groups[k] for k in sorted(groups.keys())]

def group_indices_by_key(num_items: int, key_fn: Callable[[int], int | str]) -> List[List[int]]:
    groups: Dict[int | str, List[int]] = {}
    for idx in range(num_items):
        key = key_fn(idx)
        groups.setdefault(key, []).append(idx)
    # stable order by key
    keys = sorted(groups.keys())
    return [groups[k] for k in keys]


class ProbGroupedBatchSampler(Sampler[List[int]]):
    """
    Yields batches of indices where:
      - At least one sample is taken from each group per batch.
      - Remaining slots are allocated across groups according to group_probs.
    """
    def __init__(
        self,
        groups: List[List[int]],
        batch_size: int,
        group_probs: Sequence[float],
        *,
        replacement: bool = True,
        generator: Optional[torch.Generator] = None,
        num_batches: Optional[int] = None,
    ) -> None:
        super().__init__(None)
        if len(groups) == 0:
            raise ValueError("groups must be non-empty")
        self.groups = [list(g) for g in groups]
        self.K = len(self.groups)
        if batch_size < self.K:
            raise ValueError(f"batch_size ({batch_size}) must be >= number of groups ({self.K})")
        self.batch_size = int(batch_size)

        p = torch.as_tensor(group_probs, dtype=torch.double)
        if p.numel() != self.K:
            raise ValueError(f"len(group_probs) must equal number of groups ({self.K})")
        p = p.clamp_min(0)
        if p.sum() <= 0:
            raise ValueError("sum(group_probs) must be > 0")
        self.probs = (p / p.sum()).cpu()

        self.replacement = bool(replacement)
        self.gen = generator
        total_items = sum(len(g) for g in self.groups)
        # sensible default for steps per epoch
        if num_batches is None:
            if self.replacement:
                self.num_batches = math.ceil(total_items / self.batch_size)
            else:
                # upper bound: when always needing at least 1 per group
                self.num_batches = max(1, min(len(g) for g in self.groups))
        else:
            self.num_batches = int(num_batches)

        # Prebuild tensors of indices for torch ops
        self.group_tensors = [torch.tensor(g, dtype=torch.long) for g in self.groups]

    def __len__(self) -> int:
        return self.num_batches

    def _sample_counts(self) -> List[int]:
        # guarantee 1 per group, distribute the rest by multinomial
        remaining = self.batch_size - self.K
        if remaining == 0:
            return [1] * self.K
        # sample how many extra per group
        draws = torch.multinomial(self.probs, num_samples=remaining, replacement=True, generator=self.gen)
        extra = torch.bincount(draws, minlength=self.K).tolist()
        return [1 + e for e in extra]

    def _sample_from_group(self, gidx: int, k: int) -> List[int]:
        pool = self.group_tensors[gidx]
        n = pool.numel()
        if n == 0:
            raise RuntimeError(f"group {gidx} is empty")
        if self.replacement or k > n:
            # allow repeats if needed
            pos = torch.randint(low=0, high=n, size=(k,), generator=self.gen)
            return pool[pos].tolist()
        else:
            # without replacement within batch
            perm = torch.randperm(n, generator=self.gen)[:k]
            return pool[perm].tolist()

    def __iter__(self) -> Iterator[List[int]]:
        # If replacement=False and we want epoch-wise no repeats, we can track cursors
        if not self.replacement:
            # shuffle each group's pool once per epoch and consume sequentially
            shuffled = [g[torch.randperm(len(g), generator=self.gen)].tolist() for g in self.group_tensors]
            cursors = [0] * self.K

            for _ in range(self.num_batches):
                counts = self._sample_counts()
                batch: List[int] = []
                for gi, need in enumerate(counts):
                    remain = len(shuffled[gi]) - cursors[gi]
                    if remain < need:
                        # not enough to satisfy this batch -> end epoch early
                        return
                    take = shuffled[gi][cursors[gi]: cursors[gi] + need]
                    cursors[gi] += need
                    batch.extend(take)
                yield batch
        else:
            for _ in range(self.num_batches):
                counts = self._sample_counts()
                batch: List[int] = []
                for gi, need in enumerate(counts):
                    batch.extend(self._sample_from_group(gi, need))
                yield batch


class ProbabilitySampler(Sampler[int]):
    def __init__(
        self,
        probs: Sequence[float] | torch.Tensor,
        num_samples: Optional[int] = None,
        replacement: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__(None)
        p = torch.as_tensor(probs, dtype=torch.double)
        if p.ndim != 1:
            raise ValueError("probs must be a 1D array/tensor")
        if p.numel() == 0:
            raise ValueError("probs must be non-empty")
        p = p.clamp_min(0)
        s = p.sum().item()
        if s <= 0:
            raise ValueError("sum(probs) must be > 0")
        self.probs = (p / s).cpu()
        self.N = self.probs.numel()
        self.num_samples = int(num_samples) if num_samples is not None else self.N
        if not replacement and self.num_samples > self.N:
            raise ValueError("num_samples cannot exceed dataset size when replacement=False")
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        idx = torch.multinomial(
            self.probs,
            num_samples=self.num_samples,
            replacement=self.replacement,
            generator=self.generator,
        )
        return iter(idx.tolist())

    def __len__(self) -> int:
        return self.num_samples
    

class FixedPerGroupEpochBatchSampler(Sampler[List[int]]):
    """
    For N groups (possibly N >> batch_size):
      - Each epoch draws exactly m (samples_per_group) samples from every group.
      - Total epoch size = N * m.
      - Batches are contiguous slices of the epoch sample list.
    """
    def __init__(
        self,
        groups: List[List[int]],
        samples_per_group: int,
        batch_size: int,
        *,
        replacement: bool = False,
        shuffle_epoch: bool = True,
        drop_last: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if samples_per_group <= 0:
            raise ValueError("samples_per_group must be > 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.groups = [list(g) for g in groups]
        self.K = len(self.groups)
        if self.K == 0:
            raise ValueError("groups must be non-empty")
        self.m = int(samples_per_group)
        self.batch_size = int(batch_size)
        self.replacement = bool(replacement)
        self.shuffle_epoch = bool(shuffle_epoch)
        self.drop_last = bool(drop_last)
        self.gen = generator
        # Preconvert groups to tensors
        self.group_tensors = [torch.tensor(g, dtype=torch.long) for g in self.groups]
        self._epoch_size = self.K * self.m
        # K * m rarely divides batch_size - 18 groups at m=1 into batches of 8 leaves a
        # remainder of 2 - and that short batch is not a harmless rounding detail. It is a third
        # of the optimizer steps at a quarter of the nominal batch size, and with pos and neg as
        # separate groups it drew both of its two samples from negative groups 25% of the time,
        # which is where the NaN losses came from before SAMLoss was guarded. Dropping it makes
        # every step the batch size that was asked for; the epoch is reshuffled, so the two
        # samples left out differ each time and nothing is systematically starved.
        self._num_batches = (self._epoch_size // self.batch_size if self.drop_last
                             else math.ceil(self._epoch_size / self.batch_size))
        if self._num_batches == 0:
            raise ValueError(
                f"{self.K} groups x {self.m} samples/group = {self._epoch_size} per epoch is "
                f"below batch_size {self.batch_size}; raise samples_per_group")
        dropped = self._epoch_size - self._num_batches * self.batch_size
        print(f"FixedPerGroupEpochBatchSampler: {self.K} groups, {self.m} samples/group, "
              f"{self._num_batches} batches of {self.batch_size}"
              + (f", {dropped} left over per epoch and reshuffled" if dropped else ""))

    def __len__(self) -> int:
        return self._num_batches

    def _sample_group(self, gidx: int) -> List[int]:
        pool = self.group_tensors[gidx]
        n = pool.numel()
        k = self.m
        if n == 0:
            raise RuntimeError(f"group {gidx} is empty")
        if self.replacement:
            pos = torch.randint(0, n, (k,), generator=self.gen)
            return pool[pos].tolist()
        else:
            if k > n:
                raise ValueError(f"Not enough samples in group {gidx} for k={k} without replacement (size={n})")
            perm = torch.randperm(n, generator=self.gen)[:k]
            return pool[perm].tolist()

    def _build_epoch_indices(self) -> List[int]:
        collected: List[int] = []
        for gi in range(self.K):
            collected.extend(self._sample_group(gi))
        if self.shuffle_epoch:
            idx_tensor = torch.tensor(collected, dtype=torch.long)
            perm = torch.randperm(idx_tensor.numel(), generator=self.gen)
            return idx_tensor[perm].tolist()
        return collected

    def __iter__(self) -> Iterator[List[int]]:
        epoch_indices = self._build_epoch_indices()
        for i in range(self._num_batches):
            yield epoch_indices[i * self.batch_size: (i + 1) * self.batch_size]


