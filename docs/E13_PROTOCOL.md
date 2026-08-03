# E13 — run protocol

One section per run. `docs/E13_PLAN.md` says why; this says exactly what to execute and what
each run is compared against. Every number below is measured from the corpus, not estimated.

## Common to every run

| | |
| --- | --- |
| corpus | 26 sheets — 15 Hameln, 8 Bayern, 2 Siegfried, with `hameln.3921_2023` excluded as a duplicate scan |
| candidate pool | 10,831 tiles, **12,918 (tile, class) pairs** above the foreground threshold (0.05; 0.01 for Siegfried) |
| classes | forest, grass, settlement, water, railway, vineyard |
| budget unit | one annotation pass = one class digitised on one tile |
| coverage floor | 1 pair per (cell, class) before the strategy spends the rest |
| validation | 10% of each run's pairs, held out of training, paid for from the budget |
| model | SAM ViT-B, DoRA rank 16, `prompt_encoder=example`, `mask_decoder=default` |
| training | batch 8, AMP, `accum_steps=4`, lr 1e-3, poly decay, 1000 warmup iterations, seed 2026 |
| sources | `num_src=3`, `num_notions=1`, 384 px, `src_fusion=max` |
| evaluation | full test split of all 26 sheets, natural prevalence, per-class IoU and the mean |
| selection seed | 2026 everywhere, so a rerun reproduces the same tiles |

**Retrieval scope always equals the cell.** `--cell` for `select_budget.py` and `--scope` for
`make_pool_pairing.py` take the same value in the same run; they are one decision.

**Every run trains from scratch.** No earlier checkpoint enters E13, and the historical 85.89 is
not a baseline for any of it.

### The cells, and how many groups each definition makes

The cell is set by `--cell-gap`: sheets whose mean embeddings lie within that cosine distance
share a pool, transitively. 0 is sheet-wise.

| `--cell-gap` | pools | (cell, class) combinations = floor cost |
| ---: | ---: | ---: |
| **0** | **26** | **88** |
| 0.01 | 18 | 60 |
| 0.03 | 13 | 38 |
| 0.05 | 12 | 35 |
| 0.10 | 5 | 15 |
| 0.114 | 1 | 6 |

`era` chains survey years less than five apart within one corpus: `bayern.1910`,
`bayern.1938-1941`, `bayern.1959`, `hameln.1898`, `hameln.1973-1975`, `hameln.1982`,
`hameln.1996`, `hameln.2002-2005`, `hameln.2012`, `hameln.2017`, `hameln.2023`, plus `railway`
and `vineyard`, which carry no year and stand alone.

The grouping is measured rather than declared, because the two archives behave oppositely. In Hameln, sheets of the
same year are 0.008 apart and one sheet across years 0.119, so period drives style. In Bayern it
inverts: 0.030 for the same sheet across years, 0.110 for the same year across sheets, because a
Positionsblatt is one plate revised rather than re-drawn. The recovered groups are:

    hameln.s0   3821_1974 3822_1975 3921_1973 3922_1974 3922_1982 3922_1996
    hameln.s1   3821_2002 3822_2005 3921_2002 3922_2005
    hameln.s2   3821_2012 3821_2023 3822_2023 3921_2017 3922_2017
    hameln.s10  3922_1898
    bayern.s0   489_1910 489_1939          bayern.s2   490_1910 490_1938
    bayern.s4   518_1941   s5 519_1910   s6 7230_1959   s7 7330_1959
    siegfried   railway, vineyard

Emptiness was ruled out as the driver: one embedding direction predicts a tile's ink fraction at
r=0.933, but projecting it out leaves the merge order identical for sixteen steps. Corpora are
kept apart by default, which is a caution rather than a measured necessity — the first
cross-corpus merge, `bayern.7330_1959` with the Hameln 1970s at 0.031, looks wrong to the eye
but survives that test, so whether to allow it is for the sweep to answer.

### The density guard, per budget

Measured at `cell=sheet` against a density-neutral reference — random selection under the same
cell and balance. The figure is how much more of the sparsest tenth of the pool the selection
takes than that reference does:

| budget | neutral | g=0 | g=5 | g=10 | g=15 | **use** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | 22.1% | 2.0x | 1.4x | **1.1x** | 1.0x | **10** |
| 1000 | 20.0% | 2.4x | 1.5x | **1.1x** | 0.9x | **10** |
| 3000 | 15.8% | 1.8x | **1.1x** | 0.8x | 0.6x | **5** |
| 6160 | 12.8% | **1.3x** | 0.8x | 0.5x | 0.4x | **0** |

The rule is the smallest guard that brings farthest-point sampling to the neutral reference. It
falls as the budget grows because a budget that must take half the pool cannot be choosy.

---

## E13.1 — same training-set size, new sampling

The only factor that moves is how the pairs were chosen. Both arms are paired under the same
class-restricted rules, at the same cell, with the same number of training pairs.

### E13.1-A — baseline, the n-shot scheme

| | |
| --- | --- |
| selection | the existing `cls_selection_100`, uniform 100 per class and sheet |
| cell / scope | `sheet` |
| passes | **6,160** |
| trainable pairs | **6,158** |
| tiles touched | 4,859 |
| groups covered | 88 / 88 |
| density vs neutral | 1.0x |
| training `data_info` | `sample_pair_cls_selection_100_cls_selection_100_sheet` |
| test `data_info` | `sample_pair_cls_selection_100_test_samples_sheet` |

48 of its 6,160 pairs fall outside the candidate pool — the class sits below the foreground
threshold on that tile, or the tile is not in `train_samples`. They are kept, because this arm is
the scheme as it exists rather than a reconstruction of it.

### E13.1-B — proposed, farthest-point

| | |
| --- | --- |
| selection | `--strategy fps --cell sheet --balance cell-class --density-guard 0 --budget 6160` |
| cell / scope | `sheet` |
| passes | **6,160** |
| trainable pairs | **6,158** |
| tiles touched | 4,778 |
| groups covered | 88 / 88, quota 70 per group |
| density vs neutral | 1.3x |
| training `data_info` | `sample_pair_e13x1_fps_e13x1_fps_sheet` |
| test `data_info` | `sample_pair_e13x1_fps_test_samples_sheet` |

**Read-out.** Per-class IoU and the mean, A against B. A difference here is attributable to the
selection alone: same cell, same pair count, same pairing rules, same recipe. If B does not win,
the later experiments are not worth running.

---

## E13.2 — budget

Sampling and cell fixed at whichever arm of E13.1 wins; `cell=sheet` throughout. The guard varies
per rung by the table above, so what the ladder holds fixed is *neutral density*, not the guard.
Stated in advance so it is not a choice made after seeing the curve.

| run | budget | guard | quota per group | trainable pairs | groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| E13.2-300 | 300 | 10 | 3 | 298 | 88 / 88 |
| E13.2-1000 | 1000 | 10 | 11 | 998 | 88 / 88 |
| E13.2-3000 | 3000 | 5 | 34 | 2,998 | 88 / 88 |
| E13.2-6160 | 6160 | 0 | 70 | 6,158 | 88 / 88 |

E13.2-6160 is E13.1-B; it is not run twice. **Read-out:** mIoU against passes, with the E13.1-A
baseline plotted as a single point at 6,160. The claim is about how far left the curve stays
flat.

---

## E13.3 — cell

Budget fixed at 1000, sampling at the E13.1 winner, guard 10.

| run | `--cell-gap` | pools | floor | trainable pairs | pools coherent? |
| --- | ---: | ---: | ---: | ---: | --- |
| E13.3-gap0 | 0 | 26 | 88 | 998 | trivially — one sheet each |
| E13.3-gap0.03 | 0.03 | 13 | 38 | 1,000 | yes |
| E13.3-gap0.05 | 0.05 | 12 | 35 | 1,000 | yes |
| E13.3-gap0.07 | 0.07 | 10 | 32 | 1,000 | yes — nothing incoherent has merged yet |
| **E13.3-gap0.09** | 0.09 | 6 | 19 | 1,000 | **no — deliberate failure case** |

E13.3-gap0 is E13.2-1000 and is not run twice, so this is four new runs.

The ladder is not evenly spaced, because the grouping does not fail gradually. Every merge up to
0.071 is sound — the Bayern stipple sheets, the two hachure plates, the 1959 blue pair, railway
with vineyard — and the break is a single link at **0.073**, where `3922_1982 + 3922_2005` chains
the 1970s–90s Hameln group to the 2000s group through one sheet. By 0.086 `3921_1973 + 3922_1898`
pulls in the 1898 sheet as well.

**0.09 is included on purpose as the failure case.** Its largest pool holds eleven sheets across
three conventions at once: fine black line work from 1898, red-and-black contour on white from
the 1970s, and green-and-pink area fills from the 2000s. It is the cheapest way to find out what
an incoherent pool actually costs. If mixing conventions inside a retrieval pool matters, the
drop from 0.07 to 0.09 shows it; if it does not, that is worth knowing too, because it would mean
the pool boundary is looser than the visual evidence suggests. Without it the sweep only tests
groupings that already look right, and could not tell "wider pools help" from "wider pools help
until they stop being one style". `results/pools_gap0.07.png` and `results/pools_gap0.09.png` are
the visual record.

---

## E13.4 — pairing

Same selection and the same pairing file as E13.2-1000. Only the training config changes, so no
data preparation is needed and the two runs are exactly matched on annotation.

| run | `src_selection` | meaning |
| --- | --- | --- |
| E13.4-topk | `topk` | the k nearest annotated tiles by DINOv2 cosine — retrieval on |
| E13.4-random | `random` | k drawn uniformly from the same cell — retrieval off |

E13.4-topk is E13.2-1000. **Read-out:** what retrieval is worth at a fixed annotation budget,
which is the E1 control repeated under a budget instead of the full training set.

---

## Iteration count

Not chosen by early stopping: the poly decay is a function of `max_iterations`, so a run halted
at k is not the run that would have been trained for k. Calibrated once, then frozen.

1. Two pilots at the ends of the ladder — B=300 and B=6160 — with `--lr-scheduler constant` so
   the curve measures convergence rather than the schedule, validating every 2,000 iterations
   against the held-out 10% and writing `val_curve.txt`.
2. From each curve take the smallest iteration count within 0.5 mIoU of its peak, fit the
   simplest rule through the two points, and use it for every run.
3. Restore `--lr-scheduler exp` so each run anneals over its own length.

Until the pilots are read, all runs use `max_iterations=100000`, which is provisional. At batch 8
that is 800,000 samples: 2,667 epochs over 300 pairs but 130 over 6,160, and closing that gap is
what the calibration is for.

## Execution

```
bash syntra/tools/e13_schedule.sh prepare   # selections and pairings, CPU only
bash syntra/tools/e13_schedule.sh pilot     # calibration
bash syntra/tools/e13_schedule.sh ladder    # the runs, then collect_results
```

Selections and pairings are written under `/home/yuan/data/syntra_e13`, never into the dataset.
Data tools run in the `hismap` env (needs scikit-learn), training in `syntra` (needs torch).
