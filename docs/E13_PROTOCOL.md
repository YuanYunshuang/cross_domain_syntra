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

| cell | groups | (cell, class) combinations = floor cost |
| --- | ---: | ---: |
| `sheet` | 26 | **88** |
| `style` | 12 | 35 |
| `era` | 13 | 43 |
| `corpus` | 3 | 9 |
| `global` | 1 | 6 |

`era` chains survey years less than five apart within one corpus: `bayern.1910`,
`bayern.1938-1941`, `bayern.1959`, `hameln.1898`, `hameln.1973-1975`, `hameln.1982`,
`hameln.1996`, `hameln.2002-2005`, `hameln.2012`, `hameln.2017`, `hameln.2023`, plus `railway`
and `vineyard`, which carry no year and stand alone.

`style` measures the grouping instead of declaring it — single-link over sheet centroids at 0.05
cosine, within a corpus — because the two archives behave oppositely. In Hameln, sheets of the
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

`7230_1959` and `7330_1959` fall just outside the threshold at 0.055 and stay separate; the
threshold is exposed if that boundary should move.

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

| run | cell / scope | groups | floor | trainable pairs |
| --- | --- | ---: | ---: | ---: |
| E13.3-sheet | `sheet` | 88 | 88 | 998 |
| E13.3-style | `style` | 35 | 35 | 1,000 |
| E13.3-era | `era` | 43 | 43 | 1,000 |
| E13.3-corpus | `corpus` | 9 | 9 | 1,000 |

E13.3-sheet is E13.2-1000; it is not run twice. **Read-out:** whether widening the pool helps or
hurts, between the two points §3c of the old plan already has — sheet-scope 82.38 against
global-scope 81.12 on an identical mixture. Note the confound to report rather than hide: a wider
cell also means a lower floor and more exemplars per target, so this measures the cell as a
package, not the scope in isolation.

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
