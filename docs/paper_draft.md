# SYNTRA — draft

Prose draft. Numbers are wired to the result files noted in `PAPER_OUTLINE.md`; every figure
quoted here traces to `results/*.csv`. Placeholders are marked **[TODO]**.

---

## 1. Introduction

Historical topographic maps are among the richest records of how landscapes changed before aerial
survey, but using them quantitatively means converting cartographic symbols back into land-cover
classes — and that conversion is manual. A national archive holds thousands of sheets, each
drawn in the conventions of its era, and a model trained on one decade's symbology transfers
poorly to another's. The binding constraint on map segmentation is therefore not model capacity
but annotation: somebody has to digitise enough of each style for a model to learn it.

Recent work adapts foundation segmentation models to this setting by conditioning on a labelled
exemplar. SMOL-MapSeg fine-tunes SAM with a low-rank adapter and prompts it with one annotated
tile drawn from the same sheet, which lets a single set of weights serve many cartographic styles.
The exemplar tells the model what the target class looks like *in this map's conventions*, so
style transfer becomes a matter of supplying the right example rather than retraining. But the
exemplar is drawn at random from whatever has been annotated, and the annotation budget is
accounted per class: labelling water and forest on one tile counts as two units of work.

We keep that architecture unchanged and change only what the model is shown. Two things follow.
First, exemplars are **retrieved** rather than drawn at random: a DINOv2 index, built once
offline, supplies the nearest annotated neighbours of each target tile, so inference remains a
lookup. Second, the budget is counted in **tiles, not class-passes** — an annotator who opens a
tile digitises everything on it, so a tile carrying forest and water costs one unit and yields
supervision for both. On our corpus a bought tile yields 2.9 training pairs on average, and the
classes it does *not* carry come back as negatives at no cost.

The consequence is a large reduction in how much labelling is needed. On a 27-sheet German and
Swiss corpus of 26,017 tiles, our method trained on **1,000 tiles reaches 83.50 mIoU**, above a
matched SMOL baseline trained on 5,087 tiles (81.09) and above the same baseline trained on the
full 14,675-tile split (82.57) — **a fifteenth of the annotation**, with every seed of ours above
every seed of the baseline. The same mechanism adapts to a survey year the model never saw:
given **ten annotated tiles** of that year and no gradient step, it comes within 3–7 IoU per class
of a model trained on it.

We also report what did *not* work, because two of those results change how the method should be
used. The number of exemplars must match between training and inference — supplying more at test
time than a model trained with *lowers* accuracy, monotonically. And the positive-to-negative
ratio of training samples, a detail neither method exposes as a hyperparameter, moves accuracy by
1.72 mIoU, more than the retrieval mechanism itself; the baseline's published setting is the worst
of the four we measured, and tuning *it* rather than ours cost us 1.7 points of apparent margin.

**Contributions.**

1. **Annotation efficiency.** Matching a fine-tuned baseline's full-data accuracy with a fifteenth
   of the annotation, and exceeding it at equal budget with per-seed separation.
2. **Training-free adaptation to a new survey year.** Ten annotated tiles, no gradient step,
   within 3–7 IoU per class of a model trained on that year.
3. **Tile-denominated budgeting.** Counting annotation by tile rather than by class-pass both
   matches how digitising is commissioned and yields 2.9× the supervision per unit.
4. **A decomposition, with negative results.** We separate the contributions of data organisation,
   exemplar count and retrieval, and report where each fails to help.

---

## 4. Annotation efficiency

**[Figure 1: budget curve — mIoU against annotated tiles, log-x, with the two baseline levels as
horizontal rules and seed ranges as error bars at 1,000.]**

We train at 300, 1,000 and 3,000 tiles drawn by corpus-class balanced random selection, holding
everything else at the configuration of §2 and scoring each model on the full 37,935-pair test set
with exemplars drawn from its own budget.

| annotated tiles | mIoU |
| ---: | ---: |
| 300 | 76.90 |
| 1,000 | **83.50** (82.96 / 83.29 / 84.24) |
| 3,000 | 85.78 |
| 5,102 | 86.09 |
| SMOL, matched, 5,087 | 81.09 (79.55 / 81.36 / 82.37) |
| SMOL, full split, 14,675 | 82.57 |

At 1,000 tiles the method exceeds the baseline trained on **five times** as much annotation, and
does so with complete separation: the lowest of our three seeds (82.96) is above the highest of
the baseline's three (82.37). It also exceeds the baseline trained on the entire 14,675-tile
split, a **14.7× reduction**. We report the mean difference at 5,087-equivalent budget as +2.40
with a pooled standard error of 0.91 (t = 2.64), which does not reach significance at three seeds
per arm; the claim rests on the separation of the seed ranges and on the monotonicity of the
curve, not on that test.

Two properties of the curve are worth noting. It is **steep below 1,000 tiles and flat above
3,000**: 3,000 → 5,102 tiles adds 0.31, inside run-to-run noise. So the operating point that
matters is in the low hundreds, which is also where run-to-run variance is smallest (sd 0.66 at
1,000 against 1.95 at 5,087). Second, *how* a budget is allocated matters as well as its size.
At the same ~5,100 tiles, the per-sheet-per-class n-shot selection the baseline uses scores 83.97
against 86.09 for corpus-class balancing. The n-shot selection gives the two single-sheet classes
ten tiles each rather than a ninth of the budget. **[TODO: the two runs also differ in sampler
composition (18 vs 16 groups, 50% vs 56% positives). Either state that or match it before
attributing the 2.1 points to allocation.]**

---

## 5. Adapting to a new survey year

**[Figure 2: per-class IoU against exemplar-pool size, held-out year, with the trained-on model as
a dashed ceiling.]**

**[TODO: draft prose. Data in `results/e5_unseen_sheet.csv`. Key points: ten tiles suffice, 10→100
changes nothing, gap to trained-on is 3–7 per class, n=1 degenerate, retrieved ≈ random and why.]**

---

## 6. Where the gain comes from

**[TODO: draft prose. Data in `results/ablation_all_results.csv` and
`results/per_class_results.csv`.]**
