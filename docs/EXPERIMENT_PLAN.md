# SYNTRA — experiment design and manuscript plan

## 0. Objective

**Given a pool of unannotated map datasets, decide what to annotate so that one model trained on
it segments all of them as accurately as possible, for the fewest annotation passes.** Everything
below is judged against that: a change earns its place if it raises accuracy at a fixed
annotation budget, or holds accuracy at a smaller one.

The deliverable is therefore an **annotation policy**, not a checkpoint. A practitioner arrives
with a stack of sheets and a fixed number of hours; the answer has to be a rule for choosing the
next thing to label, defensible before any of it is labelled. That the rule is computed from
DINOv2 embeddings of unannotated tiles is what makes it usable - it needs the images, not the
ground truth.

The pool being *heterogeneous* is the whole difficulty. It spans cartographic traditions with
different styles and different class vocabularies - Hameln, Bayern and Siegfried separate almost
cleanly in the embedding, and Siegfried shares no class with the other two - so a policy that
merely picks diverse-looking tiles will spend the budget wherever the pool happens to be densest
or strangest. Diversity has to be sought *within* each corpus and class and balanced across
them, which is what section 3d measures.

**The unit is one annotation pass: one class digitised on one tile.** Not one tile. A tile
carrying forest and water costs two passes, because it is two turns round the sheet for the
annotator and two training samples for the model - `cd_syntra2.py` builds each sample as a
binary mask for the class it was drawn for, with every other class's pixels as negatives.
Counting tiles instead makes the second class on a tile free, which quietly inflates any budget:
on the E13 selection at B=300 it would hand the training set 134 class-instances nobody paid
for, about a third again on top of the stated budget.

Retrieval is *one mechanism* for that, not the goal. It matters here because it lets a single
annotated tile serve many targets - at k=1 choosing the exemplar by similarity is worth +0.6 to
+8.1 mIoU depending on how heterogeneous the pool is (E9) - i.e. the same label works harder.
Where retrieval stops paying, it should be dropped rather than defended: global-pool *training*
buys 1.26 mIoU less than subset-pool training on the identical mixture (82.38 vs 81.12, section
3c), so it is not part of the answer.

**Two budgets, kept separate.** They trade off differently and must not be conflated:

1. *Training budget* - passes needed to train the model once. Current evidence: SYNTRA at
   100-shot reaches 85.89 mIoU where SMOL-Uni on the full training set reaches 82.6, so fewer
   labels already beat more.
2. *Adaptation budget* - passes needed to bring an existing model to a new sheet or year.
   Current evidence: the unseen 1960 sheet reaches ~60 mIoU from a pool of 10-100 annotated
   tiles with no gradient step.

**Numbers quoted in tiles have not all been re-counted.** The unit changed on 2026-08-03, and
only the E13 selection figures are stated in passes. Where an older result says "n-shot" or "a
pool of n tiles" it still means tiles, and converting it is not a multiplication - it depends on
how many classes those particular tiles carry. `cls_selection_100`, the n-shot baseline, is
**6,238 passes over 4,887 tiles** across the 25 hismap subsets, so the ratio there is 1.28. Any
figure that puts a budget on the x-axis has to fix one unit and say which.

**Priorities that follow.** E10 (budget curve per style gap) and E11 (which tiles to annotate, at
a fixed budget) become the main experiments rather than ablations; E12 turns "annotate more" into
a rule for *which tile to annotate next*, closing the loop with the active-learning GUI already in
the repository. The comparison against MapSAM2 becomes an equal-budget curve - they fine-tune per
sheet, we add tiles to a pool - rather than a comparison of headline IoU.

Training breadth is the other lever on the same goal: a model that has seen more cartographic
traditions should degrade more gracefully on an unseen style, which is testable by rerunning the
E10 ladder with Siegfried (384-native build) in training and comparing per-subset IoU against the
measured style gap, not pooled mIoU.

Status: draft, revised 2026-08-03. Scope decided with the author: **historical maps**, centred on
**annotation-efficient adaptation**, benchmarked against **SMOL-MapSeg** (accepted) and
**MapSAM2** (arXiv 2510.27547).

Assumption still open: target venue (planned for an ISPRS-Journal-style scope, cuttable to a
conference submission). Compute: **RTX 4090** plus an **RTX 3090 desktop**; see §4 for the
measured training recipe and the schedule that follows from it.

---

## 1. Positioning

| | Adaptation to a new sheet / year | Supervision at adaptation time | Weakness we exploit |
| --- | --- | --- | --- |
| **SMOL-MapSeg** | one hand-picked legend prompt (image + mask) | 1 annotated tile | a single exemplar cannot cover style variation inside one sheet; the user must choose it well |
| **MapSAM2** | few-shot fine-tuning; memory attention over tiles/time series | n annotated tiles **+ a training run** | adaptation requires gradient updates; context is implicit and not inspectable |
| **SYNTRA** | drop n annotated tiles into the retrieval pool | n annotated tiles, **no gradient update** | — |

**One-sentence claim.** Explicit content-based retrieval of annotated exemplars turns
adaptation to an unseen map sheet or year into a *data* operation instead of a *training*
operation, at equal or lower annotation cost than fine-tuning.

### Hypotheses

- **H1 (mechanism).** The gain over SMOL-MapSeg comes from *retrieval*, not from seeing more
  images: top-k retrieved exemplars beat k randomly drawn exemplars from the same pool, and
  k = 1 random reproduces SMOL-MapSeg.
- **H2 (efficiency).** At a fixed annotation budget of n tiles, pool-only adaptation matches or
  beats few-shot fine-tuning (MapSAM2-style) without any training.
- **H3 (temporal).** Explicit retrieval across years handles style drift on a fixed footprint at
  least as well as video-memory propagation, and degrades gracefully over long time gaps.
- **H4 (curation).** *Which* tiles enter the pool matters more than how many; cluster-based
  selection beats random selection at small n.

Existing evidence: SYNTRA 100-shot = 85.89 mIoU vs SMOL-Uni all-data = 82.6 / 100-shot = 81.8;
unseen 1960 sheet with pool only = 58.4 / 60.0 / 60.6 for pools of 10 / 50 / 100 tiles.

---

## 2. Benchmark

**Main data.** The 17 Hameln + 8 Donauwörth sheets and Siegfried RW/VY already used by
SMOL-MapSeg — identical splits, so every SMOL number is directly comparable without rerunning.

**Multi-temporal benchmark (new, and the direct answer to MapSAM2).** All Hameln subsets share
an identical tile grid (900 tiles, classes WL/GL/SM/WT), so the same ground location can be
traced across years:

| Footprint | Years | Span |
| --- | --- | --- |
| 3922 | 1898, 1974, 1982, 1996, 2005, 2017 | 119 a |
| 3821 | 1974, 2002, 2012, 2023 | 49 a |
| 3921 | 1973, 2002, 2017, 2023 | 50 a |
| 3822 | 1975, 2005, 2023 | 48 a |

Protocol: train on a subset of years, adapt to a held-out year with a pool of n tiles, evaluate
per-year IoU **and** temporal consistency of the predicted class maps on co-registered tiles.

**To verify before relying on it:** that the tile grids are geometrically co-registered and not
just identically named. Check a handful of tiles across years visually; if they are only
grid-aligned per sheet crop, restrict the temporal claim to per-year IoU and drop the
change-consistency metric.

---

## 3. Experiment matrix

~22 training runs, split over the 4090 and the 3090 (schedule in §4): **~5 days wall-clock**
instead of ~7 on the 4090 alone. Priority P0 = the paper does not exist without it,
P1 = a reviewer will demand it, P2 = fills out the ablation section.

| # | Experiment | Tests | Runs | Prio | Existing |
| --- | --- | --- | ---: | --- | --- |
| **E0** | Consolidation: freeze model + config family, results collector, re-evaluate old checkpoints | — | 0 | P0 | — |
| **E1** | Retrieval ablation: retrieved vs random sources, k ∈ {1,3,5,7}, train-k ≠ test-k | H1 | 6 | P0 | none |
| **E2** | Retrieval encoder: DINOv2 vs SAM features vs colour histogram | H1 | 3 | P2 | none |
| **E3** | Source fusion: max (current) vs mean vs attention | H1 | 2 | P2 | none |
| **E4** | Main table: SYNTRA vs SMOL-Uni/-Ind vs MapSAM/MapSAM2, all / 100 / 50 / 10-shot | H2 | 4 | P0 | maps partly done |
| **E5** | Unseen-sheet adaptation: pool size 0/1/10/50/100, no fine-tuning, vs SMOL (1 prompt) and MapSAM2 (fine-tuned) at equal budget | H2 | 0 (inference) | P0 | 1960 pilot |
| **E6** | Pool curation: random vs cluster-based vs uncertainty selection (uses `data_tools/indexing/clustering.py`, active-learning GUI) | H4 | 3 | P1 | none |
| **E7** | Multi-temporal: held-out year, per-year IoU + temporal consistency, long vs short gap | H3 | 2 | P1 | none |
| **E8** | Backbone: SAM ViT-B vs SAM2 Hiera + high-res decoder | fairness vs MapSAM2 | 2 | P1 | `SyntraModel` unused |

**E5 is the money figure**: x = annotated tiles spent on the new sheet, y = IoU, three curves
(SYNTRA pool-only, SMOL single prompt, MapSAM2 fine-tuned). If SYNTRA's curve sits above the
others at every budget, the paper writes itself.

**E8 matters more than it looks.** MapSAM2 is SAM2-based; without it, a reviewer cannot tell
whether our gains come from retrieval or from the backbone.

### Baselines, practically

- **SMOL-MapSeg** — code and checkpoints in hand, identical splits. Rerun directly; airtight.
- **MapSAM2** — code release was promised but must be confirmed. If it is unavailable, compare
  on the published MapSAM benchmark with their reported numbers and state the limitation
  explicitly; do **not** re-implement it silently.
- **MapSAM** — numbers already reported in the SMOL-MapSeg paper, reusable.

---

## 3b. Cross-year retrieval and annotation budget (E9-E12)

Added after the E1 inference sweep (results in `results/e1_sweep_natural_prevalence.csv`), which
showed that **retrieval pays for itself in precision, not recall**: at the natural
positive/negative ratio of a map sheet, `topk` beats `random` by 1.4-1.6 mIoU at every k and
roughly halves the false-positive rate (0.119% vs 0.195% of pixels at k=1). It also showed that
`max` fusion degrades as k grows (86.07 → 84.32 for k = 1 → 7) because each additional source is
another chance to fire on a tile that does not contain the class, while `mean` fusion does not.
Best measured configuration: `mean` + `topk` + k=3, at 86.35 mIoU and the lowest false-positive
rate of all 18 combinations.

The pool so far is built per (subset, class), i.e. within one map sheet of one year, where the
style is already homogeneous and retrieval has little left to do. The experiments below open the
pool across years and sheets, which is both the harder and the realistic setting.

**Why this is cheap.** The pool is an inference-time object. One trained model supports
arbitrarily many pool variations - which years are in it, how large it is, how it was selected -
at minutes per evaluation. That is also the claim itself: adapting to a new sheet must not
require a gradient step.

**What already exists.** Every subset carries a DINOv2 ViT-S/14 index (`train_samples.faiss`,
384-d, `IndexFlatL2`, reconstructable) plus its name map, so the embeddings for all 15,300
training tiles are available without recomputation. Test tiles are not indexed yet (~12,000
tiles, minutes on the 4090).

| # | Experiment | Tests | Training runs |
| --- | --- | --- | ---: |
| **E9** | Pool scope: subset-local vs same-footprint-other-years vs global-all-sheets vs random-global, for targets of seen and unseen years | does retrieval matter once the pool is stylistically heterogeneous | 0 |
| **E10** | Held-out subsets along a measured style-gap ladder (0.11 to 0.23), each adapted with a pool of n annotated tiles and no gradient step | H2 and H3: what a new cartographic style costs in annotations | 1 |
| **E11** | Annotation budget: fixed total budget B ∈ {50, 100, 300, 1000} tiles, selected by per-sheet uniform / random / k-center greedy in DINOv2 space / k-means centroids | H4, and the user's diverse-selection hypothesis | 0 for pool, 2 for training-set selection |
| **E12** | Coverage diagnostic: distance to the nearest pool exemplar vs per-tile IoU | turns "annotate more" into a measurable criterion; feeds the active-learning loop | 0 |

The style gap used throughout is one minus the cosine similarity between a subset's DINOv2 tile
embeddings and their nearest neighbours in any other subset. Measured over the 25 subsets it
ranges from 0.00 to 0.234, and it doubles as the x-axis of E12.

### E10 retired (2026-08-03)

The style-gap ladder held four subsets out of training and then asked the model to adapt to them
from a pool. It was run: adaptation tracks the style gap almost perfectly (gap 0.109 -> 86.9
mIoU, 0.140 -> 69.9, 0.181 -> 36.0, 0.234 -> 43.4) but the budget curve is **flat** - n=1 and
n=50 pool tiles are within a point of each other on every sheet, and bayern.7330_1959 degrades
as the pool grows (69.9 -> 62.6).

The design is the reason it is uninformative rather than the method: holding out four subsets
removed exactly the styles the model was then asked to generalize to, so the numbers measure a
deliberately impoverished training distribution. It is replaced by E13 below, which asks the
question the objective actually poses.

### Superseded: the ladder design

An earlier draft held out the entire 3922 footprint, which would have staked the headline on
1898, the oldest and most distinctive sheet. With ~15k training tiles that is a bet the model is
unlikely to win, and a failed zero-shot number teaches little.

Instead, the held-out subsets are chosen along the **measured** style gap - one minus the cosine
similarity between a subset's DINOv2 tile embeddings and their nearest neighbours anywhere else
in the corpus. The gap spans 0.00 to 0.234 across the 25 subsets, so difficulty becomes a
continuous design variable rather than a guess:

| Held out | Style gap | Nearest other subset | Expectation |
| --- | ---: | --- | --- |
| `hameln.3822_2005` | 0.109 | `hameln.3821_2002` | well covered by other modern sheets; zero-shot should hold |
| `bayern.7330_1959` | 0.140 | `hameln.3921_1973` | mid-century, partly covered |
| `hameln.3922_1898` | 0.181 | `hameln.3922_1974` | the 19th-century stress case |
| `bayern.519_1910` | 0.234 | `bayern.490_1910` | the actual extreme of the corpus |

One training run excludes all four. Each held-out subset is then evaluated with a pool of
n ∈ {0, 1, 5, 10, 50} annotated tiles of that subset, **without any gradient step**.

The primary result is IoU against annotation budget, one curve per gap level, which answers the
question the paper is really about: *how many annotated tiles does a new cartographic style
cost?* The expectation that a model trained on limited data will not generalize zero-shot across
a large style gap is then a measured threshold - the gap beyond which n=0 fails and the budget
at which it recovers - rather than a failed experiment.

The temporal comparison survives at no extra training cost: with 1898 held out, restricting the
pool to each other 3922 year in turn (1974, 1982, 1996, 2005, 2017) gives a year-to-year
transfer row, and the same for the other held-out subsets.

Baselines at equal annotation budget: SMOL-MapSeg with one hand-picked legend (k=1) and, where
the code is available, MapSAM2 with few-shot fine-tuning.

**Constraint from the data.** `hameln.3921_2017` and `hameln.3921_2023` are the same scan under
two year labels (see `DATA_ISSUE_hameln_3921_duplicate.md`). They must be treated as one unit in
every split - holding out one while the other trains is not a hold-out - and one of the two
should be dropped from the experiments entirely.

### E11 in detail - budget allocation

Two knobs that must be separated, because they answer different questions:

1. **Pool composition** (inference only): the model is fixed, B annotated tiles form the pool.
2. **Training-set selection** (needs training): B annotated tiles are all that the model trains on.

Strategies: per-sheet uniform (the current n-shot scheme), random global, k-center greedy
(farthest-point sampling in DINOv2 space, maximising coverage), and k-means centroids.

Expected failure mode to guard against: farthest-point sampling maximises spread and therefore
gravitates to outliers - sheet borders, blank tiles, scanning artefacts. Restrict the candidates
to tiles that actually contain the class above the foreground threshold, and report k-means
centroids next to it, which trade coverage for typicality.

### Implementation steps

1. `build_global_index.py` - merge the per-subset faiss indices and name maps into one
   `{name: vector}` store; extend it to test tiles.
2. `make_pool_pairing.py` - write `sample_pair_*`-format files with a configurable scope
   (`subset` | `footprint` | `global`) and a class restriction. **No dataset change is needed**:
   the dataset resolves each source name to its own subset folder, so cross-subset sources
   already work.
3. `select_budget.py` - k-center greedy / k-means / random / per-sheet-uniform selection over
   the embedding store, writing a `cls_selection_*.json`-compatible file.
4. `exclude_keys` in the dataset, so leave-years-out is a config field rather than a hand-edited
   subset list (`filtering_keys` only includes).

### Protocol note

All of the above is evaluated at **natural prevalence** - every class keeps its true
positive/negative tile ratio, as a sliding window over a whole sheet meets it. Balanced sampling
understates false positives and, as the E1 sweep showed, can hide the retrieval effect entirely.

## 3d. E13 - training-sample selection under a fixed budget

**The question.** Given a total budget of N annotation passes - N classes digitised, each on
some tile - which N should be labelled so that one model trained on them performs best across
*all* test samples of *all* subsets? No sheet is held out; every subset is available to select
from and every subset is evaluated.

This is the objective of section 0 stated directly, and it is the training budget rather than the
adaptation budget. The retrieval pool is held fixed at subset scope throughout, so selection is
the only variable.

**Budgets. The "roughly 2,900 tiles" written here was wrong, whatever unit it meant.** Counted
from the files, `cls_selection_100` is uniform per class and subset and comes to **6,238 passes
over 4,887 tiles** across the 25 hismap subsets (Hameln 5,170 over 4,055; Bayern 1,068 over 832;
Siegfried adds 200 more of each). No scope of the corpus produces 2,900 of anything, so the
figure was an estimate that was never checked, and 300 / 1,000 / 3,000 does not bracket the
baseline - 3,000 is under half of it.

That does not by itself condemn the ladder: the interesting claim is that a well-chosen few
hundred passes matches a uniform 6,238, and 300 / 1,000 / 3,000 spans the region where that
either holds or fails. But it has to be stated as "does a well-chosen 300 match a uniform
6,238", a 21x reduction rather than the 10x implied before, and any figure with the baseline on
it must plot it at 6,238. Whether to add a fourth budget nearer the baseline is open.

**Strategies**, all implemented in `data_tools/indexing/select_budget.py`:

| strategy | idea | risk |
| --- | --- | --- |
| uniform | n per class and subset - the current scheme | spends the budget evenly however redundant a sheet is |
| random | uniform over all annotated tiles | over-samples the sheets with most tiles |
| kcenter | farthest-point in DINOv2 space, maximal coverage | chases outliers; already guarded by requiring an annotated class |
| kmeans | centroids replaced by their nearest real tile | favours typicality over coverage |

**Class balance is the open design question.** Coverage-driven selection is class-blind, so a
minority class - Hameln water at 3% of foreground - can be starved at small budgets, and the
mean IoU would then be dominated by a class the budget simply skipped. Each strategy therefore
needs a class-aware variant that allocates the budget per class first and applies the strategy
within it. Report both; the difference is itself a result.

**Resolved as `--balance`, and it is about coverage, not only class counts (2026-08-03).** The
budget is split into equal quotas over cells - `class`, `corpus`, `corpus-class` or
`subset-class` - and the strategy runs *inside* each cell. Quotas are filled scarcest cell
first; a tile is charged once but counts towards every cell it belongs to, so later quotas come
partly free, and the remainder is spent over the whole pool so every setting lands on exactly B
tiles. `--balance subset-class --strategy uniform` is the classic n-per-class-and-subset scheme.

The reason this matters is not the head count. **Farthest-point sampling run over the whole pool
covers the *union* well and each part badly**: a single cell's picks are only wherever that cell
happens to meet the global cover. Over hismap+siegfried at B=300, plain k-center left Siegfried
vineyard with 7 tiles of 499 and Bayern forest with 10 of 330, both bunched in one corner of
their own cloud. Running the same walk inside each (corpus, class) cell - seeded by the tiles
that cell already owns, so the quota extends its own coverage rather than its neighbours' - is
what spreads them:

| cell | candidates | plain: n, worst | corpus-class: n, worst |
| --- | ---: | ---: | ---: |
| siegfried/railway | 4423 | 89, 0.330 | 41, 0.359 |
| hameln/grass | 2313 | 61, 0.382 | 56, **0.348** |
| hameln/forest | 2749 | 37, 0.448 | 45, **0.311** |
| hameln/settlement | 1399 | 31, 0.388 | 48, **0.296** |
| bayern/grass | 715 | 81, 0.401 | 74, **0.362** |
| siegfried/vineyard | 499 | 7, 0.412 | 42, **0.215** |
| hameln/water | 371 | 27, 0.389 | 50, **0.290** |
| bayern/forest | 330 | 10, 0.422 | 43, **0.209** |
| bayern/water | 119 | 25, 0.369 | 46, **0.245** |

Worst cell covering radius 0.448 → 0.362, mean over cells 0.393 → 0.293, at the same 300 tiles.
Only railway gives ground, and it is the cell that was over-served. The union is slightly worse
served in exchange - global covering radius 0.301 → 0.350 - which is the trade being made on
purpose: no cell is allowed to be poorly covered so that the corpus as a whole can look tidy.
`--balance corpus-class` is therefore the setting E13 should train, with plain k-center kept as
the control that shows what class- and corpus-blind coverage costs.

**The budget is counted in (tile, class) pairs, not tiles (2026-08-03).** One unit is one class
digitised on one tile - one annotation pass, and one training sample, since `cd_syntra2.py`
builds each sample as a binary mask for the class it was drawn for, with every other class's
pixels as negatives (only label 255 is ignored). A tile carrying forest and water therefore
costs two units if both are wanted, and is written out only under the classes actually bought.

Counting tiles instead made the second class on an already-chosen tile free, which it is not.
It also forced an awkward correction: cells overlapped, a cell could find its quota already
satisfied by tiles bought elsewhere and sample nothing of its own - bayern/grass got 0 picks at
B=300 - and a quarter of the budget fell through to a global top-up. Under the pair budget every
pair lands in exactly one cell, so the cells are disjoint, each simply buys its quota, and the
free-fill credit and its repair both disappear. At B=300 with `corpus-class` every cell now
takes exactly 33 or 34 pairs.

The cost is visible and intended: 300 pairs buy 291 tiles, and 134 class-instances present in
those tiles are left unwritten because the budget did not pay for them. Under the old accounting
those 134 were handed to training free of charge, which flattered every reported budget by
roughly a third.

**Farthest-point sampling buys the periphery, and that needs a guard (2026-08-03).** The
selected tiles sit visibly on the rim of every t-SNE cluster, and the measurement is blunt:
their local density - mean cosine distance to their 20 nearest neighbours - is 1.78x the pool
median, and **68% of the picks come from the sparsest tenth of the pool** (random 19%, k-means
16%). This is minimax working as specified, since the tile furthest from everything chosen is
by construction the most isolated one, but as a training set it spends two thirds of the budget
on tiles that stand for almost nothing else.

Two answers, both implemented, and they sit at different points on the same trade-off:

| B=300 pairs, balance corpus-class | median density | from sparsest 10% | worst cell | mean cell |
| --- | ---: | ---: | ---: | ---: |
| random | 0.1302 | 16% | 0.578 | 0.465 |
| kmeans | 0.1263 | 22% | 0.502 | 0.405 |
| fps-medoid | 0.1624 | 40% | 0.390 | 0.314 |
| fps | 0.2426 | **77%** | 0.383 | 0.316 |
| **fps + `--density-guard 15`** | 0.1620 | **24%** | **0.303** | **0.242** |

`fps-medoid` places by farthest-point and then moves each pick to the tile nearest the mean of
what it covers - k-medoids seeded by FPS. It halves the extremity for almost no coverage, but
the guard does better on both axes at once, so it earns no run of its own.

`--density-guard 15` is the answer: drop the sparsest 15% of each cell from candidacy and run
FPS as before. It is the only setting that improves coverage *and* typicality together - the
guard removes the pairs that were consuming the budget, not the ones worth annotating, so the
walk is no longer dragged outwards.

**The guard ranks density within each cell, not over the corpus.** A global threshold asks
whether a tile is unusual for the whole corpus, and the answer for a small distinctive cell is
"all of it": it took bayern/water from 119 candidates to 39 and bayern/grass to 61%, while
hameln/forest kept 96% - the cells that most needed their own outliers removed were the ones
stripped of their ordinary tiles instead. Ranking inside the cell asks whether a pair is
unusual *for its own class and corpus*, and every cell then keeps the same fraction:

| cell | candidates | global guard keeps | per-cell guard keeps |
| --- | ---: | ---: | ---: |
| siegfried/railway | 4423 | 4045 (91%) | 3980 (90%) |
| hameln/forest | 2749 | 2634 (96%) | 2474 (90%) |
| bayern/grass | 715 | 434 (61%) | 643 (90%) |
| hameln/water | 371 | 281 (76%) | 334 (90%) |
| bayern/water | 119 | **39 (33%)** | 107 (90%) |

Because the guard is per cell and the unit is a pair, a tile can stay a candidate for the class
it is typical of while being dropped for one it is not.

**What the guard is, exactly.** Vectors are L2-normalised, so cosine distance is
`d(a,b) = 1 - a·b`. For a cell `C` of (tile, class) pairs and each pair `i` in it, placed in the
embedding by its tile, with `k = min(20, |C|-1)`:

    rho_i  = mean over the k nearest neighbours of i *within C* of d(v_i, v_j)
    tau_C  = percentile_(100-g)( { rho_i : i in C } )
    keep     { i in C : rho_i <= tau_C }

It is a quantile cut, not a distance cut, which is why every cell retains exactly `100-g`
percent of its candidates whatever its density scale. Cells of fewer than 3 pairs are untouched.

**Why g = 15.** Not from the shape of the data: the density distribution has no knee. The ratio
`rho_q / rho_median` averaged over cells climbs smoothly - 1.18 at p70, 1.30 at p80, 1.49 at
p90, 1.67 at p95, 1.82 at p97 - and only accelerates past p99. There is no separate outlier
population to cut at, so no threshold is principled on the distribution alone.

The criterion that does work is a density-neutral reference: random selection under the same
`corpus-class` balancing puts **18% +- 2** of its picks in the globally-sparsest tenth of the
candidate pairs. Not 10%, because the balancing itself over-weights small cells that are
globally sparse. Farthest-point sampling against that reference, at B=300 pairs:

| guard | pairs cut | from sparsest 10% | vs neutral | worst cell | mean cell | removed per % cut |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0% | 77% | 4.3x | 0.383 | 0.316 | - |
| 3 | 3% | 56% | 3.1x | 0.361 | 0.286 | **6.7** |
| 5 | 5% | 45% | 2.5x | 0.343 | 0.273 | 6.3 |
| 10 | 10% | 28% | 1.5x | 0.319 | 0.254 | 4.9 |
| **15** | 15% | 24% | **1.3x** | 0.303 | 0.242 | 3.5 |
| 20 | 20% | 20% | 1.1x | 0.294 | 0.228 | 2.8 |

15 is the smallest guard that *neutralises* the density bias rather than merely reducing it.

**The pull is a small-budget problem, and the same guard covers both budgets.** At B=1000 the
reference is unchanged at 18% +- 1, but unguarded farthest-point starts from 57% rather than
77%: with 111 picks per cell instead of 33 the walk exhausts the extremes early and spends the
rest in the body of the distribution.

| guard | B=1000: from sparsest 10% | vs neutral | worst cell | mean cell |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 57% | 3.2x | 0.288 | 0.214 |
| 3 | 43% | 2.4x | 0.266 | 0.199 |
| 5 | 34% | 1.9x | 0.256 | 0.190 |
| 10 | 20% | 1.1x | 0.241 | 0.171 |
| **15** | 17% | **1.0x** | 0.228 | 0.161 |
| 20 | 15% | 0.9x | 0.217 | 0.152 |

B=1000 alone would be satisfied by g = 10 (1.1x). B=300 would not (1.5x). **g = 15 is the one
setting that meets the criterion at both**, at 1.3x and 1.0x, so runs 1 and 2 can share a single
selection rule and the budget curve measures budget rather than budget-and-guard. Fixing the
guard per budget instead would confound the two.

This also says where the guard earns its place: at small budgets, which is exactly where the
paper's claim is strongest.

**It was 10 under the old tile budget, and the move is real rather than a change of yardstick.**
The sparsest-10% threshold is 0.1924 measured over tiles and 0.1932 over pairs, so the same
selection scores the same either way; unguarded farthest-point went from 68% to 77% because
every cell now runs a full-length walk of its own into its own extremes instead of having part
of its quota handed to it by a neighbour under the old free-fill credit.

Efficiency alone - bias removed per percent of pairs cut - would instead pick g = 3, which peaks
at 6.7 against 3.5 at g = 15, but leaves the selection three times more outlier-heavy than an
unbiased sampler.

Both arguments assume the neutral profile is the right target, and that is an assumption, not a
result: a mild preference for atypical tiles might help a segmentation model rather than hurt
it. Run 6 below is what tests it. The flag defaults to 0, so E13 must pass it deliberately.

**The figures are laid out by t-SNE, because PCA was lying by omission.** Two principal
components hold ~25% of the variance here, and under that projection the selection looked full
of holes that were not there: tiles in 2D bins containing no selected tile sat a mean 0.189
(max 0.300) from their nearest annotated tile - indistinguishable from tiles in bins that did
contain one (0.191, max 0.301). t-SNE preserves neighbourhoods, so an empty region now means
the budget really did miss that part of the cell, and it resolves structure PCA smeared
together: Hameln separates into three style clusters, Bayern into one, Siegfried into a single
broad mass. The axes carry no scale in exchange, and distances between distant clusters are not
comparable - counts and covering radii must still come from the 384-d vectors, never the
picture. `--projection pca` keeps the old behaviour; layouts are cached under
`results/.projections`, ~40 s to build over 10.8k tiles and ~1 s per figure thereafter.

The plotting scripts need scikit-learn, which the `syntra` env does not have. Run them with the
`hismap` env (`/home/yuan/anaconda3/envs/hismap/bin/python`), which already carries sklearn
1.5.0 and matplotlib against numpy 1.24 - installing it into `syntra` risks moving numpy under
torch 2.5.1+cu118.

**The coverage numbers separate the two geometric strategies, and the mean hides it.** Over the
same pool at B=300, measuring the cosine distance from every pool tile to its nearest selected
tile:

| strategy | mean | worst (covering radius) | water tiles |
| --- | ---: | ---: | ---: |
| uniform | 0.130 | 0.417 | 35 |
| random | 0.130 | 0.504 | 24 |
| kcenter | 0.177 | **0.267** | 72 |
| kmeans | **0.119** | 0.429 | 31 |

k-center is the *worst* strategy on the mean and by far the best on the covering radius, which
is the quantity it actually minimises: it pays typicality to leave no corner of the corpus
unserved, and k-means does the reverse. That is the coverage-versus-typicality trade the plan
predicted, now measured rather than assumed, and it is why E13 must report both numbers - a
percentile would hide it, since the fringe is exactly where the difference lives.

`data_tools/indexing/plot_selection.py` draws the selections in the DINOv2 CLS space (PCA of
the candidate pool, one panel per strategy) with those numbers per panel. It imports the
selector rather than re-implementing it, so the figure shows the tiles that would actually be
written. In the k-center panel the densest cluster of the corpus is left almost untouched while
the sparse fringe is saturated, which is the outlier-chasing risk made visible.

**Cost.** 4 strategies x 3 budgets = 12 runs at ~10 h each is too much for the two GPUs.
Superseded by the pruned set in "E13 - what to actually run" below, which is 7 runs: the method
and the n-shot scheme at 300 and 1,000, a random floor, the density-guard ablation, and the
n-shot baseline at its own size. Evaluation is the full test set of every subset, identical for
all runs, and every run trains from scratch - no earlier checkpoint enters E13.

**What the result would say.** If a coverage-selected 300 passes matches a uniform 6,238, the
paper's headline is that most annotation in this corpus is redundant, and the diagnostic from
E12 tells a practitioner which tiles are worth their time.

### E13 - what to actually run

The selector grew several knobs while the questions above were being settled. This is the
decision, so it does not have to be reconstructed from the reasoning:

```
PYTHONPATH=. python data_tools/indexing/select_budget.py \
  -d /home/yuan/data/syntra --index /home/yuan/data/syntra/global_index.npz \
  --budget 300 --strategy fps --balance corpus-class --density-guard 15 \
  -dkw hismap,siegfried -xkw hameln.3921_2023
```

| flag | why this value |
| --- | --- |
| `--strategy fps` | minimises the covering radius, the only coverage quantity that can be measured |
| `--balance corpus-class` | without it vineyard gets 7 tiles of 499; with it ~40, worst cell 0.448 → 0.362 |
| `--density-guard 15` | without it 77% of the budget goes to the sparsest tenth of the pool; with it 24% |
| `--budget 300` | counted in (tile, class) pairs - one class digitised on one tile |
| `--seed 2026` | leave alone, it is what makes a selection reproducible |

`--projection`, `--perplexity` and `--cache-dir` belong to the plotting scripts and change no
selection. `fps-medoid` can be retired: it sits behind `fps` + guard on coverage and behind
`kmeans` on typicality, so it earns no run of its own.

**E13 is self-contained.** Every run is trained from scratch under the same recipe and evaluated
on the same test set, so nothing here rests on an earlier checkpoint. The 85.89 from the old
100-shot run is not a baseline for it: that run predates AMP, predates the pair accounting, and
would be comparing across two changes at once. Where E13 needs the n-shot scheme it trains it.

The runs that answer the question, within the two-GPU schedule below. Budgets are in annotation
passes, so B=300 buys ~291 tiles and 300 training samples:

| # | selection | budget | what it answers |
| --- | --- | ---: | --- |
| 1 | fps / corpus-class / guard 15 | 300 | the method |
| 2 | fps / corpus-class / guard 15 | 1000 | whether it scales with budget |
| 3 | uniform / subset-class / guard 0 | 300 | the current n-shot scheme at equal budget |
| 4 | uniform / subset-class / guard 0 | 1000 | same, higher budget |
| 5 | random / none / guard 0 | 300 | the floor |
| 6 | fps / corpus-class / **guard 0** | 300 | isolates the density guard |
| 7 | uniform / subset-class / guard 0 | **6238** | the n-shot baseline itself, re-run |

**Run 7 is what the headline claim is measured against** and was missing while the baseline was
believed to be a quotable number. "A well-chosen 300 matches a uniform 6,238" is a comparison
between two trained models, and with no earlier checkpoint in play the 6,238 model has to exist.
It is `cls_selection_100` exactly - the current scheme at its natural size - so run 7 doubles as
the recipe re-run that section 4 already asks for.

It is also nearly free. `max_iterations` is fixed at 100,000, so wall-clock is set by the
iteration count and not by how many samples the run selects from: run 7 costs the same ~9.9 h as
run 1.

**Which exposes a confound worth deciding before launching.** At batch 8, 100,000 iterations
draw 800,000 samples. Over B=300 that is 2,667 epochs; over B=6,238 it is 128. The small-budget
runs therefore see each annotation twenty times more often, so "budget" and "epochs over the
data" move together and a small-budget win could be either. Fixing iterations is the usual
compute-matched choice and is probably right, but it should be a stated decision rather than an
accident, and the B=300 runs need their overfitting checked on the test curve rather than
assumed away.

Run 6 is the one to protect if the schedule slips. Every coverage number says the guard helps,
but coverage is a proxy: the guard is a bet that outlier tiles are not worth their annotation
cost, and only training tests it. The retained-pool covering radius improves with the guard
while the full-pool one degrades - 0.362 → 0.468 including the tiles it discarded - and which
of those two matters is exactly what run 6 settles.

## 4. Training recipe and two-GPU schedule

### Measured recipe (RTX 4090, `syntra_maps_s100`, num_src=3, 384 px)

| batch | AMP | samples/s | peak GB | h per run |
| ---: | --- | ---: | ---: | ---: |
| 8 | off | 15.7 | 14.3 | 14.1 |
| 8 | **on** | **22.4** | 12.4 | **9.9** |
| 12 | off | 14.1 | 21.1 | 15.7 |
| 12 | on | 19.8 | 18.2 | 11.2 |
| 16 | either | OOM | | |

**Decision: batch 8 with AMP.** Mixed precision is worth +43% throughput and 1.9 GB, and is
free of any recipe risk beyond numerics. Larger batches are *slower per sample*: each training
sample carries `num_src` source tiles plus a target, so the encoder already processes
`batch × 4` = 32 images per iteration at batch 8 and the GPU is saturated; a larger batch only
adds memory pressure, and batch 16 does not fit at all.

**Correction to an earlier estimate.** The ~7.7 h per run used when this plan was first drafted
was read from `smol_hismap_n1_s100`, a SMOL run with `num_src: 1`, whose encoder sees half as
many images. A SYNTRA run costs **14.1 h in fp32 and 9.9 h with AMP**, so the schedule below
only holds because of AMP.

**Caveat.** AMP changes numerics, so the 85.89 baseline is not comparable to AMP runs and has to
be re-run under the new recipe. Training loss over 40 iterations is indistinguishable
(1.5147 fp32 vs 1.5267 AMP), which is not evidence about final quality.

`num_src` is also a throughput lever: k=1 would roughly halve the cost. E9 justifies keeping
k=3 for the global-pool setting, where k=3 beats k=1 (80.95 vs 80.01 mIoU).

### Schedule

Both cards have 24 GB, so the recipe carries over unchanged. The 3090 is roughly 1.5× slower per
iteration, so give it the runs that are either short or not on the critical path.

**Data to copy to the 3090 box** — 3.9 GB total, everything the map-scoped paper needs:

| | Size | |
| --- | ---: | --- |
| `hismap/` | 2.9 G | 25 sheets |
| `siegfried/railway` + `vineyard` | 0.9 G | **skip `siegfried/feats_np` (33 G)** — cached retrieval features, regenerate locally |
| `hameln_new/hameln_1960` | 102 M | unseen-sheet experiment |
| `fewshot/` | 16 M | new-class adaptation |

| Day | 4090 (critical path) | 3090 (parallel track) |
| --- | --- | --- |
| 0 | E0 consolidation (no GPU) | copy data, set up env, reproduce one known run to validate the box |
| 1 | **E1 retrieval ablation** (top-k vs random-k first — a negative result here rewrites the claim) | E8 backbone: SAM2 Hiera variant |
| 2 | E1 remainder (k sweep) | E8 remainder + E2 retrieval encoder |
| 3 | **E4 main comparison** (all / 100 / 50 / 10-shot) | E3 fusion + E6 pool curation |
| 4 | E4 remainder, then **E5** (inference only, hours not days) | E6 remainder |
| 5 | **E7 multi-temporal** | rerun whatever E1–E4 showed needs a seed repeat |

Rules that keep this honest: the 4090 always owns the P0 runs, so a 3090 failure costs an
ablation and not the paper; run E1's random-k control **before** committing the rest of the
matrix; keep both boxes on the same commit and log the commit hash per run.

Practical notes: the existing `training/scripts/train_test_*.sh` already parameterize `DATADIR`
per machine, and `train_test_kaliska.sh` shows the Docker path if the 3090 box is easier to set
up that way. Sync only `logs/<run>/config.yaml`, `log.txt`, `loss.txt` and `test_*.txt` back —
not the checkpoints, unless a run is needed for E5 inference.

## 5. Consolidation work before any training (E0)

1. Fix the model identity: the strong runs use `model_name: smol` with
   `dataset_name: cd_syntra2_efficient` — decide whether SYNTRA *is* SMOL + retrieval (then E8
   answers whether the SAM2 variant is worth keeping) and name it accordingly.
2. Retire `cd_syntra` and `cd_syntra2`; keep the efficient dataset only.
3. Collapse the v1–v4 config sprawl into one family with explicit `k`, retrieval-encoder and
   fusion switches, so E1–E3 are config flips rather than code edits.
4. `tools/collect_results.py`: parse every `test_*.txt` into one CSV keyed by
   (run, dataset, class) — the tables are currently assembled by hand.
5. Re-evaluate the existing medical / driving / trees checkpoints with the correct `-dkw`;
   several test files are empty or carry the wrong dataset name.

---

## 6. Manuscript

### Target venue

Journal, not necessarily ISPRS. The choice shifts emphasis but not the core matrix, so it can
wait until E1 and E5 are in — those results decide whether this is a top-tier paper or a solid
domain paper.

| Venue | Fit | What it demands | Note |
| --- | --- | --- | --- |
| **ISPRS J. Photogramm. Remote Sens.** | high | method + benchmark rigour, strong baselines (E1, E4, E8) | SMOL-MapSeg's venue; familiar reviewers are an asset only if SYNTRA's claim reads as clearly distinct — *training-free* adaptation — and not as SMOL + k exemplars |
| **GIScience & Remote Sensing** | high | direct comparison with MapSAM/MapSAM2 | **MapSAM's own venue** (Xia et al. 2025). Puts SYNTRA beside its competitor; editors and reviewers already know the benchmark. Reviewers may be the MapSAM authors — a risk and an opportunity |
| **IJGIS** | medium-high | the human-in-the-loop workflow and annotation cost | would promote E6 (pool curation, active learning) to a main result and reduce the demand for heavy CV ablations |
| **IEEE TGRS** | medium | exhaustive ablations, general RS relevance | historical maps sit slightly off-centre for the readership |
| **Int. J. Appl. Earth Obs. Geoinf.** | medium | applied value | sensible fallback, faster turnaround |

Recommendation: aim at **ISPRS J** if E5 shows the budget curve dominating both baselines, and
fall back to **GIScience & Remote Sensing** otherwise — the latter is the natural home for a
head-to-head with MapSAM2 and is a stronger placement than a diluted top-tier attempt.

Do not split the work across two papers to hit both venues; the SMOL/SYNTRA pair is already
close enough that a third increment would read as salami slicing.

### Writing order

Write in this order, since only the results sections depend on the runs:

1. **Now, in parallel with E1–E3:** method (retrieval, pooling, fusion), the multi-temporal
   benchmark description, and the experimental-setup section.
2. **After E1/E4/E5:** results and discussion.
3. **Last:** introduction and related work, once the numbers fix which claim leads.

Figures: (1) pipeline with the retrieval pool; (2) qualitative — target tile, its k retrieved
exemplars, prediction; (3) the E5 annotation-budget curve; (4) multi-temporal strip of one
footprint across years with predictions; (5) pool-curation comparison.

Tables: main comparison vs SMOL/MapSAM/MapSAM2; retrieval ablation; encoder + fusion ablation;
per-year temporal results.

**Action item outside this repo:** MapSAM2 (Oct 2025, same group as MapSAM, same benchmark)
should be cited in the SMOL-MapSeg paper if the proofs are still open.

---

## 7. Risks

| Risk | Mitigation |
| --- | --- |
| MapSAM2 code/dataset unavailable | compare on the MapSAM benchmark with published numbers, state it |
| Retrieval gain turns out to be an ensembling effect | E1 with random-k is exactly this control; if random-k matches top-k, pivot the claim to "multi-exemplar prompting" and report it honestly |
| Multi-temporal tiles not truly co-registered | drop the consistency metric, keep per-year IoU |
| Multi-notion (n4) mode underperforms | keep n1 as the main setting and report n4 as a limitation, or fix it first — do not quietly drop it |

---

## 3c. Global pools: negative result, and how retrieval would have to change

**Decision (2026-08-02): the experiments proceed with the subset-scope pool.** Global-pool
training was tried and did not pay off; the strategies below are parked until the subset-scope
line of experiments is finished.

### What was measured

A model trained on global-scope pairings (`syntra_maps_s100_global`, 100k iterations, 7 h 18 min,
mean fusion, topk, k=3, hismap only) evaluated against the subset-trained checkpoint on the same
2,552 target-class pairs at natural prevalence:

| pool scope at inference | subset-trained (E9 ckpt) | globally-trained |
| --- | ---: | ---: |
| subset, topk k=3 | 85.57 | 80.06 |
| footprint, topk k=3 | 85.57 | 79.89 |
| global, topk k=3 | 80.95 | 79.54 |
| global, random k=3 | — | 77.81 |

**Corrected 2026-08-03.** The comparison above is against a checkpoint trained on a different
mixture (it also saw Siegfried at 384-native), so it overstated the effect. The matched control
`syntra_maps_s100_subset` - identical in every respect except the pool scope - has since been
trained and evaluated on the full test set:

| run (identical hismap-only mixture) | mIoU |
| --- | ---: |
| subset pool (control) | **82.38** |
| global pool | 81.12 |

**Global-pool training costs 1.26 mIoU, not ~5.5.** The effect is real but small, and most of
the apparent gap was the training-mixture confound flagged when this section was first written.
The control's advantage concentrates in the minority class hameln.water (84.14 vs 70.75).
Training on global pools still gained nothing where it was meant to help. Two further symptoms: retrieval's advantage collapsed from +8.1 to +1.7 mIoU at global
scope, and the model became k-dependent (74.1 at k=1 against 80.1 at k=3, where the
subset-trained model was flat in k), with random *beating* topk at k=1.

Its full-test-set score of 81.12 mIoU looked encouraging but is not comparable to the numbers
above, which come from a stratified sample - the mismatch is what made the result look positive
at first.

**Confound, resolved.** The matched control above closes it: on the identical hismap-only
mixture the gap is 1.26 mIoU. What remains open is whether the conclusion survives a broader
training distribution - see the Siegfried rerun noted in section 0.

### Why it likely fails, and what would fix it

DINOv2 embeds the whole tile, so retrieval matches terrain and general appearance rather than
*how the class is drawn*. Two tiles can be near neighbours while rendering woodland with
different symbology - which is precisely the prompt that should not be retrieved.

1. **Class-masked embeddings.** Pool DINOv2 patch tokens over the annotated class region instead
   of using the CLS token, so the descriptor means "this is what woodland looks like here".
   `indexdb.py` already exposes patch tokens; it is an indexing change plus a rebuild of the
   global index, with no training. The obstacle: at inference the target's class region is
   unknown, so source and target descriptors would live in different spaces.
2. **Predict-then-retrieve (two passes).** Use a first pass with any reasonable prompt to get a
   rough mask, build the target's class-masked descriptor from it, re-retrieve, and run a second
   pass. Inference only, works on the checkpoints that already exist, and degrades gracefully
   because the second pass is only used where the first found something. **Try this first**: if
   better-matched prompts do not move the numbers, the bottleneck is the model's use of the
   prompt rather than the retrieval, and 1 and 3 will not help either.
3. **Learned pattern embedding.** A small projection head on frozen DINOv2, trained
   contrastively with positives drawn from the same class *and the same sheet-year* (identical
   symbology by construction) and negatives from the same class in other years. The supervision
   is free metadata, training takes minutes, and the space is symmetric so it applies to whole
   tiles on both sides.

All three are inference-time changes to a model that already works well with same-sheet prompts,
which is the useful way to read the negative result above.
