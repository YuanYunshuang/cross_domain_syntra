# SYNTRA — experiment design and manuscript plan

## 0. Objective

**Segment historical maps to the highest accuracy for the fewest annotated tiles.** Everything
below is judged against that: a change earns its place if it raises accuracy at a fixed
annotation budget, or holds accuracy at a smaller one.

Retrieval is *one mechanism* for that, not the goal. It matters here because it lets a single
annotated tile serve many targets - at k=1 choosing the exemplar by similarity is worth +0.6 to
+8.1 mIoU depending on how heterogeneous the pool is (E9) - i.e. the same label works harder.
Where retrieval stops paying, it should be dropped rather than defended: global-pool *training*
buys 1.26 mIoU less than subset-pool training on the identical mixture (82.38 vs 81.12, section
3c), so it is not part of the answer.

**Two budgets, kept separate.** They trade off differently and must not be conflated:

1. *Training budget* - tiles needed to train the model once. Current evidence: SYNTRA at
   100-shot reaches 85.89 mIoU where SMOL-Uni on the full training set reaches 82.6, so fewer
   labels already beat more.
2. *Adaptation budget* - tiles needed to bring an existing model to a new sheet or year. Current
   evidence: the unseen 1960 sheet reaches ~60 mIoU from a pool of 10-100 annotated tiles with
   no gradient step.

**Priorities that follow.** E10 (budget curve per style gap) and E11 (which tiles to annotate, at
a fixed budget) become the main experiments rather than ablations; E12 turns "annotate more" into
a rule for *which tile to annotate next*, closing the loop with the active-learning GUI already in
the repository. The comparison against MapSAM2 becomes an equal-budget curve - they fine-tune per
sheet, we add tiles to a pool - rather than a comparison of headline IoU.

Training breadth is the other lever on the same goal: a model that has seen more cartographic
traditions should degrade more gracefully on an unseen style, which is testable by rerunning the
E10 ladder with Siegfried (384-native build) in training and comparing per-subset IoU against the
measured style gap, not pooled mIoU.

Status: draft, 2026-08-01. Scope decided with the author: **historical maps**, centred on
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

### E10 in detail - a style-gap ladder, not a single hard case

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

| pool scope at inference | subset-trained (E9) | globally-trained |
| --- | ---: | ---: |
| subset, topk k=3 | **85.57** | 80.06 |
| footprint, topk k=3 | 85.57 | 79.89 |
| global, topk k=3 | 80.95 | 79.54 |
| global, random k=3 | — | 77.81 |

Training on global pools gave up ~5.5 mIoU everywhere and gained nothing where it was meant to
help. Two further symptoms: retrieval's advantage collapsed from +8.1 to +1.7 mIoU at global
scope, and the model became k-dependent (74.1 at k=1 against 80.1 at k=3, where the
subset-trained model was flat in k), with random *beating* topk at k=1.

Its full-test-set score of 81.12 mIoU looked encouraging but is not comparable to the numbers
above, which come from a stratified sample - the mismatch is what made the result look positive
at first.

**Confound, unresolved.** This model trained on hismap only, while the E9 checkpoint trained on
hismap plus Siegfried at 384-native. "Globally-trained" therefore also means "trained on less
data". A subset-pool control on the identical hismap-only mixture (one 7 h run) is required
before this negative result goes into the paper.

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
