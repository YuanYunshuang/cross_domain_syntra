# SYNTRA — experiment design and manuscript plan

Status: draft, 2026-08-01. Scope decided with the author: **historical maps**, centred on
**annotation-efficient adaptation**, benchmarked against **SMOL-MapSeg** (accepted) and
**MapSAM2** (arXiv 2510.27547).

Assumption still open: target venue (planned for an ISPRS-Journal-style scope, cuttable to a
conference submission). Compute: **RTX 4090** (~7.7 h per 100k-iteration run, measured from the
existing logs) plus an **RTX 3090 desktop** (~12 h for the same run, estimated); see §4.

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

## 4. Two-GPU schedule

Both cards have 24 GB, so the training recipe is unchanged (batch 8 × 4 accumulation steps).
The 3090 is roughly 1.5× slower per iteration, so give it the runs that are either short or not
on the critical path.

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
