# SYNTRA — paper plan (redesign, 2026-09-06)

Supersedes the forward-looking parts of `EXPERIMENT_PLAN.md` and all of `E13_PLAN.md`.
`EXPERIMENT_PLAN.md` stays as the record of E1–E12 and of what was measured; where it and this
document disagree about what to *run next*, this one wins. Section numbers below are new; the old
E-numbers are kept where the experiment survives unchanged so earlier notes still resolve.

## Decisions taken 2026-09-06

| decision | choice | what it changes |
| --- | --- | --- |
| budget unit | **one tile, fully labelled** | negatives free; all pass-denominated figures rescale; §3 rewritten |
| lead claim | **retrieval mechanism** (H1–H3) | E1/E5 become the headline; annotation policy is the supporting section |
| compute | full plan, ~28 runs | 3-seed error bars affordable; ~1 week over the 4090 and the 3090 |

---

## 1. Why the plan is being redesigned

A reproduction of the 85.89 run (`/koko/SYNTRA_logs/syntra/syntra_hismap_n1_s100_v3`, whose own
config records it as `syntra_hismap_n1_s100_v2_ce_neg_1.0` — "v3" is only the archive folder's
name) produced **86.57 mIoU** against v3's 85.89 with training data, test set and hyperparameters
matched exactly. Three things follow, and together they invalidate more of the old plan than any
single result did.

### 1a. There is a noise floor, and it is not small

| | v3 | reproduction | Δ |
| --- | ---: | ---: | ---: |
| mIoU (9 classes) | 85.89 | 86.57 | +0.68 |
| worst per-class | — | — | ±3.7 |

Everything upstream matched — 18,864 training samples, 37,935 test samples, 29 subsets, identical
class set — so the spread is seed noise. **Treat ~0.7 mIoU as the minimum detectable effect (MDE)
between training runs, and ±3.7 per class.** Any claim below that needs replication or it is not
a claim.

**This bound applies to training runs only.** Experiments that vary an *inference* setting on a
frozen checkpoint carry no training stochasticity; their only randomness is the source draw, which
averages out over repeats. Retrieval ablations can therefore resolve effects an order of magnitude
smaller than 0.7, which is why the retrieval-led framing is the safe one and the
selection-strategy framing is the risky one.

### 1b. A 12.8-point gap, found in `sam_loss.py` (closed 2026-09-06)

E13's arms scored 70.93 (n-shot) and 71.10 (fps) where v3 scored 86.57. Two rounds of
single-variable work closed most of it, but not by the expected route:

| step | mIoU | attributable to |
| --- | ---: | --- |
| E13 code, 1:1 negatives | 70.93 | — |
| E13 code, v3 negative ratio | 73.09 | negative:positive ratio, **+2.16** |
| v3 code, same data | **85.88** | code, **+12.79** |
| v3 reproduction | 86.57 | residual seed noise, +0.69 |

The negative-ratio effect is real and mechanistic: over-prediction (recall/precision) fell
1.147 → 1.082, precision rose +3.05 while recall gave back −0.93, and per class the
over-prediction ratio predicted the IoU loss at **r = −0.972**. But it is the minor term.

**The major term is `sam_loss.py`, and the first reading of it stopped one bug too early.** It
found the NaN guard, correctly judged that guard **gradient-neutral** — a `mean()` over an empty
positive tensor yields an empty gradient that scatters back as zero, exactly what
`torch.zeros(())` contributes — and concluded the file was clear. The same file had drifted twice
more, and neither change is inert:

- **Dice went from per-row to pooled.** `da187e4` added `cared = gt_masks_pos != self.ignore_index`
  and indexed by it. On a binary mask `!= 255` is all-True, so it filters nothing — but
  boolean-indexing an `(M,H,W)` tensor with an `(M,H,W)` mask returns a **flat 1-D tensor**, and
  the following `.unsqueeze(1)` leaves `(P,1)`. With no spatial dimension monai has nothing to
  reduce over, and `torch.sum(dim=[])` reduces over *everything*, so `DiceLoss` returns one pooled
  scalar instead of one value per row. Macro became micro: on a batch of one large and one small
  object, 0.5705 → 0.3753, and **small objects stop contributing** — precisely wrong for water,
  railway and vineyard. `reduction='none'` had been set by `97facdc` for a per-subset weighting
  that `da187e4` itself removed; nothing needed it afterwards and nobody put `'mean'` back.
- **A Dice term on negatives was live in every run.** `28cf0f6` made `train_one_gpu.py` pass
  `dice_neg_weight=getattr(args, ..., 1.0)`. No config sets the key, so the call-site fallback beat
  `SAMLoss`'s own default of `0.0`. On an all-negative sample it is a near-constant **+1.0 with a
  gradient norm of 1e-10 to 1e-13**: it trains nothing, inflates the loss, and — since `model_best`
  selects on windowed *training* loss — makes "best" partly a function of how many negatives a
  window happened to hold.

**The tell was in every log.** Dice is bounded in [0,1]. v3's `seg_loss` stays in [0.0099, 0.8666];
the E13 branch's reaches **2.0000 on 8,906 of 20,000 logged steps**. Nobody was reading the
magnitude, which is why `syntra/modeling/losses/check_sam_loss.py` now asserts it.

Fixed in `aa8eb04`, verified two ways: equality with a verbatim copy of `49843ea`'s implementation
across seeds and `ce_neg_weight` values (7/7), and in situ on the v3 corpus the trace reproduces
`logs/v3_repro.out` to six significant figures (0.507338 at ITR=10; 0.406873 vs 0.406846 at
ITR=200, the drift being GPU nondeterminism).

**The remaining three candidates still survive inspection** and are not implicated:

- `Resize` label dtype and sampler `drop_last` — both fixes are present on **both** sides
  (uncommitted in the v3 worktree until `4e1f5e5`; inert for v3, whose corpus runs no `Resize` and
  whose 16 groups divide exactly into batches of 8).
- `RandomRotateBatch` — rewritten for 5D input, but still draws an independent rotation per
  source; the old call site flattened to 4D and got the same thing.
- `max` source fusion — `torch.where(...).max(dim=1)` is a functional rewrite of the old in-place
  scatter. Same operation.

**Phase 0 was going to be a bisect. It turned out to be a diff read after all** — the first read
was one bug short, not wrong in method. P0.1 and P0.2 are retired; see §4.

### 1c. Two latent bugs, no historical damage — but no regression tests either

- **Label dtype collapse.** `Resize` cast labels to `torch.bool`, turning class ids `{0,1,2,3,4}`
  into `{0,1}`. Only class id 1 survives: forest, railway and vineyard scored normally while
  grass (2), settlement (3) and water (4) read exactly 0.00 with 100% accuracy. Introduced at
  init (2025-11-11), fixed 2026-08-01 (`afc0176`). **Audit result: every logged run using
  `cd_syntra2*` with a `Resize` config is dated 2026-08-02 or later, so no published number is
  affected.** v3 never tripped it because v3 had no `Resize` at all.
- **Ragged batches.** v3's data yields 16 sampler groups → exactly 2 batches of 8. The E13 data
  adds `railway.siegfried.neg` and `vineyard.siegfried.neg` (v3 had those classes as positives
  only), giving 18 groups → 8/8/**2**, and roughly half those 2-sample tails are all-negative →
  empty positive branch → NaN at 1/3 × 1/2 ≈ 16.7% against v3's 0.56%. `drop_last` brings it to
  1.22%. **v3 avoided this by divisibility luck, not design.**

Both were silent. Both are cheap to assert. See P0.3.

---

## 2. What this retires or downgrades

| claim | status |
| --- | --- |
| §3c "global-pool training does not pay off" (82.38 vs 81.12) | **1.26 points, single seed — inside the noise band.** Not established. Re-run with 3 seeds or downgrade to "no difference detected". The commit message asserts more than the data supports. |
| E13.1 fps vs n-shot (1.80 / −0.17) | **Uninformative, and doomed by construction.** At B=6,160 the budget takes 48% of the pool, so selection strategies must converge. Re-run at small budgets or not at all. |
| E9 retrieval gain "+0.6 to +8.1 mIoU" | Low end is noise. Quote the range with the MDE stated, or report only the heterogeneous-pool end. |
| H4 "which tiles matter more than how many" | Testable only at small budgets. Moves to §6b. |
| E13 pass-accounting (§0 of the old plan, §"The unit" of `E13_PLAN.md`) | Retired by the tile decision. See §3. |
| E2 (retrieval encoder), E3 (source fusion) | Almost certainly below MDE as training runs. E3 is an inference sweep now; fold it into §6a. Drop E2 unless a reviewer asks. |

Claims that survive unchanged: SYNTRA 100-shot 85.89/86.57 vs SMOL-Uni 82.6 (3.3–4.0 points,
comfortably above MDE), and the high end of E9.

---

## 3. The budget unit is one fully-labelled tile

**One unit of budget is one tile, labelled for every class in its sheet's vocabulary.** The
annotator visits a tile once and digitises everything on it; a tile carrying forest and water
costs one unit, not two.

This reverses the 2026-08-03 decision recorded in `EXPERIMENT_PLAN.md` §0 and `E13_PLAN.md` §"The
unit". The argument for passes was that the second class on a tile is not free for the annotator.
The argument for tiles, taken here, is that it matches how annotation is actually commissioned —
"label these sheets" — and that per-class revisits are not how the corpus was produced.

### 3a. What follows mechanically

| | pass accounting (retired) | tile accounting (adopted) |
| --- | --- | --- |
| unit | one class on one tile | one tile, all its classes |
| cross-class negatives | bought explicitly, cost budget | **free, and automatic** |
| `cls_selection_100` baseline | 6,438 passes → 6,435 trainable pairs | **5,087 tiles → 18,664 pairs** |
| E13 class-restricted pairing | required | **wrong — remove it** |

The class restriction in `make_pool_pairing.py` (a selection's tiles serve as sources and targets
only for the classes bought on them) exists solely to enforce pass accounting. Under tile
accounting it discards labels the budget already paid for, and it is what starved the E13 arms of
negatives. **Remove it and let a selected tile serve every class in its vocabulary.**

### 3b. The pool, in tiles

| corpus | subsets | tiles | (tile, class) pairs if fully labelled | pairs/tile |
| --- | ---: | ---: | ---: | ---: |
| hameln | 17 | 15,300 | 61,200 | 4.0 |
| bayern | 8 | 2,376 | 7,128 | 3.0 |
| railway | 1 | 7,551 | 7,551 | 1.0 |
| vineyard | 1 | 790 | 790 | 1.0 |
| **total** | **27** | **26,017** | **76,669** | 2.9 |

Of those pairs **24,618 are positive and 52,051 negative — a natural ratio of 2.11**. We measured
0.97 → 1.53 buying +2.16 mIoU, so full-tile labelling lands in the good regime by construction.
That is the strongest practical argument for the decision and should be stated in the paper.

### 3c. The new allocation question

A tile buys 4.0 training pairs in Hameln, 3.0 in Bayern and 1.0 in Siegfried. **A tile-balanced
budget is not a pair-balanced budget**, and the ratio is a property of the archive's class
vocabulary rather than of anything the policy controls. Either report both axes on the budget
curve, or state that the policy optimises tiles and let the pair count fall where it falls. This
replaces the retired "cross-class negatives" open question.

### 3d. Ladder

Tile budgets: **100 / 300 / 1,000 / 3,000 / 5,087 (n-shot baseline) / 11,152 (all annotatable)**.
Every figure with a budget on the x-axis states "tiles" in the axis label. No figure mixes units.

**The two top rungs were re-derived from the corpus on 2026-09-06** and both moved, because both
had been computed with `hameln.3921_2023` excluded and the decision is to keep it (§9):

| rung | was | is | why |
| --- | ---: | ---: | --- |
| n-shot baseline | 4,887 | **5,087** | `cls_selection_100` over 27 subsets. 4,887 predates the index rebuild of `fd59ff2`; today's selection gives 4,859 without the duplicate and 5,087 with it, so the old figure reconciles to neither and is simply stale. |
| all annotatable | 10,831 | **11,152** | train tiles carrying ≥1 class above `fg 0.05`. 10,831 is exactly this count with `hameln.3921_2023` removed — verified, so the provenance is not in doubt. |

"All annotatable" deliberately excludes the **3,523** purely-negative train tiles; the full train
split is 14,675. Under tile accounting a purely-negative tile still costs a unit and still yields
negatives, so if the budget curve is ever extended to the whole split, label that rung 14,675 and
say which convention it uses — do not silently blend the two.

`train_samples` + `test_samples` = 14,675 + 11,342 = **26,017**, the corpus total in §3b, so the
splits are exhaustive and disjoint. §3b's table was re-measured at the same time and reproduces
exactly: 2,376 / 15,300 / 7,551 / 790 tiles, 76,669 pairs, 24,618 positive and 52,051 negative
for a natural ratio of **2.11**.

---

## 4. Phase 0 — unblock. No paper experiment runs until this closes.

**P0.1 and P0.2 — retired, 2026-09-06.** The cause was found by reading the diff (§1b), so
neither the fourth cell of the 2×2 nor the 28-commit bisect has anything left to establish.
**~6 runs saved.**

| | v3 data | E13 data |
| --- | --- | --- |
| **v3 code** | 86.57 ✓ | 85.88 ✓ |
| **E13 code** | not run — cancelled at 5,890/100,000 | 73.09 ✓ |

The equivalence check that replaced them costs two minutes rather than 13 hours, and is sharper:
`syntra_v3_match_v3data` at `-itr 200` reproduces `logs/v3_repro.out` to six significant figures.
**Re-run it after any change to the training path** — it is the cheapest guard the pipeline has.

Two caveats that outlived the 2×2 and still need acting on:

- **73.09 is a `model_best` number**; 86.57, 85.88 and 85.89 are all `model_latest`. `model_best.pth`
  was written at ~92k of 100k, so it is not an early freeze, but it is a different checkpoint under
  a different rule and the comparison is not like-for-like. Re-testing `model_latest` is inference
  only and the checkpoint is on disk. See P0.5.
- **Neither v3 anchor was reproducible from a commit** until `4e1f5e5`/`0ef1750`: the two fixes the
  v3 side carries existed only as uncommitted working-tree edits in `CD_SYNTRA_v3code`, and the
  config that produced 85.88 was untracked. Both are now on `syntra/v3-repro`.

**P0.3 — regression tests.** Both bugs above were silent; both are one assert.
- class ids survive the transform chain (`{0,2,3,4}` in → `{0,2,3,4}` out)
- every emitted batch is full and contains at least one positive
- a third worth adding: the test pairing is identical across arms of a comparison

**P0.4 — data hygiene.**
- **Rebuild Siegfried at 384 natively.** The deprecated tree already holds it. This lets `Resize`
  be dropped from every config, removing an upsample confound that touches 2 of 9 classes.
- **`hameln.3921_2023` — decided 2026-09-06: keep it.** It is the same scan as `hameln.3921_2017`
  under a second year label (`DATA_ISSUE_hameln_3921_duplicate.md`), so the corpus trains on it
  twice and scores it twice. Kept because every run any new number is compared against includes
  it — v3, the reproduction, both E13 arms, and SMOL (`05c964e`) — and excluding it now would put
  a second uncontrolled difference between the anchor and the archive.
  **What this obliges:** the methods section states the convention explicitly rather than leaving
  it implicit, and the tile counts in §3d are the with-duplicate ones (5,087 and 11,152, not 4,859
  and 10,831). It stays a known defect of the data, not of any experiment.
- **Canonicalise class names.** v3-era results key `siegfried.railway` / `siegfried.vineyard`
  (three vineyard subsets collapsing into one class); the current layout gives `railway.railway` /
  `vineyard.vineyard`. Same nine classes, different weighting of the mean and no join key. Fix one
  convention and translate the archive.

**P0.5 — freeze the protocol.**
- fixed iteration count, no early stop (the decay is a function of `max_iterations`)
- evaluate **`model_latest`**. With `validation_data: null`, "best" selects on training loss —
  which on the old code was a 1–2 batch epoch average, and goes NaN on all-negative batches so
  `model_best` silently freezes at an early checkpoint
- one config family; every run records its full resolved config

---

## 5. Phase 1 — build the measurement instrument

**P1.1 — seeds.** 3 seeds × 2 anchor configs (baseline, one treatment) = **6 runs**. Report mean
and range. This fixes the MDE empirically rather than inheriting the 0.68 single observation.

**P1.2 — iteration calibration.** Check whether **50k iterations preserves the *ordering* of
conditions** even if absolute mIoU is lower. If it does, every Phase-2 comparison halves in cost
and only the headline numbers need 100k. Measure on the P1.1 anchors, which are already being run
at two settings.

**P1.3 — reporting rule.** Every comparison in the paper states its effect against the MDE.
Effects below it are reported as "no difference detected at n seeds", never as a negative result.
The noise band is shaded on every curve. This is rare in the map-segmentation literature and is
worth a paragraph in the methods.

---

## 6. Phase 2 — experiments, retrieval-led

The paper's spine is H1–H3: **explicit retrieval of annotated exemplars turns adaptation into a
data operation instead of a training operation, at equal or lower annotation cost.** The
annotation policy is the supporting act that says which labels go into the pool.

### 6a. Retrieval mechanism — inference-only, effectively free

Retrieval is baked into the pairing JSON offline, so `src_selection` variants cost **no training
at all** and repeats are cheap enough to give tight confidence intervals — at runtime
`_select_src_samples` merely draws from the stored list.

**One tool builds every pairing: `make_pool_pairing.py`.** Exact cosine over L2-normalised
**DINOv2 ViT-S/14-reg, 384-d** vectors from `global_index.npz`, candidates scoped per (cell,
class) by foreground ratio, self-match removed by setting its similarity to `-inf`, `k=8`,
neighbours stored in distance order.

`pairing.py` is **retired for the paper** (2026-09-06). It produced every archive number, but it
is a different retrieval, not a different implementation of the same one: it ranks by
**unnormalised L2** on raw DINOv2 features, scopes candidates per (subset, class) over the raw
selection list, and drops self-matches with `distances > 1e-3` — which also silently drops
genuine near-duplicates and leaves 7 neighbours where the new tool keeps 8. Measured over
`cls_selection_100`, the two agree on **84.9%** of nearest neighbours at **0.888** mean Jaccard:
close, but not the same. E1's whole design is a comparison *between* retrieval variants, so its
arms cannot each come from a different index, and the main table cannot be built on one index
while E1 is built on another.

The cost of retiring it is one run: `syntra_anchor_cls100` re-establishes the 100-shot baseline
on the new pairings, and every Phase-2 comparison is made against that rather than against 86.57.
It doubles as one of P1.1's seed anchors. **Until it lands, 85.89 / 86.57 / 85.88 are
`pairing.py` numbers and must not be quoted beside a `make_pool_pairing.py` result.**

| id | tests | design | training cost |
| --- | --- | --- | ---: |
| **E1** | H1 | retrieved vs random-from-cell vs top-k, k ∈ {1,3,5,7}, train-k ≠ test-k; ≥20 repeats per cell | 0 |
| **E3′** | H1 | source fusion max vs mean (`retrieval_sweep.py` already supports it) | 0 |
| **E5** | H2 | **money figure.** Unseen sheet/year: pool 0/1/10/50/100 tiles × {retrieved, random}, no gradient step; against SMOL (1 prompt) and MapSAM2 (fine-tuned) at equal tiles | 0 |
| **E7** | H3 | multi-temporal: held-out year per footprint, per-year IoU and temporal consistency, long vs short gap | 2 |

E1 at k=1 random must reproduce SMOL-MapSeg; if it does not, the comparison is not like-for-like
and that has to be resolved before the main table is trusted.

**Verify before relying on E7's consistency metric:** that the Hameln tile grids are
geometrically co-registered across years and not merely identically named. If they are only
grid-aligned per sheet crop, restrict E7 to per-year IoU and drop change-consistency.

### 6b. Annotation policy — the supporting section

| id | design | runs |
| --- | --- | ---: |
| **E-A** | budget curve: 100 / 300 / 1,000 / 3,000 / 4,887 / 10,831 tiles at the winning strategy | 6 |
| **E-B** | selection at small budget: random vs fps (+guard) at 100 / 300 / 1,000 tiles; 3 seeds at 300, where the effect is largest | 7 |
| **E-C** | cell width: `--cell-gap` 0 / 0.05 / 0.114 (single pool) at 1,000 tiles — the in-between §3c never measured | 3 |

The guard is a small-budget instrument and its level belongs to the budget, not the method: it
was measured at 2.0× the density-neutral reference at B=300 and 1.3× at B=6,160 in pass units, so
re-derive the per-rung levels in tile units before running E-B, and state them in the caption.
Deciding the guard after seeing the curve would be choosing on the outcome.

### 6c. Main table

**E4** — SYNTRA vs SMOL-Uni/-Ind vs MapSAM/MapSAM2 at all / 4,887 / 1,000 / 300 tiles: **4 runs**.
SMOL code and checkpoints are in hand on identical splits, so those numbers are direct. MapSAM2's
release must be confirmed; if unavailable, compare on the published MapSAM benchmark with their
reported numbers and state the limitation — do not re-implement it silently.

**E8** (SAM ViT-B vs SAM2 Hiera) stays P1 at **2 runs**: MapSAM2 is SAM2-based, and without it a
reviewer cannot separate retrieval from backbone.

---

## 7. Cost

| phase | runs | notes |
| --- | ---: | --- |
| P0 bisect | 6 | 1 full + ~5 short |
| P1 instrument | 6 | 3 seeds × 2 anchors |
| E7 temporal | 2 | |
| E-A budget | 6 | |
| E-B selection | 7 | includes 3 seeds at B=300 |
| E-C cell | 3 | |
| E4 main table | 4 | |
| E8 backbone | 2 | P1 |
| E1 / E3′ / E5 | 0 | inference only |
| **total** | **36** | |

At 12.5 h that is ~19 days on the 4090 alone. Two levers bring it inside a week: P1.2's 50k
calibration (halves everything but the headline numbers) and the 3090 running Phase 2 in
parallel. If it has to be cut, drop E-C and E8 first — both are P1 or lower and neither carries a
hypothesis.

---

## 8. Held fixed

SAM ViT-B with DoRA rank 16, `num_src=3`, `num_notions=1`, 384 px, `foreground_ratio` **flat 0.05
as a scalar** (the dict form resolved inconsistently — `fg_threshold` in the data tools matches
its key against the subset name where `siegfried` does not appear in `railway`, while
`cd_syntra2` matches the full path where it does, so one run used two thresholds), and the full
test set of every subset at natural prevalence, identical across every run in a comparison.

Whether AMP is on is **not** yet frozen: v3 ran fp32 at 2.17 it/s and the AMP path runs 5.05 it/s,
and no AMP-vs-fp32 comparison has been made at matched everything else. Decide it in P0.5 and
state it; an AMP result is not comparable to 85.89 until that is measured.

---

## 9. Open

- ~~**The 12.8 points.**~~ Closed 2026-09-06: two bugs in `sam_loss.py` (§1b), fixed in `aa8eb04`.
  Worth a methods paragraph — the loss reported a quantity outside the range its own definition
  allows for seven months, and the range check that catches it is one assert.
- ~~**`hameln.3921_2023`.**~~ Closed 2026-09-06: **kept**, since SMOL and every archive run
  include it, so keeping it is what makes the main table like-for-like. See P0.4 and §3d.
- **Tile-to-pair ratio across archives** (§3c). A tile-balanced budget over-serves Siegfried by
  4× relative to Hameln in training pairs. Decide whether the policy optimises tiles, pairs, or
  reports both, before E-A is run.
- **MapSAM2 availability.** Gates E4 and the framing of E5.
