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

> **Superseded 2026-09-16 — the 0.7 is roughly 3× too optimistic.** It came from *one* pair of v3
> runs. Measured properly at n=3 per arm (§6b-ter), run-to-run spread is **~2 mIoU**: two runs of
> an identical SMOL config gave 81.36 and 82.37, and S3's three seeds span 81.61–85.04. Per class
> it reaches **7.6** on `hameln.water`. **Use ~2 as the MDE for the nine-class mean, and read
> §6b-ter's per-class table rather than the mean.** The inference half of this paragraph *is*
> confirmed, and more strongly than stated: repeated scoring of one checkpoint agrees to < 0.01
> even though it draws different sources — a ~300× asymmetry between training and evaluation
> noise. Claims in §2 that were downgraded for sitting inside 0.7 sit further inside 2.

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
| **E-A** | budget curve: 100 / 300 / 1,000 / 3,000 / 5,087 / 11,152 tiles at the winning strategy | 6 |
| **E-B** | selection: random vs fps (+guard) at **300 / 1,000** tiles; 3 seeds at 300, where the effect is largest | 5 |
| **E-C** | cell width: `--cell-gap` 0 / 0.05 / 0.114 (single pool) at 1,000 tiles — the in-between §3c never measured | 3 |

#### The guard, re-derived in tile units (2026-09-06)

`data_tools/indexing/derive_guard.py`, over the 11,152-tile pool, `--balance cell-class`,
`--cell-gap 0`, seed 2026, floor 1 per (sheet, class). The criterion is unchanged: random
selection is the **density-neutral reference**, and the guard is the smallest level bringing
farthest-point sampling back to it (≤1.1×) on the share of picks landing in the globally-sparsest
tenth of tiles.

Under `--balance corpus-class`, the balance §6b adopts:

| B (tiles) | neutral ref | g=0 | g=3 | g=5 | g=10 | g=15 | g=20 | **level** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 300 | 21.2% ±0.7 | 2.78× | 2.25× | 1.79× | 1.16× | **1.04×** | 1.04× | **15** |
| 1,000 | 19.1% ±0.6 | 2.71× | 2.11× | 1.72× | **1.01×** | 0.87× | 0.76× | **10** |

The same sweep under the old `cell-class` balance gives the **same two levels** (15 and 10) from
different ratios (1.98× and 2.21× unguarded). The level is robust to the balance choice; the
severity is not — with 9 cells rather than 92 each cell gets many more picks over a much larger
pool, so the greedy walk reaches further into the fringe before exhausting the extremes. **The
guard matters more under the balance the paper adopts, not less.**

**The level is not the same at both rungs**, which is why it had to be re-derived rather than
carried over: `g=15` over-corrects at B=1,000 to 0.87×, i.e. farthest-point sampling ends up
*avoiding* sparse tiles more than random does. State the level per rung in the caption.

**B=100 is not derivable and is dropped from E-B.** The correctness floor — one tile per (sheet,
class), needed so that no combination leaves the evaluation — costs 92 pairs on **85 distinct
tiles**, and it is mandatory at every budget:

| B (tiles) | forced by floor | free for the strategy |
| ---: | ---: | ---: |
| 100 | 85 | **15** |
| 300 | 85 | 215 |
| 1,000 | 85 | 915 |

At B=100 random and fps agree on 85 of 100 picks by construction, so no number of seeds can
separate them; the guard sweep there is flat and non-monotonic (1.25 / 1.04 / 1.17 / 1.12 / 1.08
/ 1.08×) and its apparent optimum of `g=3` is an artefact of 15 tiles of noise. This is §2's
E13.1 failure from the other end — there the budget was 48% of the pool, here the floor is 85% of
the budget, and both force strategies to converge. **B=100 stays a rung of E-A**, where "how much
annotation do you need" is a fair question and "mostly whatever the floor gives you" is a finding.

*Caveat on the covering-radius columns:* they are measured over the **guarded** pool, so the
population shrinks as `g` rises and the radii are not comparable across guard levels. The
neutrality ratio is the criterion; the radii are context only.

#### E-B's selections, generated 2026-09-06

Eight, all `--budget-unit tile --balance corpus-class --cell-gap 0`, floor 1 per (sheet, class),
27/27 subsets covered. Random carries **no guard** — it *is* the density-neutral reference, and
guarding it would change the control's candidate pool.

| selection | tiles | pairs bought | training pairs |
| --- | ---: | ---: | ---: |
| `budget_random_gap0_300t_s{2026,2027,2028}` | 300 | 302 / 303 / 303 | 880 / 878 / 878 |
| `budget_fps_g15_gap0_300t_s{2026,2027,2028}` | 300 | 311 / 309 / 309 | 873 / 879 / 885 |
| `budget_random_gap0_1000t_s2026` | 1,000 | 1,047 | 2,958 |
| `budget_fps_g10_gap0_1000t_s2026` | 1,000 | 1,070 | 2,925 |

**§3c's worry does not bite here.** A tile-balanced budget is not a pair-balanced one in general,
and the arms do buy different numbers of pairs (302 vs 311 at B=300) because farthest-point
sampling favours tiles carrying more classes. But what trains the model is the *pairing*, and
under tile accounting that draws on every class in the cell's vocabulary rather than only the
ones bought — so the training-pair counts land within 1.4% of each other across all six B=300
arms (873–885). The arms are matched in supervision as well as in tiles.

**Every arm shares one test pairing: 11,342 targets, 37,935 target-class pairs**, byte-identical
across all eight — §8's requirement and the third of P0.3's asserts, verified rather than assumed.

E-A's remaining rungs (100 and 3,000 tiles) are **not** generated yet: §6b defines the curve as
running "at the winning strategy", which E-B decides. 5,087 and 11,152 already exist as
`cls_selection_100` and `train_samples`. B=3,000 will also need its own guard level derived first
if the winner is the guarded arm.

### 6b-bis. SMOL → SYNTRA, measured (2026-09-16)

The headline is **S3, not v3's 85.89/86.57**. Those are on a different corpus — `syntra_v3repro`,
29 subsets, Siegfried native-384, vineyard split across three subsets — while every SMOL run and
every rung here shares the current 27-subset tree, verified from their configs. Quoting v3 against
SMOL would compare across corpora and reweight a mean in which each (dataset, class) counts
equally.

Run on `cls_selection_100` (5,087 tiles), same 37,935-pair test set, 100k, identical model
throughout (`model_name: smol`, SAM ViT-B, DoRA 16, example prompt encoder, `src_fusion: max`).
Only what the model is *shown* changes. Full table: `results/ablation_all_results.csv`,
per-class with seed ranges: `results/per_class_results.csv`.

| rung | dataset | k | prompts | mIoU |
| --- | --- | ---: | --- | ---: |
| **S0** | `smol` | 1 | pool | 81.09 (n=3, 79.55–82.37) |
| **S1** | `cd_syntra2` | 1 | random | 81.48 |
| **S1r** | `cd_syntra2` | 1 | retrieved | 84.10 |
| **S2** | `cd_syntra2` | 3 | random | 84.30 |
| **S3** | `cd_syntra2` | 3 | retrieved | **83.86 (n=3, 81.61–85.04)** |

**Step 0 — the ratio, tuned on the baseline.** Positive share was an uncontrolled 30-point
difference: SMOL trains at 80% via `pos_prob`, `cd_syntra2` at exactly 50% because it happens to
have 9 positive and 9 negative sampler groups. Nobody chose either. Sweeping SMOL gave 81.17 /
**82.37** / 81.47 / 80.65 at 36 / 52 / 68 / 80% — an inverted-U peaking at 50%. SMOL's published
0.8 is the **worst** point, so tuning the baseline rather than our own method cost ~1.7 points of
apparent margin and is what makes the remaining margin defensible. `r* = 0.5` is pinned on every
rung; `group_fixed_m:<m_pos>:<m_neg>` exists to set it explicitly rather than by accident.

**What the 2×2 says.** Retrieval and source count are **substitutes, not complements**: either
alone buys ~2.7 (+2.62 retrieval at k=1, +2.82 sources at random), and doing both adds only
+0.62/+0.82 more — inside the noise. **One retrieved source ≈ three random sources** (84.10 vs
84.30). Data organisation contributes **nothing** (+0.12), which is §6a's validity check passing:
S1 at k=1 random reproduces SMOL, so the comparison is like-for-like and no reviewer can
attribute the win to infrastructure.

**H2 is refuted in its naive form.** Every model peaks at its *training* k; adding exemplars at
inference hurts. Retrieval-trained models degrade far less under mismatch (S1r spans 0.81, S1
spans 2.64), which is the salvageable part.

| model | k=1 | k=3 | k=5 | k=7 |
| --- | ---: | ---: | ---: | ---: |
| S1r (train k=1) | **84.10** | 83.85 | 83.59 | 83.29 |
| S3 (train k=3) | 84.18 | **84.92** | 84.72 | 84.56 |
| S1 (train k=1, random) | **81.48** | 80.61 | 79.79 | 78.84 |
| S2 (train k=3, random) | 82.24 | **84.30** | 84.22 | 84.16 |

**Cap source prompts at 90% coverage.** SMOL always had `fg_ratio < v < 0.9`; `cd_syntra2` had no
bound, letting it prompt with near-total-coverage tiles SMOL was denied — 14.1% of the pool, 40%
of forest's. Every rung below carries it, which is what the ladder needs.

> **Partly retracted 2026-09-20 — see §6b-quater.** The "+0.95 at 5,087" quoted here was one seed
> against one seed; against S3's three-seed mean it is **−0.11**, and on the matched-selection
> budget curve the cap *costs* 0.65 / 1.93 / 0.99 at 300 / 1,000 / 3,000 tiles. **No budget shows
> it helping, so it is out of the recipe** — though it stays in this ladder, where every rung
> carries it and the comparisons are therefore unaffected. What survives is the *grass* result,
> which is a same-configuration comparison: 67.41 → 74.88 with precision 72.65 → 81.92 and recall
> flat, and sd 4.38 → 0.54. The mechanism is indirect — only 1.1% of grass prompts are that large —
> most likely via forest, grass's main confusion partner.

### 6b-ter. Report per class, and why the mean is fragile (decided 2026-09-16)

**Run-to-run noise is ~2 mIoU, not §1a's 0.7.** Two runs of an identical SMOL config scored 81.36
and 82.37, diverging from iteration 100. At n=3, S3 − S0 = +2.76 with pooled SE 1.39, **t = 1.98,
p ≈ 0.12** — the nine-class mean is *not* significant, and S3's worst seed (81.61) sits below S0's
best (82.37).

**Evaluation, by contrast, is stable to < 0.01.** Two independent scorings of one checkpoint agree
to the printed precision on every class, even though they demonstrably draw *different* source
tiles — pooling over 37,935 samples averages the draw away. So **inference-only experiments need
no repeats and training experiments need seeds**, a ~300× asymmetry. This does not transfer to
small evaluations: a held-out sheet in E5 has far fewer samples to average over and must be
re-measured.

**The variance is structural, and lives in the metric.** `Evaluator` pools pixels — per-class IoU
is `Σtp / Σunion` over every sample including negatives — so stability tracks positive pixel mass
relative to negative exposure. Regressing sd on log(mass/neg) over the nine classes gives
**r = −0.80**:

| class | pos mass | neg tiles | mass/neg | S3 sd |
| --- | ---: | ---: | ---: | ---: |
| hameln.water | 80 | 7,763 | **0.010** | **7.60** |
| bayern.water | 19 | 1,186 | **0.016** | **4.71** |
| hameln.grass | 581 | 5,328 | 0.109 | 0.54 |
| hameln.forest | 2,415 | 4,636 | 0.521 | 0.77 |

`hameln.water` has ~80 tiles' worth of water spread over 8,160 test tiles, 7,763 containing none.
A **1%** false-positive rate on those roughly doubles the denominator and halves the IoU. That is
exactly how seed 1236 failed: precision 91.4 → 80.4, recall only −4.5. It explains what scarcity
and coverage could not — `hameln.grass` has the same 20% coverage but 7× the positive mass, and
vineyard has fewer positives but only 34 negatives to be wrong on.

**So the paper reports per class with seed ranges, and treats the nine-class mean as secondary.**
A mean over nine classes, two of which are near-degenerate estimators, is a fragile statistic and
a reviewer recomputing it will meet the same variance.

**What survives replication** — S3's worst seed beats S0's best on three classes:

| class | S3 − S0 | separated? |
| --- | ---: | --- |
| hameln.settlement | **+6.40** | yes |
| hameln.grass | **+6.06** | yes |
| bayern.forest | **+3.78** | yes |
| bayern.water | +4.62 | overlap |
| hameln.water | +3.42 | overlap |
| bayern.grass | +1.67 | overlap |
| hameln.forest | +0.39 | overlap |
| vineyard | −0.58 | overlap |
| railway | −0.90 | overlap |

"Wins decisively on settlement, grass and Bayern forest, ties elsewhere, and the two thin
Siegfried classes are a draw" is defensible and specific. "+2.76 mIoU, p = 0.12" is not.

### 6b-quater. The budget curve, and the efficiency claim (measured 2026-09-20)

**This replaces the mean-difference framing as the headline.** "+2.76 mIoU, p ≈ 0.12" does not
reach significance; "matches SMOL's full-data accuracy with a fifteenth of the annotation" does
not depend on the mean being significant at all.

Six runs at the S3 recipe (k=3, retrieved, r* = 0.5, 100k) over corpus-class balanced random
selections, each scored on the 37,935-pair test set with **prompts drawn from its own budget** —
at test time you only have the tiles you bought. `results/budget_curve.csv`.

| tiles | capped | no cap | selection |
| ---: | ---: | ---: | --- |
| 300 | 76.25 | 76.90 | corpus-class random |
| 1,000 | 81.03 | **82.96** | corpus-class random |
| 3,000 | 84.79 | **85.78** | corpus-class random |
| 5,087 | 84.92 (n=3: 83.86) | 83.97 | **n-shot — different scheme** |

| | tiles | mIoU |
| --- | ---: | ---: |
| SMOL, matched baseline (S0) | 5,087 | 81.09 |
| SMOL-full, archive | 14,675 | 82.57 |
| **SYNTRA** | **1,000** | **82.96** |
| **SYNTRA** | 3,000 | **85.78** |

**1,000 tiles beats SMOL trained on 14,675 — 14.7× less annotation.** At 3,000 it is +3.21 over
SMOL-full with 4.9× less. The *capped* — i.e. handicapped — curve also clears SMOL-full at 1,000.

**Plotting caveat: 3,000 (85.78) exceeds 5,087 (83.97), and that is not a budget effect.** The
5,087 point is `cls_selection_100`, the n-shot scheme, which quotas per (sheet, class) and gives
railway and vineyard 10 tiles each rather than ~1/9 of the budget. Either add a matched
`budget_random` point at 5,087 or stop the curve at 3,000 and mark 5,087 separately.

#### The coverage cap: retracted

§6b-bis records `src_fg_max: 0.9` as worth +0.95 at 5,087. **That was one seed against one seed.**
Against S3's three-seed mean (83.86) versus the anchor (83.97) it is −0.11, and on the
matched-selection curve it costs 0.65 / 1.93 / 0.99 at 300 / 1,000 / 3,000.

**There is no budget at which the cap is shown to help. Drop it from the recipe.** It stays in the
S0–S3 ladder, which is internally consistent and where it equalises SMOL's own `fg_ratio < v < 0.9`
bound — and the ladder's conclusions are unaffected because every rung carries it. What does *not*
survive is describing it as an improvement.

#### A larger prompt pool at test time makes things worse

Found by way of a bug: the curve scripts initially hardcoded `cls_selection_100_test_samples` as
the test pairing, so every budget was prompted from the **5,087-tile** pool instead of its own.
A 300-tile model prompted from 5,087 tiles scores **66.38**; prompted from its own 300, **76.25**.
**Ten points worse with a larger, better-covered prompt pool.**

That is the same train/test mismatch the `num_src` grid shows (§6b-bis: every model peaks at its
training k), and it is worth reporting beside it: the model adapts to the *distribution* of
exemplars it trained on, not merely to their number. It also means **the test prompt pool is part
of the protocol** and must be stated for every budgeted result.

The episode is also why the earlier "cap costs 9.87 points at 300 tiles" claim was wrong: that gap
was the protocol, not the cap. Re-scoring is inference-only, so all six checkpoints were corrected
without retraining.

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
| ~~P0 bisect~~ | ~~6~~ **0** | cause found by diff read, §1b — P0.1/P0.2 retired |
| anchor | 1 | `syntra_anchor_cls100`, the baseline on the unified retrieval (§6a) |
| P1 instrument | 6 | 3 seeds × 2 anchors; the anchor above is one of them |
| E7 temporal | 2 | |
| E-A budget | 6 | |
| E-B selection | 5 | 3 seeds at B=300; B=100 dropped, floor-dominated (§6b) |
| E-C cell | 3 | |
| E4 main table | 4 | |
| E8 backbone | 2 | P1 |
| E1 / E3′ / E5 | 0 | inference only |
| **total** | **29** | was 36 |

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
