# E13 — which annotations to buy

A separate plan from `EXPERIMENT_PLAN.md`, which is left as the record of E1–E12. This one
covers a single question and the factors that bear on it.

## The question

Given a pool of unannotated map sheets and a fixed amount of annotator time, **what should be
annotated so that one model trained on it segments all of the sheets as well as possible?**

The deliverable is an annotation policy, not a checkpoint: a rule for choosing the next thing to
label, computable before anything is labelled. It runs on DINOv2 embeddings of the unannotated
tiles, so it needs the images and not the ground truth.

## What a subset is

One subset is **one map sheet**: one corpus, one sheet number, one survey year —
`hameln.3821_1974`. There are 26 in scope (17 Hameln, 8 Bayern/Donauwörth after dropping the
`3921_2023` duplicate, 2 Siegfried), holding 10,831 annotatable tiles and 12,918 (tile, class)
pairs.

## The unit: one annotation pass

**One unit of budget is one class digitised on one tile.** A tile carrying forest and water
costs two units if both are wanted, because it is two turns round the sheet for the annotator
and two training samples for the model — `cd_syntra2.py` builds each sample as a binary mask for
the class it was drawn for.

**A pass buys nothing about the classes it did not cover.** This is the sharpest consequence and
it invalidates the obvious baseline comparison. The existing `cls_selection_100` pairing holds
17,755 source–target training pairs, but only 6,843 of them are positive; the other **10,912 are
cross-class negatives** — a water sample taken from a tile where only forest was digitised,
scored against ground truth from the fully annotated release. No annotation budget produced
those. Counting them makes the n-shot scheme look like it trains on 17,755 samples for 6,238
passes of work.

So the pairing tool is class-restricted: a selection's tiles serve as sources and targets only
for the classes bought on them. Under that rule `cls_selection_100` yields **6,154** training
samples, and the whole corpus could not reach 17,755 positives even if every pair were bought.

Counted over the 26 sheets actually in scope the baseline is **6,160 passes yielding 6,158
trainable pairs**; the 6,238 quoted earlier included the `hameln.3921_2023` duplicate.

## The four factors

| # | factor | levels | where |
| --- | --- | --- | --- |
| 1 | **cell** — which sheets share a retrieval pool | sheet, style, era, corpus, global | `--cell` / `--scope` |
| 2 | **sampling** — how the budget is spent inside a cell | uniform, random, fps, fps+guard, kmeans | `--strategy`, `--density-guard` |
| 3 | **budget** — how many passes | 300, 1000, 3000, 6160 | `--budget` |
| 4 | **pairing** — how a source is chosen for a target | top-k by similarity (RAG), random from the cell | `src_selection` |

**The cell is one choice, not two.** It sets both the group the budget is balanced over and the
scope retrieval draws from, and those must agree: split apart, a cell can be given annotations
retrieval never looks at, or asked for exemplars the budget never bought. One sheet is the
validated setting; `era` groups the sheets of one corpus drawn within five years of each other,
collapsing 16 survey years into 11, because a shared calendar year across corpora is not a
shared cartography.

**No metadata rule fits both corpora, so the default cell is measured.** Hameln is a topographic
series re-drawn each edition: sheets 3821, 3822, 3921 and 3922 surveyed in the same year sit at
a mean centroid distance of **0.008**, while one sheet across its years sits at **0.119** — style
follows the period. The Bayern Positionsblätter are one engraved plate revised for decades and
run the other way: the same sheet across years is **0.030** and the same year across sheets is
**0.110**. `bayern.489_1910` and `bayern.489_1939` are visually the same plate 29 years apart,
while `bayern.519_1910` shares neither look nor engraving with `bayern.489_1910` despite the year.

**So the cell is one number, not a menu.** `--cell-gap` unions sheets whose mean embeddings lie
within that cosine distance, transitively. **0 leaves every sheet alone — the validated
setting** — and this corpus collapses into a single pool by 0.114, so the factor is a continuous
sweep rather than a choice between named rules. At 0.05 it recovers both archive behaviours
without a rule per archive: Hameln splits into 1973–1996, 2002–2005, 2012–2023 and 1898 alone,
Bayern pairs 489_1910 with 489_1939 and 490_1910 with 490_1938.

**Emptiness was checked and is not driving it.** Sheet centroids could in principle be tracking
how much ink a sheet carries rather than how it draws: a single embedding direction predicts a
tile's ink fraction at r=0.933, and the merging pairs do have similar ink (mean difference 0.063
against 0.172 for non-merging pairs). But projecting that direction out leaves the merge order
identical for sixteen steps with gaps unchanged to three decimals. Density co-varies with the
grouping without causing it.

Widening the cell is not free in either direction. It lowers the coverage floor — 88 (cell,
class) combinations at sheet, 43 at era, 9 at corpus — and raises the number of exemplars a
target can draw on, but §3c of the old plan measured global-scope training at 1.26 mIoU *below*
sheet-scope on an identical mixture. Factor 1 exists because that trade has never been measured
in between.

## Held fixed

Everything not listed above: SAM ViT-B with DoRA rank 16, batch 8 with AMP, `num_src=3`,
`num_notions=1`, 384 px, seed 2026, and the full test set of every subset. Every run trains from
scratch; no earlier checkpoint enters E13, so the old 85.89 is not a baseline for it — it
predates both AMP and the pass accounting.

## Experiment 1 — same training set size, new sampling

**The first run holds the number of source–target training pairs equal to the n-shot scheme's
and changes only how they were chosen.** Both arms are paired under the same class-restricted
rules, so the comparison is between two selections of the same size, not between two accounting
conventions:

| arm | cell | sampling | passes | training pairs |
| --- | --- | --- | ---: | ---: |
| baseline | sheet | uniform (`cls_selection_100`) | 6,160 | 6,158 |
| proposed | sheet | fps, no density guard | 6,160 | 6,158 |

Held at `cell=sheet` so this is a clean one-factor change against the validated configuration.
If the proposed arm wins here, the sampling strategy earns its place before any other factor
moves; if it does not, nothing downstream is worth running.

**No density guard, because at this budget there is nothing to guard against.** Farthest-point
sampling buys the periphery only when it can afford to be choosy. Re-measured at `cell=sheet`
against a density-neutral reference — random selection under the same cell and balance:

| guard | B=300 | B=1000 | B=6160 |
| ---: | ---: | ---: | ---: |
| 0 | 2.0x | 2.4x | **1.3x** |
| 5 | 1.3x | 1.5x | 0.8x |
| 10 | 1.1x | 1.1x | 0.5x |
| 15 | 1.0x | 0.9x | 0.4x |

6,160 of 12,918 candidate pairs is 48% of the pool: the walk has to take nearly half of
everything, so it cannot concentrate on outliers, and unguarded it already sits at the neutral
reference. Setting g=15 there would discard a further 15% of the candidates for no gain and push
the selection to 0.4x, under-representing sparse regions rather than over-representing them.

The guard is a small-budget instrument, and its level belongs to the budget rather than to the
method: g=10 at B=300, g=10 at B=1000, g=5 at B=3000, none at B=6160, all at `cell=sheet`.
The per-rung levels are tabulated in `E13_PROTOCOL.md`. The earlier g=15 was
calibrated at `cell=corpus` under a 300-pass budget and does not transfer. E13.2 has to state
which it uses at each rung, or the budget ladder measures the guard as much as the budget.

## Experiments 2–4 — one factor at a time

Anchored on whichever arm of experiment 1 wins, at `budget=1000` unless stated:

- **E13.2, budget.** 300 / 1000 / 3000 / 6160 at the winning sampling and `cell=sheet`. The
  headline curve: does a well-chosen 300 approach a uniform 6,160?
- **E13.3, cell.** sheet / era / corpus at budget 1000. Measures what the retrieval scope is
  worth, in between the two points §3c already has.
- **E13.4, pairing.** top-k against random-from-cell at budget 1000. Retrieval on or off, at
  equal annotation — the E1 control repeated under a budget rather than the full training set.

## Protocol

- **Coverage floor.** Every (cell, class) combination gets at least one pair before the strategy
  spends the rest. Without it a combination the budget misses has no exemplar, the pairing file
  gets no entry, and `cd_syntra2` — which builds its sample list from that file — silently drops
  those test targets. Unguarded, random selection covered 60 of 88 combinations at B=300, so the
  runs would not have shared a test set. All configurations now cover all of them.
- **Validation comes out of the budget.** Ten percent of each run's pairs are held out, so a
  B=300 run trains on 270 and still reports 300. Borrowing an external labelled set would
  understate the one number this is about.
- **Iteration count** is calibrated once, not early-stopped: the poly decay is a function of
  `max_iterations`, so a run stopped at k is not the run that would have been trained for k. Two
  pilots at the extremes with constant LR, then a fitted rule frozen for all runs. See §3e of the
  old plan, which stands.
- **Evaluation** is the full test set of every subset, identical for every run, at natural
  prevalence.

## Cost

Experiment 1 is 2 runs. Experiments 2–4 add 3 + 2 + 1 = 6 more, plus 2 calibration pilots: **10
runs**, ~10 h each. On the 4090 alone that is about four days; the 3090 halves it.

## Open

- **The guard on the budget ladder.** No single guard level is neutral across E13.2's rungs,
  because the outlier pull itself falls as the budget takes more of the pool. Either vary it per
  rung and state that the ladder holds "neutral density" fixed rather than the guard, or fix it
  at one value and report where the selection sits against the reference at each rung. Deciding
  after seeing the curve would be choosing on the outcome.
- **Cross-class negatives.** The rule above says a pass buys nothing about other classes. If the
  annotation protocol is really "label this tile completely", the unit is the tile, negatives are
  free, and both the budget and this plan change. Decide it before annotating anything for real.
