# SYNTRA — paper outline (2026-09-23)

Written after the experiments, not before, so the spine follows what replicated rather than what
was hypothesised. `PAPER_PLAN.md` stays as the record of how we got here; where the two disagree
about what to *claim*, this one wins.

## The reframing, and why

§6 of the plan made **retrieval** the spine (H1–H3). The measurements do not support that as the
headline:

- retrieval at k=3 is worth **+0.62** over random — inside the ~2-point noise floor, not established
- one retrieved source ≈ three random sources (84.10 vs 84.30), so retrieval buys *prompt
  efficiency*, not accuracy
- on an unseen sheet it is a **wash** at every pool size — a single sheet's tiles are too
  homogeneous for similarity ranking to have anything to choose between

What did replicate, cleanly and with separation, is **annotation efficiency** and **training-free
adaptation**. So retrieval becomes a component of the method rather than the claim, and the paper
leads with the thing a practitioner cares about: how little labelling is needed.

## One-sentence claim

> Retrieval-conditioned segmentation reaches the accuracy of a fine-tuned baseline trained on the
> whole corpus while annotating **1/15th** of it, and adapts to an unseen map sheet from **ten
> annotated tiles with no gradient step**.

## Contributions

1. **Annotation efficiency.** 1,000 tiles beats SMOL trained on 14,675, and beats matched SMOL at
   5,087 with every seed separated.
2. **Training-free adaptation.** Ten tiles from an unseen sheet close most of the gap to a model
   trained on it — a capability the single-prompt baseline structurally cannot offer.
3. **Tile accounting.** One annotated tile yields supervision for every class it carries (2.9 on
   average, and 3 even for a single tile), so per-class budgeting understates what annotation buys.
4. **A decomposition of where the gain comes from**, including two negative results that a
   practitioner needs: the exemplar count must match between training and inference, and the
   positive:negative ratio matters more than either.

## Section plan

### 1. Introduction
Frame on annotation cost, not architecture. The model is *unchanged* from the baseline — same SAM
ViT-B, same DoRA rank 16, same prompt encoder, same loss. Only what it is shown changes. That is
the paper's rhetorical advantage: no architectural novelty to defend.

### 2. Method
Retrieval-conditioned prompting; the pairing built offline so inference is a lookup. Tile
accounting (§3 of the plan) as the budget model — argue it from how annotation is actually
commissioned, and support it with the measured 2.11 natural negative ratio.

### 3. Experimental setup
Corpus (27 subsets, 26,017 tiles), the 37,935-pair test set, and **the protocol statements that
turned out to matter**:
- prompts at test time come from the run's own budget
- evaluation is pooled-pixel IoU over every (tile, class) pair including negatives
- results are per class with seed ranges; the nine-class mean is secondary

### 4. Annotation efficiency — **Figure 1, the headline**
Budget curve, `results/budget_curve.csv` + the three 1,000-tile seeds.

| tiles | SYNTRA | note |
| ---: | ---: | --- |
| 300 | 76.90 | |
| 1,000 | **83.50** (82.96 / 83.29 / 84.24) | beats SMOL@14,675 on every seed |
| 3,000 | 85.78 | |

against SMOL 81.09 @ 5,087 (matched, n=3) and 82.57 @ 14,675 (archive).
**Every SYNTRA seed at 1,000 beats every SMOL seed at 5,087** — that separation is the claim, not
the t-test (t = 2.64, p ≈ 0.12, which the text should report and not lean on).

### 5. Training-free adaptation — **Figure 2**
`results/e5_unseen_sheet.csv`. Per class against pool size, with the trained-on model as ceiling.
Ten tiles suffice; 10 → 100 changes nothing; the gap to training on the sheet is 3–7 points per
class. State that `n=1` is degenerate (no settlement in the pool, forest 0.40) rather than
plotting it as a point.

### 6. Where the gain comes from — **Table 2**
The S0 → S3 ladder, `results/ablation_all_results.csv`, per class from
`results/per_class_results.csv`.
- data organisation: **+0.12**, i.e. nothing — which is also the like-for-like check (S1 reproduces
  SMOL, so the comparison is sound)
- retrieval and source count are **substitutes**: either buys ~2.7, both buy ~2.8
- per class, three are separated at n=3: settlement +6.40, grass +6.06, bayern.forest +3.78;
  railway and vineyard go slightly the other way and should be reported as such

### 7. What practitioners should copy — the negative and methodological results
- **Exemplar count must match between train and inference.** Every model peaks at its training k;
  more prompts at test time *hurt*. Separately, growing the *pool* helps (§5) — these are different
  operations and must not be conflated.
- **The positive:negative ratio is worth more than the method choices.** Inverted-U peaking at
  50/50, 1.72 points wide. The baseline's published 0.8 is its worst setting; we tuned *it* and
  lost 1.7 points of apparent margin doing so.
- **Retrieval's value tracks pool diversity, not scarcity.** +1.70 at k=1 over a corpus-wide pool,
  a wash within one sheet.

### 8. Threats to validity — say these before a reviewer does
- **Run-to-run noise is ~2 mIoU**, measured from identical configurations (81.36 vs 82.37).
- **The metric is fragile for thin classes.** Pooled IoU over all negatives means stability tracks
  positive mass against negative exposure, r = −0.80 on log(mass/neg). `hameln.water` holds 80
  tiles' worth of water across 8,160 test tiles, so a 1% false-positive rate halves its IoU.
- **Evaluation itself is stable to < 0.01** — a ~300× asymmetry that justifies single-run
  inference sweeps and three seeds for anything trained.
- Single corpus, single backbone, one held-out sheet for §5.

## Claims we cannot make

- ❌ "Retrieval improves accuracy" as an unconditional claim — +0.62 at k=3 is inside noise.
- ❌ "SYNTRA beats SMOL by 2.76 mIoU" — p ≈ 0.12. Use the per-class separation and the budget
  crossing instead.
- ❌ "Capping exemplar coverage helps" — retracted; no budget shows it.
- ❌ Anything quoting v3's 85.89/86.57 beside SMOL — different corpus, 29 subsets vs 27.
- ❌ Anything mixing 11,342-sample and 37,935-sample evaluations; the former excludes negatives and
  runs ~4 points high.

## Gaps worth one run each

1. **Matched `budget_random` at 5,087** — without it Figure 1 shows 3,000 beating 5,087, which is a
   selection-scheme artefact and invites an obvious question.
2. **Matched SMOL at 300/1,000/3,000** — lets §4 compare two curves instead of a curve against
   three archive points with unmatched settings.
3. **E7 multi-temporal** — H3 is currently untested; either run it or drop the hypothesis from the
   framing.
