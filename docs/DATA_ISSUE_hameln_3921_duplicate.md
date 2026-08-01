# Data issue: `hameln.3921_2017` and `hameln.3921_2023` are the same scan

Found 2026-08-01 while measuring the style distance between map sheets for the SYNTRA
experiments. Two subsets of the HisMap dataset contain **byte-identical images** under two
different year labels, with two different sets of annotations.

## Evidence

Per-subset fingerprints over all 25 subsets (md5 over the sorted per-file md5s of `imgs/` and
`lbls/`) give exactly one collision:

| Subset | image set | label set | tiles |
| --- | --- | --- | ---: |
| `hameln.3921_2017` | `503a3ae1b94f` | `34f6f2dea75d` | 900 |
| `hameln.3921_2023` | `503a3ae1b94f` | `8c0515b36e30` | 900 |

- All 900 image files are byte-identical; no other pair of subsets collides.
- **496 of 900 label files are identical, 404 differ**, with a mean pixel disagreement of 3.5%
  on the differing tiles.
- The class ratios are almost the same (forest 0.3833 vs 0.3850, grass 0.0753 vs 0.0706,
  settlement 0.0413 vs 0.0422, water 0.0004 vs 0.0004), so this is the same annotation intent
  applied twice, not two survey years.
- In DINOv2 space the two subsets have a style gap of exactly 0.0000 with 100% of their nearest
  neighbours in each other, which is how the duplication surfaced.

Conclusion: one scan of sheet 3921 was annotated twice and released as two years.

## What is *not* affected

**No train/test leakage.** The two subsets use identical splits - the same 420 training and 480
test tile names - so no tile is in training under one year and in test under the other:

```
A-train ∩ B-test: 0        B-train ∩ A-test: 0
```

## What is affected

1. **Dataset description.** Hameln is described as 17 map sheets (25 subsets in total); there
   are 16 distinct scans. This appears in the Zenodo record and in the dataset table of the
   SMOL-MapSeg paper.
2. **Weighting in training and evaluation.** That scan is trained on twice, and 960 of the
   Hameln test tiles are the same 480 images under two label versions. Because IoU is
   accumulated as the ratio of summed intersections to summed unions across subsets, the sheet
   contributes roughly twice its due share to every Hameln class-wise IoU - about 11.8% of the
   Hameln subsets rather than 6.25%. The direction and size of the resulting bias have not been
   measured; doing so requires re-running the evaluation with one of the two subsets removed.
3. **Annotation consistency.** Two label versions of the same pixels disagree on 404 tiles. One
   of them is presumably a revision of the other, and which one is authoritative is unclear from
   the data alone.
4. **Hold-out designs.** Holding out `hameln.3921_2023` while `hameln.3921_2017` is in training
   - or the reverse - is not a hold-out at all. Any leave-one-sheet-out or leave-one-year-out
   protocol has to treat the two as one unit.

## Suggested actions

- Decide which label version is authoritative, and drop the other from the release.
- Correct the sheet count in the Zenodo record (and in the paper, if the proofs are still open).
- Until then, exclude one of the two subsets from all SYNTRA experiments, and treat them as one
  unit in every hold-out split.
- Optional, ~1 h of GPU: re-run the SMOL-MapSeg evaluation with one subset removed to quantify
  how much the double weighting moved the published Hameln numbers.

## Reproducing this check

```bash
cd /home/yuan/data/syntra/hismap
for d in */; do d=${d%/}
  ih=$(find $d/imgs -type f | sort | xargs md5sum | awk '{print $1}' | md5sum | cut -c1-12)
  lh=$(find $d/lbls -type f | sort | xargs md5sum | awk '{print $1}' | md5sum | cut -c1-12)
  echo "$ih $lh $d"
done | sort
```
