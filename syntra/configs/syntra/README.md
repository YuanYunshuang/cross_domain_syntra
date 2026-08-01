# SYNTRA configs

## Current family (use these)

| Config | Training data | Notes |
| --- | --- | --- |
| `syntra_maps_full.yaml` | all training tiles | base config, all switches documented inline |
| `syntra_maps_s100.yaml` | 100 tiles per class and map subset | |
| `syntra_maps_s50.yaml` | 50 per class and subset | |
| `syntra_maps_s10.yaml` | 10 per class and subset | |

All four read the standard root `/home/yuan/data/syntra` with `data_keywords: hismap,siegfried`
(27 subsets: 25 Hameln/Donauwörth sheets + Siegfried railway + vineyard, 37,735 training
samples) and share the data pipeline of the published SMOL-Uni baseline, so the two are
directly comparable: resize to 384, per-dataset `foreground_ratio` (0.01 for Siegfried),
rotation ±30°, normalization with mean 0.5 / std 0.25.

Experiment switches, overridable on the command line:

| Field | Values | Meaning |
| --- | --- | --- |
| `num_src` | k ≥ 1 | source examples per target; `TEST.num_src` may differ from training |
| `src_selection` | `retrieved`, `topk`, `random` | how the k sources are drawn. `retrieved` = uniformly among the retrieved neighbours (all runs so far), `topk` = the k closest in ranking order, `random` = uniformly from all annotated tiles of that subset and class, i.e. **retrieval disabled** — the control for the retrieval claim |
| `src_fusion` | `max`, `mean` | how the per-source predictions are merged; `mean` averages only the sources that carry the class |
| `retrieval_encoder` | `dinov2` | documents which encoder produced the `sample_pair_*` files; changing it means regenerating those files |

Example (E1 control run):

```bash
PYTHONPATH=. python syntra/train_one_gpu.py --config syntra/syntra_maps_s100.yaml \
    --src-selection random --num-src 3 -t syntra_maps_s100_random_k3
```

## Legacy configs

Kept for provenance: they document how the earlier runs were produced and are **not** directly
runnable against today's data layout (see below). Best results per family, as collected by
`syntra/tools/collect_results.py`:

| Config | Run | mIoU | Note |
| --- | --- | ---: | --- |
| `syntra_hismap_n1_s100_v3` | `main_results` | **85.89** | best so far; 3 sources in training, 7 at test |
| `syntra_hismap_n1_full_v3` | | 84.29 | all data |
| `syntra_hismap_n1_s100_v4` | | 83.48 | + colour jitter and channel swap |
| `syntra_hismap_n1_s10_v3` | | 79.75 | 10-shot |
| `syntra_hismap_n1_s100_v2` | | 75.12 | v2 dataset |
| `syntra_hismap_n1_s100_v1` | | 86.31 | Hameln only (4 classes), not comparable to the 9-class numbers |

**Why they no longer run as-is.** Those runs globbed a root in which the Siegfried subsets sat
next to the Hameln/Donauwörth sheets, resized to 384 (`siegfried.railway`,
`siegfried.vineyard_{540,545,547}_tile`); those folders now live under
`/home/yuan/data/syntra_deprecated`. The standard root holds the same train/test splits
(railway 5,872 / 1,679; the deprecated build only adds the unused 839 validation tiles) but at
224 pixels, which is why the current family resizes to 384.

**Class naming changed with it.** The subsets are now `railway` and `vineyard` instead of
`siegfried.railway` and `siegfried.vineyard`, so the evaluator reports `railway.railway` and
`vineyard.vineyard`. Old and new result tables must be joined on the class, not on that string.
