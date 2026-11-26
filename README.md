# Annotation-efficient segmentation of historical topographic maps with exemplar prompts

This code accompanies the paper of the same name. It builds on SMOL-MapSeg (SAM ViT-B
fine-tuned with DoRA) and prompts the model with three annotated exemplar tiles. Trained on
1,000 annotated tiles, it matches the baseline trained on all 14,675. The code, the derived
data and the trained weights are described under [Data and trained weights](#data-and-trained-weights);
the data and weights are on Zenodo, [10.5281/zenodo.24565352](https://doi.org/10.5281/zenodo.24565352).

## Citation

Yunshuang Yuan. *Annotation-efficient segmentation of historical topographic maps with exemplar
prompts.* ISPRS Open Journal of Photogrammetry and Remote Sensing (submitted).

```bibtex
@article{yuan_exemplar_prompts,
  author  = {Yuan, Yunshuang},
  title   = {Annotation-efficient segmentation of historical topographic maps with exemplar prompts},
  journal = {ISPRS Open Journal of Photogrammetry and Remote Sensing},
  note    = {Submitted},
}
```

## Updates
- [Nov 13] added data converting for kitti and cityscapes; loss function support for ignoring index 255; dataset support for multi-notion training; add dockerfile and pushed to dockerhub

## Version
- v1: dataset=cd_syntra
- v2: dataset=cd_syntra2_efficient
- v3: no dice loss for negative samples
- v4: use color jitter and rgb channel swap for augmentation, normalization at last

## Installation
Using the `setup.sh` to create the conda environment for this project

## Prepare data
Using the converting tools in the package `data_tools` to convert your customized data into SYNTRA format. One example of converting the "trees" dataset for segmenting different species of trees is shown in `main.py`.

## Train
```bash
PYTHONPATH=. python syntra/train_one_gpu.py --config config_file.yaml
```

## Where outputs live
Training runs write to `output_dir: logs` (checkpoints, `log.txt`, `config.yaml`, and the
`test_model_latest_*.txt` result files beside them), and the queue scripts in `syntra/tools/`
write their `logs/<task>.out` stdout there too. Since 2026-09-24 **`logs/` is a symlink to
`/koko/cd_syntra_logs_09_2026`** on the koko share (the local disk ran out; ~57 GB moved). Paths in
configs and scripts are unchanged - they still say `logs/...` - so nothing needs editing, but
`/koko` must be mounted (`/koko/mount_koko.sh`) before training or testing. If it is not, `logs`
is a dangling link and runs fail at start-up rather than writing somewhere else.

Older archives: `/koko/SYNTRA_logs/` (runs before the E13 branch, including v3).

`results/` stays in the repo: the small per-run score files and CSVs the docs cite.

1. label meaning
2. which labels are you using
3. how the evaluation metrics are calculated, e.g., IoU
4. confusing classes

## Data and trained weights

The code, the derived data and the trained weights of *Annotation-efficient segmentation of
historical topographic maps with exemplar prompts* are released separately:

| what | where |
|---|---|
| code | this repository |
| derived data (splits, selections, pairings, embedding index) | [10.5281/zenodo.24565352](https://doi.org/10.5281/zenodo.24565352) (`syntra_data_derived.tar.gz`) |
| trained weights | [10.5281/zenodo.24565352](https://doi.org/10.5281/zenodo.24565352) (`syntra_weights_tier1.tar`, `syntra_weights_tier1_part2.tar`, `syntra_weights_tier2.tar`, `syntra_weights_tier2_part2.tar`; checksums in `SHA256SUMS` and `SHA256SUMS_part2`) |
| raw maps, HisMap (25 editions, 384×384 tiles) | [10.5281/zenodo.21707982](https://doi.org/10.5281/zenodo.21707982) |
| raw maps, Siegfried railway and vineyard | [10.3929/ethz-b-000691430](https://doi.org/10.3929/ethz-b-000691430) |

The derived-data release does not contain any map tiles. It contains small files that go next
to the tiles. Check a download with

```bash
python tools/verify_release.py path/to/syntra_release --strict
```

### 1. Tiles

All paths below are relative to one data root, `data/` by default.

**HisMap.** The Zenodo archive already holds the 384×384 tiles, so no tiling step is needed.
Its 17,676 images and labels, split lists, `cls_selection_100.json` and per-edition statistics
are byte-identical to the ones the paper used (checked file by file). It
has one folder per edition (`<region>.<sheet>_<year>/`, with `imgs/`, `lbls/`, `color_map.json`,
`<edition>.json`, `train_samples.txt`, `test_samples.txt` and `cls_selection_*.json`). Unpack it
as `data/hismap/<edition>/`.

**Siegfried.** The tiles are used at their native 224×224 and are resized to 384 when they are
loaded. Run the converter once per class. It copies `full/train` and `full/test` of the ETH
release as tiles and writes the train/test lists:

```python
from data_tools.converting.siegfried import SiegfriedConverter, color_map
for cls in ("railway", "vineyard"):
    SiegfriedConverter(syntra_data_path=f"data/siegfried/{cls}",
                       customized_data_path="path/to/Siegfried_Railway_Vineyard",
                       color_map=color_map, class_name=cls).convert(["train", "test"])
```

The result is 26,017 tiles: 17,676 HisMap and 8,341 Siegfried (7,551 railway, 790 vineyard).

### 2. Derived data

Unpack `syntra_data_derived.tar.gz` over the same root. It mirrors the tile layout one
subset folder at a time, and `data/manifest.csv` lists every file with its size, sha256, what it
is and the table that uses it.

```
data/
  hismap/<edition>/            (25 editions)    siegfried/{railway,vineyard}/
    train_samples.txt, test_samples.txt          the train/test split
    test_samples_ov{10,50}.txt                   test lists without tiles overlapping training
    color_map.json, <edition>.json               class map, per-tile class ratios
    cls_selection_100.json                       per-sheet 100-shot selection (5,087 tiles)
    budget_random_gap0_{300,1000,3000,5087}t_s2026.json
                                                 class-balanced budget selections
                                                 (the 5087t file holds 5,102 tiles)
    sample_pair_<src>_<tgt>_gap0.json            top-8 DINOv2 neighbours of each target tile
    train_samples.faiss, train_samples_map.json  DINOv2 CLS vectors of the train tiles (HisMap)
    e5_pool_*.json, e7b_pool_*.json              prompt pools for the withheld survey
                                                 (hameln.3922_1996)
    fp_pool_*.json                               prompt pools for the withheld footprint
                                                 (hameln.3821_*)
  withheld_training_lists/                       exact training tiles of the two withheld models
```

Two config keys point at the release:

* `root_dir`: the data root. It is `data` in every `syntra/configs/release/*.yaml`.
  Override it with `-d /path/to/data`.
* `data_info`: the `sample_pair_*` file name without `.json`, looked up in every subset folder.
  Training uses `sample_pair_<selection>_<selection>_gap0` and testing uses
  `sample_pair_<selection>_test_samples_gap0`.

SMOL runs (`dataset_name: smol`) ignore `data_info`. They take prompts from `source_file`
and score `test_samples.txt` (or `--target-file`, e.g. `test_samples_ov50.txt`).

### 3. Regenerating the embeddings and pairings

The pairings are released, so this step is optional. Retrieval uses the CLS token of DINOv2
ViT-S/14-reg (384-d, 224×224 input, ImageNet normalisation). `global_index.npz` is not
released, because it can be rebuilt from the tiles. The command below reads the released
`train_samples.faiss` vectors and embeds every tile that no index covers (the test split and
Siegfried) on a GPU:

```bash
PYTHONPATH=. python data_tools/indexing/build_global_index.py -d data \
    --splits train_samples,test_samples -dkw hismap,siegfried --compute-missing \
    -o data/global_index.npz
```

DINOv2 is fetched with `torch.hub`. Selections and pairings are then made with
`data_tools/indexing/select_budget.py`, `make_pool_pairing.py`, `make_eval_pools.py` and
`make_overlap_filtered_tests.py`. The commands are in the docstrings of those scripts and in
`syntra/tools/gen_eb_selections.sh` and `syntra/tools/run_gaps.sh`.

### 4. Checkpoints

Each `.safetensors` file holds only the trainable tensors (209 tensors, 4.80 M parameters: DoRA
rank 16 on the qkv layers, the prompt encoder and the mask decoder). This is the run's
`model_latest.pth["model"]` after 100k iterations. The frozen SAM ViT-B encoder is rebuilt from
the stock `checkpoints/sam_vit_b_01ec64.pth`. The training config of each checkpoint is
`syntra/configs/release/<name>.yaml`, copied from the config logged at the start of that run.

All four weight archives unpack into the same `syntra_weights/tier1/` and `syntra_weights/tier2/`
folders. Each archive also carries its own `manifest.csv` / `SHA256SUMS`, or `manifest_part2.csv` /
`SHA256SUMS_part2` for the part-2 archives.

**Recommended model: `syntra_persheet_s1236`** (86.12 mIoU). It uses the per-sheet 100-shot
selection, k=3 retrieved exemplars, seed 1236.

**Tier 1: models behind the headline numbers**

| name | archive | what | paper result |
|---|---|---|---|
| `syntra_anchor_cls100` | tier1 | per-sheet 100-shot selection (5,087 tiles), k=3 retrieved, seed 1234 | per-sheet row of Table 1 (83.97); reference model of the adaptation experiments, §7 |
| `syntra_persheet_s1235`, `syntra_persheet_s1236` | tier1_part2 | same, seeds 1235/1236 | per-sheet row of Table 1 (85.07, 86.12) |
| `syntra_5102t_s1234` | tier1_part2 | class-balanced random 5,102 tiles, seed 1234 | Table 1 (86.09). Same weights as tier 2 `curve_nocap_5087t` |
| `syntra_1000t_s1234/s1235/s1236` | tier1 | class-balanced random 1,000 tiles, three seeds | Table 1 (82.96, 83.29, 84.24) |
| `syntra_3000t_s1234` | tier1 | class-balanced random 3,000 tiles, seed 1234 | Table 1 (85.78) |
| `syntra_3000t_s1235`, `syntra_3000t_s1236` | tier1_part2 | same, seeds 1235/1236 | Table 1 (85.26, 86.02) |
| `smol_full_s1234/s1235/s1236` | tier1_part2 | SMOL-MapSeg baseline, full training split, no coverage cap | Table 1, full-split row (83.03, 81.91, 83.51) |
| `syntra_minus_hameln1996` | tier1 | anchor recipe without the hameln.3922_1996 survey | withheld survey year, §7.1–7.2 (84.86 on the full test set) |
| `syntra_minus_fp3821` | tier1 | anchor recipe without the hameln.3821 footprint (all its years) | withheld map area, §7.3 (85.79 on the full test set) |

**Tier 2: ablations, controls and baseline curves**

| name | archive | what | paper result |
|---|---|---|---|
| `S0_smol_k1`, `S0_smol_k1_s1235` | tier2 | SMOL dataset, k=1, cls_selection_100 | step-by-step ablation, Table 4 (81.36, 79.55; three-seed mean 81.09) |
| `S1_k1_random`, `S1r_k1_retrieved` | tier2 | cd_syntra2, k=1, random / retrieved exemplars | step-by-step ablation, Table 4 (81.48, 84.10) |
| `S2_k3_random` | tier2 | k=3, random exemplars | step-by-step ablation, Table 4 (84.30) |
| `S3_k3_retrieved`, `_s1235`, `_s1236` | tier2 | k=3, retrieved exemplars, three seeds | step-by-step ablation, Table 4 (84.92, 85.04, 81.61; mean 83.86) |
| `cvr1000_retrieved_s{1234,1235,1236}` | tier2 | 1,000 tiles, k=3, retrieved exemplars | exemplar count × retrieval, Table 5 (83.47, 83.97, 84.13) |
| `cvr1000_random_s{1234,1235,1236}` | tier2 | 1,000 tiles, k=3, random exemplars | exemplar count × retrieval, Table 5 (83.75, 83.15, 83.66) |
| `cvr1000k1_retrieved_s{1234,1235,1236}` | tier2 | 1,000 tiles, k=1, retrieved exemplars | exemplar count × retrieval, Table 5 (82.17, 81.39, 82.77) |
| `cvr1000k1_random_s{1234,1235,1236}` | tier2 | 1,000 tiles, k=1, random exemplars | exemplar count × retrieval, Table 5 (82.47, 80.07, 82.23) |
| `curve_nocap_300t`, `curve_nocap_5087t` | tier2 | 300 and 5,102 tiles, seed 1234 | Table 1 (76.90, 86.09) |
| `syntra_300t_s1235`, `syntra_300t_s1236` | tier2_part2 | 300 tiles, seeds 1235/1236 | Table 1 (71.60, 76.42) |
| `smol_persheet_s1234/s1235/s1236` | tier2_part2 | SMOL baseline, per-sheet 100-shot selection, no cap | Table 1, per-sheet row (80.42, 81.65, 82.48) |
| `smol_300t_s1234/s1235/s1236` | tier2_part2 | SMOL baseline, 300 tiles | Table 1 and Fig. 1 (76.41, 77.62, 77.97) |
| `smol_1000t_s1234/s1235/s1236` | tier2_part2 | SMOL baseline, 1,000 tiles | Table 1 and Fig. 1 (80.39, 81.17, 81.76) |
| `smol_3000t_s1234/s1235/s1236` | tier2_part2 | SMOL baseline, 3,000 tiles | Table 1 and Fig. 1 (82.44, 83.18, 82.32) |
| `smol_5102t_s1234/s1235/s1236` | tier2_part2 | SMOL baseline, 5,102 tiles | Table 1 and Fig. 1 (82.15, 83.66, 82.78) |

**Use the configs in `syntra/configs/release/`; they are authoritative.** The SMOL configs
inside the Zenodo v1.0 weight archives lack `TEST.target_file`.

The numbers in brackets are test mIoU over the 9 classes on the 37,935 test pairs, as reported
in the paper.

SMOL configs set `target_file: test_samples.txt` in `TEST`. `test.py` starts from the `TRAIN`
section, so without this line it would inherit the training `target_file`, which is the
selection itself, and score that instead of the test split.

### 5. Loading a checkpoint

`syntra/test.py` accepts a `.safetensors` file directly:

```bash
PYTHONPATH=. python syntra/test.py --config release/syntra_anchor_cls100 \
    -w syntra_weights/tier1/syntra_anchor_cls100.safetensors -o results/anchor \
    -d data --num-src 3 --src-selection retrieved
```

From Python:

```python
from hydra import compose, initialize_config_module
from safetensors.torch import load_file
from syntra.modeling import build_segmentation_model

initialize_config_module(config_module="syntra.configs.release", version_base=None)
cfg = compose(config_name="syntra_anchor_cls100").TRAIN
model = build_segmentation_model(**cfg)          # loads checkpoints/sam_vit_b_01ec64.pth
model.load_trainable_state_dict(load_file("syntra_weights/tier1/syntra_anchor_cls100.safetensors"))
model.eval()
```

`load_trainable_state_dict` writes the released tensors over the model's full state dict and
then loads it strictly. A key with the wrong name therefore raises an error, but a file that
leaves out a trainable tensor would load without complaint. Each release file covers all 209
trainable tensors (checked). The anchor scores 83.96 mIoU on the 37,935 test pairs at k=3
(83.97 in the paper).
