#!/usr/bin/env bash

DATADIR=/home/yuan/data/syntra/hismap


# for CONFIG in hismap_n1_s100_v4 end; do
# if [ "$CONFIG" = "end" ]; then
#     break
# fi
    
#   PYTHONPATH=. python syntra/train_one_gpu.py \
#   --config syntra_${CONFIG}.yaml -d ${DATADIR}  
#   --resume ./logs/syntra_${CONFIG}/model_latest.pth # --vis-freq 100 

#   # PYTHONPATH=. python syntra/test.py \
#   # --config syntra_${CONFIG}.yaml -d ${DATADIR} \
#   # --save-result --vis --checkpoint ./logs/syntra_${CONFIG}/model_latest.pth 

# done

for CONFIG in hismap_unet hismap_unet_group_samp end; do
if [ "$CONFIG" = "end" ]; then
    break
fi
  PYTHONPATH=. python syntra/train_unet.py \
        --config syntra_${CONFIG}.yaml \
        -d ${DATADIR}  \
        --resume logs/syntra_${CONFIG}/model_latest.pth

  PYTHONPATH=. python syntra/test_unet.py \
        --config syntra_${CONFIG}.yaml \
        -d ${DATADIR}  \
        --checkpoint logs/syntra_${CONFIG}/model_latest.pth \
        --data-info sample_pair_cls_selection_100_test_samples_intracls
done   

# CONFIG=trees_sep_n4_full
# DATADIR=/home/yuan/data/syntra/forest_monitoring

# PYTHONPATH=. python syntra/train_one_gpu.py \
#   --config syntra_${CONFIG}.yaml -d ${DATADIR} --cache-data \
#   # --resume ./logs/syntra_${CONFIG}/model_latest.pth # --vis-freq 100 --disable-runid

# PYTHONPATH=. python syntra/test.py \
#   --config syntra_${CONFIG}.yaml -d ${DATADIR} \
#   --save-result --vis # --checkpoint ./logs/syntra_${CONFIG}/model_latest.pth 