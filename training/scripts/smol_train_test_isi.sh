#!/usr/bin/env bash

DATADIR=/home/yuan/data/syntra/hismap_zs


# for CONFIG in newcls_fewshot_finetune newcls_fewshot_scratch end; do
# if [ "$CONFIG" = "end" ]; then
#     break
# fi
    
#   PYTHONPATH=. python syntra/train_one_gpu.py \
#   --config smol_${CONFIG}.yaml -d ${DATADIR} \
#   --resume ./logs/smol_${CONFIG}/model_latest.pth # --vis-freq 100   logs/smol_hismap_n1_full_uni_samp/model_90000.pth

#   PYTHONPATH=. python syntra/test.py \
#   --config smol_${CONFIG}.yaml -d ${DATADIR} \
#   --save-result --checkpoint ./logs/smol_${CONFIG}/model_latest.pth 

# done

CONFIG=newcls_fewshot_finetune

PYTHONPATH=. python syntra/train_one_gpu.py \
--config smol_${CONFIG}.yaml -d ${DATADIR} \
--resume ./logs/smol_hismap_n1_full/model_latest.pth # --vis-freq 100   logs/smol_hismap_n1_full_uni_samp/model_90000.pth

PYTHONPATH=. python syntra/test.py \
--config smol_${CONFIG}.yaml -d ${DATADIR} \
--save-result --checkpoint ./logs/smol_${CONFIG}/model_latest.pth 

CONFIG=newcls_fewshot_scratch

PYTHONPATH=. python syntra/train_one_gpu.py \
--config smol_${CONFIG}.yaml -d ${DATADIR} 

PYTHONPATH=. python syntra/test.py \
--config smol_${CONFIG}.yaml -d ${DATADIR} \
--save-result --checkpoint ./logs/smol_${CONFIG}/model_latest.pth 