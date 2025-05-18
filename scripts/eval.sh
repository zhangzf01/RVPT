#!/bin/bash

# custom config
DATA=/home/zzf/data
CFG=coop_vit_b32_bs32 

DATASET=$1
CKPT=$2
RESUME=$3


for SEED in 1 2 3
do
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python train.py \
        --root ${DATA} \
        --trainer VPT \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${CFG}.yaml \
        --model-dir ${CKPT} \
        --resume ${RESUME} \
        --eval-only\
        --alpha 2\
        --margin -1\
        TRAINER.VPT.PROMPT_DEPTH_VISION 9\
        TRAINER.VPT.N_CTX_VISION  50 \
        DATASET.NUM_SHOTS 16
    fi
done