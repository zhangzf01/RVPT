#!/bin/bash

# custom config
DATA=/home/zzf/data


TRAINER=VPT
DATASET=$1

CFG=coop_vit_b32_bs32 

SHOTS=16  # number of shots (1, 2, 4, 8, 16)
RESUME=$2
NCTX=$3
V_DEPTH=$4
MARGIN=$5
ALPHA=$6
NOTE=$7

Rdir=$(basename "$RESUME" | sed 's/_\([0-9]\+\)\.pt//')


for SEED in 1 2 3
do
    DIR=results/${NOTE}/${Rdir}/${DATASET}/${NCTX}_${V_DEPTH}_${MARGIN}_${ALPHA}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${CFG}.yaml \
        --output-dir ${DIR} \
        --margin ${MARGIN}\
        --alpha ${ALPHA}\
        --resume ${RESUME}\
        TRAINER.VPT.PROMPT_DEPTH_VISION ${V_DEPTH}\
        TRAINER.VPT.N_CTX_VISION  ${NCTX} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done