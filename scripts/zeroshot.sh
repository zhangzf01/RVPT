#!/bin/bash

# custom config
DATA=/home/zzf/data
TRAINER=ZeroshotCLIP
CFG=coop_vit_b32_bs32 

DATASET=$1
RESUME=$2


python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${CFG}.yaml \
--output-dir results/${TRAINER}/${CFG}/${DATASET} \
--resume ${RESUME} \
--eval-only
