#!/usr/bin/env bash

MODEL="efficientformer_l1"
nGPUs=1
export CUDA_VISIBLE_DEVICES="0"
OUTPUT_DIR="efficientformer_l1_2d"
DATASET="IMNET"
DATA_PATH="/data1/datasets/imagenet"
CHECKPOINT=$OUTPUT_DIR/checkpoint.pth

nohup torchrun --nproc_per_node=$nGPUs main.py \
--model $MODEL \
--resume $CHECKPOINT \
--eval \
--data-set $DATASET \
--data-path $DATA_PATH \
--output_dir $OUTPUT_DIR > test.txt 2>&1 &