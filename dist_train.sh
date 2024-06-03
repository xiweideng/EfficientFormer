#!/usr/bin/env bash

MODEL="efficientformer_l1"
DATASET="IMNET"
nGPUs=4
export CUDA_VISIBLE_DEVICES="0,1,2,7"
BATCH_SIZE=128
EPOCHS=2
DATA_PATH="/data1/datasets/imagenet"
OUTPUT_DIR="efficientformer_l1_${EPOCHS}d"
nohup torchrun --nproc_per_node=$nGPUs main.py \
--model $MODEL \
--data-set $DATASET \
--data-path $DATA_PATH \
--output_dir $OUTPUT_DIR \
--batch-size $BATCH_SIZE \
--epochs $EPOCHS > train_${EPOCHS}.txt 2>&1 &