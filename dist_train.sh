#!/usr/bin/env bash

MODEL="efficientformer_l3"
# [efficientformer_l1, efficientformer_l3, efficientformer_l7,
# efficientformerv2_s0, efficientformerv2_s1,
# efficientformerv2_s2, efficientformerv2_l]
nGPUs=4
WARMUP_EPOCHS=1
export CUDA_VISIBLE_DEVICES="0,1,2,7"
BATCH_SIZE=32
EPOCHS=10
DATASET="IMNET"
DATA_PATH="/data1/datasets/imagenet"
OUTPUT_DIR="model${MODEL}_dt${DATASET}_ep${EPOCHS}_bs${BATCH_SIZE}"
nohup torchrun --nproc_per_node=$nGPUs main.py \
--model $MODEL \
--warmup-epochs $WARMUP_EPOCHS \
--data-set $DATASET \
--data-path $DATA_PATH \
--output_dir $OUTPUT_DIR \
--batch-size $BATCH_SIZE \
--epochs $EPOCHS > train_model${MODEL}_dt${DATASET}_ep${EPOCHS}_bs${BATCH_SIZE}.txt 2>&1 &