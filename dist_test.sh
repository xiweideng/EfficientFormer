#!/usr/bin/env bash

MODEL="efficientformer_l1"
nGPUs=2
CUDA_VISIBLE_DEVICES="0,1"
OUTPUT_DIR="efficientformer_test"
DATA_PATH="/path/to/imagenet"
CHECKPOINT=$OUTPUT_DIR/checkpoint.pth

nohup CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python -m torch.distributed.launch \
--nproc_per_node=$nGPUs \
--use_env main.py \
--model $MODEL \
--resume $CHECKPOINT \
--eval \
--data-path $DATA_PATH \
--output_dir $OUTPUT_DIR > test.txt 2>&1 &