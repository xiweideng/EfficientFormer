#!/usr/bin/env bash

MODEL="efficientformer_l1"
nGPUs=2
CUDA_VISIBLE_DEVICES="0,1"
BATCH_SIZE=64
EPOCHS=5
DATA_PATH="/path/to/imagenet"
OUTPUT_DIR="efficientformer_l1_300d"
RESUME=""
EVAL=""
FINETUNE=""
DISTILLATION_TYPE=""
DISTILLATION_ALPHA=""
DISTILLATION_TAU=""

nohup CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python -m torch.distributed.launch \
--nproc_per_node=$nGPUs \
--use_env main.py \
--model $MODEL \
--data-path $DATA_PATH \
--output_dir $OUTPUT_DIR \
--resume $RESUME \
--eval $EVAL \
--finetune $FINETUNE \
--distillation_type $DISTILLATION_TYPE \
--distillation_alpha $DISTILLATION_ALPHA \
--distillation_tau $DISTILLATION_TAU \
--batch-size $BATCH_SIZE \
--epochs $EPOCHS > train.txt 2>&1 &