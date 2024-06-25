#!/usr/bin/env bash

MODEL="efficientformerv2_s0"
# [efficientformer_l1, efficientformer_l3, efficientformer_l7,
# efficientformerv2_s0, efficientformerv2_s1,
# efficientformerv2_s2, efficientformerv2_l]
nGPUs=2
WARMUP_EPOCHS=2
export CUDA_VISIBLE_DEVICES="1,4"
BATCH_SIZE=64
EPOCHS=20
LR_SCHEDULER="cosine" # 'cosine', 'tanh', 'step', 'multistep', 'plateau', 'poly'
DISTILLATION_TYPE="hard" # 'none', 'soft', 'hard'
TEACHER_MODEL="regnety_160"
TEACHER_PATH="https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth"
DATASET="IMNET"
DATA_PATH="/data1/datasets/imagenet"
OUTPUT_NAME="model_${MODEL}_lrs_${LR_SCHEDULER}_distil_${DISTILLATION_TYPE}_dt_${DATASET}_ep${EPOCHS}_bs${BATCH_SIZE}"
OUTPUT_DIR="/data1/models/EFReplication/${OUTPUT_NAME}"
mkdir -p $OUTPUT_DIR
nohup torchrun --nproc_per_node=$nGPUs main.py \
--model $MODEL \
--warmup-epochs $WARMUP_EPOCHS \
--data-set $DATASET \
--distillation-type $DISTILLATION_TYPE \
--sched $LR_SCHEDULER \
--teacher-model $TEACHER_MODEL \
--teacher-path $TEACHER_PATH \
--data-path $DATA_PATH \
--output_dir $OUTPUT_DIR \
--batch-size $BATCH_SIZE \
--epochs $EPOCHS > ${OUTPUT_DIR}/train.txt 2>&1 &