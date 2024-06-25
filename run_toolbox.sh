#!/usr/bin/env bash

MODEL="efficientformerv2_s0"
BASE_DIR="/data1/models/EFReplication/model_efficientformerv2_s0_lrs_cosine_distil_hard_dt_IMNET_ep20_bs64"
CKPT="${BASE_DIR}/${MODEL}.pth"
RESOLUTION=224

nohup python toolbox.py \
--model=$MODEL \
--ckpt=$CKPT \
--resolution=$RESOLUTION \
--profile \
--onnx \
--coreml > ${BASE_DIR}/${MODEL}_stats_export.txt 2>&1 &