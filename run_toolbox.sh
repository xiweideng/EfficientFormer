#!/usr/bin/env bash

MODEL="efficientformer_l3"
BASE_DIR="/data1/models/EFReplication/model_efficientformer_l3_distil_hard_dt_IMNET_ep20_bs64"
CKPT="${BASE_DIR}/${MODEL}.pth"
RESOLUTION=224

nohup python toolbox.py \
--model=$MODEL \
--ckpt=$CKPT \
--resolution=$RESOLUTION \
--profile \
--onnx \
--coreml > ${BASE_DIR}/${MODEL}_stats_export.txt 2>&1 &