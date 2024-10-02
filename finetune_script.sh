#!/bin/bash
# set -x  # print the commands
# Set environment variable for protobuf
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# MODEL="vit_base_patch16_224"
MODEL="vit_large_patch16_224"

OUTPUT_DIR="/home/bawolf/workspace/break/finetune/output/${MODEL}/"
DATA_PATH="/home/bawolf/workspace/break/finetune/inputs"
MODEL_PATH="/home/bawolf/workspace/break/finetune/weights/vit_g_hybrid_pt_1200e.pth"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR


torchrun --nproc_per_node=1 run_class_finetuning.py \
    --model ${MODEL} \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 1 \
    --opt adamw \
    --lr 5e-5 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 50 \
    --warmup_epochs 5 \
    --dist_eval \
    --test_num_segment 5 \
    --test_num_crop 3 \
    --drop_path 0.1 \
    --aa rand-m7-mstd0.5-inc1 \
    --reprob 0.3 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --nb_classes 3