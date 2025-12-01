#!/bin/bash

# 设置可见 GPU（可选）
export CUDA_VISIBLE_DEVICES=0

# 日志 / 输出目录
OUTPUT_DIR="inference_output/bottle"
mkdir -p "$OUTPUT_DIR"

# 配置与权重路径
CONFIG_FILE="experiments/MVTec-AD/realnet.yaml"
RL_CONFIG_FILE="rl_config.yaml"          # AFS 的 RL 配置
RRS_RL_CONFIG_FILE="rrs_rl_config.yaml"  # RRS 的 RL 配置（推理时主要影响结构一致性）
CKPT_FILE="experiments/MVTec-AD/realnet_checkpoints/bottle/ckpt_best.pth.tar"

# 待推理图片，可以是单张图片路径或“包含多级子文件夹的根目录”
# 例如：MVTec 的 bottle 测试集根目录：
INPUT_PATH="data/MVTec-AD/mvtec/bottle/test"

# 运行推理（启用 AFS/RRS 的 RL 策略）
python infer_with_rl.py \
    --config "$CONFIG_FILE" \
    --checkpoint "$CKPT_FILE" \
    --rl_config "$RL_CONFIG_FILE" \
    --input "$INPUT_PATH" \
    --output "$OUTPUT_DIR" \
    --dataset mvtec \
    --class_name bottle \
    --device cuda \
    --use_rl