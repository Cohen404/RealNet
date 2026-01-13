#!/bin/bash

# 设置可见 GPU（可选，torchrun 通常能自动处理）
export CUDA_VISIBLE_DEVICES=0

# BTAD 数据集的所有类别
CLASS_NAMES=("01" "02" "03")

# 遍历所有类别进行训练
for class_name in "${CLASS_NAMES[@]}"; do
    # 日志目录
    LOG_DIR="logs/btad/${class_name}/rl"
    mkdir -p "$LOG_DIR"

    # 配置文件路径
    CONFIG_FILE="experiments/BTAD/realnet.yaml"
    # AFS 的 RL 配置
    RL_CONFIG_FILE="rl_config.yaml"
    # RRS 的 RL 配置（与 AFS 分开）
    RRS_RL_CONFIG_FILE="rrs_rl_config.yaml"

    echo "========================================"
    echo "开始训练 BTAD 类别: $class_name"
    echo "========================================"

    # 使用 torchrun 启动分布式训练（推荐方式）
    # 注意：不再需要手动设置 MASTER_ADDR/MASTER_PORT/WORLD_SIZE，
    #       torchrun 会自动处理单机多卡的通信初始化。

    torchrun \
        --nproc_per_node=1 \
        --master_addr=localhost \
        --master_port=12345 \
        train_realnet_rl_fixed.py \
        --config "$CONFIG_FILE" \
        --use_rl \
        --rl_config "$RL_CONFIG_FILE" \
        --rrs_rl_config "$RRS_RL_CONFIG_FILE" \
        --class_name "$class_name" \
        2>&1 | tee "$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

    echo "========================================"
    echo "类别 $class_name 训练完成"
    echo "========================================"
done

echo "所有 BTAD 类别训练完成！"