#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=localhost
export MASTER_PORT=12345
export WORLD_SIZE=4

# 配置文件路径
CONFIG_FILE="experiments/MVTec-AD/realnet.yaml"
RL_CONFIG_FILE="rl_config.yaml"

# 日志目录
LOG_DIR="logs/mvtec/bottle/rl"
mkdir -p $LOG_DIR

# 训练命令
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_realnet_rl_fixed.py \
    --config $CONFIG_FILE \
    --use_rl \
    --rl_config $RL_CONFIG_FILE \
    --class_name bottle \
    2>&1 | tee $LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log