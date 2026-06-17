#!/bin/bash

# 设置可见 GPU
export CUDA_VISIBLE_DEVICES=1

# 显存碎片优化，减少"有空间但无连续块"导致的 OOM
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# BTAD 数据集的所有类别
CLASS_NAMES=("01" "02" "03")

# 配置文件路径
CONFIG_FILE="experiments/BTAD/realnet.yaml"
RL_CONFIG_FILE="rl_config.yaml"
RRS_RL_CONFIG_FILE="rrs_rl_config.yaml"

# 训练单个类别的函数（带自动降 batch_size 重试）
train_one_class() {
    local class_name="$1"

    LOG_DIR="logs/btad/${class_name}/rl"
    mkdir -p "$LOG_DIR"

    local bs=4   # 从 4 开始，OOM 则降为 2

    while [ "$bs" -ge 2 ]; do
        # 写入当前 batch_size
        sed -i "s/batch_size: [0-9]*/batch_size: ${bs}/" "$CONFIG_FILE"
        echo "当前 batch_size=${bs}"

        local log_file="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S)_bs${bs}.log"

        echo "========================================"
        echo "开始训练 BTAD 类别: $class_name (batch_size=${bs})"
        echo "========================================"

        # 记录开始时间
        local start_time=$(date +%s)

        # 用临时文件捕获 stderr 来判断 OOM
        local err_file=$(mktemp)
        torchrun \
            --nproc_per_node=1 \
            --master_addr=localhost \
            --master_port=12345 \
            train_realnet_rl_fixed.py \
            --config "$CONFIG_FILE" \
            --use_rl \
            --rl_config "$RL_CONFIG_FILE" \
            --rrs_rl_config "$RRS_RL_CONFIG_FILE" \
            --dataset BTAD \
            --class_name "$class_name" \
            > >(tee "$log_file") 2> "$err_file"

        local exit_code=$?

        # 检查是否 OOM
        if [ $exit_code -ne 0 ] && grep -qi "out of memory\|OOM\|CUDA error" "$err_file" "$log_file" 2>/dev/null; then
            echo "[警告] 检测到 OOM，batch_size=${bs} 爆显存"

            if [ "$bs" -eq 2 ]; then
                echo "[错误] batch_size=2 仍然 OOM，请手动排查！"
                rm -f "$err_file"
                return 1
            fi

            bs=2
            rm -f "$err_file"
            echo "自动将 batch_size 降为 2 重试..."
            sleep 3   # 等 GPU 显存释放
            continue
        fi

        rm -f "$err_file"

        if [ $exit_code -ne 0 ]; then
            echo "[错误] 训练异常退出，退出码: $exit_code"
            return 1
        fi

        echo "========================================"
        echo "类别 $class_name 训练完成"
        echo "========================================"
        return 0
    done
}

# 主循环
for class_name in "${CLASS_NAMES[@]}"; do
    if ! train_one_class "$class_name"; then
        echo "类别 $class_name 训练失败，跳过后续类别"
        exit 1
    fi
done

echo "所有 BTAD 类别训练完成！"
