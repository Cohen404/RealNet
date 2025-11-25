# RealNet RL训练指南

本指南说明如何使用强化学习(RL)配置文件训练RealNet模型。

## 文件说明

1. **rl_config.yaml** - RL配置文件，包含PPO算法参数、奖励函数权重和特征选择参数
2. **train_realnet_rl_fixed.py** - 修复后的训练脚本，正确加载和使用RL配置
3. **train_with_rl.sh** - 启动RL训练的Shell脚本

## 使用方法

### 1. 准备配置文件

确保`rl_config.yaml`文件位于RealNet根目录下。该文件包含以下关键配置：

- **ppo**: PPO算法参数
  - lr: 学习率
  - gamma: 折扣因子
  - eps_clip: 裁剪参数
  - k_epochs: 更新迭代次数
  - entropy_coef: 熵系数
  - value_coef: 价值函数系数

- **reward**: 奖励函数权重
  - alpha: 通道选择奖励权重
  - beta: 性能提升奖励权重
  - gamma: 计算效率奖励权重

- **training**: RL训练参数
  - update_every_n_epochs: 更新频率(每N个epoch更新一次)
  - batch_size: 批大小
  - max_timesteps: 最大时间步数

- **feature_selection**: 特征选择参数
  - min_channels: 最小通道数
  - max_channels_ratio: 最大通道比例
  - dynamic_inference: 是否启用动态推理

### 2. 启动训练

使用提供的Shell脚本启动训练：

```bash
./train_with_rl.sh
```

或者直接使用Python命令：

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train_realnet_rl_fixed.py \
    --config configs/mvtec/bottle.yaml \
    --use_rl \
    --rl_config rl_config.yaml \
    --class_name bottle
```

### 3. 关键参数说明

- `--config`: 主配置文件路径
- `--use_rl`: 启用RL特征选择
- `--rl_config`: RL配置文件路径
- `--class_name`: 数据集类别名称

## 工作原理

1. **配置加载**: 训练脚本会加载主配置文件和RL配置文件
2. **模型构建**: 将AFS模块替换为RLAFS，并传入RL配置
3. **RL初始化**: 在训练开始前初始化RL智能体和环境
4. **训练循环**: 在每个epoch后更新RL智能体
5. **检查点保存**: 保存模型参数和RL智能体状态

## 注意事项

1. 确保所有依赖项已正确安装
2. 根据实际硬件调整GPU数量和分布式设置
3. 可以通过修改`rl_config.yaml`调整RL参数
4. 训练日志保存在`logs/mvtec/bottle/rl/`目录下

## 故障排除

1. 如果出现导入错误，请检查Python路径设置
2. 如果CUDA内存不足，请减少批大小或GPU数量
3. 如果RL智能体不收敛，请调整学习率和奖励权重