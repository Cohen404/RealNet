# RealNet RL-based Dynamic Feature Selection for Inference

This guide explains how to use the RL-based dynamic feature selection feature during inference with RealNet.

## Overview

RealNet now supports dynamic feature selection during inference, where the trained RL agent actively selects the most relevant feature channels for each input image. This allows the model to adapt its feature selection strategy based on the specific characteristics of each input, potentially improving performance on diverse test cases.

## Key Features

- **Dynamic Channel Selection**: The RL agent selects different feature channels for each input image during inference
- **Adaptive Feature Usage**: The model can focus on different aspects of the input depending on what's most relevant
- **Trained Policy Utilization**: Uses the policy learned during training to make informed decisions about feature selection

## Files Modified

1. **models/afs/rl_afs.py**: Updated to support dynamic inference
2. **rl_config.yaml**: Added `dynamic_inference` configuration option
3. **train_realnet_rl.py**: Updated to properly load and use RL state during inference
4. **infer_with_rl.py**: New script specifically for inference with RL-based dynamic feature selection

## Configuration

To enable dynamic feature selection during inference, ensure your RL configuration has the following setting:

```yaml
rl:
  enabled: True
  feature_selection:
    enabled: True
    dynamic_inference: True  # This enables dynamic feature selection during inference
    # ... other RL settings
```

## Usage

### 1. Training with RL

First, train your model with RL-based feature selection:

```bash
python train_realnet_rl.py --config configs/realnet_mvtec.yaml --rl_config rl_config.yaml --dataset mvtec --class_name bottle --epochs 100 --use_rl
```

### 2. Inference with Dynamic Feature Selection

There are two ways to use the dynamic feature selection during inference:

#### Option A: Using the dedicated inference script (recommended)

```bash
python infer_with_rl.py --config configs/infer_rl_example.yaml --rl_config rl_config.yaml --checkpoint checkpoints/best_model.pth --input path/to/test/image/or/folder --output ./results
```

#### Option B: Using the training script in evaluation mode

```bash
python train_realnet_rl.py --config configs/realnet_mvtec.yaml --rl_config rl_config.yaml --dataset mvtec --class_name bottle --evaluate --resume checkpoints/best_model.pth
```

## Example

Here's a complete example workflow:

1. **Train the model with RL**:
   ```bash
   python train_realnet_rl.py --config configs/realnet_mvtec.yaml --rl_config rl_config.yaml --dataset mvtec --class_name bottle --epochs 100 --use_rl
   ```

2. **Run inference with dynamic feature selection**:
   ```bash
   python infer_with_rl.py --config configs/infer_rl_example.yaml --rl_config rl_config.yaml --checkpoint checkpoints/best_model.pth --input ./datasets/MVTec/bottle/test --output ./results
   ```

## How It Works

1. During training, the RL agent learns a policy for selecting feature channels that maximize the anomaly detection performance.
2. When `dynamic_inference` is enabled, the model uses this learned policy during inference to select feature channels for each input image.
3. The RL agent observes the current input features and decides which channels to keep or discard based on the learned policy.
4. The model then processes the selected features through the rest of the network to generate the final anomaly map.

## Benefits

- **Adaptability**: The model can adapt its feature selection strategy to different input images.
- **Efficiency**: By selecting only the most relevant features, the model can potentially reduce computation.
- **Performance**: Dynamic selection can improve performance on diverse test cases by focusing on the most informative features for each input.

## Notes

- The RL agent must be trained before using dynamic inference.
- The checkpoint must contain the RL state (`rl_state_dict`) to use dynamic inference.
- Performance may vary depending on the quality of the trained RL policy and the diversity of the test data.
- Dynamic inference may slightly increase inference time due to the additional RL agent processing.

## Troubleshooting

1. **Error: "RL state not found"**: Make sure your checkpoint contains the RL state. Train with `--use_rl` flag to ensure RL state is saved.
2. **Error: "Dynamic inference not enabled"**: Check that `dynamic_inference: True` is set in your RL configuration.
3. **Poor performance**: The RL agent may need more training or different hyperparameters. Consider adjusting the reward weights or training parameters in `rl_config.yaml`.