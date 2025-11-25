# Dynamic Feature Selection with PPO Policy

## Overview

This document explains how to use the trained PPO (Proximal Policy Optimization) policy for dynamic feature selection during inference. Unlike fixed channel selection, the PPO policy allows the model to dynamically select different channels for different input images while maintaining a fixed total number of channels.

## Key Features

1. **Dynamic Channel Selection**: The PPO network selects channels based on input features, allowing for adaptive feature selection.
2. **Fixed Channel Count**: Despite dynamic selection, the total number of channels remains constant, avoiding dimension changes.
3. **Policy Persistence**: The trained PPO policy is saved with the model checkpoint and can be loaded for inference.

## How It Works

### Training Phase

During training, the PPO agent learns to select optimal channels based on performance metrics. The training process includes:

1. **State Representation**: The current feature selection and performance metrics.
2. **Action Selection**: Choosing which channels to select or deselect.
3. **Reward Calculation**: Based on model performance metrics (AUC, F1, etc.).
4. **Policy Update**: Using PPO algorithm to update the policy network.

### Inference Phase

During inference, the trained PPO policy is used to dynamically select channels:

1. **Policy Loading**: The PPO policy is loaded from the checkpoint.
2. **Dynamic Selection**: For each input image, the PPO network selects channels based on the input features.
3. **Fixed Channel Count**: The selection process ensures a constant number of channels are selected.
4. **Feature Processing**: The selected channels are processed by the model.

## Usage

### Training with PPO Policy

To train a model with PPO-based dynamic feature selection:

```bash
python train_realnet_rl.py --config configs/mvtec.yaml --rl_config rl_config.yaml --save ./checkpoints/realnet_rl --class_name bottle
```

### Inference with PPO Policy

To use the trained PPO policy for inference:

```bash
python infer_with_rl.py --config configs/mvtec.yaml --checkpoint ./checkpoints/realnet_rl/model_best.pth --input ./test_images --output ./results --use_rl
```

The `--use_rl` flag enables the use of the PPO policy for dynamic feature selection during inference.

## Configuration

### RL Configuration

The RL configuration is specified in `rl_config.yaml`:

```yaml
# PPO hyperparameters
ppo:
  lr: 0.001
  eps_clip: 0.2
  k_epochs: 4
  entropy_coef: 0.01
  value_coef: 0.5
  
# Reward function weights
reward:
  auc_weight: 1.0
  f1_weight: 0.5
  efficiency_weight: 0.2
  
# RL training parameters
training:
  update_frequency: 10
  batch_size: 32
  
# Feature selection parameters
feature_selection:
  selection_ratio: 0.5
  min_channels: 8
  max_channels: 64
  
# Enable dynamic inference
dynamic_inference: true
```

### Model Configuration

The model configuration should include the RL-based feature selection:

```yaml
model:
  type: realnet
  feature_selection:
    type: rl
    structure:
      - name: block1
        layers:
          - idx: 0
            planes: 64
          - idx: 1
            planes: 64
      - name: block2
        layers:
          - idx: 0
            planes: 128
          - idx: 1
            planes: 128
```

## Implementation Details

### PPO Policy Saving

The PPO policy is saved along with the model checkpoint:

```python
# Save PPO policy
ppo_policy = model.afs.save_ppo_policy()
checkpoint['ppo_policy'] = ppo_policy
```

### PPO Policy Loading

The PPO policy is loaded during inference:

```python
# Load PPO policy
model.afs.load_ppo_policy(checkpoint['ppo_policy'])
model.afs.enable_ppo_inference()
```

### Fixed Channel Count

The implementation ensures a fixed number of channels are selected:

```python
# Ensure we maintain a fixed number of channels
if hasattr(self, '_target_channels') and block_name in self._target_channels:
    target_channels = self._target_channels[block_name]
else:
    total_channels = sum([layer['planes'] for layer in block['layers']])
    target_channels = min(total_channels // 2, total_channels)

if len(current_selection) > target_channels:
    # Remove excess channels
    current_selection = current_selection[:target_channels]
elif len(current_selection) < target_channels:
    # Add channels if needed
    available_channels = [i for i in range(total_channels) if i not in current_selection]
    needed = target_channels - len(current_selection)
    current_selection.extend(available_channels[:needed])
```

## Advantages

1. **Adaptive Feature Selection**: The model can adapt to different input images by selecting different channels.
2. **Consistent Dimensions**: The fixed channel count ensures consistent model dimensions.
3. **Improved Performance**: Dynamic selection can lead to better performance on diverse inputs.
4. **Efficiency**: The model can focus on the most relevant channels for each input.

## Troubleshooting

### Common Issues

1. **Dimension Mismatch**: Ensure the target channel count is set correctly for each block.
2. **Policy Not Loaded**: Check that the checkpoint contains the PPO policy and that `--use_rl` is specified.
3. **Performance Degradation**: Verify that the PPO policy is properly trained and configured.

### Debugging Tips

1. Check the PPO policy loading messages in the inference log.
2. Verify the channel selection process by adding debug prints.
3. Ensure the RL configuration matches the training configuration.

## Future Improvements

1. **Adaptive Channel Count**: Allow the channel count to adapt based on input complexity.
2. **Multi-Objective Optimization**: Incorporate additional objectives into the reward function.
3. **Hierarchical Selection**: Implement hierarchical channel selection for more granular control.
4. **Online Learning**: Enable online learning during inference for continuous adaptation.