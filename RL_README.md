# RealNet with Reinforcement Learning-based Feature Selection

This implementation enhances RealNet with a reinforcement learning (RL) based feature selection mechanism that dynamically optimizes channel selection during training and inference.

## Key Features

- **Dynamic Feature Selection**: Uses RL agents to dynamically select optimal channels for each block
- **Performance-driven Optimization**: Optimizes feature selection based on model performance metrics
- **Adaptive Channel Selection**: Adapts channel selection based on input data characteristics
- **Seamless Integration**: Integrates with the existing RealNet architecture without major changes

## New Files

### RL Agent Implementation
- `models/rl_agent/ppo_agent.py`: PPO (Proximal Policy Optimization) agent implementation
- `models/rl_agent/feature_selection_env.py`: Environment for feature selection task

### RL-Enhanced AFS Module
- `models/afs/rl_afs.py`: RL-enhanced version of the Adaptive Feature Selection (AFS) module

### Training Script
- `train_realnet_rl.py`: Training script with RL support and dynamic inference capabilities

### Configuration
- `rl_config.yaml`: Configuration file for RL hyperparameters and settings

## Reward Function

The RL agent optimizes channel selection using the following reward function:

```
R = α * ΔPixelAUC + β * ΔImageAUC - γ * (SelectedChannels / TotalChannels)
```

Where:
- α, β, γ are weighting factors (configurable in `rl_config.yaml`)
- ΔPixelAUC and ΔImageAUC are improvements in performance metrics
- SelectedChannels / TotalChannels penalizes selecting too many channels

## Usage

### Training with RL

To train RealNet with RL-based feature selection:

```bash
# Basic RL training
python train_realnet_rl.py --config configs/realnet_mvtec.yaml --use_rl --rl_config rl_config.yaml

# Training with specific class
python train_realnet_rl.py --config configs/realnet_mvtec.yaml --use_rl --rl_config rl_config.yaml --class_name bottle
```

### Inference with Dynamic Feature Selection

To run inference with dynamic feature selection enabled:

```bash
# Inference with dynamic feature selection
python train_realnet_rl.py --config configs/realnet_mvtec.yaml --use_rl --rl_config rl_config.yaml --evaluate --resume path/to/checkpoint.pth
```

### Training without RL

To train the original RealNet without RL:

```bash
python train_realnet_rl.py --config configs/realnet_mvtec.yaml
```

## Configuration

The RL behavior can be configured through `rl_config.yaml`:

### PPO Hyperparameters
- `lr`: Learning rate for the PPO agent
- `gamma`: Discount factor for future rewards
- `eps_clip`: Clipping parameter for PPO
- `k_epochs`: Number of epochs per update
- `entropy_coef`: Coefficient for entropy regularization
- `value_coef`: Coefficient for value function loss

### Reward Function Weights
- `alpha`: Weight for PixelAUC improvement
- `beta`: Weight for ImageAUC improvement
- `gamma`: Weight for channel selection penalty

### Training Parameters
- `update_every_n_epochs`: Frequency of RL agent updates
- `batch_size`: Batch size for RL training
- `max_timesteps`: Maximum timesteps per episode

### Feature Selection Parameters
- `min_channels`: Minimum number of channels to select
- `max_channels_ratio`: Maximum ratio of channels that can be selected
- `dynamic_inference`: Enable dynamic feature selection during inference (default: true)

## Implementation Details

### RL Agent
The PPO agent maintains a policy network that outputs action probabilities for channel selection. The agent learns to select channels that maximize the reward function based on model performance.

### Feature Selection Environment
The environment provides a state representation based on feature statistics and returns rewards based on the performance improvement from the selected channels.

### RL-AFS Module
The RL-AFS module integrates with the original AFS module and replaces the fixed channel selection with dynamic selection based on the RL agent's policy.

### Dynamic Inference
When `dynamic_inference` is enabled, the model uses the trained RL agent to dynamically select channels during inference, allowing the model to adapt its feature selection based on input data characteristics.

## Notes

- The RL agents are trained alongside the main model and are updated based on validation performance
- The RL state is saved along with the model checkpoints
- Dynamic inference can be disabled by setting `dynamic_inference: false` in `rl_config.yaml`
- The implementation is compatible with both single-GPU and multi-GPU training

## Future Improvements

- Implement more advanced RL algorithms (e.g., SAC, TD3)
- Add support for hierarchical feature selection
- Implement curriculum learning for the RL agents
- Add visualization tools for feature selection patterns