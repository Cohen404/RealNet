import torch
import torch.nn as nn
from tqdm import tqdm
from utils.misc_helper import to_device
import torch.nn.functional as F
import torch.distributed as dist
import copy
import numpy as np
from ..rl_agent.ppo_agent import PPOAgent
from ..rl_agent.feature_selection_env import FeatureSelectionEnv


class RLAFS(nn.Module):
    """
    Adaptive Feature Selection with Reinforcement Learning
    """
    
    def __init__(self, inplanes, instrides, structure, init_bsn=100, rl_config=None):
        """
        Initialize RL-based Adaptive Feature Selection
        
        Args:
            inplanes: Number of input planes for each layer
            instrides: Input strides for each layer
            structure: Network structure
            init_bsn: Initial batch size number
            rl_config: RL configuration
        """
        super(RLAFS, self).__init__()
        self.structure = structure
        self.inplanes = inplanes
        self.instrides = instrides
        self.init_bsn = init_bsn
        
        # RL configuration
        if rl_config is None:
            rl_config = {
                'lr': 0.001,
                'alpha': 1.0,
                'beta': 0.5,
                'gamma': 0.1,
                'init_epochs': 10,
                'update_freq': 5,
                'dynamic_inference': False
            }
        
        self.rl_config = rl_config
        self.rl_enabled = False
        self.update_counter = 0
        
        # Initialize indexes
        self.indexes = nn.ParameterDict()
        
        # Initialize RL agents and environments for each block
        self.rl_agents = {}
        self.rl_envs = {}
        
        # Store fixed policy for inference
        self.fixed_policy = {}
        self.use_fixed_policy = False
        
        for block in self.structure:
            block_name = block['name']
            
            # Initialize upsampling layers
            for layer in block['layers']:
                self.indexes["{}_{}".format(block_name, layer['idx'])] = nn.Parameter(
                    torch.zeros(layer['planes']).long(), requires_grad=False
                )
                self.add_module(
                    "{}_{}_upsample".format(block_name, layer['idx']),
                    nn.UpsamplingBilinear2d(scale_factor=self.instrides[layer['idx']] / block['stride'])
                )
            
            # Initialize RL agent and environment for this block
            total_channels = sum([layer['planes'] for layer in block['layers']])
            state_dim = total_channels
            action_dim = total_channels
            
            self.rl_agents[block_name] = PPOAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                lr=self.rl_config['ppo']['lr'],
                gamma=self.rl_config['ppo']['gamma'],
                eps_clip=self.rl_config['ppo']['eps_clip'],
                k_epochs=self.rl_config['ppo']['k_epochs']
            )
            
            # We'll initialize the environment later when we have initial selections
            self.rl_envs[block_name] = None
            
            # Initialize fixed policy for this block
            self.fixed_policy[block_name] = None
        
        # Store previous metrics for reward calculation
        self.prev_metrics = {}
        for block in self.structure:
            self.prev_metrics[block['name']] = {
                'pixel_auc': 0.0,
                'image_auc': 0.0
            }

    @torch.no_grad()
    def forward(self, inputs, train=False):
        """
        Forward pass
        
        Args:
            inputs: Input dictionary containing features
            train: Whether in training mode
            
        Returns:
            Dictionary containing block features
        """
        block_feats = {}
        feats = inputs["feats"]
        
        for block in self.structure:
            block_name = block['name']
            block_feats[block_name] = []
            
            # Collect all features for this block first
            block_layer_feats = []
            for layer in block['layers']:
                block_layer_feats.append(feats[layer['idx']]['feat'])
            
            # Determine channel selection method
            if self.rl_enabled and (train or self.rl_config.get('dynamic_inference', False)):
                # Use RL agent for dynamic selection
                # Build state from concatenated features
                # We need to ensure the state dimension matches the configured planes
                concat_feats = torch.cat([f.mean(dim=[2, 3]) for f in block_layer_feats], dim=1)
                
                # Get actual and configured channel dimensions
                actual_channels = concat_feats.size(1)
                configured_channels = sum([layer['planes'] for layer in block['layers']])
                
                # Adjust state dimension if needed
                if actual_channels != configured_channels:
                    # If actual channels > configured channels, truncate
                    if actual_channels > configured_channels:
                        concat_feats = concat_feats[:, :configured_channels]
                    # If actual channels < configured channels, pad with zeros
                    else:
                        padding = torch.zeros(concat_feats.size(0), configured_channels - actual_channels, 
                                              device=concat_feats.device)
                        concat_feats = torch.cat([concat_feats, padding], dim=1)
                
                state = concat_feats.cpu().numpy()
                
                # Get action from RL agent
                agent = self.rl_agents[block_name]
                action, _ = agent.select_action(state)
                
                # Convert action to channel indices
                total_channels = sum([layer['planes'] for layer in block['layers']])
                selected_mask = np.zeros(total_channels, dtype=bool)
                
                # Apply action to modify current selection
                if hasattr(self, '_current_selection') and block_name in self._current_selection:
                    current_selection = self._current_selection[block_name].copy()
                else:
                    # Initialize with default selection if not exists
                    current_selection = list(range(min(total_channels // 2, total_channels)))
                    if not hasattr(self, '_current_selection'):
                        self._current_selection = {}
                    self._current_selection[block_name] = current_selection
                
                # Toggle the selected channel
                if action in current_selection:
                    current_selection.remove(action)
                else:
                    current_selection.append(action)
                
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
                
                # Update current selection
                self._current_selection[block_name] = current_selection
                selected_mask[current_selection] = True
                
                # Apply selection to each layer, but keep channel dimension
                # consistent with the configured planes so that downstream
                # modules (e.g., recon) see a fixed number of channels.
                offset = 0
                for i, layer in enumerate(block['layers']):
                    # Use actual channel count from the feature tensor
                    layer_feat = block_layer_feats[i]
                    actual_layer_planes = layer_feat.size(1)
                    configured_layer_planes = layer['planes']

                    # Get mask for this layer based on configured planes
                    layer_mask = selected_mask[offset:offset + configured_layer_planes]
                    layer_indices = np.where(layer_mask)[0]

                    # Convert back to layer-relative indices
                    layer_relative_indices = [idx - offset for idx in layer_indices]

                    # Ensure indices are within the actual channel range
                    valid_indices = [idx for idx in layer_relative_indices if idx < actual_layer_planes]

                    # Upsample full feature map first
                    upsample = getattr(self, "{}_{}_upsample".format(block_name, layer['idx']))
                    layer_feat_up = upsample(layer_feat)

                    # Prepare output tensor with fixed channel size
                    B, C_up, H_up, W_up = layer_feat_up.shape
                    feat_c = torch.zeros(
                        B,
                        configured_layer_planes,
                        H_up,
                        W_up,
                        device=layer_feat_up.device,
                        dtype=layer_feat_up.dtype,
                    )

                    if valid_indices:
                        # Copy selected channels to their original positions
                        idx_tensor = torch.tensor(valid_indices, device=layer_feat_up.device)
                        valid_in_feat = idx_tensor[idx_tensor < C_up]
                        if valid_in_feat.numel() > 0:
                            feat_c[:, valid_in_feat, :, :] = layer_feat_up[:, valid_in_feat, :, :]
                    else:
                        # If no valid indices, fall back to first channels
                        num_default = min(configured_layer_planes, C_up)
                        if num_default > 0:
                            default_idx = torch.arange(num_default, device=layer_feat_up.device)
                            feat_c[:, default_idx, :, :] = layer_feat_up[:, default_idx, :, :]

                    block_feats[block_name].append(feat_c)

                    offset += configured_layer_planes
            else:
                # Use fixed indices
                for i, layer in enumerate(block['layers']):
                    layer_name = "{}_{}".format(block_name, layer['idx'])
                    
                    # Get selected channels
                    selected_indices = self.indexes[layer_name].data
                    feat_c = torch.index_select(block_layer_feats[i], 1, selected_indices)
                    feat_c = getattr(self, "{}_{}_upsample".format(block_name, layer['idx']))(feat_c)
                    block_feats[block_name].append(feat_c)
            
            block_feats[block_name] = torch.cat(block_feats[block_name], dim=1)
        
        if train:
            gt_block_feats = {}
            gt_feats = inputs["gt_feats"]
            
            for block in self.structure:
                block_name = block['name']
                gt_block_feats[block_name] = []
                
                for layer in block['layers']:
                    layer_name = "{}_{}".format(block_name, layer['idx'])
                    
                    # Get selected channels
                    selected_indices = self.indexes[layer_name].data
                    feat_c = torch.index_select(gt_feats[layer['idx']]['feat'], 1, selected_indices)
                    feat_c = getattr(self, "{}_{}_upsample".format(block_name, layer['idx']))(feat_c)
                    gt_block_feats[block_name].append(feat_c)
                
                gt_block_feats[block_name] = torch.cat(gt_block_feats[block_name], dim=1)
            
            return {'block_feats': block_feats, "gt_block_feats": gt_block_feats}
        
        return {'block_feats': block_feats}

    def get_outplanes(self):
        """Get output planes for each block"""
        return {block['name']: sum([layer['planes'] for layer in block['layers']]) for block in self.structure}

    def get_outstrides(self):
        """Get output strides for each block"""
        return {block['name']: block['stride'] for block in self.structure}

    @torch.no_grad()
    def init_idxs(self, model, train_loader, distributed=True):
        """
        Initialize channel indices using the original method
        
        Args:
            model: Model to initialize for
            train_loader: Training data loader
            distributed: Whether using distributed training
        """
        # Store distributed flag for later use
        self._distributed = distributed
        
        anomaly_types = copy.deepcopy(train_loader.dataset.anomaly_types)

        if 'normal' in train_loader.dataset.anomaly_types:
            del train_loader.dataset.anomaly_types['normal']

        for key in train_loader.dataset.anomaly_types:
            train_loader.dataset.anomaly_types[key] = 1.0 / len(list(train_loader.dataset.anomaly_types.keys()))

        model.eval()
        criterion = nn.MSELoss(reduce=False).to(model.device)
        
        for block in self.structure:
            self.init_block_idxs(block, model, train_loader, criterion, distributed=distributed)
            
            # Initialize RL environment for this block with the initial selection
            block_name = block['name']
            initial_selection = []
            offset = 0  # Track offset for each layer
            
            for layer in block['layers']:
                layer_name = "{}_{}".format(block_name, layer['idx'])
                selected_indices = self.indexes[layer_name].data.tolist()
                # Add offset to make indices relative to the entire block
                initial_selection.extend([idx + offset for idx in selected_indices])
                offset += layer['planes']  # Use configured planes as offset
            
            total_channels = sum([layer['planes'] for layer in block['layers']])
            
            self.rl_envs[block_name] = FeatureSelectionEnv(
                total_channels=total_channels,
                initial_selection=initial_selection,
                alpha=self.rl_config['reward']['alpha'],
                beta=self.rl_config['reward']['beta'],
                gamma=self.rl_config['reward']['gamma']
            )
        
        train_loader.dataset.anomaly_types = anomaly_types
        model.train()

    def init_block_idxs(self, block, model, train_loader, criterion, distributed=True):
        """
        Initialize channel indices for a specific block
        
        Args:
            block: Block to initialize
            model: Model to initialize for
            train_loader: Training data loader
            criterion: Loss criterion
            distributed: Whether using distributed training
        """
        if distributed:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            if rank == 0:
                tq = tqdm(range(self.init_bsn), desc="init {} index".format(block['name']))
            else:
                tq = range(self.init_bsn)
        else:
            tq = tqdm(range(self.init_bsn), desc="init {} index".format(block['name']))

        cri_sum_vec = [torch.zeros(layer['planes']).to(model.device) for layer in block['layers']]
        iterator = iter(train_loader)

        for bs_i in tq:
            try:
                input = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                input = next(iterator)

            bb_feats = model.backbone(to_device(input), train=True)

            ano_feats = bb_feats['feats']
            ori_feats = bb_feats['gt_feats']
            gt_mask = input['mask'].to(model.device)

            B = gt_mask.size(0)

            ori_layer_feats = [ori_feats[layer['idx']]['feat'] for layer in block['layers']]
            ano_layer_feats = [ano_feats[layer['idx']]['feat'] for layer in block['layers']]

            for i, (ano_layer_feat, ori_layer_feat) in enumerate(zip(ano_layer_feats, ori_layer_feats)):
                layer_name = block['layers'][i]['idx']

                C = ano_layer_feat.size(1)
                
                # Ensure we only use configured number of planes
                configured_planes = block['layers'][i]['planes']
                if C > configured_planes:
                    ano_layer_feat = ano_layer_feat[:, :configured_planes, :, :]
                    ori_layer_feat = ori_layer_feat[:, :configured_planes, :, :]
                    C = configured_planes

                ano_layer_feat = getattr(self, "{}_{}_upsample".format(block['name'], layer_name))(ano_layer_feat)
                ori_layer_feat = getattr(self, "{}_{}_upsample".format(block['name'], layer_name))(ori_layer_feat)

                layer_pred = (ano_layer_feat - ori_layer_feat) ** 2

                _, _, H, W = layer_pred.size()

                layer_pred = layer_pred.permute(1, 0, 2, 3).contiguous().view(C, B * H * W)
                (min_v, _), (max_v, _) = torch.min(layer_pred, dim=1), torch.max(layer_pred, dim=1)
                layer_pred = (layer_pred - min_v.unsqueeze(1)) / (max_v.unsqueeze(1) - min_v.unsqueeze(1) + 1e-4)

                label = F.interpolate(gt_mask, (H, W), mode='nearest')
                label = label.permute(1, 0, 2, 3).contiguous().view(1, B * H * W).repeat(C, 1)

                mse_loss = torch.mean(criterion(layer_pred, label), dim=1)

                if distributed:
                    mse_loss_list = [mse_loss for _ in range(world_size)]
                    dist.all_gather(mse_loss_list, mse_loss)
                    mse_loss = torch.mean(torch.stack(mse_loss_list, dim=0), dim=0, keepdim=False)

                cri_sum_vec[i] += mse_loss

        for i in range(len(cri_sum_vec)):
            cri_sum_vec[i][torch.isnan(cri_sum_vec[i])] = torch.max(cri_sum_vec[i][~torch.isnan(cri_sum_vec[i])])
            values, indices = torch.topk(cri_sum_vec[i], k=block['layers'][i]['planes'], dim=-1, largest=False)
            values, _ = torch.sort(indices)

            if distributed:
                tensor_list = [values for _ in range(world_size)]
                dist.all_gather(tensor_list, values)
                self.indexes["{}_{}".format(block['name'], block['layers'][i]['idx'])].data.copy_(tensor_list[0].long())
            else:
                self.indexes["{}_{}".format(block['name'], block['layers'][i]['idx'])].data.copy_(values.long())

    def enable_rl(self):
        """Enable reinforcement learning"""
        self.rl_enabled = True

    def update_rl(self, metrics, epoch):
        """
        Update RL agents based on validation metrics.
        使用验证阶段计算的 AUC 指标来更新每个 block 的 PPO 策略。
        """
        if not self.rl_enabled:
            return

        # 兼容不同的 RL 配置键名
        training_cfg = self.rl_config.get('training', {})
        init_epochs = training_cfg.get('init_epochs', 0)
        update_freq = training_cfg.get('update_freq', training_cfg.get('update_every_n_epochs', 1))

        if epoch < init_epochs:
            return

        self.update_counter += 1

        # Only update at specified frequency
        if self.update_counter % update_freq != 0:
            return
        
        # In distributed training, only update on rank 0 and then broadcast
        if hasattr(self, '_distributed') and self._distributed:
            import torch.distributed as dist
            rank = dist.get_rank()
            should_update = (rank == 0)
        else:
            should_update = True
        
        # Store selections to be synchronized
        selections_to_sync = {}
        
        # 从整体指标中取出像素级 / 图像级 AUC（按需求 3）
        # 优先使用 mean_*，其次使用当前类别的 AUC
        mean_pixel_auc = metrics.get('mean_pixel_auc', None)
        mean_image_auc = metrics.get('mean_image_auc', None)

        # 如果没有 mean_*，尝试从任意 *_pixel_auc / *_image_auc 中取值
        if mean_pixel_auc is None:
            for k, v in metrics.items():
                if k.endswith('_pixel_auc'):
                    mean_pixel_auc = v
                    break
        if mean_image_auc is None:
            for k, v in metrics.items():
                if k.endswith('_image_auc'):
                    mean_image_auc = v
                    break

        if mean_pixel_auc is None and mean_image_auc is None:
            # 没有可用指标则不更新
            return

        for block in self.structure:
            block_name = block['name']

            if should_update:
                # 当前使用统一的整体指标更新每个 block 的策略
                pixel_auc = mean_pixel_auc if mean_pixel_auc is not None else 0.0
                image_auc = mean_image_auc if mean_image_auc is not None else 0.0
                
                # Get previous metrics
                prev_pixel_auc = self.prev_metrics[block_name]['pixel_auc']
                prev_image_auc = self.prev_metrics[block_name]['image_auc']
                
                # Calculate deltas
                delta_pixel_auc = pixel_auc - prev_pixel_auc
                delta_image_auc = image_auc - prev_image_auc
                
                # Update previous metrics
                self.prev_metrics[block_name]['pixel_auc'] = pixel_auc
                self.prev_metrics[block_name]['image_auc'] = image_auc
                
                # Get current selection（基于索引参数还原 block 级通道选择）
                current_selection = []
                offset = 0  # Track offset for each layer

                for layer in block['layers']:
                    layer_name = "{}_{}".format(block_name, layer['idx'])
                    selected_indices = self.indexes[layer_name].data.tolist()
                    # Add offset to make indices relative to the entire block
                    current_selection.extend([idx + offset for idx in selected_indices])
                    offset += layer['planes']  # Use configured planes as offset
                
                # Get environment
                env = self.rl_envs[block_name]
                if env is None:
                    continue
                
                # Reset environment with current selection
                state = env.reset(initial_selection=current_selection)
                
                # Select action
                agent = self.rl_agents[block_name]
                action, action_logprob = agent.select_action(state)
                
                # Calculate reward using:
                # R = alpha * Δpixel_auc + beta * Δimage_auc
                #     - lambda * ((N_curr - N_target) / N_target)^2
                total_channels = sum([layer['planes'] for layer in block['layers']])

                # 目标通道数与前向过程保持一致：优先使用已缓存的 target_channels，其次用 total_channels // 2
                if hasattr(self, '_target_channels') and block_name in self._target_channels:
                    target_channels = self._target_channels[block_name]
                else:
                    target_channels = min(total_channels // 2, total_channels)

                target_channels = max(1, target_channels)  # 避免除零

                N_curr = len(current_selection)
                norm_diff = (N_curr - target_channels) / float(target_channels)

                alpha = self.rl_config['reward'].get('alpha', 1.0)
                beta = self.rl_config['reward'].get('beta', 0.5)
                lambda_ = self.rl_config['reward'].get('gamma', 0.1)

                reward = (
                    alpha * delta_pixel_auc
                    + beta * delta_image_auc
                    - lambda_ * (norm_diff ** 2)
                )
                
                # Take step in environment
                next_state, _, _, _ = env.step(action, pixel_auc, image_auc)
                
                # Store experience
                agent.memory.states.append(state)
                agent.memory.actions.append(action)
                agent.memory.logprobs.append(action_logprob)
                agent.memory.rewards.append(reward)
                
                # Update agent
                agent.update()
                
                # Get new selection
                new_selection = env.get_current_selection()
                selections_to_sync[block_name] = new_selection
        
        # Synchronize selections across all processes
        if hasattr(self, '_distributed') and self._distributed:
            import torch.distributed as dist

            # 如果实际上只有 1 个进程，就不需要做任何同步
            world_size = dist.get_world_size()
            if world_size > 1:
                # Convert selections to tensors for broadcasting
                for block_name, selection in selections_to_sync.items():
                    # 使用与模块参数相同的 device，确保在 NCCL 后端下为 CUDA tensor
                    device = next(self.parameters()).device
                    selection_tensor = torch.tensor(selection, dtype=torch.long, device=device)
                    dist.broadcast(selection_tensor, src=0)
                    selections_to_sync[block_name] = selection_tensor.tolist()
            
            # Synchronize RL agent states
            if should_update:
                rl_state = self.get_rl_state_dict()
            else:
                rl_state = None
                
            # Broadcast RL states from rank 0 to all processes
            if world_size > 1:
                for block in self.structure:
                    block_name = block['name']
                    agent_key = f"{block_name}_agent"

                    if should_update and agent_key in rl_state:
                        # Get state dict for this agent
                        agent_state = rl_state[agent_key]

                        # Convert each tensor to a list for broadcasting
                        broadcast_state = {}
                        for key, value in agent_state.items():
                            if isinstance(value, torch.Tensor):
                                broadcast_state[key] = value
                            else:
                                # Handle non-tensor values (if any)
                                broadcast_state[key] = value

                        # Broadcast each tensor in the state
                        for key, value in broadcast_state.items():
                            if isinstance(value, torch.Tensor):
                                dist.broadcast(value, src=0)
                    else:
                        # Receive broadcasted state
                        agent = self.rl_agents[block_name]
                        agent_state = agent.state_dict()

                        # Receive each tensor in the state
                        for key in agent_state.keys():
                            if isinstance(agent_state[key], torch.Tensor):
                                received_tensor = torch.zeros_like(agent_state[key])
                                dist.broadcast(received_tensor, src=0)
                                agent_state[key] = received_tensor

                        # Load the received state
                        agent.load_state_dict(agent_state)
        
        # Apply the synchronized selections
        for block_name, selection in selections_to_sync.items():
            self._apply_selection(block_name, selection)

    def _apply_selection(self, block_name, selection):
        """
        Apply a new channel selection
        
        Args:
            block_name: Name of the block
            selection: List of selected channel indices (block-level, 0~total_channels-1)
        """
        # Convert flat block-level selection to per-layer selections.
        # 注意：selection 中保存的是“带 offset 的全局通道索引”，
        # 需要先按 layer 范围过滤并减去 offset，才能写回每层固定长度的 indexes。
        offset = 0
        for block in self.structure:
            if block['name'] != block_name:
                continue
                
            for i, layer_info in enumerate(block['layers']):
                layer_name = "{}_{}".format(block_name, layer_info['idx'])
                planes = layer_info['planes']

                # 1) 取出属于该 layer 的全局通道索引
                layer_global = [
                    idx for idx in selection
                    if offset <= idx < offset + planes
                ]

                # 2) 转成 layer 内部的局部索引，并去重排序
                layer_local = sorted(set([idx - offset for idx in layer_global]))
                layer_local = [idx for idx in layer_local if 0 <= idx < planes]

                # 3) 保证长度 == planes：
                #    - 如果不足 planes，用 [0..planes-1] 中未出现的索引补齐
                #    - 如果多于 planes，只取前 planes 个
                if len(layer_local) < planes:
                    default_indices = [j for j in range(planes) if j not in layer_local]
                    need = planes - len(layer_local)
                    layer_local = layer_local + default_indices[:need]
                elif len(layer_local) > planes:
                    layer_local = layer_local[:planes]

                layer_selection_tensor = torch.tensor(
                    layer_local,
                    dtype=torch.long,
                    device=self.indexes[layer_name].data.device,
                )

                # Update indices（大小与 Parameter 一致，避免维度不匹配）
                self.indexes[layer_name].data.copy_(layer_selection_tensor)

                offset += planes
        
        # Synchronize across all processes in distributed training
        if hasattr(self, '_distributed') and self._distributed:
            import torch.distributed as dist
            for block in self.structure:
                if block['name'] != block_name:
                    continue
                    
                for i, layer_info in enumerate(block['layers']):
                    layer_name = "{}_{}".format(block_name, layer_info['idx'])
                    # Ensure all processes have the same indices
                    dist.broadcast(self.indexes[layer_name].data, src=0)
    
    def save_ppo_policy(self):
        """
        Save the trained PPO policy for inference
        
        Returns:
            Dictionary containing the PPO policy state
        """
        ppo_policy = {}
        
        # Save RL agent states
        ppo_policy['rl_agents'] = self.get_rl_state_dict()
        
        # Save current channel selections as initial state
        ppo_policy['initial_selections'] = {}
        for block in self.structure:
            block_name = block['name']
            
            # Get current selection
            current_selection = []
            for layer in block['layers']:
                layer_name = "{}_{}".format(block_name, layer['idx'])
                selected_indices = self.indexes[layer_name].data.tolist()
                current_selection.extend(selected_indices)
            
            # Save the current selection as initial state
            ppo_policy['initial_selections'][block_name] = current_selection
        
        # Save target channel count for each block
        ppo_policy['target_channels'] = {}
        for block in self.structure:
            block_name = block['name']
            total_channels = sum([layer['planes'] for layer in block['layers']])
            ppo_policy['target_channels'][block_name] = min(total_channels // 2, total_channels)
        
        return ppo_policy
    
    def load_ppo_policy(self, ppo_policy):
        """
        Load the trained PPO policy for inference
        
        Args:
            ppo_policy: Dictionary containing the PPO policy state
        """
        # Load RL agent states
        if 'rl_agents' in ppo_policy:
            self.load_rl_state_dict(ppo_policy['rl_agents'])
        
        # Load initial selections
        if 'initial_selections' in ppo_policy:
            self._current_selection = ppo_policy['initial_selections']
        
        # Load target channel counts
        if 'target_channels' in ppo_policy:
            self._target_channels = ppo_policy['target_channels']
        
        # Enable RL for inference
        self.rl_enabled = True
        self.rl_config['dynamic_inference'] = True
    
    def enable_ppo_inference(self):
        """Enable using PPO policy for inference"""
        self.rl_enabled = True
        self.rl_config['dynamic_inference'] = True
    
    def disable_ppo_inference(self):
        """Disable using PPO policy for inference"""
        self.rl_config['dynamic_inference'] = False
    
    def save_fixed_policy(self):
        """
        Save the current RL policy as a fixed policy for inference
        
        Returns:
            Dictionary containing the fixed policy
        """
        fixed_policy = {}
        
        for block in self.structure:
            block_name = block['name']
            
            # Get current selection
            current_selection = []
            for layer in block['layers']:
                layer_name = "{}_{}".format(block_name, layer['idx'])
                selected_indices = self.indexes[layer_name].data.tolist()
                current_selection.extend(selected_indices)
            
            # Save the current selection as the fixed policy
            fixed_policy[block_name] = current_selection
            
        return fixed_policy
    
    def load_fixed_policy(self, fixed_policy):
        """
        Load a fixed policy for inference
        
        Args:
            fixed_policy: Dictionary containing the fixed policy
        """
        self.fixed_policy = fixed_policy
        self.use_fixed_policy = True
        
        # Apply the fixed policy to the indexes
        for block in self.structure:
            block_name = block['name']
            
            if block_name not in fixed_policy:
                continue
                
            selection = fixed_policy[block_name]
            
            # Convert flat selection to per-layer selections
            offset = 0
            for block in self.structure:
                if block['name'] != block_name:
                    continue
                    
                for i, layer_info in enumerate(block['layers']):
                    layer_name = "{}_{}".format(block_name, layer_info['idx'])
                    planes = layer_info['planes']
                    
                    # Get selection for this layer
                    layer_selection = selection[offset:offset + planes]
                    
                    # Update indices
                    self.indexes[layer_name].data.copy_(torch.tensor(layer_selection).long())
                    
                    offset += planes
    
    def enable_fixed_policy_mode(self):
        """Enable using fixed policy for inference"""
        self.use_fixed_policy = True
    
    def disable_fixed_policy_mode(self):
        """Disable using fixed policy for inference"""
        self.use_fixed_policy = False
    
    def get_rl_state_dict(self):
        """
        Get the state dictionary for RL agents
        
        Returns:
            Dictionary containing RL agent states
        """
        rl_state = {}
        
        for block_name, agent in self.rl_agents.items():
            rl_state[f"{block_name}_agent"] = agent.state_dict()
            
        return rl_state
    
    def load_rl_state_dict(self, rl_state):
        """
        Load the state dictionary for RL agents
        
        Args:
            rl_state: Dictionary containing RL agent states
        """
        for block_name, agent in self.rl_agents.items():
            agent_key = f"{block_name}_agent"
            if agent_key in rl_state:
                agent.load_state_dict(rl_state[agent_key])