import copy
import math
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..rl_agent.ppo_agent import PPOAgent
from ..rl_agent.feature_selection_env import FeatureSelectionEnv


class Residual(nn.Module):
    def __init__(self, in_channels):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_residual_layers):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [Residual(in_channels) for _ in range(self._num_residual_layers)]
        )

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class RLRRS(torch.nn.Module):
    """
    RL-enhanced Reconstruction Residuals Selection (RRS).

    与原始 RRS 不同，本模块不再使用 max/mean 两种模式来选择特征块，
    而是使用一个独立于 AFS 的 PPO 智能体，根据验证阶段的像素级 / 图像级 AUC
    以及与目标块数（rl_number）的偏差构造奖励来学习选择策略。
    """

    def __init__(
        self,
        inplanes: Dict[str, int],
        instrides: Dict[str, int],
        modes: List[str],
        mode_numbers: List[int],
        num_residual_layers: int,
        stop_grad: bool,
        rl_number: int,
        rl_config: Dict[str, Any] = None,
    ):

        super(RLRRS, self).__init__()

        # ----------- 基本结构参数（与原始 RRS 一致或兼容）-----------
        self.inplanes = inplanes
        self.instrides = instrides
        self.mode_numbers = mode_numbers
        self.modes = modes
        self.num_residual_layers = num_residual_layers
        self.stop_grad = stop_grad

        # RL 目标选择块数（来自配置文件中的 rl_number）
        self.rl_number = rl_number

        # 为了兼容原始解码器结构，decoder1 的输入通道数 = 目标选择块数
        self.total_select_number = self.rl_number

        # 对齐 stride
        align_stride = min([self.instrides[block] for block in self.instrides])
        for block in self.instrides:
            self.add_module(
                "{}_upsample".format(block),
                nn.UpsamplingBilinear2d(
                    scale_factor=self.instrides[block] / align_stride
                ),
            )

        # 对齐通道数（拼接所有 block 的 residual）
        align_inplane = sum([self.inplanes[block] for block in self.inplanes])
        self.align_inplane = align_inplane

        self.bn_idx = nn.BatchNorm2d(align_inplane, momentum=0.9, affine=False)

        # 解码器结构与原始 RRS 一致，只是输入通道改为 rl_number
        self.decoder1 = nn.Sequential(
            ResidualStack(self.total_select_number, self.num_residual_layers),
            nn.Conv2d(self.total_select_number, 128, (3, 3), padding=(1, 1), bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 32, (3, 3), padding=(1, 1), bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 8, (3, 3), padding=(1, 1), bias=True),
            nn.ReLU(),
        )

        self.decoder3 = nn.Sequential(
            nn.Conv2d(8, 4, (3, 3), padding=(1, 1), bias=True),
            nn.ReLU(),
            nn.Conv2d(4, 2, (3, 3), padding=(1, 1), bias=True),
        )

        # ----------- RRS 专用 RL 智能体与环境（与 AFS 解耦）-----------
        if rl_config is None:
            rl_config = {
                "ppo": {
                    "lr": 0.001,
                    "gamma": 0.99,
                    "eps_clip": 0.2,
                    "k_epochs": 4,
                },
                "reward": {
                    "alpha": 1.0,
                    "beta": 0.5,
                    "gamma": 0.1,
                },
                "training": {
                    "update_every_n_epochs": 5,
                    "init_epochs": 0,
                },
                "feature_selection": {
                    "dynamic_inference": True,
                },
            }

        self.rl_config = rl_config
        self.rl_enabled = False
        self.update_counter = 0

        # 单个 RRS 智能体（与 AFS 的每 block 一个不同）
        self.rrs_agent = PPOAgent(
            state_dim=self.align_inplane,
            action_dim=self.align_inplane,
            lr=self.rl_config["ppo"]["lr"],
            gamma=self.rl_config["ppo"]["gamma"],
            eps_clip=self.rl_config["ppo"]["eps_clip"],
            k_epochs=self.rl_config["ppo"]["k_epochs"],
        )

        # RRS 专用环境（与 AFS 的环境实例区分开）
        self.rrs_env = FeatureSelectionEnv(
            total_channels=self.align_inplane,
            initial_selection=list(range(min(self.rl_number, self.align_inplane))),
            alpha=self.rl_config["reward"]["alpha"],
            beta=self.rl_config["reward"]["beta"],
            gamma=self.rl_config["reward"]["gamma"],
        )

        # 保存当前选择以及目标块数（用于目标数偏差二次惩罚）
        self._current_selection: List[int] = list(
            range(min(self.rl_number, self.align_inplane))
        )
        self._target_blocks = max(1, min(self.rl_number, self.align_inplane))

        # 记录历史指标，用于计算 AUC 增量
        self.prev_metrics: Dict[str, float] = {
            "pixel_auc": 0.0,
            "image_auc": 0.0,
        }

        # 是否分布式训练（由外部设置）
        self._distributed = False

    # ------------------------------------------------------------------
    # 前向：使用 RL 智能体选择特征块，完全替代 max/mean
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _select_blocks_with_rl(self, residual: torch.Tensor) -> torch.Tensor:
        """
        使用 RRS 专用 RL 智能体，根据当前 residual 特征选择 rl_number 个特征块。
        若选择数不足则补齐，超出则截断。
        """
        B, C, H, W = residual.size()

        # 构造状态：每个通道的均值，形状 [B, C]
        state = residual.mean(dim=[2, 3])

        # 对齐维度（理论上 C == self.align_inplane）
        actual_channels = state.size(1)
        configured_channels = self.align_inplane
        if actual_channels != configured_channels:
            if actual_channels > configured_channels:
                state = state[:, :configured_channels]
            else:
                padding = torch.zeros(
                    state.size(0),
                    configured_channels - actual_channels,
                    device=state.device,
                )
                state = torch.cat([state, padding], dim=1)

        state_np = state.cpu().numpy()

        # 从智能体获取一个要 toggle 的通道 index
        action, _ = self.rrs_agent.select_action(state_np)

        # 更新当前选择集合
        current_selection = copy.copy(self._current_selection)

        if action in current_selection:
            current_selection.remove(action)
        else:
            current_selection.append(action)

        # 保持选择块数接近目标数：不足则补齐，超出则截断
        target = self._target_blocks
        total_channels = self.align_inplane

        if len(current_selection) > target:
            current_selection = current_selection[:target]
        elif len(current_selection) < target:
            available = [i for i in range(total_channels) if i not in current_selection]
            needed = target - len(current_selection)
            current_selection.extend(available[:needed])

        # 更新内部状态
        self._current_selection = current_selection

        # 根据当前选择从 residual 中采样通道，得到 [B, rl_number, H, W]
        idx_tensor = torch.tensor(
            current_selection, dtype=torch.long, device=residual.device
        )
        idx_expand = idx_tensor.view(1, -1, 1, 1).repeat(B, 1, H, W)
        residual_selected = torch.gather(residual, dim=1, index=idx_expand)

        return residual_selected

    def forward(self, inputs: Dict[str, Any], train: bool = False) -> Dict[str, Any]:

        residual = inputs["residual"]

        if self.stop_grad:
            residual = {block: residual[block].detach() for block in residual}

        # 将各个 block 的 residual 上采样后拼接
        residual = torch.cat(
            [
                getattr(self, "{}_upsample".format(block))(residual[block])
                for block in residual
            ],
            dim=1,
        )

        residual_idx = self.bn_idx(residual)

        # 使用 RL 智能体进行块选择（不再使用 max/mean）
        if self.rl_enabled and (
            train or self.rl_config.get("feature_selection", {}).get(
                "dynamic_inference", False
            )
        ):
            residual = self._select_blocks_with_rl(residual_idx)
        else:
            # 若未启用 RL，则使用前 rl_number 个通道作为退化策略
            B, C, H, W = residual_idx.size()
            num = min(self.rl_number, C)
            idx = torch.arange(num, device=residual_idx.device, dtype=torch.long)
            idx_expand = idx.view(1, -1, 1, 1).repeat(B, 1, H, W)
            residual = torch.gather(residual_idx, dim=1, index=idx_expand)

        # 解码过程与原始 RRS 一致
        decoded_residual = self.decoder1(residual)
        decoded_residual = self.decoder2(decoded_residual)

        upsample_size = (decoded_residual.size(-1) * 2,) * 2
        decoded_residual = F.interpolate(
            decoded_residual, upsample_size, mode="bilinear", align_corners=True
        )
        logit_mask = self.decoder3(decoded_residual)

        _, _, ht, wt = inputs["image"].size()
        logit_mask = F.interpolate(
            logit_mask, (ht, wt), mode="bilinear", align_corners=True
        )
        pred = torch.softmax(logit_mask, dim=1)
        pred = pred[:, 1, :, :].unsqueeze(1)

        return {"logit": logit_mask, "anomaly_score": pred}

    # ------------------------------------------------------------------
    # RL 控制接口（与 AFS 的 RL 智能体严格区分）
    # ------------------------------------------------------------------
    def enable_rl(self):
        """启用 RRS 的 RL 智能体。"""
        self.rl_enabled = True

    def disable_rl(self):
        """关闭 RRS 的 RL 智能体。"""
        self.rl_enabled = False

    def update_rl(self, metrics: Dict[str, float], epoch: int):
        """
        使用验证阶段的指标更新 RRS 的 PPO 智能体。

        奖励函数与 AFS 基本一致：
        R = alpha * Δpixel_auc + beta * Δimage_auc
            - gamma * ((N_curr - N_target) / N_target)^2
        其中 N_target = rl_number（或其裁剪值），惩罚项逼迫智能体选择的块数
        逐渐逼近配置文件中的 rl_number。
        """
        if not self.rl_enabled:
            return

        training_cfg = self.rl_config.get("training", {})
        init_epochs = training_cfg.get("init_epochs", 0)
        update_freq = training_cfg.get(
            "update_every_n_epochs",
            training_cfg.get("update_freq", 1),
        )

        if epoch < init_epochs:
            return

        self.update_counter += 1
        if self.update_counter % update_freq != 0:
            return

        # 仅在 rank 0 上更新，然后再做必要的同步
        if self._distributed:
            import torch.distributed as dist

            rank = dist.get_rank()
            should_update = rank == 0
        else:
            should_update = True

        # 提取 mean_pixel_auc / mean_image_auc（与 AFS 一致）
        mean_pixel_auc = metrics.get("mean_pixel_auc", None)
        mean_image_auc = metrics.get("mean_image_auc", None)

        if mean_pixel_auc is None:
            for k, v in metrics.items():
                if k.endswith("_pixel_auc"):
                    mean_pixel_auc = v
                    break

        if mean_image_auc is None:
            for k, v in metrics.items():
                if k.endswith("_image_auc"):
                    mean_image_auc = v
                    break

        if mean_pixel_auc is None and mean_image_auc is None:
            return

        if not should_update:
            return

        # 当前指标
        pixel_auc = mean_pixel_auc if mean_pixel_auc is not None else 0.0
        image_auc = mean_image_auc if mean_image_auc is not None else 0.0

        # 历史指标
        prev_pixel_auc = self.prev_metrics["pixel_auc"]
        prev_image_auc = self.prev_metrics["image_auc"]

        # 增量
        delta_pixel_auc = pixel_auc - prev_pixel_auc
        delta_image_auc = image_auc - prev_image_auc

        # 更新历史记录
        self.prev_metrics["pixel_auc"] = pixel_auc
        self.prev_metrics["image_auc"] = image_auc

        # 当前选择集合
        current_selection = copy.copy(self._current_selection)

        # 使用环境重置 state，并从智能体选一个 action
        state = self.rrs_env.reset(initial_selection=current_selection)
        action, action_logprob = self.rrs_agent.select_action(state)

        # 计算目标块数偏差惩罚
        total_channels = self.align_inplane
        target_blocks = max(1, min(self._target_blocks, total_channels))
        N_curr = len(current_selection)
        norm_diff = (N_curr - target_blocks) / float(target_blocks)

        alpha = self.rl_config["reward"].get("alpha", 1.0)
        beta = self.rl_config["reward"].get("beta", 0.5)
        gamma = self.rl_config["reward"].get("gamma", 0.1)

        reward = alpha * delta_pixel_auc + beta * delta_image_auc - gamma * (
            norm_diff ** 2
        )

        # 让环境执行一步，以获得新的选择集合
        next_state, _, _, _ = self.rrs_env.step(action, pixel_auc, image_auc)

        # 存入记忆
        self.rrs_agent.memory.states.append(state)
        self.rrs_agent.memory.actions.append(action)
        self.rrs_agent.memory.logprobs.append(action_logprob)
        self.rrs_agent.memory.rewards.append(reward)

        # 更新智能体（需要开启梯度）
        with torch.enable_grad():
            self.rrs_agent.update()

        # 从环境读取新的选择，并做一次长度修正（不足补齐，超出截断）
        new_selection = self.rrs_env.get_current_selection()
        if len(new_selection) > target_blocks:
            new_selection = new_selection[:target_blocks]
        elif len(new_selection) < target_blocks:
            available = [
                i for i in range(total_channels) if i not in new_selection
            ]
            need = target_blocks - len(new_selection)
            new_selection.extend(available[:need])

        self._current_selection = new_selection

    # ------------------------------------------------------------------
    # RRS 专用 RL 状态保存 / 加载接口
    # （与 AFS 的 get_rl_state_dict / save_ppo_policy 等方法严格区分）
    # ------------------------------------------------------------------
    def get_rrs_rl_state_dict(self) -> Dict[str, Any]:
        """
        返回 RRS 智能体的状态字典（仅包含 PPO 策略网络参数）。
        """
        return {"rrs_agent": self.rrs_agent.state_dict()}

    def load_rrs_rl_state_dict(self, rl_state: Dict[str, Any]):
        """
        从外部提供的状态字典中恢复 RRS 智能体参数。
        """
        if "rrs_agent" in rl_state:
            self.rrs_agent.load_state_dict(rl_state["rrs_agent"])

    def save_rrs_ppo_policy(self) -> Dict[str, Any]:
        """
        保存 RRS 的 PPO 策略，用于推理阶段。
        这里只保存“策略本身”，而不保存具体的特征块编号。
        """
        return {"rrs_agent": self.rrs_agent.state_dict()}

    def load_rrs_ppo_policy(self, ppo_policy: Dict[str, Any]):
        """
        加载已经训练好的 RRS PPO 策略。
        """
        if "rrs_agent" in ppo_policy:
            self.rrs_agent.load_state_dict(ppo_policy["rrs_agent"])
        # 启用动态推理
        self.rl_enabled = True
        self.rl_config.setdefault("feature_selection", {})
        self.rl_config["feature_selection"]["dynamic_inference"] = True


