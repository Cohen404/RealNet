"""
Model factory utilities for RealNet.

训练脚本 `train_realnet_rl_fixed.py` 直接使用 `ModelHelper`，
而推理脚本 `infer_with_rl.py` 通过 `from models import create_model` 来构建模型。
这里提供统一的 `create_model` 接口，避免导入错误。
"""

from .model_helper import ModelHelper


def create_model(net_config):
    """
    构建完整的 RealNet 模型。

    Args:
        net_config: 配置文件中 `net:` 段对应的列表（realnet.yaml 里的 config.net）

    Returns:
        nn.Module: 由 ModelHelper 串起来的整体模型
    """
    return ModelHelper(net_config)


