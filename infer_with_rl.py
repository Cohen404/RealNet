#!/usr/bin/env python3

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import yaml
from easydict import EasyDict
from PIL import Image
import torchvision.transforms as transforms
from skimage.segmentation import mark_boundaries
from pytorch_grad_cam.utils.image import show_cam_on_image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_model

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Inference with RealNet RL-based dynamic feature selection")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--rl_config", type=str, default=None, help="Path to RL config file")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--output", type=str, default="./output", help="Path to output directory")
    parser.add_argument("--dataset", type=str, default="mvtec", help="Dataset name")
    parser.add_argument("--class_name", type=str, default="all", help="Class name")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--use_rl", action="store_true", help="Use RL for dynamic feature selection during inference")
    return parser.parse_args()

def load_config(config_path, rl_config_path=None, class_name="bottle"):
    """
    Load configuration files and make them consistent with training-time structure.

    与 `train_realnet_rl_fixed.update_config` 保持一致：
    - 根据 `structure` 计算 backbone 的 `outlayers`
    - 将 `structure` 传入 AFS（RLAFS）模块
    - 可选加载 AFS 的 RL 配置
    """
    with open(config_path) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    # 根据结构计算 backbone 输出的层索引（与训练脚本一致）
    if hasattr(config, "structure"):
        layers = []
        for block in config.structure:
            # block.layers 可能是 EasyDict 列表
            layers.extend([layer.idx for layer in block.layers])
        layers = sorted(list(set(layers)))

        # backbone 的 outlayers
        if len(config.net) > 0:
            config.net[0].kwargs = config.net[0].get("kwargs", {})
            config.net[0].kwargs["outlayers"] = layers

        # AFS / RLAFS 需要 structure 信息
        if len(config.net) > 1:
            config.net[1].kwargs = config.net[1].get("kwargs", {})
            config.net[1].kwargs["structure"] = config.structure

    # 加载 AFS 的 RL 配置（可选）
    if rl_config_path and os.path.exists(rl_config_path):
        with open(rl_config_path) as f:
            rl_config = yaml.load(f, Loader=yaml.FullLoader)
        config.rl = rl_config

        # 将 RL 配置传入 RLAFS（若存在）
        for module in config.net:
            if module["type"] == "models.afs.RLAFS":
                module["kwargs"] = module.get("kwargs", {})
                module["kwargs"]["rl_config"] = config.rl

    return config

def load_model(config, checkpoint_path, device, use_rl=False):
    """Load model with RL state"""
    print(f"Loading model from {checkpoint_path}")
    
    # Create model
    model = create_model(config.net)

    # 使用 ModelHelper 自带的 cuda()/cpu()，以正确设置 model.device
    if device.type == "cuda":
        model = model.cuda()
    else:
        model = model.cpu()
    
    # Load checkpoint with RL state
    model = load_checkpoint(model, checkpoint_path, use_rl)
    
    model.eval()
    return model

def preprocess_image(image_path, transform):
    """Preprocess a single image"""
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def save_visual_result(orig_image_path, anomaly_map, output_path, input_size):
    """
    将 anomaly map 叠加在原图上，并用红色边界标出异常区域，类似 evaluation_realnet.py 的可视化效果。

    orig_image_path: 原始输入图像路径
    anomaly_map:     H x W 的异常得分图（numpy）
    output_path:     输出保存路径
    input_size:      (H, W)，与训练时的 dataset.input_size 一致
    """
    # 读取并按训练分辨率缩放原图
    img = Image.open(orig_image_path).convert("RGB").resize(input_size)
    img_np = np.array(img)

    score = anomaly_map.astype(np.float32)
    # 防止全零导致除零
    max_v = np.max(score)
    if max_v < 1e-6:
        score_norm = score
    else:
        score_norm = score / max_v

    # 简单阈值：均值 + 1 标准差（没有 GT 时的近似分割阈值）
    mu = score_norm.mean()
    sigma = score_norm.std()
    threshold = float(mu + sigma)
    threshold = max(0.0, min(1.0, threshold))

    score_mask = np.zeros_like(score_norm, dtype=np.uint8)
    score_mask[score_norm > threshold] = 1

    # heatmap 叠加 & 红色边界
    heat = show_cam_on_image(img_np / 255.0, score_norm, use_rgb=True)
    score_img = mark_boundaries(heat, score_mask, color=(1, 0, 0), mode="thick")
    score_img = (255.0 * score_img).astype(np.uint8)

    # 左右拼接：原图 | 预测结果
    merge = np.concatenate([img_np, score_img], axis=1).astype(np.uint8)
    Image.fromarray(merge).save(output_path)

def inference_single_image(model, image_tensor, device):
    """Run inference on a single image"""
    with torch.no_grad():
        # 与训练时一致：模型期望输入中包含 "image" 键
        inputs = {"image": image_tensor.to(device)}

        # Forward pass（经过 backbone -> AFS -> recon -> RRS）
        outputs = model(inputs, train=False)

        # 优先使用 RRS 输出的 anomaly_score
        if "anomaly_score" in outputs:
            return outputs["anomaly_score"].squeeze().cpu().numpy()

        # 兼容旧实现：如果有 anomaly_map 就用之
        if "anomaly_map" in outputs:
            return outputs["anomaly_map"].squeeze().cpu().numpy()

        # 再退一步，从 block_feats 简单聚合
        if "block_feats" in outputs:
            block_feats = outputs["block_feats"]
            last_block = list(block_feats.keys())[-1]
            feat = block_feats[last_block]
            anomaly_map = feat.mean(dim=1).squeeze().cpu().numpy()
            return anomaly_map

        # Fallback: return a simple placeholder
        print("Warning: Could not extract anomaly map from model output")
        return np.zeros((image_tensor.shape[2], image_tensor.shape[3]))

def load_checkpoint(model, checkpoint_path, use_rl=False):
    """Load model checkpoint with RL state (支持 AFS 与 RRS 的 RL)"""
    print(f"Loading checkpoint from {checkpoint_path}")

    # .pth.tar 这里只是命名习惯，本质仍然是 torch.save 的字典，直接 load 即可
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # -------- 加载 backbone + 主网络权重 --------
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # 兼容最早期的保存格式：直接就是 state_dict
        model.load_state_dict(checkpoint)

    # -------- 可选：加载 RL 状态（AFS & RRS）--------
    if use_rl:
        # AFS 模块（RLAFS）
        afs_mod = getattr(model, "afs", None)
        if afs_mod is not None:
            # 旧 key：统一 RL 状态 / 策略
            if "ppo_policy" in checkpoint and hasattr(afs_mod, "load_ppo_policy"):
                print("Loading AFS PPO policy for inference (ppo_policy)")
                afs_mod.load_ppo_policy(checkpoint["ppo_policy"])
                afs_mod.enable_ppo_inference()
            elif "rl_state_dict" in checkpoint and hasattr(
                afs_mod, "load_rl_state_dict"
            ):
                print("Loading AFS RL state for inference (rl_state_dict)")
                afs_mod.load_rl_state_dict(checkpoint["rl_state_dict"])
                afs_mod.enable_ppo_inference()

            # 新 key：与训练脚本 train_realnet_rl_fixed.py 对齐
            if "afs_ppo_policy" in checkpoint and hasattr(
                afs_mod, "load_ppo_policy"
            ):
                print("Loading AFS PPO policy for inference (afs_ppo_policy)")
                afs_mod.load_ppo_policy(checkpoint["afs_ppo_policy"])
                afs_mod.enable_ppo_inference()

            if "afs_rl_state_dict" in checkpoint and hasattr(
                afs_mod, "load_rl_state_dict"
            ):
                print("Loading AFS RL state for inference (afs_rl_state_dict)")
                afs_mod.load_rl_state_dict(checkpoint["afs_rl_state_dict"])
                afs_mod.enable_ppo_inference()

        # RRS 模块（RLRRS）
        # 训练时我们是通过 model.module.rrs_module 保存的，这里优先查 rrs_module，其次 rrs
        rrs_mod = getattr(model, "rrs_module", None)
        if rrs_mod is None:
            rrs_mod = getattr(model, "rrs", None)

        if rrs_mod is not None:
            # RRS RL 策略（只保存 PPO policy，不保存编号）
            if "rrs_ppo_policy" in checkpoint and hasattr(
                rrs_mod, "load_rrs_ppo_policy"
            ):
                print("Loading RRS PPO policy for inference (rrs_ppo_policy)")
                rrs_mod.load_rrs_ppo_policy(checkpoint["rrs_ppo_policy"])

            if "rrs_rl_state_dict" in checkpoint and hasattr(
                rrs_mod, "load_rrs_rl_state_dict"
            ):
                print("Loading RRS RL state for inference (rrs_rl_state_dict)")
                rrs_mod.load_rrs_rl_state_dict(checkpoint["rrs_rl_state_dict"])

    return model

def main():
    """Main inference function"""
    # Parse arguments
    args = parse_args()
    
    # Load configuration（包含 outlayers / structure / RL 配置）
    config = load_config(args.config, args.rl_config, args.class_name)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load model
    model = load_model(config, args.checkpoint, device, use_rl=args.use_rl)
    
    # Setup image transform
    # 训练配置中字段名为 dataset.input_size / pixel_mean / pixel_std
    image_h, image_w = config.dataset.input_size
    mean = config.dataset.pixel_mean
    std = config.dataset.pixel_std

    transform = transforms.Compose([
        transforms.Resize((image_h, image_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Process input
    if os.path.isfile(args.input):
        # Single image
        print(f"Processing single image: {args.input}")
        image_tensor = preprocess_image(args.input, transform)
        anomaly_map = inference_single_image(model, image_tensor, device)

        # Save visual result (原图 + 红色圈叠加)
        input_name = os.path.splitext(os.path.basename(args.input))[0]
        output_path = os.path.join(args.output, f"{input_name}_anomaly.png")
        save_visual_result(args.input, anomaly_map, output_path, (image_h, image_w))
        print(f"Saved visual result to: {output_path}")
        
    elif os.path.isdir(args.input):
        # Directory of images (递归遍历子文件夹)
        print(f"Processing images in directory (recursive): {args.input}")

        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        total_count = 0

        for root, _, files in os.walk(args.input):
            # 相对路径，用于在输出目录中还原子目录结构
            rel_dir = os.path.relpath(root, args.input)
            if rel_dir == ".":
                out_dir = args.output
            else:
                out_dir = os.path.join(args.output, rel_dir)
            os.makedirs(out_dir, exist_ok=True)

            # 当前目录下的图片
            image_files = [
                f for f in files
                if os.path.splitext(f)[1].lower() in image_extensions
            ]

            for image_file in image_files:
                total_count += 1
                image_path = os.path.join(root, image_file)
                print(f"Processing image {total_count}: {image_path}")

                try:
                    image_tensor = preprocess_image(image_path, transform)
                    anomaly_map = inference_single_image(model, image_tensor, device)

                    # 保持相对子目录结构保存（原图 + 红色圈叠加）
                    input_name = os.path.splitext(image_file)[0]
                    output_path = os.path.join(out_dir, f"{input_name}_anomaly.png")
                    save_visual_result(image_path, anomaly_map, output_path, (image_h, image_w))
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
                    continue

        print(f"Processed {total_count} images. Results saved to: {args.output}")
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return
    
    print("Inference completed!")


if __name__ == "__main__":
    main()