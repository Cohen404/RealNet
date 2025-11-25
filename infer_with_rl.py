#!/usr/bin/env python3

import argparse
import os
import sys
import logging
import pprint
import time
import numpy as np
import torch
import torch.nn as nn
import yaml
from easydict import EasyDict
from PIL import Image
import torchvision.transforms as transforms

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_model
from utils import AverageMeter, get_current_time, create_logger, setup_distributed, set_seed, save_checkpoint
from datasets import build_dataloader
from utils import build_criterion, get_optimizer, log_metrics, summary_model

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

def load_config(config_path, rl_config_path=None):
    """Load configuration files"""
    with open(config_path) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    
    # Load RL config if provided
    if rl_config_path and os.path.exists(rl_config_path):
        with open(rl_config_path) as f:
            rl_config = yaml.load(f, Loader=yaml.FullLoader)
        config.rl = rl_config
    
    return config

def load_model(config, checkpoint_path, device, use_rl=False):
    """Load model with RL state"""
    print(f"Loading model from {checkpoint_path}")
    
    # Create model
    model = create_model(config.net)
    model.to(device)
    
    # Load checkpoint with RL state
    model = load_checkpoint(model, checkpoint_path, use_rl)
    
    model.eval()
    return model

def preprocess_image(image_path, transform):
    """Preprocess a single image"""
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def save_anomaly_map(anomaly_map, output_path):
    """Save anomaly map as image"""
    # Normalize to [0, 255]
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
    anomaly_map = (anomaly_map * 255).astype(np.uint8)
    
    # Save as image
    Image.fromarray(anomaly_map).save(output_path)

def inference_single_image(model, image_tensor, device):
    """Run inference on a single image"""
    with torch.no_grad():
        # Create input dictionary
        inputs = {"feats": {"0": {"feat": image_tensor.to(device)}}}
        
        # Forward pass
        outputs = model(inputs, train=False)
        
        # Extract anomaly map (assuming it's in the output)
        if 'anomaly_map' in outputs:
            return outputs['anomaly_map'].cpu().numpy()
        else:
            # If anomaly_map is not directly in outputs, try to extract it from block_feats
            if 'block_feats' in outputs:
                # Simple aggregation of block features (this might need adjustment based on actual model)
                block_feats = outputs['block_feats']
                # Use the last block's features
                last_block = list(block_feats.keys())[-1]
                feat = block_feats[last_block]
                # Convert to anomaly map (simple approach)
                anomaly_map = feat.mean(dim=1).squeeze().cpu().numpy()
                return anomaly_map
            
            # Fallback: return a simple placeholder
            print("Warning: Could not extract anomaly map from model output")
            return np.zeros((image_tensor.shape[2], image_tensor.shape[3]))

def load_checkpoint(model, checkpoint_path, use_rl=False):
    """Load model checkpoint with RL state"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load RL state if available
    if use_rl and hasattr(model, 'afs') and 'ppo_policy' in checkpoint:
        print("Loading PPO policy for inference")
        model.afs.load_ppo_policy(checkpoint['ppo_policy'])
        model.afs.enable_ppo_inference()
        print("PPO policy loaded and enabled for inference")
    elif use_rl and hasattr(model, 'afs') and 'rl_state_dict' in checkpoint:
        print("Loading RL state for inference")
        model.afs.load_rl_state_dict(checkpoint['rl_state_dict'])
        model.afs.enable_ppo_inference()
        print("RL state loaded and enabled for inference")
    
    return model

def main():
    """Main inference function"""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config, args.rl_config)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load model
    model = load_model(config, args.checkpoint, device, use_rl=args.use_rl)
    
    # Setup image transform
    transform = transforms.Compose([
        transforms.Resize((config.data.image_size, config.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.data.mean, std=config.data.std)
    ])
    
    # Process input
    if os.path.isfile(args.input):
        # Single image
        print(f"Processing single image: {args.input}")
        image_tensor = preprocess_image(args.input, transform)
        anomaly_map = inference_single_image(model, image_tensor, device)
        
        # Save result
        input_name = os.path.splitext(os.path.basename(args.input))[0]
        output_path = os.path.join(args.output, f"{input_name}_anomaly.png")
        save_anomaly_map(anomaly_map, output_path)
        print(f"Saved anomaly map to: {output_path}")
        
    elif os.path.isdir(args.input):
        # Directory of images
        print(f"Processing images in directory: {args.input}")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend([f for f in os.listdir(args.input) if f.lower().endswith(ext)])
        
        # Process each image
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(args.input, image_file)
            print(f"Processing image {i+1}/{len(image_files)}: {image_file}")
            
            try:
                image_tensor = preprocess_image(image_path, transform)
                anomaly_map = inference_single_image(model, image_tensor, device)
                
                # Save result
                input_name = os.path.splitext(image_file)[0]
                output_path = os.path.join(args.output, f"{input_name}_anomaly.png")
                save_anomaly_map(anomaly_map, output_path)
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")
                continue
        
        print(f"Processed {len(image_files)} images. Results saved to: {args.output}")
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return
    
    print("Inference completed!")


if __name__ == "__main__":
    main()