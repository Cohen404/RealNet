#!/usr/bin/env python3

import argparse
import os
import sys
import logging
import pprint
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import yaml
from easydict import EasyDict
from collections import OrderedDict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_helper import ModelHelper
from utils.misc_helper import AverageMeter, get_current_time, create_logger, set_seed, save_checkpoint, summary_model
from utils.dist_helper import setup_distributed
from utils.optimizer_helper import get_optimizer
from utils.criterion_helper import build_criterion
from utils.eval_helper import log_metrics
from utils.categories import Categories
from datasets.data_builder import build_dataloader
from evaluation_realnet import validate

def create_model(net_config):
    """Create model from configuration"""
    return ModelHelper(net_config)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train RealNet with RL-based feature selection")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--use_rl", action="store_true", help="Use RL-based feature selection")
    parser.add_argument("--rl_config", type=str, default=None, help="Path to AFS RL config file")
    parser.add_argument("--rrs_rl_config", type=str, default=None, help="Path to RRS RL config file (separate from AFS)")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation only")
    parser.add_argument("--dataset", type=str, default="mvtec", help="Dataset name")
    parser.add_argument("--class_name", type=str, default="all", help="Class name")
    return parser.parse_args()

def update_config(config_path, use_rl=False, rl_config_path=None, class_name="bottle", rrs_rl_config_path=None):
    """Update configuration with RL settings for AFS and RRS (two separate agents)"""
    with open(config_path) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    
    # Update dataset paths with class name
    config.dataset.train.meta_file = config.dataset.train.meta_file.replace("{}", class_name)
    config.dataset.test.meta_file = config.dataset.test.meta_file.replace("{}", class_name)
    if hasattr(config.dataset.train, 'sdas_dir'):
        config.dataset.train.sdas_dir = config.dataset.train.sdas_dir.replace("{}", class_name)
    
    # Update network structure with class-specific layers
    layers = []
    for block in config.structure:
        layers.extend([layer.idx for layer in block.layers])
    layers = list(set(layers))
    layers.sort()
    config.net[0].kwargs['outlayers'] = layers
    config.net[1].kwargs = config.net[1].get('kwargs', {})
    config.net[1].kwargs['structure'] = config.structure
    
    if use_rl:
        # ---------------- AFS RL 配置（保持与原逻辑兼容） ----------------
        if rl_config_path and os.path.exists(rl_config_path):
            with open(rl_config_path) as f:
                rl_config = yaml.load(f, Loader=yaml.FullLoader)
            config.rl = rl_config
        else:
            # Default RL config for AFS
            config.rl = {
                "ppo": {
                    "lr": 0.001,
                    "gamma": 0.99,
                    "eps_clip": 0.2,
                    "k_epochs": 4,
                    "entropy_coef": 0.01,
                    "value_coef": 0.5,
                },
                "reward": {
                    "alpha": 1.0,
                    "beta": 0.5,
                    "gamma": 0.1,
                },
                "training": {
                    "update_every_n_epochs": 5,
                    "batch_size": 32,
                    "max_timesteps": 1000,
                },
                "feature_selection": {
                    "min_channels": 1,
                    "max_channels_ratio": 0.8,
                    "dynamic_inference": True,
                },
            }

        # Pass RL config to RLAFS if it's already configured
        for module in config.net:
            if module["type"] == "models.afs.RLAFS":
                module["kwargs"] = module.get("kwargs", {})
                module["kwargs"]["rl_config"] = config.rl

        # ---------------- RRS RL 配置（与 AFS 分开） ----------------
        if rrs_rl_config_path and os.path.exists(rrs_rl_config_path):
            with open(rrs_rl_config_path) as f:
                rrs_rl_config = yaml.load(f, Loader=yaml.FullLoader)
            config.rrs_rl = rrs_rl_config
        else:
            # Default RL config for RRS (结构上与 AFS 一致，但为独立配置)
            config.rrs_rl = {
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

        # 把 RRS RL 配置传给 RLRRS 模块（若存在）
        for module in config.net:
            if module["type"] == "models.rrs.RLRRS":
                module["kwargs"] = module.get("kwargs", {})
                module["kwargs"]["rl_config"] = config.rrs_rl
    
    return config

def load_checkpoint(model, checkpoint_path, device, use_rl=False):
    """Load model checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Load RL state if available and using RL
    if use_rl:
        # -------- AFS RL 状态（保持向后兼容）--------
        if hasattr(model.module, "afs_module"):
            afs_mod = model.module.afs_module
            # 旧 key：rl_state_dict / ppo_policy
            if "rl_state_dict" in checkpoint and hasattr(afs_mod, "load_rl_state_dict"):
                print("Loading AFS RL state")
                afs_mod.load_rl_state_dict(checkpoint["rl_state_dict"])

            if "ppo_policy" in checkpoint and hasattr(afs_mod, "load_ppo_policy"):
                print("Loading AFS PPO policy for inference")
                afs_mod.load_ppo_policy(checkpoint["ppo_policy"])
                if hasattr(afs_mod, "rl_config") and afs_mod.rl_config.get(
                    "dynamic_inference", False
                ):
                    print("Enabling dynamic inference with AFS RL agent")
                    afs_mod.enable_ppo_inference()

            # 新 key：afs_rl_state_dict / afs_ppo_policy
            if "afs_rl_state_dict" in checkpoint and hasattr(
                afs_mod, "load_rl_state_dict"
            ):
                print("Loading AFS RL state (afs_rl_state_dict)")
                afs_mod.load_rl_state_dict(checkpoint["afs_rl_state_dict"])

            if "afs_ppo_policy" in checkpoint and hasattr(afs_mod, "load_ppo_policy"):
                print("Loading AFS PPO policy (afs_ppo_policy)")
                afs_mod.load_ppo_policy(checkpoint["afs_ppo_policy"])
                if hasattr(afs_mod, "rl_config") and afs_mod.rl_config.get(
                    "dynamic_inference", False
                ):
                    print("Enabling dynamic inference with AFS RL agent")
                    afs_mod.enable_ppo_inference()

        # -------- RRS RL 状态（单独智能体）--------
        if hasattr(model.module, "rrs_module"):
            rrs_mod = model.module.rrs_module
            if (
                "rrs_rl_state_dict" in checkpoint
                and hasattr(rrs_mod, "load_rrs_rl_state_dict")
            ):
                print("Loading RRS RL state")
                rrs_mod.load_rrs_rl_state_dict(checkpoint["rrs_rl_state_dict"])

            if (
                "rrs_ppo_policy" in checkpoint
                and hasattr(rrs_mod, "load_rrs_ppo_policy")
            ):
                print("Loading RRS PPO policy for inference")
                rrs_mod.load_rrs_ppo_policy(checkpoint["rrs_ppo_policy"])
    
    return checkpoint.get('epoch', 0), checkpoint.get('best_metric', 0.0)

def train_one_epoch(config, train_loader, model, optimizer, epoch, start_iter, criterion, class_name):
    """Train model for one epoch"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        logger = logging.getLogger("realnet_logger_{}".format(class_name))

    losses = AverageMeter(config.trainer.print_freq_step)
    model.train()

    for i, input in enumerate(train_loader):
        curr_step = start_iter + i

        # Forward pass
        outputs = model(input, train=True)

        # Compute loss
        loss = []
        for name, criterion_loss in criterion.items():
            weight = criterion_loss.weight
            loss.append(weight * criterion_loss(outputs))

        loss = torch.sum(torch.stack(loss))

        # Backward pass
        reduced_loss = loss.clone()
        dist.all_reduce(reduced_loss)
        reduced_loss = reduced_loss / world_size
        losses.update(reduced_loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if rank == 0:
            logger.info(
                "Epoch: [{0}][{1}/{2}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})".format(
                    epoch, i, len(train_loader), loss=losses
                )
            )

    return losses.avg

def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Update config
    config = update_config(
        args.config,
        args.use_rl,
        args.rl_config,
        args.class_name,
        args.rrs_rl_config,
    )
    
    # Setup distributed training
    rank, world_size = setup_distributed()
    
    # Set random seeds
    set_seed(config.random_seed)
    
    # Update config with args
    config.exp_path = os.path.dirname(args.config)
    config.checkpoints_path = os.path.join(config.exp_path, config.saver.checkpoints_dir)
    config.log_path = os.path.join(config.exp_path, config.saver.log_dir)
    # Visualization path for evaluation_realnet.validate / export_segment_images
    config.vis_path = os.path.join(config.exp_path, config.saver.vis_dir)

    if rank == 0:
        os.makedirs(config.checkpoints_path, exist_ok=True)
        os.makedirs(config.log_path, exist_ok=True)
        os.makedirs(config.vis_path, exist_ok=True)

        current_time = get_current_time()

        logger = create_logger(
            "realnet_logger_{}".format(args.class_name), 
            config.log_path + "/realnet_{}_{}.log".format(args.class_name, current_time)
        )

        logger.info("args: {}".format(pprint.pformat(args)))
        logger.info("config: {}".format(pprint.pformat(config)))
        logger.info("class name is : {}".format(args.class_name))
        logger.info("Using RL: {}".format(args.use_rl))

    # Build dataloaders
    train_loader, val_loader = build_dataloader(config.dataset, distributed=True)

    # Create model
    model = create_model(config.net)
    model.cuda()

    if rank == 0:
        summary_model(model, logger)

    # Initialize AFS indices
    model.afs_module.init_idxs(model, train_loader, distributed=True)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Setup distributed training
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    # Setup optimizer
    layers = []
    for module in config.net:
        layers.append(module["name"])

    frozen_layers = model.module.frozen_layers
    active_layers = [layer for layer in layers if layer not in frozen_layers]

    if rank == 0:
        logger.info("layers: {}".format(layers))
        logger.info("frozen layers: {}".format(frozen_layers))
        logger.info("active layers: {}".format(active_layers))

    parameters = [
        {"params": getattr(model.module, layer).parameters()} for layer in active_layers
    ]

    optimizer = get_optimizer(parameters, config.trainer.optimizer)

    key_metric = config.evaluator["key_metric"]

    best_metric = 0
    last_epoch = 0

    criterion = build_criterion(config.criterion)

    # Enable RL if specified
    if args.use_rl:
        # AFS RL
        if hasattr(model.module, "afs") and hasattr(model.module.afs, "enable_rl"):
            model.module.afs.enable_rl()
            if rank == 0:
                logger.info("AFS RL-based feature selection enabled")

        # RRS RL（如果网络中使用了 RLRRS）
        if hasattr(model.module, "rrs") and hasattr(model.module.rrs, "enable_rl"):
            model.module.rrs.enable_rl()
            if rank == 0:
                logger.info("RRS RL-based residual selection enabled")

    # Load checkpoint if specified
    if args.resume:
        last_epoch, best_metric = load_checkpoint(model, args.resume, torch.device("cuda"), args.use_rl)
        if rank == 0:
            logger.info(f"Resumed from epoch {last_epoch} with best metric {best_metric:.4f}")
    
    # Run evaluation only
    if args.evaluate:
        if not args.resume:
            raise ValueError("Please specify a checkpoint for evaluation using --resume")
        
        if rank == 0:
            logger.info("Running evaluation...")
        test_metrics = validate(config, val_loader, model, last_epoch, args.class_name)
        
        if rank == 0:
            logger.info(f"Test metrics: {test_metrics}")
        
        return

    # Training loop
    for epoch in range(last_epoch, config.trainer.max_epoch):
        last_iter = epoch * len(train_loader)
        train_loader.sampler.set_epoch(epoch)
        
        # Train for one epoch
        train_one_epoch(
            config,
            train_loader,
            model,
            optimizer,
            epoch,
            last_iter,
            criterion,
            args.class_name
        )

        # Validate
        if (epoch + 1) % config.trainer.val_freq_epoch == 0:
            # evaluation_realnet.validate(config, val_loader, model, class_name)
            ret_metrics = validate(config, val_loader, model, args.class_name)

            # Update RL agents if enabled
            if args.use_rl:
                # AFS RL agent
                if hasattr(model.module, "afs") and hasattr(
                    model.module.afs, "update_rl"
                ):
                    model.module.afs.update_rl(ret_metrics, epoch)
                    if rank == 0:
                        logger.info("AFS RL agents updated")

                # RRS RL agent
                if hasattr(model.module, "rrs_module") and hasattr(
                    model.module.rrs_module, "update_rl"
                ):
                    model.module.rrs_module.update_rl(ret_metrics, epoch)
                    if rank == 0:
                        logger.info("RRS RL agent updated")

            if rank == 0:
                ret_key_metric = np.mean([ret_metrics[key] for key in ret_metrics if key.find(key_metric) != -1])

                is_best = ret_key_metric >= best_metric
                best_metric = max(ret_key_metric, best_metric)

                if is_best:
                    best_record = {key.replace("mean", 'best') : ret_metrics[key] for key in ret_metrics if key.find("mean") != -1}

                ret_metrics.update(best_record)
                log_metrics(ret_metrics, config.evaluator.metrics, "realnet_logger_{}".format(args.class_name))
                
                if is_best:
                    # Save checkpoint with RL state if enabled
                    checkpoint_data = {
                        "epoch": epoch + 1,
                        "arch": config.net,
                        "state_dict": model.module.state_dict(),
                        "best_metric": best_metric,
                    }
                    
                    # Save RL state if enabled
                    if args.use_rl:
                        # AFS RL
                        if hasattr(model.module, "afs") and hasattr(
                            model.module.afs, "get_rl_state_dict"
                        ):
                            checkpoint_data["afs_rl_state_dict"] = (
                                model.module.afs.get_rl_state_dict()
                            )

                            if hasattr(model.module.afs, "save_ppo_policy"):
                                checkpoint_data["afs_ppo_policy"] = (
                                    model.module.afs.save_ppo_policy()
                                )

                        # RRS RL（单独智能体，保存的是“策略”而不是块编号）
                        if hasattr(model.module, "rrs_module") and hasattr(
                            model.module.rrs_module, "get_rrs_rl_state_dict"
                        ):
                            checkpoint_data["rrs_rl_state_dict"] = (
                                model.module.rrs_module.get_rrs_rl_state_dict()
                            )

                        if hasattr(model.module, "rrs_module") and hasattr(
                            model.module.rrs_module, "save_rrs_ppo_policy"
                        ):
                            checkpoint_data["rrs_ppo_policy"] = (
                                model.module.rrs_module.save_rrs_ppo_policy()
                            )
                    
                    save_checkpoint(checkpoint_data, config, args.class_name)
        
        dist.barrier()

    # Final test evaluation
    if rank == 0:
        logger.info("Running final test evaluation...")
    # evaluation_realnet.validate(config, val_loader, model, class_name)
    test_metrics = validate(config, val_loader, model, args.class_name)
    
    if rank == 0:
        logger.info(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()