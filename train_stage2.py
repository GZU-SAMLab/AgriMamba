#!/usr/bin/env python3
"""
阶段2：病害分割训练脚本

专门用于训练病害分割模型，使用阶段1生成的叶子RGB图像作为输入。
"""

import argparse
import datetime
import json
import time
import os
from pathlib import Path
from contextlib import suppress

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.scheduler import create_scheduler

# 导入我们的模块
from segmentation_model.config import SegmentationConfig, get_args_parser
from segmentation_model.combined_model import create_combined_model
from segmentation_model.dataset import StageDatasetWrapper
from segmentation_model.engine import train_stage_pipeline, evaluate_stage

# 导入现有工具
import lesion_utils as utils
from model.lesion_model.lesion_utils import create_optimizer


def main():
    parser = argparse.ArgumentParser(description='阶段2：病害分割训练')
    
    # 基础参数
    parser.add_argument('--stage1-results', type=str, required=True,
                       help='阶段1生成的叶子RGB图像目录路径（必需）')
    parser.add_argument('--data-path', default='./dataset', help='数据集根目录路径')
    parser.add_argument('--data-set', default='dataset4380_split', type=str, help='数据集名称')
    parser.add_argument('--output-dir', default='./output/stage2', help='输出目录')
    parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
    parser.add_argument('--device', default='cuda', help='设备')
    parser.add_argument('--epochs', type=int, default=25, help='训练轮数')
    parser.add_argument('--lr', type=float, default=3e-5, help='学习率')
    parser.add_argument('--resume', default='', help='恢复训练的检查点路径')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    
    # 可选参数
    parser.add_argument('--if-amp', action='store_true', default=True,
                       help='启用混合精度训练')
    parser.add_argument('--no-amp', action='store_false', dest='if_amp',
                       help='禁用混合精度训练')
    
    args = parser.parse_args()
    
    # 检查必需的参数
    if not os.path.exists(args.stage1_results):
        print(f" 错误：Stage1结果目录不存在: {args.stage1_results}")
        print("请先运行阶段1训练生成叶子RGB图像")
        return
    
    # 初始化分布式训练
    utils.init_distributed_mode(args)
    print(f"Arguments: {args}")
    
    # 设备检测和自动回退
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        args.device = 'cpu'
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # 创建配置
    # 模拟args对象以兼容现有配置系统
    full_args = argparse.Namespace(
        data_path=args.data_path,
        data_set=args.data_set,
        input_size=480,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        leaf_lr=5e-5,
        leaf_epochs=25,
        leaf_weight_decay=1e-4,
        lr_backbone=args.lr * 0.5,
        lesion_lr=args.lr,
        lesion_epochs=args.epochs,
        lesion_weight_decay=1e-4,
        lr_decoder=args.lr,
        lr_vssm=args.lr * 0.5,
        stage=2,
        freeze_leaf=True,
        stage1_results=args.stage1_results,
        generate_stage1_results=False,
        pretrain_path='./pretrain',
        output_dir=args.output_dir,
        resume=args.resume,
        eval=False,
        leaf_checkpoint='',
        lesion_checkpoint='',
        eval_stage=0,
        save_predictions=False,
        save_visualization=False,
        device=args.device,
        seed=args.seed,
        num_workers=8,
        pin_mem=True,
        if_amp=args.if_amp,
        distributed=False,
        world_size=1,
        dist_url='env://',
        local_rank=0,
        opt='adamw',
        sched='cosine',
        warmup_epochs=0,
        min_lr=1e-6,
        warmup_lr=1e-6,
        decay_epochs=30,
        cooldown_epochs=10,
        patience_epochs=10,
        decay_rate=0.1,
        drop=0.0,
        drop_path=0.1
    )
    
    config = SegmentationConfig.from_args(full_args)
    config.device = args.device
    config.stage1_results_path = args.stage1_results  # 设置阶段1结果路径
    print(f"Configuration: {config.to_dict()}")
    print(f"Using Stage1 results from: {args.stage1_results}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    if utils.is_main_process():
        with (output_dir / "config.json").open("w") as f:
            json.dump(config.to_dict(), f, indent=4)
    
    # 设置随机种子
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    
    # 创建数据集包装器
    dataset_wrapper = StageDatasetWrapper(config)
    
    # AMP设置
    amp_autocast = suppress
    loss_scaler = "none"
    if config.training_config['if_amp'] and torch.cuda.is_available():
        amp_autocast = torch.cuda.amp.autocast
        from torch.cuda.amp import GradScaler
        loss_scaler = GradScaler()
        print("Using PyTorch GradScaler for AMP")
    else:
        print("AMP disabled (CUDA not available or disabled)")
    
    # 创建模型
    print("=== Starting Stage 2: Lesion Segmentation Training ===")
    model = create_combined_model(config, stage=2, device=device)
    
    # 数据加载器
    train_loader = dataset_wrapper.get_dataloader('train', stage=2)
    val_loader = dataset_wrapper.get_dataloader('val', stage=2)
    
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size: {len(val_loader.dataset)}")
    
    # 优化器和调度器
    stage_config = config.get_stage_config(2)
    args_stage = argparse.Namespace(**vars(full_args))
    
    # 从stage_config更新参数
    for key, value in stage_config.items():
        setattr(args_stage, key, value)
    
    optimizer = create_optimizer(args_stage, model.get_model_for_stage(2), model.get_new_params_for_stage(2))
    lr_scheduler, _ = create_scheduler(args_stage, optimizer)
    
    # 训练
    print(f"Starting Stage 2 training for {args.epochs} epochs")
    training_stats = train_stage_pipeline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        stage=2,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        amp_autocast=amp_autocast,
        loss_scaler=loss_scaler,
        output_dir=str(output_dir)
    )
    
    print(f"Stage 2 training completed.")
    
    # 测试评估
    print(f"\n=== Stage 2 Test Evaluation ===")
    test_loader = dataset_wrapper.get_dataloader('test', stage=2, batch_size=1)
    test_stats = evaluate_stage(
        model=model,
        data_loader=test_loader,
        device=device,
        stage=2,
        config=config,
        amp_autocast=amp_autocast
    )
    
    print(f"Test evaluation: {test_stats}")
    
    # 保存结果
    if utils.is_main_process():
        results = {
            'training_stats': training_stats,
            'test_stats': test_stats,
            'stage': 2,
            'config': config.to_dict(),
            'stage1_results_used': args.stage1_results
        }
        with (output_dir / "stage2_results.json").open("w") as f:
            json.dump(results, f, indent=4)
    
    print(f"\nStage 2 training completed successfully!")


if __name__ == '__main__':
    # 设置环境变量
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main() 