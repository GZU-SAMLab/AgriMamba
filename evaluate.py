#!/usr/bin/env python3

import argparse
import datetime
import json
import time
import os
from pathlib import Path
from contextlib import suppress
import shutil

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageOps
import torch.nn.functional as F
from tqdm import tqdm

# 导入我们的模块
from segmentation_model.config import SegmentationConfig, get_args_parser
from segmentation_model.combined_model import create_combined_model
from segmentation_model.dataset import StageDatasetWrapper
from segmentation_model.engine import evaluate_stage
from segmentation_model.utils import get_leaf_rgb_from_mask, calculate_segmentation_metrics

# 导入现有工具
import lesion_utils as utils


def create_leaf_model(checkpoint_path: str, device: str = 'cuda'):
    """
    从检查点创建叶片分割模型
    """
    # 导入模型创建函数
    import sys
    import os.path as osp
    current_dir = osp.dirname(__file__)
    parent_dir = osp.dirname(current_dir)
    sys.path.append(parent_dir)
    
    # 导入模型定义以注册模型
    from model.leaf_model.leaf_model import LMLS
    from timm.models import create_model
    
    # 创建模型（使用和训练时相同的配置）
    model, _ = create_model(
        'LMLS',
        img_size=480,  # 统一使用480尺寸
        model_size='base'  # 使用base模型，和训练时一致
    )
    
    # 加载检查点
    print(f"Loading leaf model checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 处理检查点键名
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 移除可能的前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('leaf_model.'):
            new_state_dict[k[11:]] = v
        else:
            new_state_dict[k] = v
    
    # 加载权重
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"Leaf model loaded successfully on {device}")
    return model


def extract_rgb_from_mask(original_image: Image.Image, mask: np.ndarray) -> Image.Image:
    """
    通过mask从原图中提取RGB图（通用函数，可用于叶子和病害）
    
    Args:
        original_image: 原始RGB图像
        mask: 分割mask (0-1范围)
    
    Returns:
        rgb_image: 提取的RGB图像
    """
    # 确保原图和mask尺寸一致
    if original_image.size != (mask.shape[1], mask.shape[0]):
        # 将mask调整到原图尺寸
        mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        mask_image = mask_image.resize(original_image.size, Image.NEAREST)
        mask = np.array(mask_image) / 255.0
    
    # 转换原图为numpy数组
    original_np = np.array(original_image)
    
    # 确保mask是二维的
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    
    # 二值化mask
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    # 创建三通道mask
    mask_3ch = np.stack([binary_mask, binary_mask, binary_mask], axis=2)
    
    # 像素点求交集：原图 * mask
    rgb_np = original_np * mask_3ch
    
    # 转换回PIL图像
    return Image.fromarray(rgb_np.astype(np.uint8))


def create_stage1_dataset_for_evaluation(data_path: str, split: str = 'test', dataset_name: str = 'dataset4380_split'):
    """
    创建第一阶段数据集用于评估
    
    Args:
        data_path: 数据集路径
        split: 数据集划分
        dataset_name: 数据集名称
        
    Returns:
        dataset: 数据集对象
    """
    # 导入数据集（从正确的模块）
    from segmentation_model.dataset import SegmentationDataset
    
    # 创建数据集（只加载图像，不加载mask）
    dataset = SegmentationDataset(
        root=data_path,
        split=split,
        stage=1,  # 只需要图像
        input_size=(480, 480),  # 统一使用480尺寸
        dataset_name=dataset_name  # 传递数据集名称
    )
    
    return dataset


def evaluate_stage1_and_generate_leaf_rgb(leaf_model, data_path: str, split: str,
                                        device, config, output_dir: Path, 
                                        batch_size: int = 4, amp_autocast=suppress):
    """
    评估第一阶段并生成叶子RGB图和mask
    
    Args:
        leaf_model: 叶片分割模型
        data_path: 数据集路径
        split: 数据集划分
        device: 设备
        config: 配置
        output_dir: 输出目录
        batch_size: 批次大小
        amp_autocast: 混合精度上下文
        
    Returns:
        stage1_stats: 第一阶段评估结果
        image_info_dict: 图像信息字典（用于第二阶段）
    """
    print("=== Stage 1 Evaluation: Leaf Segmentation ===")
    
    leaf_model.eval()
    
    # 创建输出目录
    leaf_rgb_dir = output_dir / "stage1_leaf_rgb"
    leaf_mask_dir = output_dir / "leafClass"
    leaf_rgb_dir.mkdir(parents=True, exist_ok=True)
    leaf_mask_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用与generate_stage1_results.py相同的数据集创建逻辑
    dataset = create_stage1_dataset_for_evaluation(data_path, split, config.data_set)
    
    # 创建数据加载器
    from torch.utils.data import DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Processing {split} set with {len(dataset)} images...")
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Stage 1 Evaluation'
    
    all_ious = []
    all_mious = []
    all_dices = []
    all_precisions = []
    all_recalls = []
    image_info_dict = {}  # 存储图像信息用于第二阶段
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Generating leaf RGB images and masks")):
            # 将数据移到设备
            images = batch['query_img'].to(device, non_blocking=True)
            img_names = batch['img_name']
            
            # 使用混合精度推理
            if str(device).startswith('cuda') and amp_autocast != suppress:
                with torch.amp.autocast(device_type='cuda'):
                    predictions = leaf_model(images)
            else:
                predictions = leaf_model(images)
            
            # 应用sigmoid并获取预测结果
            predictions = torch.sigmoid(predictions)
            
            # 计算评估指标（如果有GT）
            if 'leaf_mask' in batch:
                leaf_gt = batch['leaf_mask'].to(device, non_blocking=True).float()
                metrics = calculate_segmentation_metrics(predictions, leaf_gt)
                
                all_ious.append(metrics['iou'])
                all_mious.append(metrics['miou'])
                all_dices.append(metrics['dice'])
                all_precisions.append(metrics['precision'])
                all_recalls.append(metrics['recall'])
                
                metric_logger.update(iou=metrics['iou'])
                metric_logger.update(miou=metrics['miou'])
                metric_logger.update(dice=metrics['dice'])
                metric_logger.update(precision=metrics['precision'])
                metric_logger.update(recall=metrics['recall'])
            
            # 处理每个预测结果
            for i, img_name in enumerate(img_names):
                pred_mask = predictions[i].cpu().numpy()[0]  # 取出第一个通道
                
                # 加载对应的原图
                original_img_path = Path(data_path) / config.data_set / split / 'img' / f"{img_name}.jpg"
                
                try:
                    # 加载原图并处理EXIF旋转，确保与训练时一致
                    original_image = Image.open(original_img_path).convert('RGB')
                    original_image = ImageOps.exif_transpose(original_image)
                    
                    # 通过mask提取叶子RGB图
                    leaf_rgb_image = extract_rgb_from_mask(original_image, pred_mask)
                    
                    # 保存叶子RGB图
                    leaf_rgb_path = leaf_rgb_dir / f"{img_name}.jpg"
                    leaf_rgb_image.save(leaf_rgb_path, quality=95)
                    
                    # 保存叶片mask（保持与原图尺寸一致）
                    leaf_mask_path = leaf_mask_dir / f"{img_name}.png"
                    mask_image = Image.fromarray((pred_mask * 255).astype(np.uint8), mode='L')
                    mask_image = mask_image.resize(original_image.size, Image.NEAREST)
                    mask_image.save(leaf_mask_path)
                    resized_mask = np.array(mask_image, dtype=np.float32) / 255.0  # 保存用于后续处理

                    # 存储图像信息用于第二阶段
                    image_info_dict[img_name] = {
                        'original_image_path': str(original_img_path),
                        'leaf_rgb_path': str(leaf_rgb_path),
                        'leaf_mask_path': str(leaf_mask_path),
                        'original_image': original_image,
                        'leaf_rgb_image': leaf_rgb_image,
                        'leaf_mask': resized_mask
                    }
                    
                except Exception as e:
                    raise RuntimeError(f"Failed to process leaf segmentation for {img_name}: {e}") from e
    
    # 收集统计信息
    metric_logger.synchronize_between_processes()
    stage1_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    if all_ious:
        stage1_stats['mean_iou'] = sum(all_ious) / len(all_ious)
        stage1_stats['mean_miou'] = sum(all_mious) / len(all_mious)
        stage1_stats['mean_dice'] = sum(all_dices) / len(all_dices)
        stage1_stats['mean_precision'] = sum(all_precisions) / len(all_precisions)
        stage1_stats['mean_recall'] = sum(all_recalls) / len(all_recalls)
    
    print(f"Stage 1 evaluation stats: {stage1_stats}")
    print(f"Generated leaf RGB images in: {leaf_rgb_dir}")
    print(f"Generated leaf masks in: {leaf_mask_dir}")
    
    return stage1_stats, image_info_dict


def evaluate_stage2_and_generate_lesion_rgb(combined_model, image_info_dict, device, config, 
                                           output_dir: Path, batch_size: int = 4, 
                                           amp_autocast=suppress):
    """
    评估第二阶段并生成病害RGB图和mask
    
    Args:
        combined_model: 组合模型
        image_info_dict: 图像信息字典（来自第一阶段）
        device: 设备
        config: 配置
        output_dir: 输出目录
        batch_size: 批次大小
        amp_autocast: 混合精度上下文
        
    Returns:
        stage2_stats: 第二阶段评估结果
    """
    print("=== Stage 2 Evaluation: Lesion Segmentation ===")
    
    combined_model.eval()
    combined_model.set_stage(2)  # 设置为病害分割模式
    
    # 创建输出目录
    lesion_rgb_dir = output_dir / "stage2_lesion_rgb"
    lesion_mask_dir = output_dir / "lesionClass"
    lesion_rgb_dir.mkdir(parents=True, exist_ok=True)
    lesion_mask_dir.mkdir(parents=True, exist_ok=True)
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Stage 2 Evaluation'
    
    all_ious = []
    all_mious = []
    all_dices = []
    all_precisions = []
    all_recalls = []
    
    # 图像变换（第二阶段使用480x480）
    import torchvision.transforms as T
    lesion_transform = T.Compose([
        T.Resize((480, 480)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 准备批次处理
    img_names = list(image_info_dict.keys())
    total_samples = len(img_names)
    
    with torch.no_grad():
        for batch_start in tqdm(range(0, total_samples, batch_size), desc="Generating lesion RGB images and masks"):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_img_names = img_names[batch_start:batch_end]
            
            # 准备batch数据
            batch_images = []
            batch_sentences = []
            batch_original_images = []
            batch_lesion_gts = []  # 存储病害GT masks
            
            for img_name in batch_img_names:
                info = image_info_dict[img_name]
                
                # 加载叶子RGB图并应用变换
                leaf_rgb_image = info['leaf_rgb_image']
                leaf_rgb_tensor = lesion_transform(leaf_rgb_image)
                batch_images.append(leaf_rgb_tensor)
                
                # 加载文本描述
                text_path = Path(config.data_path) / config.data_set / config.split / 'txt' / f"{img_name}.txt"
                sentence = ""
                if text_path.exists():
                    with open(text_path, 'r', encoding='utf-8') as f:
                        sentence = f.read().strip()
                    if not sentence:
                        raise ValueError(f"Text file is empty for {img_name}: {text_path}")
                else:
                    raise FileNotFoundError(f"Text file not found for {img_name}: {text_path}")
                
                batch_sentences.append(sentence)
                batch_original_images.append(info['original_image'])
                
                # 加载病害GT mask（如果存在）
                lesion_gt_path = Path(config.data_path) / config.data_set / config.split / 'lesionClass' / f"{img_name}.png"
                if lesion_gt_path.exists():
                    lesion_gt = Image.open(lesion_gt_path).convert('L')
                    lesion_gt = lesion_gt.resize((480, 480), Image.NEAREST)  # 调整到模型输出尺寸
                    lesion_gt_np = np.array(lesion_gt) / 255.0  # 归一化到0-1
                    lesion_gt_tensor = torch.from_numpy(lesion_gt_np).float().unsqueeze(0)  # [1, H, W]
                    batch_lesion_gts.append(lesion_gt_tensor)
                else:
                    raise FileNotFoundError(f"Lesion GT mask not found for {img_name}: {lesion_gt_path}")
            
            # 转换为tensor并移到设备
            images_tensor = torch.stack(batch_images).to(device)
            
            # 使用混合精度推理
            if str(device).startswith('cuda') and amp_autocast != suppress:
                with torch.amp.autocast(device_type='cuda'):
                    lesion_pred = combined_model.forward_stage2(images_tensor, batch_sentences)[0]
            else:
                lesion_pred = combined_model.forward_stage2(images_tensor, batch_sentences)[0]
            
            # 应用sigmoid激活
            lesion_pred_prob = torch.sigmoid(lesion_pred)
            
            # 计算批次的评估指标（如果有GT）
            valid_gts = [gt for gt in batch_lesion_gts if gt is not None]
            if valid_gts and len(valid_gts) == len(batch_lesion_gts):
                # 只有当所有样本都有GT时才计算指标
                try:
                    lesion_gt_batch = torch.stack(valid_gts).to(device)  # [B, 1, H, W]
                    metrics = calculate_segmentation_metrics(lesion_pred_prob, lesion_gt_batch)
                    
                    all_ious.append(metrics['iou'])
                    all_mious.append(metrics['miou'])
                    all_dices.append(metrics['dice'])
                    all_precisions.append(metrics['precision'])
                    all_recalls.append(metrics['recall'])
                    
                    metric_logger.update(iou=metrics['iou'])
                    metric_logger.update(miou=metrics['miou'])
                    metric_logger.update(dice=metrics['dice'])
                    metric_logger.update(precision=metrics['precision'])
                    metric_logger.update(recall=metrics['recall'])
                except Exception as e:
                    print(f"Error calculating metrics for batch: {e}")
            
            # 处理每个预测结果
            for i, img_name in enumerate(batch_img_names):
                pred_mask = lesion_pred_prob[i].cpu().numpy()[0]  # 取出第一个通道
                original_image = batch_original_images[i]
                
                try:
                    # 通过mask提取病害RGB图
                    lesion_rgb_image = extract_rgb_from_mask(original_image, pred_mask)
                    
                    # 保存病害RGB图
                    lesion_rgb_path = lesion_rgb_dir / f"{img_name}.jpg"
                    lesion_rgb_image.save(lesion_rgb_path, quality=95)
                    
                    # 保存病害mask（保持与原图尺寸一致）
                    lesion_mask_path = lesion_mask_dir / f"{img_name}.png"
                    mask_image = Image.fromarray((pred_mask * 255).astype(np.uint8), mode='L')
                    mask_image = mask_image.resize(original_image.size, Image.NEAREST)
                    mask_image.save(lesion_mask_path)

                except Exception as e:
                    raise RuntimeError(f"Failed to process lesion segmentation for {img_name}: {e}") from e
    
    # 收集统计信息
    metric_logger.synchronize_between_processes()
    stage2_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    if all_ious:
        stage2_stats['mean_iou'] = sum(all_ious) / len(all_ious)
        stage2_stats['mean_miou'] = sum(all_mious) / len(all_mious)
        stage2_stats['mean_dice'] = sum(all_dices) / len(all_dices)
        stage2_stats['mean_precision'] = sum(all_precisions) / len(all_precisions)
        stage2_stats['mean_recall'] = sum(all_recalls) / len(all_recalls)
        stage2_stats['processed_samples'] = len(img_names)
        stage2_stats['samples_with_gt'] = len(all_ious)
    else:
        stage2_stats = {
            'processed_samples': len(img_names),
            'samples_with_gt': 0,
            'mean_iou': 0.0,
            'mean_miou': 0.0,
            'mean_dice': 0.0,
            'mean_precision': 0.0,
            'mean_recall': 0.0,
            'note': 'No GT masks found for evaluation'
        }
    
    print(f"Stage 2 evaluation stats: {stage2_stats}")
    print(f"Generated lesion RGB images in: {lesion_rgb_dir}")
    print(f"Generated lesion masks in: {lesion_mask_dir}")
    
    return stage2_stats


def main():
    parser = argparse.ArgumentParser(description='增强版两阶段评估：植物病害分割系统（生成叶子和病害的RGB图和mask）')
    
    # 必需参数
    parser.add_argument('--leaf-checkpoint', type=str, required=True,
                       help='叶片分割模型检查点路径（必需）')
    parser.add_argument('--lesion-checkpoint', type=str, required=True,
                       help='病害分割模型检查点路径（必需）')
    
    # 基础参数
    parser.add_argument('--data-path', default='./dataset', help='数据集根目录路径')
    parser.add_argument('--data-set', default='dataset4380_split', type=str, help='数据集名称')
    parser.add_argument('--output-dir', default='./output/evaluation_enhanced', help='输出目录')
    parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
    parser.add_argument('--device', default='cuda', help='设备')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--split', default='test', help='评估数据集划分 (train/val/test)')
    
    # 可选参数
    parser.add_argument('--save-intermediate', action='store_true', default=True,
                       help='保存中间结果（叶子和病害的RGB图和mask）')
    parser.add_argument('--cleanup-intermediate', action='store_true', default=False,
                       help='评测结束后删除中间文件（叶子和病害的RGB图和mask图）')
    parser.add_argument('--if-amp', action='store_true', default=True,
                       help='启用混合精度推理')
    parser.add_argument('--no-amp', action='store_false', dest='if_amp',
                       help='禁用混合精度推理')
    
    args = parser.parse_args()
    
    # 检查必需的检查点文件
    if not os.path.exists(args.leaf_checkpoint):
        print(f" 错误：叶片模型检查点不存在: {args.leaf_checkpoint}")
        return
        
    if not os.path.exists(args.lesion_checkpoint):
        print(f" 错误：病害模型检查点不存在: {args.lesion_checkpoint}")
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
    full_args = argparse.Namespace(
        data_path=args.data_path,
        data_set=args.data_set,
        input_size=480,  # 第二阶段用480
        batch_size=args.batch_size,
        epochs=50,
        lr=5e-5,
        leaf_lr=5e-5,
        leaf_epochs=25,
        leaf_weight_decay=1e-4,
        lr_backbone=2.5e-5,
        lesion_lr=3e-5,
        lesion_epochs=25,
        lesion_weight_decay=1e-4,
        lr_decoder=3e-5,
        lr_vssm=2.5e-5,
        stage=0,
        freeze_leaf=True,
        stage1_results='',
        generate_stage1_results=False,
        pretrain_path='./pretrain',
        output_dir=args.output_dir,
        resume='',
        eval=True,
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
        drop_path=0.1,
        split=args.split  # 添加split到配置中
    )
    
    config = SegmentationConfig.from_args(full_args)
    config.device = args.device
    config.split = args.split  # 确保split在config中可用
    
    print(f"Configuration: {config.to_dict()}")
    
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
    
    # AMP设置
    amp_autocast = suppress
    if config.training_config['if_amp'] and torch.cuda.is_available():
        amp_autocast = torch.cuda.amp.autocast
        print("Using AMP for inference")
    else:
        print("AMP disabled (CUDA not available or disabled)")
    
    # 创建模型
    print("=== Starting Enhanced Two-Stage Evaluation ===")
    
    # 第一阶段：创建直接的叶片模型
    print(f"Creating direct leaf model from checkpoint: {args.leaf_checkpoint}")
    leaf_model = create_leaf_model(args.leaf_checkpoint, device)
    
    # 第二阶段：创建病害模型
    print("Creating combined model for lesion segmentation...")
    combined_model = create_combined_model(config, stage=0, device=device)
    print(f"Loading lesion model checkpoint from {args.lesion_checkpoint}")
    combined_model.load_lesion_checkpoint(args.lesion_checkpoint)
    
    print(" Both models loaded successfully")
    
    # 阶段1评估和叶子RGB图生成
    print(f"\n=== Step 1: Evaluating Stage 1 and Generating Leaf RGB Images and Masks (Split: {args.split}) ===")
    start_time = time.time()
    
    stage1_stats, image_info_dict = evaluate_stage1_and_generate_leaf_rgb(
        leaf_model=leaf_model,
        data_path=args.data_path,
        split=args.split,
        device=device,
        config=config,
        output_dir=output_dir,
        batch_size=args.batch_size,
        amp_autocast=amp_autocast
    )
    
    stage1_time = time.time() - start_time
    print(f" Stage 1 evaluation completed in {stage1_time:.2f} seconds")
    
    # 阶段2评估和病害RGB图生成
    print("\n=== Step 2: Evaluating Stage 2 and Generating Lesion RGB Images and Masks ===")
    start_time = time.time()
    
    stage2_stats = evaluate_stage2_and_generate_lesion_rgb(
        combined_model=combined_model,
        image_info_dict=image_info_dict,
        device=device,
        config=config,
        output_dir=output_dir,
        batch_size=args.batch_size,
        amp_autocast=amp_autocast
    )
    
    stage2_time = time.time() - start_time
    print(f" Stage 2 evaluation completed in {stage2_time:.2f} seconds")
    
    # 汇总结果
    total_time = stage1_time + stage2_time
    
    print(f"\n Enhanced two-stage evaluation completed in {total_time:.2f} seconds")
    print(f" Evaluation Results:")
    print(f"   - Stage 1 (Leaf) IoU: {stage1_stats.get('mean_iou', 0):.4f}")
    print(f"   - Stage 1 (Leaf) mIoU: {stage1_stats.get('mean_miou', 0):.4f}")
    print(f"   - Stage 1 (Leaf) Dice: {stage1_stats.get('mean_dice', 0):.4f}")
    print(f"   - Stage 1 (Leaf) Precision: {stage1_stats.get('mean_precision', 0):.4f}")
    print(f"   - Stage 1 (Leaf) Recall: {stage1_stats.get('mean_recall', 0):.4f}")
    print(f"   - Stage 2 (Lesion) IoU: {stage2_stats.get('mean_iou', 0):.4f}")
    print(f"   - Stage 2 (Lesion) mIoU: {stage2_stats.get('mean_miou', 0):.4f}")
    print(f"   - Stage 2 (Lesion) Dice: {stage2_stats.get('mean_dice', 0):.4f}")
    print(f"   - Stage 2 (Lesion) Precision: {stage2_stats.get('mean_precision', 0):.4f}")
    print(f"   - Stage 2 (Lesion) Recall: {stage2_stats.get('mean_recall', 0):.4f}")
    print(f"   - Stage 2 processed samples: {stage2_stats.get('processed_samples', 0)}")
    print(f"   - Stage 2 samples with GT: {stage2_stats.get('samples_with_gt', 0)}")
    print(f"   - Total Samples: {len(image_info_dict)}")
    
    # 保存结果
    if utils.is_main_process():
        # 汇总评估结果
        final_results = {
            'stage1_results': stage1_stats,
            'stage2_results': stage2_stats,
            'summary': {
                'leaf_iou': stage1_stats.get('mean_iou', 0),
                'leaf_miou': stage1_stats.get('mean_miou', 0),
                'leaf_dice': stage1_stats.get('mean_dice', 0),
                'leaf_precision': stage1_stats.get('mean_precision', 0),
                'leaf_recall': stage1_stats.get('mean_recall', 0),
                'lesion_iou': stage2_stats.get('mean_iou', 0),
                'lesion_miou': stage2_stats.get('mean_miou', 0),
                'lesion_dice': stage2_stats.get('mean_dice', 0),
                'lesion_precision': stage2_stats.get('mean_precision', 0),
                'lesion_recall': stage2_stats.get('mean_recall', 0),
                'lesion_processed_samples': stage2_stats.get('processed_samples', 0),
                'lesion_samples_with_gt': stage2_stats.get('samples_with_gt', 0),
                'total_samples': len(image_info_dict),
                'stage1_time': stage1_time,
                'stage2_time': stage2_time,
                'total_time': total_time
            },
            'evaluation_metadata': {
                'leaf_checkpoint': args.leaf_checkpoint,
                'lesion_checkpoint': args.lesion_checkpoint,
                'evaluation_type': 'enhanced_two_stage',
                'batch_size': args.batch_size,
                'device': args.device,
                'split': args.split,
                'img_size_stage1': 480,
                'img_size_stage2': 480,
                'timestamp': datetime.datetime.now().isoformat()
            }
        }
        
        # 保存详细结果
        result_filename = "evaluation_results_enhanced.json"
        with (output_dir / result_filename).open("w") as f:
            json.dump(final_results, f, indent=4, default=str)
        
        print(f"\n Evaluation results saved to: {output_dir / result_filename}")
        
        # 保存简要报告
        summary_report = {
            'method': 'enhanced_two_stage_evaluation',
            'leaf_iou': stage1_stats.get('mean_iou', 0),
            'leaf_miou': stage1_stats.get('mean_miou', 0),
            'leaf_dice': stage1_stats.get('mean_dice', 0),
            'leaf_precision': stage1_stats.get('mean_precision', 0),
            'leaf_recall': stage1_stats.get('mean_recall', 0),
            'lesion_iou': stage2_stats.get('mean_iou', 0),
            'lesion_miou': stage2_stats.get('mean_miou', 0),
            'lesion_dice': stage2_stats.get('mean_dice', 0),
            'lesion_precision': stage2_stats.get('mean_precision', 0),
            'lesion_recall': stage2_stats.get('mean_recall', 0),
            'lesion_samples_processed': stage2_stats.get('processed_samples', 0),
            'lesion_samples_with_gt': stage2_stats.get('samples_with_gt', 0),
            'total_time': total_time,
            'output_directories': {
                'leaf_rgb': str(output_dir / 'stage1_leaf_rgb'),
                'leaf_masks': str(output_dir / 'leafClass'),
                'lesion_rgb': str(output_dir / 'stage2_lesion_rgb'),
                'lesion_masks': str(output_dir / 'lesionClass')
            }
        }
        
        with (output_dir / "evaluation_summary_enhanced.json").open("w") as f:
            json.dump(summary_report, f, indent=4)
        
        print(f" Summary report saved to: {output_dir / 'evaluation_summary_enhanced.json'}")
        
        if args.save_intermediate:
            print(f" Leaf RGB images saved in: {output_dir / 'stage1_leaf_rgb'}")
            print(f" Leaf masks saved in: {output_dir / 'leafClass'}")
            print(f" Lesion RGB images saved in: {output_dir / 'stage2_lesion_rgb'}")
            print(f" Lesion masks saved in: {output_dir / 'lesionClass'}")
    
    # 清理中间文件（如果启用）
    if args.cleanup_intermediate and args.save_intermediate:
        print(f"\n🧹 Cleaning up intermediate files...")
        try:
            # 清理叶子相关文件
            leaf_rgb_dir = output_dir / "stage1_leaf_rgb"
            leaf_mask_dir = output_dir / "leafClass"
            
            if leaf_rgb_dir.exists():
                shutil.rmtree(leaf_rgb_dir)
                print(f" Deleted leaf RGB images directory: {leaf_rgb_dir}")
            
            if leaf_mask_dir.exists():
                shutil.rmtree(leaf_mask_dir)
                print(f" Deleted leaf masks directory: {leaf_mask_dir}")
            
            # 清理病害相关文件
            lesion_rgb_dir = output_dir / "stage2_lesion_rgb"
            lesion_mask_dir = output_dir / "lesionClass"
            
            if lesion_rgb_dir.exists():
                shutil.rmtree(lesion_rgb_dir)
                print(f" Deleted lesion RGB images directory: {lesion_rgb_dir}")
            
            if lesion_mask_dir.exists():
                shutil.rmtree(lesion_mask_dir)
                print(f" Deleted lesion masks directory: {lesion_mask_dir}")
                
        except Exception as e:
            print(f" Warning: Failed to clean up intermediate files: {e}")
    elif args.cleanup_intermediate and not args.save_intermediate:
        print(f" Note: --cleanup-intermediate has no effect when --save-intermediate is disabled")
    
    print(f"\n Enhanced two-stage evaluation completed successfully!")
    
    return final_results


if __name__ == '__main__':
    # 设置环境变量
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main() 
