"""
生成第一阶段叶片分割结果

在第一阶段训练完成后，使用训练好的模型对所有数据生成叶片分割结果，
供第二阶段训练使用。
"""

import os
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def create_leaf_model(checkpoint_path: str, device: str = 'cuda'):
    """
    从检查点创建叶片分割模型
    
    Args:
        checkpoint_path: 第一阶段模型检查点路径
        device: 设备
    
    Returns:
        model: 加载了权重的叶片分割模型
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
        img_size=480,
        model_size='base'  # 使用base模型，和训练时一致
    )
    
    # 加载检查点
    print(f"Loading checkpoint from {checkpoint_path}")
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
    
    print(f"Model loaded successfully on {device}")
    return model

def extract_leaf_rgb_from_mask(original_image: Image.Image, leaf_mask: np.ndarray) -> Image.Image:
    """
    通过mask从原图中提取叶子RGB图
    
    Args:
        original_image: 原始RGB图像
        leaf_mask: 叶子mask (0-1范围)
    
    Returns:
        leaf_rgb_image: 提取的叶子RGB图像
    """
    # 确保原图和mask尺寸一致
    if original_image.size != (leaf_mask.shape[1], leaf_mask.shape[0]):
        # 将mask调整到原图尺寸
        mask_image = Image.fromarray((leaf_mask * 255).astype(np.uint8), mode='L')
        mask_image = mask_image.resize(original_image.size, Image.NEAREST)
        leaf_mask = np.array(mask_image) / 255.0
    
    # 转换原图为numpy数组
    original_np = np.array(original_image)
    
    # 确保mask是二维的
    if leaf_mask.ndim == 3:
        leaf_mask = leaf_mask[:, :, 0]
    
    # 二值化mask
    binary_mask = (leaf_mask > 0.5).astype(np.uint8)
    
    # 创建三通道mask
    mask_3ch = np.stack([binary_mask, binary_mask, binary_mask], axis=2)
    
    # 像素点求交集：原图 * mask
    leaf_rgb_np = original_np * mask_3ch
    
    # 转换回PIL图像
    return Image.fromarray(leaf_rgb_np.astype(np.uint8))

def generate_predictions(model, data_loader, output_dir: str, device: str = 'cuda', data_path: str = './dataset', dataset_name: str = 'dataset4380_split'):
    """
    生成叶片RGB图（通过mask和原图计算）
    
    Args:
        model: 叶片分割模型
        data_loader: 数据加载器
        output_dir: 输出目录
        device: 设备
        data_path: 数据集路径
        dataset_name: 数据集名称
    """
    model.eval()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating leaf RGB images to {output_dir}")
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating leaf RGB images"):
            images = batch['query_img'].to(device)
            img_names = batch['img_name']
            
            # 前向传播
            predictions = model(images)
            
            # 应用sigmoid并二值化
            predictions = torch.sigmoid(predictions)
            
            # 处理每个预测结果
            for i, img_name in enumerate(img_names):
                pred_mask = predictions[i].cpu().numpy()[0]  # 取出第一个通道
                
                # 加载对应的原图 - 使用动态数据集名称
                split = data_loader.dataset.split
                original_img_path = Path(data_path) / dataset_name / split / 'img' / f"{img_name}.jpg"
                
                try:
                    # 加载原图并处理EXIF旋转，确保与训练时一致
                    original_image = Image.open(original_img_path).convert('RGB')
                    original_image = ImageOps.exif_transpose(original_image)
                    
                    # 通过mask提取叶子RGB图
                    leaf_rgb_image = extract_leaf_rgb_from_mask(original_image, pred_mask)
                    
                    # 保存叶子RGB图
                    save_path = output_path / f"{img_name}.jpg"  # 保存为RGB图像
                    leaf_rgb_image.save(save_path, quality=95)
                    
                except Exception as e:
                    raise RuntimeError(f"Failed to process {img_name}: {e}") from e

def generate_stage1_results(checkpoint_path: str, 
                          data_path: str, 
                          output_dir: str,
                          device: str = 'cuda',
                          batch_size: int = 8,
                          dataset_name: str = 'dataset4380_split'):
    """
    生成第一阶段的所有分割结果
    
    Args:
        checkpoint_path: 第一阶段模型检查点路径
        data_path: 数据集路径
        output_dir: 输出目录
        device: 设备
        batch_size: 批次大小
        dataset_name: 数据集名称
    """
    # 创建模型
    model = create_leaf_model(checkpoint_path, device)
    
    # 导入数据集
    from dataset import SegmentationDataset
    from torch.utils.data import DataLoader
    
    # 为每个数据集划分生成结果
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} set...")
        
        # 创建数据集（只加载图像，不加载mask）
        dataset = SegmentationDataset(
            root=data_path,
            split=split,
            stage=1,  # 只需要图像
            input_size=(480, 480),
            dataset_name=dataset_name  # 传递数据集名称
        )
        
        # 创建数据加载器
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # 创建输出目录（改为leaf_rgb以区分）
        split_output_dir = os.path.join(output_dir, split, 'leaf_rgb')
        
        # 生成预测
        generate_predictions(model, data_loader, split_output_dir, device, data_path, dataset_name)
        
        print(f"Completed {split} set: {len(dataset)} images processed")
    
    print(f"\nAll stage1 results generated in: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate Stage1 Leaf Segmentation Results')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to stage1 model checkpoint')
    parser.add_argument('--data-path', type=str,  default='./dataset', 
                       help='Path to dataset root directory')
    parser.add_argument('--data-set', type=str, default='dataset4380_split',
                       help='数据集名称')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for stage1 results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for inference')
    
    args = parser.parse_args()
    
    # 检查输入路径
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data path not found: {args.data_path}")
    
    # 生成结果
    generate_stage1_results(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        dataset_name=args.data_set
    )

if __name__ == '__main__':
    main() 