import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Optional
import cv2


def get_leaf_rgb_from_mask(original_image: torch.Tensor, leaf_mask: torch.Tensor, 
                          threshold: float = 0.5) -> torch.Tensor:
    """
    从原图和叶片mask中提取叶片区域的RGB图像
    
    Args:
        original_image: 原始RGB图像 [B, 3, H, W]
        leaf_mask: 叶片分割mask [B, 1, H, W] 或 [B, H, W]
        threshold: 二值化阈值
        
    Returns:
        leaf_rgb: 叶片区域RGB图像 [B, 3, H, W]，非叶片区域设为0
    """
    if leaf_mask.dim() == 3:
        leaf_mask = leaf_mask.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
    
    # 确保mask和image尺寸一致
    if leaf_mask.shape[-2:] != original_image.shape[-2:]:
        leaf_mask = F.interpolate(leaf_mask, size=original_image.shape[-2:], 
                                mode='bilinear', align_corners=True)
    
    # 二值化mask
    binary_mask = (leaf_mask > threshold).float()
    
    # 应用mask到RGB图像，每个通道都乘以mask
    leaf_rgb = original_image * binary_mask
    
    return leaf_rgb


def apply_mask_to_image(image: torch.Tensor, mask: torch.Tensor, 
                       background_value: float = 0.0) -> torch.Tensor:
    """
    将mask应用到图像上
    
    Args:
        image: 输入图像 [B, C, H, W]
        mask: 二值mask [B, 1, H, W]
        background_value: 背景值
        
    Returns:
        masked_image: 应用mask后的图像
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    
    # 确保尺寸一致
    if mask.shape[-2:] != image.shape[-2:]:
        mask = F.interpolate(mask, size=image.shape[-2:], 
                           mode='bilinear', align_corners=True)
    
    binary_mask = (mask > 0.5).float()
    masked_image = image * binary_mask + background_value * (1 - binary_mask)
    
    return masked_image


def calculate_segmentation_metrics(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> Dict[str, float]:
    """
    计算分割指标：IoU、Dice Coefficient、Precision、Recall、mIoU
    
    Args:
        pred_mask: 预测mask [B, 1, H, W] - 对于阶段1是叶片分割预测，对于阶段2是病害分割预测
        gt_mask: 真实mask [B, 1, H, W] - 对于阶段1来自leafClass/*.png，对于阶段2来自lesionClass/*.png
        
    Returns:
        metrics: 包含IoU、Dice、Precision、Recall、mIoU的字典
    """
    # 确保输入是正确的形状和类型
    if pred_mask.dim() == 3:
        pred_mask = pred_mask.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
    if gt_mask.dim() == 3:
        gt_mask = gt_mask.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
    
    # 确保是浮点类型
    pred_mask = pred_mask.float()
    gt_mask = gt_mask.float()
    
    # 二值化预测mask (阈值0.5)
    pred_binary = (pred_mask > 0.5).float()
    gt_binary = (gt_mask > 0.5).float()
    
    # 计算混淆矩阵组件
    pred_bool = pred_binary.bool()
    gt_bool = gt_binary.bool()
    tp = (pred_bool & gt_bool).sum(dim=(1, 2, 3)).float()
    tn = (~pred_bool & ~gt_bool).sum(dim=(1, 2, 3)).float()
    fp = (pred_bool & ~gt_bool).sum(dim=(1, 2, 3)).float()
    fn = (~pred_bool & gt_bool).sum(dim=(1, 2, 3)).float()

    # 1. IoU (Intersection over Union)
    # IoU = TP / (TP + FP + FN)，空集对空集记为1
    iou_den = tp + fp + fn
    iou_terms = torch.where(iou_den > 0, tp / iou_den, torch.ones_like(iou_den))
    iou = iou_terms.mean().item()

    # 2. Dice Coefficient (Sørensen-Dice Index)
    # Dice = 2×TP / (2×TP + FP + FN)，空集对空集记为1
    dice_den = (2.0 * tp) + fp + fn
    dice_terms = torch.where(dice_den > 0, (2.0 * tp) / dice_den, torch.ones_like(dice_den))
    dice = dice_terms.mean().item()

    # 3. Precision (精确率)
    # Precision = TP / (TP + FP); 若无预测且GT也为空则记为1，否则为0
    precision_den = tp + fp
    precision_terms = torch.where(
        precision_den > 0,
        tp / precision_den,
        torch.where((tp + fn) == 0, torch.ones_like(precision_den), torch.zeros_like(precision_den))
    )
    precision = precision_terms.mean().item()

    # 4. Recall (召回率/敏感度)
    # Recall = TP / (TP + FN); 若GT为空则记为1
    recall_den = tp + fn
    recall_terms = torch.where(recall_den > 0, tp / recall_den, torch.ones_like(recall_den))
    recall = recall_terms.mean().item()

    # 5. mIoU (前景IoU与背景IoU均值)
    # 背景IoU：把TN作为true_matches，FP/FN对调
    bg_iou_den = tn + fn + fp
    bg_iou_terms = torch.where(bg_iou_den > 0, tn / bg_iou_den, torch.ones_like(bg_iou_den))
    miou_terms = (iou_terms + bg_iou_terms) / 2.0
    miou = miou_terms.mean().item()
    
    return {
        'iou': iou,
        'miou': miou,
        'dice': dice,
        'precision': precision,
        'recall': recall
    }


def visualize_predictions(original_image: torch.Tensor,
                         leaf_mask: torch.Tensor,
                         disease_mask: torch.Tensor,
                         disease_ratio: Optional[float] = None,
                         disease_level: Optional[str] = None,
                         save_path: Optional[str] = None) -> np.ndarray:
    """
    可视化预测结果
    
    Args:
        original_image: 原始图像 [3, H, W]
        leaf_mask: 叶片mask [1, H, W] 或 [H, W]
        disease_mask: 病害mask [1, H, W] 或 [H, W]
        disease_ratio: 病害比例（可选）
        disease_level: 病害等级（可选）
        save_path: 保存路径
        
    Returns:
        vis_image: 可视化图像 [H, W, 3]
    """
    # 转换为numpy格式
    if isinstance(original_image, torch.Tensor):
        img = original_image.cpu().numpy().transpose(1, 2, 0)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    if isinstance(leaf_mask, torch.Tensor):
        leaf_mask = leaf_mask.cpu().numpy()
        if leaf_mask.ndim == 3:
            leaf_mask = leaf_mask[0]
    
    if isinstance(disease_mask, torch.Tensor):
        disease_mask = disease_mask.cpu().numpy()
        if disease_mask.ndim == 3:
            disease_mask = disease_mask[0]
    
    # 创建彩色overlay
    h, w = img.shape[:2]
    overlay = img.copy()
    
    # 叶片区域用绿色边框标出
    leaf_binary = (leaf_mask > 0.5).astype(np.uint8)
    leaf_contours, _ = cv2.findContours(leaf_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, leaf_contours, -1, (0, 255, 0), 2)
    
    # 病害区域用红色填充
    disease_binary = (disease_mask > 0.5).astype(np.uint8)
    red_overlay = np.zeros_like(overlay)
    red_overlay[disease_binary > 0] = [255, 0, 0]
    overlay = cv2.addWeighted(overlay, 0.8, red_overlay, 0.2, 0)
    
    # 添加文本信息
    if disease_ratio is not None:
        if disease_level is not None:
            text = f"Disease Ratio: {disease_ratio:.3f} ({disease_level})"
        else:
            text = f"Disease Ratio: {disease_ratio:.3f}"
        cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 保存图像
    if save_path:
        cv2.imwrite(save_path, overlay)
    
    return overlay


def resize_and_pad(image: torch.Tensor, target_size: Tuple[int, int], 
                  mode: str = 'bilinear') -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """
    调整图像大小并填充到目标尺寸
    
    Args:
        image: 输入图像 [B, C, H, W]
        target_size: 目标尺寸 (H, W)
        mode: 插值模式
        
    Returns:
        resized_image: 调整后的图像
        padding: 填充信息 (top, bottom, left, right)
    """
    _, _, h, w = image.shape
    target_h, target_w = target_size
    
    # 计算缩放比例
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # 调整大小
    resized = F.interpolate(image, size=(new_h, new_w), mode=mode, align_corners=True)
    
    # 计算填充
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    # 应用填充
    padded = F.pad(resized, (left, right, top, bottom), mode='constant', value=0)
    
    return padded, (top, bottom, left, right) 
