import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Optional, NamedTuple
import cv2


def get_leaf_rgb_from_mask(original_image: torch.Tensor, leaf_mask: torch.Tensor,
                          threshold: float = 0.5) -> torch.Tensor:

    if leaf_mask.dim() == 3:
        leaf_mask = leaf_mask.unsqueeze(1)


    if leaf_mask.shape[-2:] != original_image.shape[-2:]:
        leaf_mask = F.interpolate(leaf_mask, size=original_image.shape[-2:],
                                mode='bilinear', align_corners=True)


    binary_mask = (leaf_mask > threshold).float()


    leaf_rgb = original_image * binary_mask

    return leaf_rgb


def apply_mask_to_image(image: torch.Tensor, mask: torch.Tensor,
                       background_value: float = 0.0) -> torch.Tensor:

    if mask.dim() == 3:
        mask = mask.unsqueeze(1)


    if mask.shape[-2:] != image.shape[-2:]:
        mask = F.interpolate(mask, size=image.shape[-2:],
                           mode='bilinear', align_corners=True)

    binary_mask = (mask > 0.5).float()
    masked_image = image * binary_mask + background_value * (1 - binary_mask)

    return masked_image


class Confusion(NamedTuple):
    tp: int
    tn: int
    fp: int
    fn: int


def _compute_confusion(pred: np.ndarray, gt: np.ndarray) -> Confusion:
    pred_flat = pred.reshape(-1).astype(bool)
    gt_flat = gt.reshape(-1).astype(bool)

    tp = np.logical_and(pred_flat, gt_flat).sum()
    tn = np.logical_and(~pred_flat, ~gt_flat).sum()
    fp = np.logical_and(pred_flat, ~gt_flat).sum()
    fn = np.logical_and(~pred_flat, gt_flat).sum()
    return Confusion(int(tp), int(tn), int(fp), int(fn))


def _iou(true_matches: int, fp: int, fn: int) -> float:
    denom = true_matches + fp + fn
    if denom == 0:
        return 1.0
    return true_matches / denom


def _dice(tp: int, fp: int, fn: int) -> float:
    denom = (2 * tp) + fp + fn
    if denom == 0:
        return 1.0
    return (2 * tp) / denom


def _recall(tp: int, fn: int) -> float:
    denom = tp + fn
    if denom == 0:
        return 1.0
    return tp / denom


def _precision_robust(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp
    if denom == 0:
        return 1.0 if (tp + fn) == 0 else 0.0
    return tp / denom


def _accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    total = tp + tn + fp + fn
    if total == 0:
        return 1.0
    return (tp + tn) / total


def calculate_segmentation_metrics(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> Dict[str, float]:


    if pred_mask.dim() == 3:
        pred_mask = pred_mask.unsqueeze(1)
    if gt_mask.dim() == 3:
        gt_mask = gt_mask.unsqueeze(1)


    pred_mask = pred_mask.float().detach().cpu().numpy()
    gt_mask = gt_mask.float().detach().cpu().numpy()

    if pred_mask.ndim != 4 or gt_mask.ndim != 4:
        raise ValueError(f"Expected 4D masks, got pred={pred_mask.shape}, gt={gt_mask.shape}")

    fg_ious = []
    bg_ious = []
    mious = []
    dices = []
    precisions = []
    recalls = []
    accuracies = []
    f2_scores = []

    batch_size = pred_mask.shape[0]
    for idx in range(batch_size):
        y_pred = pred_mask[idx]
        y_true = gt_mask[idx]

        if y_true.max() > 1.0:
            y_true = y_true / 255.0

        # Training-time predictions are detached logits, while evaluation code
        # often passes probabilities after sigmoid. Only GT masks should use the
        # 0-255 normalization path; predictions outside [0, 1] are treated as logits.
        if y_pred.min() < 0.0 or y_pred.max() > 1.0:
            y_pred = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -60.0, 60.0)))

        y_pred_bin = (y_pred > 0.5).astype(np.uint8)
        y_true_bin = (y_true > 0.5).astype(np.uint8)

        conf = _compute_confusion(y_pred_bin, y_true_bin)
        tp, tn, fp, fn = conf.tp, conf.tn, conf.fp, conf.fn

        fg_iou = _iou(tp, fp, fn)
        bg_iou = _iou(tn, fn, fp)
        miou = (fg_iou + bg_iou) / 2.0
        dice = _dice(tp, fp, fn)
        recall = _recall(tp, fn)
        precision = _precision_robust(tp, fp, fn)
        accuracy = _accuracy(tp, tn, fp, fn)

        beta2 = 4.0
        if (beta2 * precision + recall) == 0:
            f2 = 1.0 if precision == 1.0 and recall == 1.0 else 0.0
        else:
            f2 = (1 + beta2) * (precision * recall) / ((beta2 * precision) + recall)

        fg_ious.append(fg_iou)
        bg_ious.append(bg_iou)
        mious.append(miou)
        dices.append(dice)
        precisions.append(precision)
        recalls.append(recall)
        accuracies.append(accuracy)
        f2_scores.append(f2)

    return {
        'iou': float(np.mean(fg_ious)),
        'miou': float(np.mean(mious)),
        'bg_iou': float(np.mean(bg_ious)),
        'dice': float(np.mean(dices)),
        'precision': float(np.mean(precisions)),
        'recall': float(np.mean(recalls)),
        'accuracy': float(np.mean(accuracies)),
        'f2': float(np.mean(f2_scores)),
    }


def visualize_predictions(original_image: torch.Tensor,
                         leaf_mask: torch.Tensor,
                         disease_mask: torch.Tensor,
                         disease_ratio: Optional[float] = None,
                         disease_level: Optional[str] = None,
                         save_path: Optional[str] = None) -> np.ndarray:


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


    h, w = img.shape[:2]
    overlay = img.copy()


    leaf_binary = (leaf_mask > 0.5).astype(np.uint8)
    leaf_contours, _ = cv2.findContours(leaf_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, leaf_contours, -1, (0, 255, 0), 2)


    disease_binary = (disease_mask > 0.5).astype(np.uint8)
    red_overlay = np.zeros_like(overlay)
    red_overlay[disease_binary > 0] = [255, 0, 0]
    overlay = cv2.addWeighted(overlay, 0.8, red_overlay, 0.2, 0)


    if disease_ratio is not None:
        if disease_level is not None:
            text = f"Disease Ratio: {disease_ratio:.3f} ({disease_level})"
        else:
            text = f"Disease Ratio: {disease_ratio:.3f}"
        cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


    if save_path:
        cv2.imwrite(save_path, overlay)

    return overlay


def resize_and_pad(image: torch.Tensor, target_size: Tuple[int, int],
                  mode: str = 'bilinear') -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:

    _, _, h, w = image.shape
    target_h, target_w = target_size


    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)


    resized = F.interpolate(image, size=(new_h, new_w), mode=mode, align_corners=True)


    pad_h = target_h - new_h
    pad_w = target_w - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left


    padded = F.pad(resized, (left, right, top, bottom), mode='constant', value=0)

    return padded, (top, bottom, left, right)
