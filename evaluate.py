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
import pandas as pd


from segmentation_model.config import SegmentationConfig, get_args_parser
from segmentation_model.combined_model import create_combined_model
from segmentation_model.dataset import StageDatasetWrapper
from segmentation_model.engine import evaluate_stage
from segmentation_model.utils import get_leaf_rgb_from_mask, calculate_segmentation_metrics


import lesion_utils as utils


GRADE_THRESHOLDS = {
    "Healthy": 0,
    "lesion_level_1": (0.0000001, 10),
    "lesion_level_3": (10, 25),
    "lesion_level_5": (25, 40),
    "lesion_level_7": (40, 65),
    "lesion_level_9": (65, 100),
}

MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_matching_mask(folder: Path, stem: str):
    for ext in sorted(MASK_EXTENSIONS):
        candidate = folder / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    for path in folder.glob(f"{stem}.*"):
        if path.suffix.lower() in MASK_EXTENSIONS:
            return path
    return None


def calculate_mask_area(mask_path: Path) -> int:
    image = Image.open(mask_path).convert("L")
    return int(np.sum(np.asarray(image) > 128))


def classify_grade(percentage: float) -> str:
    for grade, threshold in GRADE_THRESHOLDS.items():
        if isinstance(threshold, tuple):
            lower, upper = threshold
            if lower < percentage <= upper:
                return grade
        else:
            if percentage == threshold:
                return grade
    if percentage == 0:
        return "Healthy"
    return "lesion_level_9"


def compute_grade_stats(mask_root: Path, stem: str) -> dict:
    leaf_path = find_matching_mask(mask_root / "leafClass", stem)
    lesion_path = find_matching_mask(mask_root / "lesionClass", stem)
    if leaf_path is None or lesion_path is None:
        missing = []
        if leaf_path is None:
            missing.append(str(mask_root / "leafClass" / f"{stem}.*"))
        if lesion_path is None:
            missing.append(str(mask_root / "lesionClass" / f"{stem}.*"))
        raise FileNotFoundError(", ".join(missing))

    leaf_area = calculate_mask_area(leaf_path)
    lesion_area = calculate_mask_area(lesion_path)
    percentage = 0.0 if leaf_area == 0 else (lesion_area / leaf_area) * 100
    return {
        "leaf_area": leaf_area,
        "lesion_area": lesion_area,
        "percentage": round(float(percentage), 6),
        "grade": classify_grade(percentage),
    }


def evaluate_grade_and_save_excel(data_path: str, dataset_name: str, split: str,
                                  prediction_dir: Path, output_dir: Path,
                                  image_names) -> dict:
    answer_dir = Path(data_path) / dataset_name / split
    records = []

    for image_name in sorted(image_names):
        stem = str(image_name)
        record = {"image_name": stem}
        try:
            answer_stats = compute_grade_stats(answer_dir, stem)
            predict_stats = compute_grade_stats(prediction_dir, stem)
            is_correct = answer_stats["grade"] == predict_stats["grade"]

            record.update({
                "predict_grade": predict_stats["grade"],
                "correct_grade": answer_stats["grade"],
                "answer_grade": answer_stats["grade"],
                "is_correct": is_correct,
                "answer_percentage": answer_stats["percentage"],
                "predict_percentage": predict_stats["percentage"],
                "absolute_diff_percentage": round(
                    abs(answer_stats["percentage"] - predict_stats["percentage"]), 6
                ),
                "answer_leaf_area": answer_stats["leaf_area"],
                "answer_lesion_area": answer_stats["lesion_area"],
                "predict_leaf_area": predict_stats["leaf_area"],
                "predict_lesion_area": predict_stats["lesion_area"],
                "error": "",
            })
        except Exception as exc:
            record.update({
                "predict_grade": "",
                "correct_grade": "",
                "answer_grade": "",
                "is_correct": False,
                "answer_percentage": np.nan,
                "predict_percentage": np.nan,
                "absolute_diff_percentage": np.nan,
                "answer_leaf_area": np.nan,
                "answer_lesion_area": np.nan,
                "predict_leaf_area": np.nan,
                "predict_lesion_area": np.nan,
                "error": str(exc),
            })
        records.append(record)

    records_df = pd.DataFrame(records)
    preferred_columns = [
        "image_name",
        "predict_grade",
        "correct_grade",
        "is_correct",
        "predict_percentage",
        "answer_percentage",
        "absolute_diff_percentage",
        "predict_leaf_area",
        "predict_lesion_area",
        "answer_leaf_area",
        "answer_lesion_area",
        "error",
    ]
    records_df = records_df[[col for col in preferred_columns if col in records_df.columns]]
    valid_df = records_df[records_df["error"] == ""]
    total_images = len(records_df)
    evaluated_images = len(valid_df)
    correct_predictions = int(valid_df["is_correct"].sum()) if evaluated_images else 0
    overall_accuracy = correct_predictions / evaluated_images if evaluated_images else 0.0

    summary_df = pd.DataFrame([
        {
            "total_images": total_images,
            "evaluated_images": evaluated_images,
            "correct_predictions": correct_predictions,
            "overall_grade_accuracy": overall_accuracy,
            "overall_grade_accuracy_percent": round(overall_accuracy * 100, 4),
            "failed_images": total_images - evaluated_images,
        }
    ])

    excel_path = output_dir / "grade_evaluation_results.xlsx"
    output_dir.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        records_df.to_excel(writer, sheet_name="per_image", index=False)

    result = {
        "total_images": total_images,
        "evaluated_images": evaluated_images,
        "correct_predictions": correct_predictions,
        "overall_grade_accuracy": overall_accuracy,
        "overall_grade_accuracy_percent": round(overall_accuracy * 100, 4),
        "failed_images": total_images - evaluated_images,
        "excel_path": str(excel_path),
    }
    print(f" Grade evaluation saved to: {excel_path}")
    print(
        " Grade accuracy: "
        f"{correct_predictions}/{evaluated_images} "
        f"({overall_accuracy * 100:.2f}%)"
    )
    return result


def create_leaf_model(checkpoint_path: str, device: str = 'cuda'):


    import sys
    import os.path as osp
    current_dir = osp.dirname(__file__)
    parent_dir = osp.dirname(current_dir)
    sys.path.append(parent_dir)


    from model.leaf_model.leaf_model import LMLS
    from timm.models import create_model


    model, _ = create_model(
        'LMLS',
        img_size=480,
        model_size='base'
    )


    print(f"Loading leaf model checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')


    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint


    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('leaf_model.'):
            new_state_dict[k[11:]] = v
        else:
            new_state_dict[k] = v


    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    print(f"Leaf model loaded successfully on {device}")
    return model


def extract_rgb_from_mask(original_image: Image.Image, mask: np.ndarray) -> Image.Image:


    if original_image.size != (mask.shape[1], mask.shape[0]):

        mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        mask_image = mask_image.resize(original_image.size, Image.NEAREST)
        mask = np.array(mask_image) / 255.0


    original_np = np.array(original_image)


    if mask.ndim == 3:
        mask = mask[:, :, 0]


    binary_mask = (mask > 0.5).astype(np.uint8)


    mask_3ch = np.stack([binary_mask, binary_mask, binary_mask], axis=2)


    rgb_np = original_np * mask_3ch


    return Image.fromarray(rgb_np.astype(np.uint8))


def create_stage1_dataset_for_evaluation(data_path: str, split: str = 'test', dataset_name: str = 'dataset4380_split'):


    from segmentation_model.dataset import SegmentationDataset


    dataset = SegmentationDataset(
        root=data_path,
        split=split,
        stage=1,
        input_size=(480, 480),
        dataset_name=dataset_name
    )

    return dataset


def evaluate_stage1_and_generate_leaf_rgb(leaf_model, data_path: str, split: str,
                                        device, config, output_dir: Path,
                                        batch_size: int = 4, amp_autocast=suppress):

    print("=== Stage 1 Evaluation: Leaf Segmentation ===")

    leaf_model.eval()


    leaf_rgb_dir = output_dir / "stage1_leaf_rgb"
    leaf_mask_dir = output_dir / "leafClass"
    leaf_rgb_dir.mkdir(parents=True, exist_ok=True)
    leaf_mask_dir.mkdir(parents=True, exist_ok=True)


    dataset = create_stage1_dataset_for_evaluation(data_path, split, config.data_set)


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
    image_info_dict = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Generating leaf RGB images and masks")):

            images = batch['query_img'].to(device, non_blocking=True)
            img_names = batch['img_name']


            if str(device).startswith('cuda') and amp_autocast != suppress:
                with torch.amp.autocast(device_type='cuda'):
                    predictions = leaf_model(images)
            else:
                predictions = leaf_model(images)


            predictions = torch.sigmoid(predictions)


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


            for i, img_name in enumerate(img_names):
                pred_mask = predictions[i].cpu().numpy()[0]


                original_img_path = Path(data_path) / config.data_set / split / 'img' / f"{img_name}.jpg"

                try:

                    original_image = Image.open(original_img_path).convert('RGB')
                    original_image = ImageOps.exif_transpose(original_image)


                    leaf_rgb_image = extract_rgb_from_mask(original_image, pred_mask)


                    leaf_rgb_path = leaf_rgb_dir / f"{img_name}.jpg"
                    leaf_rgb_image.save(leaf_rgb_path, quality=95)


                    leaf_mask_path = leaf_mask_dir / f"{img_name}.png"
                    mask_image = Image.fromarray((pred_mask * 255).astype(np.uint8), mode='L')
                    mask_image = mask_image.resize(original_image.size, Image.NEAREST)
                    mask_image.save(leaf_mask_path)
                    resized_mask = np.array(mask_image, dtype=np.float32) / 255.0


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

    print("=== Stage 2 Evaluation: Lesion Segmentation ===")

    combined_model.eval()
    combined_model.set_stage(2)


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


    import torchvision.transforms as T
    lesion_transform = T.Compose([
        T.Resize((480, 480)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    img_names = list(image_info_dict.keys())
    total_samples = len(img_names)

    with torch.no_grad():
        for batch_start in tqdm(range(0, total_samples, batch_size), desc="Generating lesion RGB images and masks"):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_img_names = img_names[batch_start:batch_end]


            batch_images = []
            batch_sentences = []
            batch_original_images = []
            batch_lesion_gts = []

            for img_name in batch_img_names:
                info = image_info_dict[img_name]


                leaf_rgb_image = info['leaf_rgb_image']
                leaf_rgb_tensor = lesion_transform(leaf_rgb_image)
                batch_images.append(leaf_rgb_tensor)


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


                lesion_gt_path = Path(config.data_path) / config.data_set / config.split / 'lesionClass' / f"{img_name}.png"
                if lesion_gt_path.exists():
                    lesion_gt = Image.open(lesion_gt_path).convert('L')
                    lesion_gt = lesion_gt.resize((480, 480), Image.NEAREST)
                    lesion_gt_np = np.array(lesion_gt) / 255.0
                    lesion_gt_tensor = torch.from_numpy(lesion_gt_np).float().unsqueeze(0)
                    batch_lesion_gts.append(lesion_gt_tensor)
                else:
                    raise FileNotFoundError(f"Lesion GT mask not found for {img_name}: {lesion_gt_path}")


            images_tensor = torch.stack(batch_images).to(device)


            if str(device).startswith('cuda') and amp_autocast != suppress:
                with torch.amp.autocast(device_type='cuda'):
                    lesion_pred = combined_model.forward_stage2(images_tensor, batch_sentences)[0]
            else:
                lesion_pred = combined_model.forward_stage2(images_tensor, batch_sentences)[0]


            lesion_pred_prob = torch.sigmoid(lesion_pred)


            valid_gts = [gt for gt in batch_lesion_gts if gt is not None]
            if valid_gts and len(valid_gts) == len(batch_lesion_gts):

                try:
                    lesion_gt_batch = torch.stack(valid_gts).to(device)
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


            for i, img_name in enumerate(batch_img_names):
                pred_mask = lesion_pred_prob[i].cpu().numpy()[0]
                original_image = batch_original_images[i]

                try:

                    lesion_rgb_image = extract_rgb_from_mask(original_image, pred_mask)


                    lesion_rgb_path = lesion_rgb_dir / f"{img_name}.jpg"
                    lesion_rgb_image.save(lesion_rgb_path, quality=95)


                    lesion_mask_path = lesion_mask_dir / f"{img_name}.png"
                    mask_image = Image.fromarray((pred_mask * 255).astype(np.uint8), mode='L')
                    mask_image = mask_image.resize(original_image.size, Image.NEAREST)
                    mask_image.save(lesion_mask_path)

                except Exception as e:
                    raise RuntimeError(f"Failed to process lesion segmentation for {img_name}: {e}") from e


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
    parser = argparse.ArgumentParser(description='Two-stage evaluation for plant disease segmentation')


    parser.add_argument('--leaf-checkpoint', type=str, required=True,
                       help='Path to the leaf segmentation checkpoint (required)')
    parser.add_argument('--lesion-checkpoint', type=str, required=True,
                       help='Path to the lesion segmentation checkpoint (required)')


    parser.add_argument('--data-path', default='./dataset', help='Dataset root path')
    parser.add_argument('--data-set', default='dataset4380_split', type=str, help='Dataset name')
    parser.add_argument('--output-dir', default='./output/evaluation_enhanced', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--device', default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--split', default='test', help='Evaluation split (train/val/test)')


    parser.add_argument('--save-intermediate', action='store_true', default=True,
                       help='Save intermediate leaf/lesion RGB images and masks')
    parser.add_argument('--cleanup-intermediate', action='store_true', default=False,
                       help='Delete intermediate leaf/lesion RGB images and masks after evaluation')
    parser.add_argument('--if-amp', action='store_true', default=True,
                       help='Enable mixed precision inference')
    parser.add_argument('--no-amp', action='store_false', dest='if_amp',
                       help='Disable mixed precision inference')

    args = parser.parse_args()


    if not os.path.exists(args.leaf_checkpoint):
        print(f"Error: leaf checkpoint does not exist: {args.leaf_checkpoint}")
        return

    if not os.path.exists(args.lesion_checkpoint):
        print(f"Error: lesion checkpoint does not exist: {args.lesion_checkpoint}")
        return


    utils.init_distributed_mode(args)
    print(f"Arguments: {args}")


    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        args.device = 'cpu'
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")


    full_args = argparse.Namespace(
        data_path=args.data_path,
        data_set=args.data_set,
        input_size=480,
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
        split=args.split
    )

    config = SegmentationConfig.from_args(full_args)
    config.device = args.device
    config.split = args.split

    print(f"Configuration: {config.to_dict()}")


    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    if utils.is_main_process():
        with (output_dir / "config.json").open("w") as f:
            json.dump(config.to_dict(), f, indent=4)


    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True


    amp_autocast = suppress
    if config.training_config['if_amp'] and torch.cuda.is_available():
        amp_autocast = torch.cuda.amp.autocast
        print("Using AMP for inference")
    else:
        print("AMP disabled (CUDA not available or disabled)")


    print("=== Starting Enhanced Two-Stage Evaluation ===")


    print(f"Creating direct leaf model from checkpoint: {args.leaf_checkpoint}")
    leaf_model = create_leaf_model(args.leaf_checkpoint, device)


    print("Creating combined model for lesion segmentation...")
    combined_model = create_combined_model(config, stage=0, device=device)
    print(f"Loading lesion model checkpoint from {args.lesion_checkpoint}")
    combined_model.load_lesion_checkpoint(args.lesion_checkpoint)

    print(" Both models loaded successfully")


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


    grade_results = {}
    if utils.is_main_process():
        print("\n=== Step 3: Evaluating Disease Grade from Leaf/Lesion Masks ===")
        grade_results = evaluate_grade_and_save_excel(
            data_path=args.data_path,
            dataset_name=args.data_set,
            split=args.split,
            prediction_dir=output_dir,
            output_dir=output_dir,
            image_names=image_info_dict.keys(),
        )


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
    print(f"   - Grade Accuracy: {grade_results.get('overall_grade_accuracy', 0):.4f}")
    print(f"   - Total Samples: {len(image_info_dict)}")


    if utils.is_main_process():

        final_results = {
            'stage1_results': stage1_stats,
            'stage2_results': stage2_stats,
            'grade_results': grade_results,
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
                'grade_accuracy': grade_results.get('overall_grade_accuracy', 0),
                'grade_correct_predictions': grade_results.get('correct_predictions', 0),
                'grade_evaluated_images': grade_results.get('evaluated_images', 0),
                'grade_failed_images': grade_results.get('failed_images', 0),
                'grade_excel_path': grade_results.get('excel_path', ''),
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


        result_filename = "evaluation_results_enhanced.json"
        with (output_dir / result_filename).open("w") as f:
            json.dump(final_results, f, indent=4, default=str)

        print(f"\n Evaluation results saved to: {output_dir / result_filename}")


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
            'grade_accuracy': grade_results.get('overall_grade_accuracy', 0),
            'grade_correct_predictions': grade_results.get('correct_predictions', 0),
            'grade_evaluated_images': grade_results.get('evaluated_images', 0),
            'grade_failed_images': grade_results.get('failed_images', 0),
            'grade_excel_path': grade_results.get('excel_path', ''),
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


    if args.cleanup_intermediate and args.save_intermediate:
        print(f"\n🧹 Cleaning up intermediate files...")
        try:

            leaf_rgb_dir = output_dir / "stage1_leaf_rgb"
            leaf_mask_dir = output_dir / "leafClass"

            if leaf_rgb_dir.exists():
                shutil.rmtree(leaf_rgb_dir)
                print(f" Deleted leaf RGB images directory: {leaf_rgb_dir}")

            if leaf_mask_dir.exists():
                shutil.rmtree(leaf_mask_dir)
                print(f" Deleted leaf masks directory: {leaf_mask_dir}")


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

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
