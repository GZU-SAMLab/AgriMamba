

import os
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

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


    print(f"Loading checkpoint from {checkpoint_path}")
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

    print(f"Model loaded successfully on {device}")
    return model

def extract_leaf_rgb_from_mask(original_image: Image.Image, leaf_mask: np.ndarray) -> Image.Image:


    if original_image.size != (leaf_mask.shape[1], leaf_mask.shape[0]):

        mask_image = Image.fromarray((leaf_mask * 255).astype(np.uint8), mode='L')
        mask_image = mask_image.resize(original_image.size, Image.NEAREST)
        leaf_mask = np.array(mask_image) / 255.0


    original_np = np.array(original_image)


    if leaf_mask.ndim == 3:
        leaf_mask = leaf_mask[:, :, 0]


    binary_mask = (leaf_mask > 0.5).astype(np.uint8)


    mask_3ch = np.stack([binary_mask, binary_mask, binary_mask], axis=2)


    leaf_rgb_np = original_np * mask_3ch


    return Image.fromarray(leaf_rgb_np.astype(np.uint8))

def generate_predictions(model, data_loader, output_dir: str, device: str = 'cuda', data_path: str = './dataset', dataset_name: str = 'dataset4380_split'):

    model.eval()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating leaf RGB images to {output_dir}")

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating leaf RGB images"):
            images = batch['query_img'].to(device)
            img_names = batch['img_name']


            predictions = model(images)


            predictions = torch.sigmoid(predictions)


            for i, img_name in enumerate(img_names):
                pred_mask = predictions[i].cpu().numpy()[0]


                split = data_loader.dataset.split
                original_img_path = Path(data_path) / dataset_name / split / 'img' / f"{img_name}.jpg"

                try:

                    original_image = Image.open(original_img_path).convert('RGB')
                    original_image = ImageOps.exif_transpose(original_image)


                    leaf_rgb_image = extract_leaf_rgb_from_mask(original_image, pred_mask)


                    save_path = output_path / f"{img_name}.jpg"
                    leaf_rgb_image.save(save_path, quality=95)

                except Exception as e:
                    raise RuntimeError(f"Failed to process {img_name}: {e}") from e

def generate_stage1_results(checkpoint_path: str,
                          data_path: str,
                          output_dir: str,
                          device: str = 'cuda',
                          batch_size: int = 8,
                          dataset_name: str = 'dataset4380_split'):


    model = create_leaf_model(checkpoint_path, device)


    from dataset import SegmentationDataset
    from torch.utils.data import DataLoader


    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} set...")


        dataset = SegmentationDataset(
            root=data_path,
            split=split,
            stage=1,
            input_size=(480, 480),
            dataset_name=dataset_name
        )


        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )


        split_output_dir = os.path.join(output_dir, split, 'leaf_rgb')


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
                       help='Dataset name')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for stage1 results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for inference')

    args = parser.parse_args()


    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data path not found: {args.data_path}")


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
