import os
import torch
import torch.utils.data as data
from PIL import Image, ImageOps
import numpy as np
from typing import Optional, List, Dict, Any
import sys
import os.path as osp


current_dir = osp.dirname(__file__)
parent_dir = osp.dirname(current_dir)
dataset_dir = osp.join(parent_dir, 'dataset')
sys.path.append(dataset_dir)

try:
    from transform import get_transform
except ImportError:
    print("Warning: Could not import transform from dataset directory")
    print(f"Tried to import from: {dataset_dir}")

    def get_transform(size, train=True):
        import torchvision.transforms as T
        from PIL import Image
        import torch

        class SimpleTransform:
            def __init__(self, size, train=True):
                self.size = size
                self.train = train

            def __call__(self, image, target=None):

                image = image.resize(self.size, Image.BILINEAR)
                if target is not None:
                    target = target.resize(self.size, Image.NEAREST)


                image = T.ToTensor()(image)
                if target is not None:
                    target = torch.from_numpy(np.array(target)).float()

                return image, target

        return SimpleTransform(size, train)


class SegmentationDataset(data.Dataset):


    def __init__(self,
                 root: str,
                 split: str = 'train',
                 stage: int = 0,
                 input_size: tuple = (480, 480),
                 stage1_results_path: Optional[str] = None,
                 dataset_name: str = 'dataset4380_split'):

        self.root = root
        self.split = split
        self.stage = stage
        self.input_size = input_size
        self.stage1_results_path = stage1_results_path
        self.dataset_name = dataset_name


        self.split_dir = os.path.join(root, dataset_name, split)
        self.img_dir = os.path.join(self.split_dir, 'img')
        self.leaf_mask_dir = os.path.join(self.split_dir, 'leafClass')
        self.lesion_mask_dir = os.path.join(self.split_dir, 'lesionClass')
        self.txt_dir = os.path.join(self.split_dir, 'txt')


        if self.stage1_results_path:
            self.pred_leaf_rgb_dir = os.path.join(self.stage1_results_path, split, 'leaf_rgb')
            if not os.path.exists(self.pred_leaf_rgb_dir):
                print(f"Warning: Stage1 results not found at {self.pred_leaf_rgb_dir}")
                print("Will use ground truth leaf masks for leaf RGB extraction instead")
                self.stage1_results_path = None


        self.images = self._get_image_list()


        self.transform = get_transform(input_size, train=(split == 'train'))

        print(f"Dataset initialized: {len(self.images)} images, stage={stage}, split={split}")
        if self.stage1_results_path:
            print(f"Using stage1 results from: {self.pred_leaf_rgb_dir}")

    def _get_image_list(self) -> List[str]:

        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")


        valid_exts = ('.jpg', '.jpeg', '.png')
        images = [
            f for f in os.listdir(self.img_dir)
            if os.path.isfile(os.path.join(self.img_dir, f)) and f.lower().endswith(valid_exts)
        ]
        images.sort()
        return images

    def _load_text_description(self, img_name: str) -> str:

        base_name = os.path.splitext(img_name)[0]
        txt_path = os.path.join(self.txt_dir, f"{base_name}.txt")

        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                sentence = f.read().strip()

            if not sentence:
                raise ValueError(f"Text file is empty for '{base_name}': {txt_path}")

            return sentence

        except FileNotFoundError:
            raise FileNotFoundError(f"Text file not found for '{base_name}': {txt_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load text file {txt_path}: {e}") from e

    def _load_leaf_mask(self, img_name: str, use_predicted: bool = False) -> Optional[Image.Image]:

        base_name = os.path.splitext(img_name)[0]

        if use_predicted and self.stage1_results_path:


            mask_path = os.path.join(self.leaf_mask_dir, f"{base_name}.png")
        else:

            mask_path = os.path.join(self.leaf_mask_dir, f"{base_name}.png")

        try:
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')

                return mask
            else:
                if use_predicted:

                    return self._load_leaf_mask(img_name, use_predicted=False)
                else:
                    raise FileNotFoundError(f"Leaf mask not found: {mask_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load leaf mask {mask_path}: {e}") from e

    def _load_lesion_mask(self, img_name: str) -> Optional[Image.Image]:

        base_name = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.lesion_mask_dir, f"{base_name}.png")

        try:
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')

                return mask
            else:
                raise FileNotFoundError(f"Lesion mask not found: {mask_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load lesion mask {mask_path}: {e}") from e

    def _extract_leaf_rgb(self, image: Image.Image, leaf_mask: Image.Image) -> Image.Image:


        if leaf_mask is None:
            return image


        if image.size != leaf_mask.size:
            leaf_mask = leaf_mask.resize(image.size, Image.NEAREST)


        image_np = np.array(image)
        mask_np = np.array(leaf_mask)


        binary_mask = (mask_np > 128).astype(np.uint8)


        mask_3ch = np.stack([binary_mask, binary_mask, binary_mask], axis=2)


        leaf_rgb_np = image_np * mask_3ch


        leaf_rgb_image = Image.fromarray(leaf_rgb_np.astype(np.uint8))

        return leaf_rgb_image

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        for _ in range(3):
            img_name = self.images[index]
            base_name = os.path.splitext(img_name)[0]


            img_path = os.path.join(self.img_dir, img_name)
            try:
                image = Image.open(img_path).convert('RGB')
                image = ImageOps.exif_transpose(image)
                break
            except FileNotFoundError as e:
                print(f"Skip bad image: {img_path} ({e})")
                index = (index + 1) % len(self.images)
                continue
            except Exception as e:
                print(f"Skip bad image: {img_path} ({e})")
                index = (index + 1) % len(self.images)
                continue
        else:
            raise RuntimeError("Too many bad images encountered while loading data.")


        sample = {
            'query_img': None,
            'img_name': base_name,
            'sentence': ""
        }


        if self.stage in [0, 1]:
            leaf_mask = self._load_leaf_mask(img_name, use_predicted=False)
        elif self.stage == 2:

            leaf_mask = None
        else:
            leaf_mask = None


        if self.stage in [0, 2]:
            lesion_mask = self._load_lesion_mask(img_name)
        else:
            lesion_mask = None


        if self.stage in [0, 2]:
            sample['sentence'] = self._load_text_description(img_name)


        try:

            if self.stage == 2:
                leaf_rgb_image = None

                if self.stage1_results_path:


                    leaf_rgb_path = os.path.join(self.pred_leaf_rgb_dir, img_name)
                    if os.path.exists(leaf_rgb_path):
                        leaf_rgb_image = Image.open(leaf_rgb_path).convert('RGB')


                if leaf_rgb_image is None:
                    raise FileNotFoundError(f"Stage2 training requires pregenerated leaf RGB images. "
                                          f"Missing leaf RGB image: {os.path.join(self.pred_leaf_rgb_dir, img_name)}. "
                                          f"Please run stage1 result generation first.")


                image, _ = self.transform(leaf_rgb_image, None)

            else:

                if leaf_mask is not None:
                    image, leaf_target = self.transform(image, leaf_mask)

                    if isinstance(leaf_target, torch.Tensor):
                        leaf_target = leaf_target.float()

                        if leaf_target.dim() == 2:
                            leaf_target = leaf_target.unsqueeze(0)
                    else:
                        leaf_target = torch.tensor(leaf_target).float()
                        if leaf_target.dim() == 2:
                            leaf_target = leaf_target.unsqueeze(0)
                    sample['leaf_mask'] = leaf_target
                else:
                    image, _ = self.transform(image, None)

            if lesion_mask is not None:

                dummy_img = Image.new('RGB', self.input_size)
                _, lesion_target = self.transform(dummy_img, lesion_mask)

                if isinstance(lesion_target, torch.Tensor):
                    lesion_target = lesion_target.float()
                    if lesion_target.dim() == 2:
                        lesion_target = lesion_target.unsqueeze(0)
                else:
                    lesion_target = torch.tensor(lesion_target).float()
                    if lesion_target.dim() == 2:
                        lesion_target = lesion_target.unsqueeze(0)
                sample['lesion_mask'] = lesion_target

            sample['query_img'] = image

        except Exception as e:
            raise RuntimeError(f"Failed to process sample {index}: {e}") from e

        return sample


def segmentation_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:

    batched_data = {}

    for key in batch[0].keys():
        if key == 'sentence' and isinstance(batch[0][key], list):

            batched_data[key] = [d[key] for d in batch]
        elif key in ['leaf_org_gt', 'lesion_org_gt']:

            batched_data[key] = [d[key] for d in batch if key in d]
        elif key in ['base_name', 'image_filename']:

            batched_data[key] = [d[key] for d in batch]
        else:

            batch_values = [d[key] for d in batch if key in d]
            if batch_values:
                try:
                    batched_data[key] = torch.utils.data.default_collate(batch_values)
                except:

                    batched_data[key] = batch_values

    return batched_data


def build_segmentation_dataset(config, split: str = 'train', stage: int = 0):

    is_train = (split == 'train')


    dataset = SegmentationDataset(
        root=config.data_path,
        split=split,
        stage=stage,
        input_size=(config.input_size, config.input_size),
        stage1_results_path=config.stage1_results_path if hasattr(config, 'stage1_results_path') else None,
        dataset_name=config.data_set
    )

    return dataset


class StageDatasetWrapper:


    def __init__(self, config):
        self.config = config
        self.datasets = {}

    def get_dataset(self, split: str, stage: int):

        key = f"{split}_stage{stage}"

        if key not in self.datasets:
            self.datasets[key] = build_segmentation_dataset(self.config, split, stage)

        return self.datasets[key]

    def get_dataloader(self, split: str, stage: int, batch_size: Optional[int] = None,
                      shuffle: Optional[bool] = None, num_workers: Optional[int] = None):

        dataset = self.get_dataset(split, stage)

        if batch_size is None:
            batch_size = self.config.batch_size
        if shuffle is None:
            shuffle = (split == 'train')
        if num_workers is None:
            num_workers = self.config.num_workers

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self.config.pin_mem,
            collate_fn=segmentation_collate_fn,
            drop_last=(split == 'train')
        )

        return dataloader
