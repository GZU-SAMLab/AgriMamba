import os
import torch
import torch.utils.data as data
from PIL import Image, ImageOps
import numpy as np
from typing import Optional, List, Dict, Any
import sys
import os.path as osp

# 添加dataset路径到sys.path
current_dir = osp.dirname(__file__)
parent_dir = osp.dirname(current_dir)
dataset_dir = osp.join(parent_dir, 'dataset')
sys.path.append(dataset_dir)

try:
    from transform import get_transform
except ImportError:
    print("Warning: Could not import transform from dataset directory")
    print(f"Tried to import from: {dataset_dir}")
    # 提供一个简单的备用变换
    def get_transform(size, train=True):
        import torchvision.transforms as T
        from PIL import Image
        import torch
        
        class SimpleTransform:
            def __init__(self, size, train=True):
                self.size = size
                self.train = train
            
            def __call__(self, image, target=None):
                # 简单的resize变换
                image = image.resize(self.size, Image.BILINEAR)
                if target is not None:
                    target = target.resize(self.size, Image.NEAREST)
                
                # 转换为tensor
                image = T.ToTensor()(image)
                if target is not None:
                    target = torch.from_numpy(np.array(target)).float()
                
                return image, target
        
        return SimpleTransform(size, train)


class SegmentationDataset(data.Dataset):
    """
    植物病害分割数据集
    
    支持两种模式：
    1. 标准模式：加载原始数据，适用于第一阶段训练和联合训练
    2. 预生成叶片模式：加载预生成的叶片分割结果，适用于第二阶段训练
    """
    
    def __init__(self, 
                 root: str, 
                 split: str = 'train',
                 stage: int = 0,
                 input_size: tuple = (480, 480),
                 stage1_results_path: Optional[str] = None,
                 dataset_name: str = 'dataset4380_split'):
        """
        Args:
            root: 数据集根目录
            split: 数据集划分 ('train', 'val', 'test')
            stage: 训练阶段 (0=联合, 1=叶片, 2=病害)
            input_size: 输入图像尺寸
            stage1_results_path: 第一阶段结果路径，用于第二阶段训练
            dataset_name: 数据集名称 (如 'dataset4380_split', 'my_dataset' 等)
        """
        self.root = root
        self.split = split
        self.stage = stage
        self.input_size = input_size
        self.stage1_results_path = stage1_results_path
        self.dataset_name = dataset_name
        
        # 数据路径设置 - 使用动态数据集名称
        self.split_dir = os.path.join(root, dataset_name, split)
        self.img_dir = os.path.join(self.split_dir, 'img')
        self.leaf_mask_dir = os.path.join(self.split_dir, 'leafClass')
        self.lesion_mask_dir = os.path.join(self.split_dir, 'lesionClass')
        self.txt_dir = os.path.join(self.split_dir, 'txt')
        
        # 如果指定了第一阶段结果路径，设置预生成叶片RGB图路径
        if self.stage1_results_path:
            self.pred_leaf_rgb_dir = os.path.join(self.stage1_results_path, split, 'leaf_rgb')
            if not os.path.exists(self.pred_leaf_rgb_dir):
                print(f"Warning: Stage1 results not found at {self.pred_leaf_rgb_dir}")
                print("Will use ground truth leaf masks for leaf RGB extraction instead")
                self.stage1_results_path = None
        
        # 获取图像列表
        self.images = self._get_image_list()
        
        # 数据变换
        self.transform = get_transform(input_size, train=(split == 'train'))
        
        print(f"Dataset initialized: {len(self.images)} images, stage={stage}, split={split}")
        if self.stage1_results_path:
            print(f"Using stage1 results from: {self.pred_leaf_rgb_dir}")
    
    def _get_image_list(self) -> List[str]:
        """获取图像文件列表"""
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        
        # Allow common uppercase image extensions in addition to lowercase ones
        valid_exts = ('.jpg', '.jpeg', '.png')
        images = [
            f for f in os.listdir(self.img_dir)
            if os.path.isfile(os.path.join(self.img_dir, f)) and f.lower().endswith(valid_exts)
        ]
        images.sort()
        return images
    
    def _load_text_description(self, img_name: str) -> str:
        """加载完整文本描述"""
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
        """
        加载叶片mask
        
        Args:
            img_name: 图像文件名
            use_predicted: 是否使用预测的叶片mask
        """
        base_name = os.path.splitext(img_name)[0]
        
        if use_predicted and self.stage1_results_path:
            # 第二阶段训练中，不使用预测的mask，只使用叶子RGB图
            # 如果调用这里说明需要fallback到GT mask
            mask_path = os.path.join(self.leaf_mask_dir, f"{base_name}.png")
        else:
            # 使用ground truth叶片mask
            mask_path = os.path.join(self.leaf_mask_dir, f"{base_name}.png")
        
        try:
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
                # PNG格式的mask通常不包含EXIF信息，不需要旋转处理
                return mask
            else:
                if use_predicted:
                    # 回退到GT mask
                    return self._load_leaf_mask(img_name, use_predicted=False)
                else:
                    raise FileNotFoundError(f"Leaf mask not found: {mask_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load leaf mask {mask_path}: {e}") from e
    
    def _load_lesion_mask(self, img_name: str) -> Optional[Image.Image]:
        """加载病害mask"""
        base_name = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.lesion_mask_dir, f"{base_name}.png")
        
        try:
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
                # PNG格式的mask通常不包含EXIF信息，不需要旋转处理
                return mask
            else:
                raise FileNotFoundError(f"Lesion mask not found: {mask_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load lesion mask {mask_path}: {e}") from e
    
    def _extract_leaf_rgb(self, image: Image.Image, leaf_mask: Image.Image) -> Image.Image:
        """
        从原始图像中提取叶片区域的RGB图像
        
        Args:
            image: 原始RGB图像
            leaf_mask: 叶片分割mask (可能为None)
            
        Returns:
            leaf_rgb_image: 叶片区域RGB图像（非叶片区域为黑色）
        """
        # 如果没有leaf_mask，直接返回原图
        if leaf_mask is None:
            return image
            
        # 确保图像和mask尺寸一致
        if image.size != leaf_mask.size:
            leaf_mask = leaf_mask.resize(image.size, Image.NEAREST)
        
        # 转换为numpy数组
        image_np = np.array(image)  # [H, W, 3]
        mask_np = np.array(leaf_mask)  # [H, W]
        
        # 二值化mask（大于128的认为是叶片区域）
        binary_mask = (mask_np > 128).astype(np.uint8)
        
        # 将mask扩展到3个通道
        mask_3ch = np.stack([binary_mask, binary_mask, binary_mask], axis=2)  # [H, W, 3]
        
        # 应用mask到图像
        leaf_rgb_np = image_np * mask_3ch
        
        # 转换回PIL图像
        leaf_rgb_image = Image.fromarray(leaf_rgb_np.astype(np.uint8))
        
        return leaf_rgb_image
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        img_name = self.images[index]
        base_name = os.path.splitext(img_name)[0]
        
        # 加载原始图像
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
            image = ImageOps.exif_transpose(image)  # 处理EXIF旋转
        except FileNotFoundError:
            raise FileNotFoundError(f" 错误：找不到图像文件: {img_path}")
        except Exception as e:
            raise RuntimeError(f" 错误：加载图像文件失败 {img_path}: {e}")
        
        # 根据训练阶段加载不同的数据
        sample = {
            'query_img': None,
            'img_name': base_name,
            'sentence': ""
        }
        
        # 加载叶片mask
        if self.stage in [0, 1]:  # 联合训练或第一阶段训练
            leaf_mask = self._load_leaf_mask(img_name, use_predicted=False)
        elif self.stage == 2:  # 第二阶段训练
            # 第二阶段可能需要GT叶片mask作为回退
            leaf_mask = None  # 先设为None，在需要时再加载
        else:
            leaf_mask = None
        
        # 加载病害mask（第二阶段和联合训练需要）
        if self.stage in [0, 2]:
            lesion_mask = self._load_lesion_mask(img_name)
        else:
            lesion_mask = None
        
        # 加载文本描述（第二阶段和联合训练需要）
        if self.stage in [0, 2]:
            sample['sentence'] = self._load_text_description(img_name)
        
        # 应用数据变换
        try:
            # 第二阶段的特殊处理：直接加载预生成的叶片区域RGB图像
            if self.stage == 2:
                leaf_rgb_image = None
                
                if self.stage1_results_path:
                    # 尝试加载预生成的叶子RGB图像
                    # img_name已经包含.jpg扩展名，不需要再加
                    leaf_rgb_path = os.path.join(self.pred_leaf_rgb_dir, img_name)
                    if os.path.exists(leaf_rgb_path):
                        leaf_rgb_image = Image.open(leaf_rgb_path).convert('RGB')
                
                # 如果没有预生成的RGB图像，报错
                if leaf_rgb_image is None:
                    raise FileNotFoundError(f"Stage2 training requires pregenerated leaf RGB images. "
                                          f"Missing leaf RGB image: {os.path.join(self.pred_leaf_rgb_dir, img_name)}. "
                                          f"Please run stage1 result generation first.")
                
                # 应用变换
                image, _ = self.transform(leaf_rgb_image, None)
                
            else:
                # 第一阶段或联合训练：正常处理
                if leaf_mask is not None:
                    image, leaf_target = self.transform(image, leaf_mask)
                    # 确保mask是float类型和正确维度
                    if isinstance(leaf_target, torch.Tensor):
                        leaf_target = leaf_target.float()
                        # 确保维度为 [1, H, W]
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
                # 创建一个dummy图像进行transform
                dummy_img = Image.new('RGB', self.input_size)
                _, lesion_target = self.transform(dummy_img, lesion_mask)
                # 确保mask是float类型和正确维度
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
    """
    自定义collate函数，处理不同模态的数据
    """
    batched_data = {}
    
    for key in batch[0].keys():
        if key == 'sentence' and isinstance(batch[0][key], list):
            # 处理多句子文本
            batched_data[key] = [d[key] for d in batch]
        elif key in ['leaf_org_gt', 'lesion_org_gt']:
            # 保持原始GT格式
            batched_data[key] = [d[key] for d in batch if key in d]
        elif key in ['base_name', 'image_filename']:
            # 字符串列表
            batched_data[key] = [d[key] for d in batch]
        else:
            # 使用默认collate
            batch_values = [d[key] for d in batch if key in d]
            if batch_values:
                try:
                    batched_data[key] = torch.utils.data.default_collate(batch_values)
                except:
                    # 如果collate失败，保持为列表
                    batched_data[key] = batch_values
    
    return batched_data


def build_segmentation_dataset(config, split: str = 'train', stage: int = 0):
    """
    构建分割数据集
    
    Args:
        config: 配置对象
        split: 数据集分割
        stage: 训练阶段
    
    Returns:
        dataset: 数据集对象
    """
    is_train = (split == 'train')
    
    # 支持任意数据集名称，不再限制为特定数据集
    dataset = SegmentationDataset(
        root=config.data_path,
        split=split,
        stage=stage,
        input_size=(config.input_size, config.input_size),
        stage1_results_path=config.stage1_results_path if hasattr(config, 'stage1_results_path') else None,
        dataset_name=config.data_set  # 使用配置中的数据集名称
    )
    
    return dataset


class StageDatasetWrapper:
    """
    分阶段数据集包装器，管理不同阶段的数据集
    """
    
    def __init__(self, config):
        self.config = config
        self.datasets = {}
    
    def get_dataset(self, split: str, stage: int):
        """获取指定阶段和分割的数据集"""
        key = f"{split}_stage{stage}"
        
        if key not in self.datasets:
            self.datasets[key] = build_segmentation_dataset(self.config, split, stage)
        
        return self.datasets[key]
    
    def get_dataloader(self, split: str, stage: int, batch_size: Optional[int] = None, 
                      shuffle: Optional[bool] = None, num_workers: Optional[int] = None):
        """获取数据加载器"""
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
