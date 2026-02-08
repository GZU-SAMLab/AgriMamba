import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from timm.models import create_model
import sys
import os.path as osp

# 添加model路径到sys.path
sys.path.append(osp.join(osp.dirname(__file__), '..'))
import model.leaf_model.leaf_model
import model.lesion_model.lesion_model

from .utils import get_leaf_rgb_from_mask


class CombinedSegmentationModel(nn.Module):
    """
    两阶段组合分割模型
    第一阶段：叶片分割 (Leaf Model)
    第二阶段：病害分割 (Lesion Model)
    """
    
    def __init__(self, 
                 config,
                 leaf_model_name: str = 'LMLS',
                 lesion_model_name: str = 'TMLS',
                 img_size: int = 480,
                 model_size: str = "base",
                 stage: int = 0):  # 0: 两阶段, 1: 仅叶片, 2: 仅病害
        """
        Args:
            config: 配置对象
            leaf_model_name: 叶片模型名称
            lesion_model_name: 病害模型名称
            img_size: 输入图像尺寸
            model_size: 模型大小
            stage: 训练阶段
        """
        super().__init__()
        
        self.config = config
        self.img_size = img_size
        self.model_size = model_size
        self.stage = stage
        
        # 第一阶段：叶片分割模型
        if self.stage == 1 or self.stage == 0:
            print(f"Creating leaf model: {leaf_model_name}")
            self.leaf_model, self.leaf_new_param = create_model(
                leaf_model_name,
                img_size=img_size,
                model_size=model_size,
            )
        else:
            self.leaf_model = None
            self.leaf_new_param = None
        
        # 第二阶段：病害分割模型
        if self.stage == 2 or self.stage == 0:
            print(f"Creating lesion model: {lesion_model_name}")
            self.lesion_model, self.lesion_new_param = create_model(
                lesion_model_name,
                img_size=img_size,
                model_size=model_size,
            )
        else:
            self.lesion_model = None
            self.lesion_new_param = None
        
        # 训练阶段设置
        self._set_training_mode()
    
    def _set_training_mode(self):
        """设置训练模式"""
        if self.stage == 1:  # 仅训练叶片模型
            if self.leaf_model is not None:
                self.leaf_model.train()
        elif self.stage == 2:  # 仅训练病害模型
            if self.lesion_model is not None:
                self.lesion_model.train()
            # 如果第二阶段需要冻结叶片模型
            if self.leaf_model is not None and self.config.training_config.get('freeze_leaf_model', True):
                self.leaf_model.eval()
                for param in self.leaf_model.parameters():
                    param.requires_grad = False
        elif self.stage == 0:  # 两阶段联合
            if self.leaf_model is not None:
                self.leaf_model.train()
            if self.lesion_model is not None:
                self.lesion_model.train()
    
    def freeze_leaf_model(self):
        """冻结叶片模型"""
        if self.leaf_model is not None:
            self.leaf_model.eval()
            for param in self.leaf_model.parameters():
                param.requires_grad = False
    
    def unfreeze_leaf_model(self):
        """解冻叶片模型"""
        if self.leaf_model is not None:
            self.leaf_model.train()
            for param in self.leaf_model.parameters():
                param.requires_grad = True
    
    def forward_stage1(self, x: torch.Tensor, leaf_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        第一阶段前向传播：叶片分割
        
        Args:
            x: 输入图像 [B, 3, H, W]
            leaf_mask: 叶片真实mask [B, 1, H, W] (训练时)
            
        Returns:
            leaf_pred: 叶片预测mask [B, 1, H, W]
            leaf_loss: 叶片分割损失 (训练时)
        """
        if self.leaf_model is None:
            raise ValueError("Leaf model not initialized for stage 1")
        
        # 确保mask是float类型
        if leaf_mask is not None and leaf_mask.dtype != torch.float32:
            leaf_mask = leaf_mask.float()
        
        if self.training and leaf_mask is not None:
            # 训练模式
            leaf_pred, leaf_mask_processed, leaf_loss = self.leaf_model(x, leaf_mask)
            return leaf_pred, leaf_loss
        else:
            # 推理模式
            leaf_pred = self.leaf_model(x)
            return leaf_pred, None
    
    def forward_stage2(self, x: torch.Tensor, text: List[str], 
                      lesion_mask: Optional[torch.Tensor] = None,
                      leaf_pred: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        第二阶段前向传播：病害分割
        
        Args:
            x: 输入图像 [B, 3, H, W] 或叶片区域图像
            text: 文本描述
            lesion_mask: 病害真实mask [B, 1, H, W] (训练时)
            leaf_pred: 叶片预测mask [B, 1, H, W] (用于生成叶片区域图像)
            
        Returns:
            lesion_pred: 病害预测mask [B, 1, H, W]
            lesion_loss: 病害分割损失 (训练时)
        """
        if self.lesion_model is None:
            raise ValueError("Lesion model not initialized for stage 2")
        
        # 如果提供了叶片mask，提取叶片区域
        if leaf_pred is not None:
            # 确保叶片预测经过sigmoid激活（从logits转为概率）
            if leaf_pred.dtype == torch.float32 and (leaf_pred.min() < 0 or leaf_pred.max() > 1):
                leaf_pred_prob = torch.sigmoid(leaf_pred)
            else:
                leaf_pred_prob = leaf_pred
            leaf_rgb = get_leaf_rgb_from_mask(x, leaf_pred_prob)
            x = leaf_rgb
        # 如果没有提供叶片mask，说明输入的x已经是叶片RGB图像（第二阶段训练时）
        
        # 确保mask是float类型
        if lesion_mask is not None and lesion_mask.dtype != torch.float32:
            lesion_mask = lesion_mask.float()
        
        # 确保text是字符串列表
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list):
            try:
                text = text.tolist()
            except Exception as e:
                raise ValueError(f"Cannot convert text to list: {text}, error: {e}") from e
        
        # 确保text长度与batch大小一致
        batch_size = x.shape[0]
        if len(text) == 1 and batch_size > 1:
            text = text * batch_size
        elif len(text) != batch_size:
            if len(text) < batch_size:
                text = text + [text[0]] * (batch_size - len(text))
            else:
                text = text[:batch_size]
        
        if self.training and lesion_mask is not None:
            # 训练模式
            lesion_pred, lesion_mask_processed, lesion_loss = self.lesion_model(x, text, lesion_mask)
            return lesion_pred, lesion_loss
        else:
            # 推理模式
            lesion_pred = self.lesion_model(x, text)
            return lesion_pred, None
    
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        完整前向传播
        
        Args:
            batch: 批次数据
            
        Returns:
            results: 包含所有预测结果和损失的字典
        """
        x = batch['query_img']
        results = {}
        
        # 第一阶段：叶片分割
        if self.stage == 1 or self.stage == 0:
            leaf_mask = batch.get('leaf_mask', None)
            leaf_pred, leaf_loss = self.forward_stage1(x, leaf_mask)
            
            results['leaf_pred'] = leaf_pred
            if leaf_loss is not None:
                results['leaf_loss'] = leaf_loss
        else:
            leaf_pred = None
        
        # 第二阶段：病害分割
        if self.stage == 2 or self.stage == 0:
            text = batch.get('sentence', None)
            lesion_mask = batch.get('lesion_mask', None)
            
            # 确保text是有效的字符串列表
            if text is None:
                text = [""] * x.shape[0]
            elif isinstance(text, str):
                text = [text]
            elif not isinstance(text, list):
                try:
                    text = text.tolist()
                except:
                    text = [str(text)] * x.shape[0]
            
            # 确保text长度与batch大小一致
            if len(text) == 1 and x.shape[0] > 1:
                text = text * x.shape[0]
            elif len(text) != x.shape[0]:
                if len(text) < x.shape[0]:
                    text = text + [text[0]] * (x.shape[0] - len(text))
                else:
                    text = text[:x.shape[0]]
            
            # 第二阶段训练时，数据集已经提供了叶片RGB图像，直接使用
            if self.stage == 2:
                # 第二阶段单独训练时，x已经是叶片RGB图像，不需要额外处理
                lesion_pred, lesion_loss = self.forward_stage2(x, text, lesion_mask, None)
            else:
                # 联合训练或推理时，需要使用叶片预测来提取叶片区域
                lesion_pred, lesion_loss = self.forward_stage2(x, text, lesion_mask, leaf_pred)
            
            results['lesion_pred'] = lesion_pred
            if lesion_loss is not None:
                results['lesion_loss'] = lesion_loss
        else:
            lesion_pred = None
        
        # 计算总损失
        total_loss = 0.0
        loss_count = 0
        
        if 'leaf_loss' in results:
            total_loss += results['leaf_loss']
            loss_count += 1
        
        if 'lesion_loss' in results:
            total_loss += results['lesion_loss']
            loss_count += 1
        
        if loss_count > 0:
            results['total_loss'] = total_loss / loss_count  # 平均损失
        
        return results
    
    def load_leaf_checkpoint(self, checkpoint_path: str):
        """加载叶片模型检查点"""
        if self.leaf_model is None:
            raise ValueError("Leaf model not initialized")
        
        print(f"Loading leaf model checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 尝试多种可能的检查点格式
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model_state = checkpoint['model']
            elif 'state_dict' in checkpoint:
                model_state = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
            else:
                # 假设整个字典就是模型状态
                model_state = checkpoint
        else:
            model_state = checkpoint
        
        # 处理键名前缀问题
        # 检查是否存在leaf_model前缀
        has_leaf_prefix = any(k.startswith('leaf_model.') for k in model_state.keys())
        
        if has_leaf_prefix:
            # 移除leaf_model前缀
            new_state_dict = {}
            for k, v in model_state.items():
                if k.startswith('leaf_model.'):
                    new_state_dict[k[11:]] = v  # 去掉'leaf_model.'前缀
                else:
                    new_state_dict[k] = v
            model_state = new_state_dict
        
        # 尝试直接加载到leaf_model
        try:
            ret = self.leaf_model.load_state_dict(model_state, strict=False)
            print(f"Leaf model load result: missing={len(ret.missing_keys)}, unexpected={len(ret.unexpected_keys)}")
            
            # 如果有太多缺失或意外的键，给出警告
            if len(ret.missing_keys) > 10 or len(ret.unexpected_keys) > 10:
                print("⚠️ 警告: 叶片模型加载了部分权重，但有大量键不匹配。模型性能可能受到影响。")
                print(f"缺失的前5个键: {ret.missing_keys[:5]}")
                print(f"意外的前5个键: {ret.unexpected_keys[:5]}")
        except Exception as e:
            raise RuntimeError(f"Failed to load leaf model checkpoint from {checkpoint_path}: {e}") from e
    
    def load_lesion_checkpoint(self, checkpoint_path: str):
        """加载病害模型检查点"""
        if self.lesion_model is None:
            raise ValueError("Lesion model not initialized")
        
        print(f"Loading lesion model checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 尝试多种可能的检查点格式
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model_state = checkpoint['model']
            elif 'state_dict' in checkpoint:
                model_state = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
            else:
                # 假设整个字典就是模型状态
                model_state = checkpoint
        else:
            model_state = checkpoint
        
        # 处理键名前缀问题
        # 检查是否存在lesion_model前缀
        has_lesion_prefix = any(k.startswith('lesion_model.') for k in model_state.keys())
        
        if has_lesion_prefix:
            # 移除lesion_model前缀
            new_state_dict = {}
            for k, v in model_state.items():
                if k.startswith('lesion_model.'):
                    new_state_dict[k[13:]] = v  # 去掉'lesion_model.'前缀
                else:
                    new_state_dict[k] = v
            model_state = new_state_dict
        
        # 尝试直接加载到lesion_model
        try:
            ret = self.lesion_model.load_state_dict(model_state, strict=False)
            print(f"Lesion model load result: missing={len(ret.missing_keys)}, unexpected={len(ret.unexpected_keys)}")
            
            # 如果有太多缺失或意外的键，给出警告
            if len(ret.missing_keys) > 10 or len(ret.unexpected_keys) > 10:
                print("⚠️ 警告: 病害模型加载了部分权重，但有大量键不匹配。模型性能可能受到影响。")
                print(f"缺失的前5个键: {ret.missing_keys[:5]}")
                print(f"意外的前5个键: {ret.unexpected_keys[:5]}")
        except Exception as e:
            raise RuntimeError(f"Failed to load lesion model checkpoint from {checkpoint_path}: {e}") from e
    
    def get_model_for_stage(self, stage: int) -> nn.Module:
        """获取指定阶段的模型"""
        if stage == 1:
            return self.leaf_model
        elif stage == 2:
            return self.lesion_model
        else:
            raise ValueError(f"Invalid stage: {stage}")
    
    def get_new_params_for_stage(self, stage: int):
        """获取指定阶段的新参数"""
        if stage == 1:
            return self.leaf_new_param
        elif stage == 2:
            return self.lesion_new_param
        else:
            raise ValueError(f"Invalid stage: {stage}")
    
    def set_stage(self, stage: int):
        """设置训练阶段"""
        self.stage = stage
        self._set_training_mode()
    
    def eval_mode(self):
        """设置为评估模式"""
        if self.leaf_model is not None:
            self.leaf_model.eval()
        if self.lesion_model is not None:
            self.lesion_model.eval()
    
    def train_mode(self):
        """设置为训练模式"""
        self._set_training_mode()


def create_combined_model(config, stage: int = 0, device: str = 'cuda') -> CombinedSegmentationModel:
    """
    创建组合分割模型
    
    Args:
        config: 配置对象
        stage: 训练阶段
        device: 设备
        
    Returns:
        model: 组合模型
    """
    # 检查设备可用性
    if device == 'cuda' and not torch.cuda.is_available():
        print(f"Warning: CUDA is not available, falling back to CPU")
        device = 'cpu'
    elif device == 'cuda':
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Using device: {device}")
    
    model = CombinedSegmentationModel(
        config=config,
        img_size=config.input_size,
        model_size="base",
        stage=stage
    )
    
    model.to(device)
    print(f"Model successfully moved to {device}")
    
    return model 
