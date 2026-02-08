from typing import List, Optional, Tuple
import math
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
from timm.models.layers import DropPath
import torch.utils.checkpoint as checkpoint
from .base_segmenter import BaseSegmenter

from vmamba_model.vmamba import SS2D, VSSM, LayerNorm2d, Linear2d

from .lesion_utils import Fusion as FuseLayer
from .lesion_utils import conv_layer, load_ckpt, update_mamba_config
from .my_backbone import MyModel


class VSSBlock(nn.Module):
    def __init__(
        self,
        forward_coremm='SS2D',
        **kwargs,
    ):
        super().__init__()
        norm_layer = kwargs['norm_layer']
        dim = kwargs['dim']
        drop_path = kwargs['drop_path']
        self.ln_1 = norm_layer(dim)
        self.forward_coremm = forward_coremm
        if not forward_coremm:
            raise
        elif forward_coremm == 'SS2D':
            self.self_attention = SS2D(
                d_model=dim,
                d_state=kwargs['ssm_d_state'],
                dt_rank=kwargs['ssm_dt_rank'],
                act_layer=kwargs['ssm_act_layer'],
                d_conv=kwargs['ssm_conv'],
                conv_bias=kwargs['ssm_conv_bias'],
                dropout=kwargs['ssm_drop_rate'],
                initialize=kwargs['ssm_init'],
                **kwargs,
            )
        else:
            raise
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        if isinstance(input, torch.Tensor):
            out = self.ln_1(input)
            out = self.self_attention(out)
            out = input + self.drop_path(out)
            x = out
        else:
            # input should be a list (img and global / local conditions)
            out = [self.ln_1(i) if i is not None else None for i in input]
            out = self.self_attention(out)
            out = [i + self.drop_path(o) if i is not None else None for i, o in zip(input, out)]
            x = out
        return x


class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        depth,
        downsample=None,
        **kwargs,
    ):
        super().__init__()
        drop_path = 0.
        dim = kwargs['dim']
        use_checkpoint = kwargs['use_checkpoint']
        norm_layer = kwargs['norm_layer']

        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                forward_coremm='SS2D',
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                **kwargs,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, channel_first=kwargs['channel_first'])
        else:
            self.downsample = None

    def forward(self, x, l_feat, l_mask):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        # x: b w h c
        inner = x
        if self.downsample is not None:
            x = self.downsample(x)

        return x, inner



def pixel_unshuffle(x: torch.Tensor, downscale_factor: int) -> torch.Tensor:
    """Pixel unshuffle operation to reduce spatial dimensions and increase channels.
    
    Args:
        x: Input tensor of shape (B, C, H, W)
        downscale_factor: Factor to reduce spatial dimensions
    Returns:
        Tensor of shape (B, C*downscale_factor^2, H/downscale_factor, W/downscale_factor)
    """
    b, c, h, w = x.shape
    out_h = h // downscale_factor
    out_w = w // downscale_factor
    x = x.view(b, c, out_h, downscale_factor, out_w, downscale_factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(b, c * (downscale_factor ** 2), out_h, out_w)


class FeatureMambaEnhancer(nn.Module):
    """单层特征Mamba增强器 - 稳定性优化版本"""
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        
        # 按原VSSLayer方式调用
        # 处理norm_layer字符串转换
        norm_layer = kwargs.get('norm_layer', 'ln2d')
        _NORMLAYERS = {'ln': nn.LayerNorm, 'ln2d': LayerNorm2d, 'bn': nn.BatchNorm2d}
        norm_layer = _NORMLAYERS.get(norm_layer.lower() if isinstance(norm_layer, str) else 'ln2d', LayerNorm2d)
        
        # 处理激活层
        _ACTLAYERS = {'silu': nn.SiLU, 'gelu': nn.GELU, 'relu': nn.ReLU, 'sigmoid': nn.Sigmoid}
        ssm_act_layer = _ACTLAYERS.get(kwargs.get('ssm_act_layer', 'silu').lower(), nn.SiLU)
        
        # 添加输入LayerNorm提升稳定性
        self.input_norm = LayerNorm2d(dim)
        
        self.feature_mamba = VSSLayer(
            dim=dim,
            depth=1,
            use_checkpoint=kwargs.get('use_checkpoint', False),
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            downsample=None,
            channel_first=True,
            ssm_d_state=kwargs.get('ssm_d_state', 16),
            ssm_ratio=kwargs.get('ssm_ratio', 2.0),
            ssm_dt_rank=kwargs.get('ssm_dt_rank', "auto"),
            ssm_conv=kwargs.get('ssm_conv', 3),
            ssm_conv_bias=kwargs.get('ssm_conv_bias', True),
            ssm_drop_rate=kwargs.get('ssm_drop_rate', 0.0),
            ssm_init=kwargs.get('ssm_init', "v0"),
        )
        
        # 残差连接权重 - 改为固定值，提升稳定性
        self.residual_weight = 0.1  # 固定小权重，避免梯度爆炸
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入归一化提升数值稳定性
        x_norm = self.input_norm(x)
        enhanced_x, _ = self.feature_mamba(x_norm, None, None)
        # 使用固定小权重进行残差连接
        return x + self.residual_weight * enhanced_x


class AdaptiveScaleWeighting(nn.Module):
    """自适应多尺度权重学习"""
    def __init__(self, num_scales: int = 4, dim: int = 128):
        super().__init__()
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 权重学习网络
        self.weight_mlp = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, num_scales),
            nn.Softmax(dim=-1)
        )
        
        # 可学习的基础权重
        self.base_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # 使用第一个特征来计算权重
        feat_for_weight = self.global_pool(features[0]).flatten(1)  # (B, C)
        adaptive_weights = self.weight_mlp(feat_for_weight)  # (B, num_scales)
        
        # 结合基础权重和自适应权重
        final_weights = 0.7 * self.base_weights.unsqueeze(0) + 0.3 * adaptive_weights
        
        # 应用权重
        weighted_features = []
        for i, feat in enumerate(features):
            weight = final_weights[:, i:i+1, None, None]  # (B, 1, 1, 1)
            weighted_features.append(feat * weight)
            
        return weighted_features


class UltraLightMMSCopE(nn.Module):
    """超轻量级多尺度上下文提取模块 - 大幅简化版本"""
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        
        # 极简的多尺度处理 - 移除复杂的pixel_unshuffle
        self.multi_scale_conv = nn.Sequential(
            # 第一层：局部特征提取
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1, groups=dim // 4),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            
            # 第二层：扩展感受野
            nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            
            # 第三层：恢复通道数
            nn.Conv2d(dim // 2, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
        )
        
        # 轻量级多尺度融合
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, f: torch.Tensor) -> torch.Tensor:
        # 原始特征保存
        identity = f
        
        # 多尺度卷积增强
        enhanced = self.multi_scale_conv(f)
        
        # 全局上下文调制
        global_weights = self.global_context(f)
        enhanced = enhanced * global_weights
        
        # 简单残差连接
        output = identity + enhanced
        
        return output



class LightweightMambaDecoder(nn.Module):
    """轻量级5层Mamba decoder - 显存优化设计"""
    
    def __init__(self, **kwargs) -> None:
        super().__init__()
        
        # 解析输入维度
        dims = self._parse_dimensions(kwargs['dims'])
        self.f4_dim, self.f3_dim, self.f2_dim, self.f1_dim = dims[3], dims[2], dims[1], dims[0]
        self.target_dim = self.f1_dim
        
        # 1. 特征投影层
        self.feature_projections = self._build_feature_projections()
        
        # 2. 每层特征的Mamba增强 (4层)
        self.feature_enhancers = nn.ModuleDict({
            'f4_enhancer': FeatureMambaEnhancer(self.target_dim, **kwargs),
            'f3_enhancer': FeatureMambaEnhancer(self.target_dim, **kwargs),
            'f2_enhancer': FeatureMambaEnhancer(self.target_dim, **kwargs),
            'f1_enhancer': FeatureMambaEnhancer(self.target_dim, **kwargs),
        })
        
        # 3. 自适应权重学习
        self.adaptive_weighting = AdaptiveScaleWeighting(4, self.target_dim)
        
        # 4. 4路拼接和维度减少
        self.concat_conv = nn.Sequential(
            nn.Conv2d(4 * self.target_dim, self.target_dim, kernel_size=1),
            nn.BatchNorm2d(self.target_dim),
            nn.ReLU(inplace=True)
        )
        
        # 5. 轻量级文本融合 (移除复杂的TextGuidedMamba)
        self.simple_text_fusion = SimpleLightweightTextFusion(self.target_dim)
        
        # 6. 简化版多尺度上下文提取
        self.enhanced_mmscope = UltraLightMMSCopE(self.target_dim, **kwargs)
        
        # 7. 移除FinalMambaRefinement - 节省45%显存消耗
        # 前面已有充分的特征处理：4个FeatureMambaEnhancer + TextFusion + MMSCopE
        
        # 8. 预测层
        self.prediction_mlp = nn.Sequential(
            nn.Conv2d(4 * self.target_dim, self.target_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.target_dim, self.target_dim // 2, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Conv2d(self.target_dim // 2, 1, kernel_size=1)
        self.last_information_aggregator = None
    
    def _parse_dimensions(self, dims) -> List[int]:
        """解析和验证输入维度"""
        if isinstance(dims, (list, tuple)):
            return list(dims)
        else:
            base_dim = dims
            return [base_dim, base_dim*2, base_dim*4, base_dim*8]
    
    def _build_feature_projections(self) -> nn.ModuleDict:
        """构建特征投影层"""
        projections = nn.ModuleDict()
        
        for i, (name, in_dim) in enumerate([
            ('f4_proj', self.f4_dim), ('f3_proj', self.f3_dim),
            ('f2_proj', self.f2_dim), ('f1_proj', self.f1_dim)
        ]):
            projections[name] = nn.Sequential(
                nn.Conv2d(in_dim, self.target_dim, kernel_size=1),
                nn.BatchNorm2d(self.target_dim),
                nn.ReLU(inplace=True)
            )
        
        return projections

    def forward(self, x: List[torch.Tensor], l_feat: torch.Tensor, 
                l_mask: torch.Tensor, pooler_out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """轻量级5层MambaDecoder前向传播"""
        f4, f3, f2, f1 = x[0], x[1], x[2], x[3]
        target_size = (f1.shape[2], f1.shape[3])
        
        # 1. 特征投影
        f4_proj = self.feature_projections['f4_proj'](f4)
        f3_proj = self.feature_projections['f3_proj'](f3)
        f2_proj = self.feature_projections['f2_proj'](f2)
        f1_proj = self.feature_projections['f1_proj'](f1)
        
        # 2. 每层特征Mamba增强 (4层Mamba: #1-4)
        f4_enhanced = self.feature_enhancers['f4_enhancer'](f4_proj)
        f3_enhanced = self.feature_enhancers['f3_enhancer'](f3_proj)
        f2_enhanced = self.feature_enhancers['f2_enhancer'](f2_proj)
        f1_enhanced = self.feature_enhancers['f1_enhancer'](f1_proj)
        
        # 3. 空间对齐到F1分辨率
        f_down4 = F.interpolate(f4_enhanced, size=target_size, mode='bilinear', align_corners=True)
        f_down3 = F.interpolate(f3_enhanced, size=target_size, mode='bilinear', align_corners=True)
        f_down2 = F.interpolate(f2_enhanced, size=target_size, mode='bilinear', align_corners=True)
        
        # 4. 自适应权重学习
        features = [f1_enhanced, f_down2, f_down3, f_down4]
        weighted_features = self.adaptive_weighting(features)
        
        # 5. 4路特征拼接和降维
        concat_features = torch.cat(weighted_features, dim=1)
        fused_features = self.concat_conv(concat_features)
        
        # 6. 轻量级文本融合 (替代复杂的TextGuidedMamba)
        if l_feat is not None:
            text_enhanced_features = self.simple_text_fusion(fused_features, l_feat, l_mask)
        else:
            text_enhanced_features = fused_features
        
        # 7. 简化版多尺度上下文提取
        context_features = self.enhanced_mmscope(text_enhanced_features)
        self.last_information_aggregator = context_features.detach()
        
        # 8. 多尺度增强 (residual connections)
        enhanced_features = self._apply_multiscale_enhancement(weighted_features, context_features)
        
        # 9. 最终特征融合
        final_features = torch.cat(enhanced_features, dim=1)
        
        # 10. 直接进入预测 (移除FinalMambaRefinement，节省45%显存)
        pred_features = self.prediction_mlp(final_features)
        output = self.final_conv(pred_features)
        
        return output
    
    def _apply_multiscale_enhancement(self, features: List[torch.Tensor], 
                                    context: torch.Tensor) -> List[torch.Tensor]:
        """应用多尺度增强"""
        return [feat + context for feat in features]


class SimpleLightweightTextFusion(nn.Module):
    """超轻量级文本融合模块 - 稳定性优化版本"""
    def __init__(self, visual_dim: int, text_dim: int = 768):
        super().__init__()
        
        # 添加输入LayerNorm
        self.text_input_norm = nn.LayerNorm(text_dim)
        
        # 极简文本投影
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, visual_dim),
            nn.ReLU()
        )
        
        # 简单的注意力权重
        self.attention_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(visual_dim, visual_dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(visual_dim // 4, visual_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, visual_feat: torch.Tensor, text_feat: torch.Tensor, 
                text_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_feat: (B, C, H, W) 视觉特征
            text_feat: (B, D, L) 文本特征
            text_mask: (B, L, 1) 文本掩码
        """
        B, C, H, W = visual_feat.shape
        
        # 调整文本特征形状并聚合
        text_feat = text_feat.permute(0, 2, 1)  # (B, L, D)
        text_mask = text_mask.squeeze(-1)  # (B, L)
        
        # 文本特征归一化提升稳定性
        text_feat = self.text_input_norm(text_feat)
        
        # 简单的mask-aware平均 - 改进数值稳定性
        if text_mask is not None:
            text_mask_float = text_mask.unsqueeze(-1).float()  # (B, L, 1)
            masked_text = text_feat * text_mask_float
            # 提高clamp最小值，避免除零风险
            mask_sum = torch.clamp(text_mask_float.sum(dim=1), min=1.0)
            text_global = masked_text.sum(dim=1) / mask_sum  # (B, D)
        else:
            text_global = text_feat.mean(dim=1)  # (B, D)
        
        # 文本特征投影
        text_projected = self.text_proj(text_global)  # (B, C)
        
        # 生成注意力权重
        attention_weights = self.attention_weight(visual_feat)  # (B, C, 1, 1)
        
        # 更保守的文本调制 (只有2%的文本影响，进一步降低风险)
        text_modulation = text_projected.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        modulated_visual = visual_feat * (0.98 + 0.02 * attention_weights * text_modulation)
        
        return modulated_visual


# 修改主要的Segmentor类
class MambaSegmentor(BaseSegmenter):
    """轻量级5层Mamba segmentor - 显存优化设计 + 梯度稳定性增强
    
    重要提示：为避免梯度爆炸，强烈建议在训练时设置梯度裁剪：
    - 推荐配置：clip_grad = 1.0 (在main.py的args中设置)
    - 或在config.json中设置："clip_grad": 1.0
    - 263M参数的大模型必须使用梯度裁剪！
    """
    
    def __init__(self, backbone: nn.Module, **kwargs):
        super().__init__(backbone)
        self.decoder = LightweightMambaDecoder(**kwargs)


@register_model
def TMLS(img_size: int = 256, model_size: str = "tiny", **kwargs):
    """创建轻量级5层Mamba segmentor模型"""
    config_dict = update_mamba_config(model_size)
    backbone = MyModel(img_size=img_size, **config_dict)
    backbone, ret = load_ckpt(backbone, model_size)
    return MambaSegmentor(backbone, **config_dict), ret[0] 

