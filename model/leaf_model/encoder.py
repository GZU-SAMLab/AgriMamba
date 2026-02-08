import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

import einops

from vmamba_model.vmamba import SS2D, VSSM, LayerNorm2d, Linear2d






class Encoder(VSSM):
    """still extract feature"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classifier = None
        dims = kwargs['dims']
        use_checkpoint = kwargs['use_checkpoint']
        norm_layer = kwargs['norm_layer']
        self.pos_embed = None
        # 移除不再使用的text_guidencee


        # 处理dims参数：如果是整数，转换为列表
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]


        img_size = kwargs.get('img_size', None)
        patch_size = kwargs.get('patch_size', 4)
        if img_size is not None:
            grid_size = img_size // patch_size
            if self.channel_first:
                self.pos_embed = nn.Parameter(torch.zeros(1, dims[0], grid_size, grid_size))
            else:
                self.pos_embed = nn.Parameter(torch.zeros(1, grid_size, grid_size, dims[0]))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        ) 
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)



    def _add_pos_embed(self, x):
        if self.pos_embed is None:
            return x
        if self.channel_first:
            pos = self.pos_embed
            if pos.shape[-2:] != x.shape[-2:]:
                pos = F.interpolate(pos, size=x.shape[-2:], mode="bicubic", align_corners=False)
            return x + pos

        pos = self.pos_embed
        if pos.shape[1:3] != x.shape[1:3]:
            pos = F.interpolate(
                pos.permute(0, 3, 1, 2),
                size=x.shape[1:3],
                mode="bicubic",
                align_corners=False,
            ).permute(0, 2, 3, 1)
        return x + pos


    def forward_layer(self, x, layer):
        inner = layer.blocks(x)
        out = layer.downsample(inner)
        return out, inner
    
    def forward(self, x):
        # 移除文本相关参数：l_feat, l_mask, pooler_out
        x = self.patch_embed(x)
        x = self._add_pos_embed(x)
        outs = []            # 存储各层输出特征
        
        for i, layer in enumerate(self.layers):
            x, inner = self.forward_layer(x, layer)     #每一个layer由一系列block(VSS2S)  inner:当前层特征, x:下采样后特征
            _, c, h, w = inner.shape                       # 获取特征图尺寸
            out = inner


            if layer.downsample is not None:
                x = layer.downsample(out)
            else:
                x = out

            outs.append(out)

        return outs

