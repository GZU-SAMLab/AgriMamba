import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# import mamba_ssm.selective_scan_fn (in which causal_conv1d is needed)
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass


# =====================================================
class SSM(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=4,
            ssm_ratio=2,
            dt_rank="auto",
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # x proj; dt proj ============================
        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)

        self.dt_proj = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                    **factory_kwargs)

        # A, D =======================================
        self.A_log = self.A_log_init(self.d_state, self.d_inner)  # (D, N)
        self.D = self.D_init(self.d_inner)  # (D)

        # out norm ===================================
        self.out_norm = nn.LayerNorm(self.d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward(self, x: torch.Tensor):
        selective_scan = selective_scan_fn
        B, L, d = x.shape
        x = x.permute(0, 2, 1)
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=L)
        A = -torch.exp(self.A_log.float())  # (k * d, d_state)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()

        y = selective_scan(
            x, dt,
            A, B, C, self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        # assert out_y.dtype == torch.float
        y = rearrange(y, "b d l -> b l d")
        y = self.out_norm(y)
        return y


class OT_TextToImage_Alignment(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, img_feat, text_feat):
        """
        img_feat:  [B, N_img, C] (作为 Anchor，保持不动)
        text_feat: [B, N_txt, C] (作为 Source，被搬运)

        返回:
        text_aligned_to_img: [B, N_img, C]
        (形状和图像完全一样，但内容是重组后的文本特征)
        """
        img_norm = F.normalize(img_feat, p=2, dim=-1)
        text_norm = F.normalize(text_feat, p=2, dim=-1)

        similarity = torch.matmul(img_norm, text_norm.transpose(1, 2))
        cost_matrix = 1 - similarity

        _, min_indices = torch.min(cost_matrix, dim=-1)

        B, N_img, N_txt = cost_matrix.shape
        weight = 1.0

        M = torch.zeros(B, N_img, N_txt, device=img_feat.device)
        src_weights = torch.full_like(min_indices, weight, dtype=torch.float)
        M.scatter_(2, min_indices.unsqueeze(2), src_weights.unsqueeze(2))

        text_aligned = torch.matmul(M, text_feat)

        return text_aligned

class TextGuidedMamba(nn.Module):
    def __init__(
            self,
            d_model=96,
            d_state=4,
            ssm_ratio=2,
            dt_rank="auto",
            d_conv=3,
            conv_bias=True,
            dropout=0.,
            bias=False,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            softmax_version=False,
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.bi = False

        # in proj
        self.in_proj_img = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_text = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        # conv_sep
        self.conv2d_img = nn.Sequential(nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        ), nn.BatchNorm2d(self.d_inner), nn.SiLU())

        self.conv2d_text = nn.Sequential(nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        ), nn.BatchNorm2d(self.d_inner), nn.SiLU())

        self.SSM_2_f = SSM(d_model=self.d_model, d_state=self.d_state, ssm_ratio=self.expand, dt_rank=self.dt_rank)

        # out proj
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        
        # 融合投影层
        self.fusion_proj = nn.Linear(2 * self.d_model, self.d_model, bias=bias, **factory_kwargs)
        nn.init.xavier_uniform_(self.fusion_proj.weight)

    def ssm_stage_2(self, img, text):
        # 连接所有模态的输入
        x_for = torch.cat([img, text], dim=1)
        
        # 处理连接后的输入
        y_for = self.SSM_2_f(x_for)
        
        return y_for

    def conv_sep(self, img, text):
        B, N, D = img.shape
        H = int(math.sqrt(N))
        W = H

        xz_img = self.in_proj_img(img)
        xz_text = self.in_proj_text(text)

        x_img, z_img = xz_img.chunk(2, dim=-1)
        x_text, z_text = xz_text.chunk(2, dim=-1)

        # 重塑为2D格式
        x_img = x_img.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_text = x_text.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # 应用卷积
        x_img = self.conv2d_img(x_img)
        x_text = self.conv2d_text(x_text)

        # 重塑回原始格式
        x_img = x_img.permute(0, 2, 3, 1).reshape(B, N, -1)
        x_text = x_text.permute(0, 2, 3, 1).reshape(B, N, -1)

        # 处理SSM
        z = torch.cat([z_img, z_text], dim=1)
        y = self.ssm_stage_2(x_img, x_text)
        y = y * F.silu(z)

        # 输出投影
        out = self.dropout(self.out_proj(y))
        
        return out

    def forward(self, img, text, **kwargs):
        # 保存原始形状
        B, C, H, W = img.shape
        N = H * W
        
        # 将输入转换为(B, N, D)格式
        img_flat = img.permute(0, 2, 3, 1).reshape(B, -1, C)
        text_flat = text.permute(0, 2, 3, 1).reshape(B, -1, C)
        
        # 处理
        out = self.conv_sep(img_flat, text_flat)
        
        # 空间重排，将2*N维度拆分为2个N
        out = out.reshape(B, 2, N, C)
        
        # 将空间维度N重塑为H,W
        out = out.reshape(B, 2, H, W, C)
        
        # 在空间维度上拼接两个模态的特征 (B, 2, H, W, C) -> (B, H, W, 2*C)
        out = out.permute(0, 2, 3, 1, 4).reshape(B, H, W, 2*C)
        
        # 使用投影层将2*C维度投影回C维度
        out = self.fusion_proj(out)  # (B, H, W, C)
        
        # 转换为(B, C, H, W)输出格式
        out = out.permute(0, 3, 1, 2).contiguous()
        
        return out


class LocalContextMamba(nn.Module):
    def __init__(
            self,
            d_model=96,
            d_state=16,
            ssm_ratio=2,
            dt_rank="auto",
            d_conv=3,
            conv_bias=True,
            dropout=0.,
            bias=False,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            softmax_version=False,
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.bi = False

        # in proj
        self.in_proj_img = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_text = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        # conv_sep
        self.conv2d_img = nn.Sequential(nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        ), nn.BatchNorm2d(self.d_inner), nn.SiLU())

        self.conv2d_text = nn.Sequential(nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        ), nn.BatchNorm2d(self.d_inner), nn.SiLU())

        self.SSM_img_f = SSM(d_model=self.d_model, d_state=self.d_state, ssm_ratio=self.expand, dt_rank=self.dt_rank)
        self.SSM_text_f = SSM(d_model=self.d_model, d_state=self.d_state, ssm_ratio=self.expand, dt_rank=self.dt_rank)

        # out proj
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        
        
        self.fusion_proj = nn.Linear(2 * self.d_model, self.d_model, bias=bias, **factory_kwargs)
        nn.init.xavier_uniform_(self.fusion_proj.weight)

    def ssm_stage_1(self, img, text):
        # 处理每个模态
        y_img = self.SSM_img_f(img)
        y_text = self.SSM_text_f(text)
        
        # 连接所有模态的输出
        return torch.cat([y_img, y_text], dim=1)

    def conv_sep(self, img, text):
        B, N, D = img.shape
        H = int(math.sqrt(N))
        W = H

        xz_img = self.in_proj_img(img)
        xz_text = self.in_proj_text(text)

        x_img, z_img = xz_img.chunk(2, dim=-1)
        x_text, z_text = xz_text.chunk(2, dim=-1)

        # 重塑为2D格式
        x_img = x_img.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_text = x_text.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # 应用卷积
        x_img = self.conv2d_img(x_img)
        x_text = self.conv2d_text(x_text)

        # 重塑回原始格式
        x_img = x_img.permute(0, 2, 3, 1).reshape(B, N, -1)
        x_text = x_text.permute(0, 2, 3, 1).reshape(B, N, -1)

        # 处理SSM
        z = torch.cat([z_img, z_text], dim=1)
        y = self.ssm_stage_1(x_img, x_text)
        y = y * F.silu(z)

        # 输出投影
        out = self.dropout(self.out_proj(y))
        
        return out

    def forward(self, img, text, **kwargs):
        # 保存原始形状
        B, C, H, W = img.shape
        N = H * W
        
        # 将输入转换为(B, N, D)格式
        img_flat = img.permute(0, 2, 3, 1).reshape(B, -1, C)
        text_flat = text.permute(0, 2, 3, 1).reshape(B, -1, C)
        
        # 处理
        out = self.conv_sep(img_flat, text_flat)
        
        # 空间重排，将2*N维度拆分为2个N
        out = out.reshape(B, 2, N, C)
        
        # 将空间维度N重塑为H,W
        out = out.reshape(B, 2, H, W, C)
        
        # 在空间维度上拼接两个模态的特征 (B, 2, H, W, C) -> (B, H, W, 2*C)
        out = out.permute(0, 2, 3, 1, 4).reshape(B, H, W, 2*C)
        
        # 使用投影层将2*C维度投影回C维度
        out = self.fusion_proj(out)  # (B, H, W, C)
        
        # 转换为(B, C, H, W)输出格式
        out = out.permute(0, 3, 1, 2).contiguous()
        
        # 注意: 不再返回img和text的残差连接，而是直接返回融合特征
        return out


class BiBranchMambaFusion(nn.Module):
    def __init__(self, d_model, dt_rank="auto", d_state=16, **kwargs):
        super(BiBranchMambaFusion, self).__init__()
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        #分别SSM
        self.SSM_parallel1 = LocalContextMamba(d_model=d_model, d_state=d_state, dt_rank=dt_rank)

        #融合后  再SSM
        self.SSM_parallel2 = TextGuidedMamba(d_model=d_model, d_state=d_state, dt_rank=dt_rank)
        self.text_align = OT_TextToImage_Alignment(d_model=d_model)
  
    def forward(self, img, text, **kwargs):
        B, C, H, W = img.shape
        img_feat = img.permute(0, 2, 3, 1).reshape(B, -1, C)
        text_feat = text.permute(0, 2, 3, 1).reshape(B, -1, C)
        text_feat = self.text_align(img_feat, text_feat)
        text = text_feat.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # 并行处理
        out1 = self.SSM_parallel1(img, text)
        out2 = self.SSM_parallel2(img, text)

        # 将两个分支的输出相加
        out = out1 + out2     

        return out
