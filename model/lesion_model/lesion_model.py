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

            out = [self.ln_1(i) if i is not None else None for i in input]
            out = self.self_attention(out)
            out = [i + self.drop_path(o) if i is not None else None for i, o in zip(input, out)]
            x = out
        return x


class VSSLayer(nn.Module):


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

        if True:
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()
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

        inner = x
        if self.downsample is not None:
            x = self.downsample(x)

        return x, inner


def pixel_unshuffle(x: torch.Tensor, downscale_factor: int) -> torch.Tensor:

    b, c, h, w = x.shape
    out_h = h // downscale_factor
    out_w = w // downscale_factor
    x = x.view(b, c, out_h, downscale_factor, out_w, downscale_factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(b, c * (downscale_factor ** 2), out_h, out_w)


class FeatureMambaEnhancer(nn.Module):

    def __init__(self, dim: int, **kwargs):
        super().__init__()


        norm_layer = kwargs.get('norm_layer', 'ln2d')
        _NORMLAYERS = {'ln': nn.LayerNorm, 'ln2d': LayerNorm2d, 'bn': nn.BatchNorm2d}
        norm_layer = _NORMLAYERS.get(norm_layer.lower() if isinstance(norm_layer, str) else 'ln2d', LayerNorm2d)


        _ACTLAYERS = {'silu': nn.SiLU, 'gelu': nn.GELU, 'relu': nn.ReLU, 'sigmoid': nn.Sigmoid}
        ssm_act_layer = _ACTLAYERS.get(kwargs.get('ssm_act_layer', 'silu').lower(), nn.SiLU)


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


        self.residual_weight = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_norm = self.input_norm(x)
        enhanced_x, _ = self.feature_mamba(x_norm, None, None)

        return x + self.residual_weight * enhanced_x


class AdaptiveScaleWeighting(nn.Module):

    def __init__(self, num_scales: int = 4, dim: int = 128):
        super().__init__()


        self.global_pool = nn.AdaptiveAvgPool2d(1)


        self.weight_mlp = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, num_scales),
            nn.Softmax(dim=-1)
        )


        self.base_weights = nn.Parameter(torch.ones(num_scales) / num_scales)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:

        feat_for_weight = self.global_pool(features[0]).flatten(1)
        adaptive_weights = self.weight_mlp(feat_for_weight)


        final_weights = 0.7 * self.base_weights.unsqueeze(0) + 0.3 * adaptive_weights


        weighted_features = []
        for i, feat in enumerate(features):
            weight = final_weights[:, i:i+1, None, None]
            weighted_features.append(feat * weight)

        return weighted_features


class UltraLightMMSCopE(nn.Module):

    def __init__(self, dim: int, **kwargs):
        super().__init__()


        self.multi_scale_conv = nn.Sequential(

            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1, groups=dim // 4),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),


            nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),


            nn.Conv2d(dim // 2, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
        )


        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:

        identity = f


        enhanced = self.multi_scale_conv(f)


        global_weights = self.global_context(f)
        enhanced = enhanced * global_weights


        output = identity + enhanced

        return output


class LightweightMambaDecoder(nn.Module):


    def __init__(self, **kwargs) -> None:
        super().__init__()


        dims = self._parse_dimensions(kwargs['dims'])
        self.f4_dim, self.f3_dim, self.f2_dim, self.f1_dim = dims[3], dims[2], dims[1], dims[0]
        self.target_dim = self.f1_dim


        self.feature_projections = self._build_feature_projections()


        self.feature_enhancers = nn.ModuleDict({
            'f4_enhancer': FeatureMambaEnhancer(self.target_dim, **kwargs),
            'f3_enhancer': FeatureMambaEnhancer(self.target_dim, **kwargs),
            'f2_enhancer': FeatureMambaEnhancer(self.target_dim, **kwargs),
            'f1_enhancer': FeatureMambaEnhancer(self.target_dim, **kwargs),
        })


        self.adaptive_weighting = AdaptiveScaleWeighting(4, self.target_dim)


        self.concat_conv = nn.Sequential(
            nn.Conv2d(4 * self.target_dim, self.target_dim, kernel_size=1),
            nn.BatchNorm2d(self.target_dim),
            nn.ReLU(inplace=True)
        )


        self.simple_text_fusion = SimpleLightweightTextFusion(self.target_dim)


        self.enhanced_mmscope = UltraLightMMSCopE(self.target_dim, **kwargs)


        self.prediction_mlp = nn.Sequential(
            nn.Conv2d(4 * self.target_dim, self.target_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.target_dim, self.target_dim // 2, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(self.target_dim // 2, 1, kernel_size=1)
        self.last_information_aggregator = None

    def _parse_dimensions(self, dims) -> List[int]:

        if isinstance(dims, (list, tuple)):
            return list(dims)
        else:
            base_dim = dims
            return [base_dim, base_dim*2, base_dim*4, base_dim*8]

    def _build_feature_projections(self) -> nn.ModuleDict:

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

        f4, f3, f2, f1 = x[0], x[1], x[2], x[3]
        target_size = (f1.shape[2], f1.shape[3])


        f4_proj = self.feature_projections['f4_proj'](f4)
        f3_proj = self.feature_projections['f3_proj'](f3)
        f2_proj = self.feature_projections['f2_proj'](f2)
        f1_proj = self.feature_projections['f1_proj'](f1)


        f4_enhanced = self.feature_enhancers['f4_enhancer'](f4_proj)
        f3_enhanced = self.feature_enhancers['f3_enhancer'](f3_proj)
        f2_enhanced = self.feature_enhancers['f2_enhancer'](f2_proj)
        f1_enhanced = self.feature_enhancers['f1_enhancer'](f1_proj)


        f_down4 = F.interpolate(f4_enhanced, size=target_size, mode='bilinear', align_corners=True)
        f_down3 = F.interpolate(f3_enhanced, size=target_size, mode='bilinear', align_corners=True)
        f_down2 = F.interpolate(f2_enhanced, size=target_size, mode='bilinear', align_corners=True)


        features = [f1_enhanced, f_down2, f_down3, f_down4]
        weighted_features = self.adaptive_weighting(features)


        concat_features = torch.cat(weighted_features, dim=1)
        fused_features = self.concat_conv(concat_features)


        if l_feat is not None:
            text_enhanced_features = self.simple_text_fusion(fused_features, l_feat, l_mask)
        else:
            text_enhanced_features = fused_features


        context_features = self.enhanced_mmscope(text_enhanced_features)
        self.last_information_aggregator = context_features.detach()


        enhanced_features = self._apply_multiscale_enhancement(weighted_features, context_features)


        final_features = torch.cat(enhanced_features, dim=1)


        pred_features = self.prediction_mlp(final_features)
        output = self.final_conv(pred_features)

        return output

    def _apply_multiscale_enhancement(self, features: List[torch.Tensor],
                                    context: torch.Tensor) -> List[torch.Tensor]:

        return [feat + context for feat in features]


class SimpleLightweightTextFusion(nn.Module):

    def __init__(self, visual_dim: int, text_dim: int = 768):
        super().__init__()


        self.text_input_norm = nn.LayerNorm(text_dim)


        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, visual_dim),
            nn.ReLU()
        )


        self.attention_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(visual_dim, visual_dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(visual_dim // 4, visual_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, visual_feat: torch.Tensor, text_feat: torch.Tensor,
                text_mask: torch.Tensor) -> torch.Tensor:

        B, C, H, W = visual_feat.shape


        text_feat = text_feat.permute(0, 2, 1)
        text_mask = text_mask.squeeze(-1)


        text_feat = self.text_input_norm(text_feat)


        if text_mask is not None:
            text_mask_float = text_mask.unsqueeze(-1).float()
            masked_text = text_feat * text_mask_float

            mask_sum = torch.clamp(text_mask_float.sum(dim=1), min=1.0)
            text_global = masked_text.sum(dim=1) / mask_sum
        else:
            text_global = text_feat.mean(dim=1)


        text_projected = self.text_proj(text_global)


        attention_weights = self.attention_weight(visual_feat)


        text_modulation = text_projected.unsqueeze(-1).unsqueeze(-1)
        modulated_visual = visual_feat * (0.98 + 0.02 * attention_weights * text_modulation)

        return modulated_visual


class MambaSegmentor(BaseSegmenter):


    def __init__(self, backbone: nn.Module, **kwargs):
        super().__init__(backbone)
        self.decoder = LightweightMambaDecoder(**kwargs)


@register_model
def TMLS(img_size: int = 256, model_size: str = "tiny", **kwargs):

    config_dict = update_mamba_config(model_size)
    backbone = MyModel(img_size=img_size, **config_dict)
    backbone, ret = load_ckpt(backbone, model_size)
    return MambaSegmentor(backbone, **config_dict), ret[0]
