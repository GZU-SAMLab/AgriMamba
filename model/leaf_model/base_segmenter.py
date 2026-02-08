import torch.nn as nn
import torch.nn.functional as F

from .leaf_utils import dice_loss, sigmoid_focal_loss


class BaseSegmenter(nn.Module):
    def __init__(self, backbone, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.decoder = None

    def forward(self, x, mask=None, **kwargs):
        input_shape = x.shape[-2:]
        
        # 纯视觉特征提取
        features = self.backbone(x)
        x_c1, x_c2, x_c3 = features
        
        # 移除文本特征传递
        pred = self.decoder([x_c3, x_c2, x_c1])
        pred = F.interpolate(pred, input_shape, mode='bilinear', align_corners=True)
        
        # loss
        if self.training and mask is not None:
            loss = dice_loss(pred, mask) + sigmoid_focal_loss(pred, mask, alpha=-1, gamma=0)
            return pred.detach(), mask, loss
        else:
            return pred.detach()