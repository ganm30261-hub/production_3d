"""
ResNet-FPN Backbone（生产级）

ResNet 编码器 + Feature Pyramid Network 解码器，
用于多尺度户型图特征提取。

参考:
- ResNet: He et al., 2016
- FPN: Lin et al., 2017

使用方法：
    from models.backbone.resnet_fpn import create_segmentation_model
    model = create_segmentation_model(num_classes=12, variant='resnet50')
    logits = model(images)  # (B, 12, H, W)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import (
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    ResNet101_Weights, ResNet152_Weights,
)

from ..exceptions import BackboneError

logger = logging.getLogger(__name__)


# ============================================================================
# 配置
# ============================================================================

class ResNetVariant(Enum):
    RESNET18 = "resnet18"
    RESNET34 = "resnet34"
    RESNET50 = "resnet50"
    RESNET101 = "resnet101"
    RESNET152 = "resnet152"


RESNET_CHANNELS = {
    ResNetVariant.RESNET18:  [64, 128, 256, 512],
    ResNetVariant.RESNET34:  [64, 128, 256, 512],
    ResNetVariant.RESNET50:  [256, 512, 1024, 2048],
    ResNetVariant.RESNET101: [256, 512, 1024, 2048],
    ResNetVariant.RESNET152: [256, 512, 1024, 2048],
}

_WEIGHTS_MAP = {
    ResNetVariant.RESNET18:  ResNet18_Weights.IMAGENET1K_V1,
    ResNetVariant.RESNET34:  ResNet34_Weights.IMAGENET1K_V1,
    ResNetVariant.RESNET50:  ResNet50_Weights.IMAGENET1K_V2,
    ResNetVariant.RESNET101: ResNet101_Weights.IMAGENET1K_V2,
    ResNetVariant.RESNET152: ResNet152_Weights.IMAGENET1K_V2,
}

_MODEL_MAP = {
    ResNetVariant.RESNET18:  models.resnet18,
    ResNetVariant.RESNET34:  models.resnet34,
    ResNetVariant.RESNET50:  models.resnet50,
    ResNetVariant.RESNET101: models.resnet101,
    ResNetVariant.RESNET152: models.resnet152,
}


@dataclass
class ResNetFPNConfig:
    """ResNet-FPN 配置"""
    variant: ResNetVariant = ResNetVariant.RESNET50
    pretrained: bool = True

    # FPN
    fpn_channels: int = 256
    use_p6: bool = False

    # 归一化
    norm_layer: str = "batch_norm"   # batch_norm / group_norm / none
    num_groups: int = 32

    # 冻结
    freeze_bn: bool = False
    freeze_stages: int = 0

    # 膨胀卷积
    replace_stride_with_dilation: List[bool] = field(
        default_factory=lambda: [False, False, False]
    )

    # 额外卷积
    extra_convs: int = 0


# ============================================================================
# 基础模块
# ============================================================================

class ConvBNReLU(nn.Module):
    """Conv2d + Norm + (optional) ReLU"""

    def __init__(
        self, in_ch: int, out_ch: int, ks: int = 3, stride: int = 1,
        padding: int = 1, norm: str = "batch_norm", groups: int = 32,
        act: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, ks, stride, padding, bias=(norm == "none"))
        if norm == "batch_norm":
            self.norm = nn.BatchNorm2d(out_ch)
        elif norm == "group_norm":
            self.norm = nn.GroupNorm(groups, out_ch)
        else:
            self.norm = nn.Identity()
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class FPNBlock(nn.Module):
    """FPN 侧向连接 + 自顶向下路径"""

    def __init__(self, in_ch: int, out_ch: int, norm: str = "batch_norm", groups: int = 32):
        super().__init__()
        self.lateral = nn.Conv2d(in_ch, out_ch, 1)
        self.output = ConvBNReLU(out_ch, out_ch, norm=norm, groups=groups, act=False)

    def forward(self, x: torch.Tensor, top_down: Optional[torch.Tensor] = None) -> torch.Tensor:
        lat = self.lateral(x)
        if top_down is not None:
            lat = lat + F.interpolate(top_down, size=lat.shape[-2:], mode='nearest')
        return self.output(lat)


# ============================================================================
# 编码器 / 解码器 / 主干
# ============================================================================

class ResNetEncoder(nn.Module):
    """ResNet 多尺度特征提取器"""

    def __init__(self, config: ResNetFPNConfig):
        super().__init__()
        self.channels = RESNET_CHANNELS[config.variant]

        weights = _WEIGHTS_MAP[config.variant] if config.pretrained else None
        try:
            backbone = _MODEL_MAP[config.variant](
                weights=weights,
                replace_stride_with_dilation=config.replace_stride_with_dilation,
            )
        except Exception as e:
            raise BackboneError(f"ResNet 初始化失败: {e}") from e

        del backbone.fc, backbone.avgpool
        self.backbone = backbone

        if config.freeze_stages > 0:
            self._freeze_stages(config.freeze_stages)
        if config.freeze_bn:
            self._freeze_bn()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        c2 = b.layer1(x)
        c3 = b.layer2(c2)
        c4 = b.layer3(c3)
        c5 = b.layer4(c4)
        return {'c2': c2, 'c3': c3, 'c4': c4, 'c5': c5}

    def _freeze_stages(self, n: int):
        if n >= 1:
            for p in list(self.backbone.conv1.parameters()) + list(self.backbone.bn1.parameters()):
                p.requires_grad = False
        for idx, layer in [(2, self.backbone.layer1), (3, self.backbone.layer2),
                           (4, self.backbone.layer3), (5, self.backbone.layer4)]:
            if n >= idx:
                for p in layer.parameters():
                    p.requires_grad = False

    def _freeze_bn(self):
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False


class FPNDecoder(nn.Module):
    """FPN 解码器：自顶向下融合"""

    def __init__(self, in_channels_list: List[int], out_ch: int, config: ResNetFPNConfig):
        super().__init__()
        self.config = config
        self.fpn_blocks = nn.ModuleDict()
        for i, in_ch in enumerate(reversed(in_channels_list)):
            lvl = len(in_channels_list) - i + 1
            self.fpn_blocks[f'p{lvl}'] = FPNBlock(in_ch, out_ch, config.norm_layer, config.num_groups)
        if config.use_p6:
            self.p6_conv = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        td = None
        for lvl in [5, 4, 3, 2]:
            ck = f'c{lvl}'
            pk = f'p{lvl}'
            if ck in feats:
                out[pk] = self.fpn_blocks[pk](feats[ck], td)
                td = out[pk]
        if self.config.use_p6 and 'p5' in out:
            out['p6'] = self.p6_conv(out['p5'])
        return out


class ResNetFPN(nn.Module):
    """
    完整 ResNet-FPN 主干。

    输入 (B,3,H,W) → 输出 {'p2': ..., 'p3': ..., 'p4': ..., 'p5': ...}
    """

    def __init__(self, config: Optional[ResNetFPNConfig] = None):
        super().__init__()
        self.config = config or ResNetFPNConfig()
        self.encoder = ResNetEncoder(self.config)
        self.decoder = FPNDecoder(self.encoder.channels, self.config.fpn_channels, self.config)
        self.out_channels = self.config.fpn_channels
        self.feature_strides = {'p2': 4, 'p3': 8, 'p4': 16, 'p5': 32}
        if self.config.use_p6:
            self.feature_strides['p6'] = 64
        logger.info("ResNet-FPN: %s, channels=%d, pretrained=%s",
                     self.config.variant.value, self.config.fpn_channels, self.config.pretrained)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.decoder(self.encoder(x))


class SegmentationHead(nn.Module):
    """分割头：融合 FPN 多尺度特征 → 分类输出"""

    def __init__(self, in_ch: int, num_classes: int, hidden: int = 256, num_convs: int = 4):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers.append(ConvBNReLU(in_ch if i == 0 else hidden, hidden))
        self.convs = nn.Sequential(*layers)
        self.cls = nn.Conv2d(hidden, num_classes, 1)

    def forward(self, feats: Dict[str, torch.Tensor], target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        x = feats['p2']
        for k in ('p3', 'p4', 'p5'):
            if k in feats:
                x = x + F.interpolate(feats[k], size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = self.cls(self.convs(x))
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x


class ResNetFPNSegmentation(nn.Module):
    """
    ResNet-FPN 语义分割模型（端到端）。

    输入 (B,3,H,W) → 输出 (B, num_classes, H, W) logits。
    """

    def __init__(
        self, num_classes: int,
        backbone_config: Optional[ResNetFPNConfig] = None,
        hidden_channels: int = 256, num_head_convs: int = 4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = ResNetFPN(backbone_config)
        self.head = SegmentationHead(self.backbone.out_channels, num_classes, hidden_channels, num_head_convs)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feats = self.backbone(x)
        logits = self.head(feats, target_size=x.shape[-2:])
        return (logits, feats) if return_features else logits


# ============================================================================
# 工厂函数
# ============================================================================

_VARIANT_MAP = {v.value: v for v in ResNetVariant}


def create_resnet_fpn(variant: str = "resnet50", pretrained: bool = True, fpn_channels: int = 256, **kw) -> ResNetFPN:
    """创建 ResNet-FPN 主干。"""
    cfg = ResNetFPNConfig(variant=_VARIANT_MAP[variant], pretrained=pretrained, fpn_channels=fpn_channels, **kw)
    return ResNetFPN(cfg)


def create_segmentation_model(num_classes: int, variant: str = "resnet50", pretrained: bool = True, **kw) -> ResNetFPNSegmentation:
    """创建端到端分割模型。"""
    cfg = ResNetFPNConfig(variant=_VARIANT_MAP[variant], pretrained=pretrained, **kw)
    return ResNetFPNSegmentation(num_classes, backbone_config=cfg)
