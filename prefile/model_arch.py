"""
model_arch.py — 模型骨架（核心共享区）
训练脚本和推理脚本都从这里 import 模型类，不能各自定义。
包含：SharedBackbone / SegmentationHead / FloorplanModel
      AffinityFieldLoss / KendallWeighting / SegmentationLoss
"""

import logging
from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import (
    AnchorGenerator, RPNHead, RegionProposalNetwork,
)
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 网络模块
# ══════════════════════════════════════════════════════════════

class SharedBackbone(nn.Module):
    """
    共享 ResNet50 Backbone
    提取 4 个尺度的特征图供两个任务头使用
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet  = resnet50(weights=weights)

        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1   # /4,  256ch
        self.layer2 = resnet.layer2   # /8,  512ch
        self.layer3 = resnet.layer3   # /16, 1024ch
        self.layer4 = resnet.layer4   # /32, 2048ch

        self.out_channels = [256, 512, 1024, 2048]

    def forward(self, x: torch.Tensor) -> OrderedDict:
        x0 = self.layer0(x)
        c1 = self.layer1(x0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return OrderedDict([('0', c1), ('1', c2), ('2', c3), ('3', c4)])


class SegmentationHead(nn.Module):
    """
    墙体分割头（Section 2.3）
    FPN + 轻量分割 decoder
    """

    def __init__(self, fpn_out_channels: int = 256, num_classes: int = 2):
        super().__init__()
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=fpn_out_channels,
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(fpn_out_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, features: OrderedDict, input_size: Tuple) -> torch.Tensor:
        fpn_out     = self.fpn(features)
        target_size = fpn_out['0'].shape[-2:]
        fused       = fpn_out['0']
        for k in ['1', '2', '3']:
            fused = fused + F.interpolate(
                fpn_out[k], size=target_size, mode='bilinear', align_corners=False
            )
        logits = self.decoder(fused)
        logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        return logits


class FloorplanModel(nn.Module):
    """
    论文完整模型：共享 backbone 的双头架构

    forward 返回：
      训练时：{'seg_logits': ..., 'det_losses': {...}}
      推理时：{'seg_logits': ..., 'det_outputs': [...]}
    """

    def __init__(self, cfg):
        """cfg 为 PaperConfig 实例（避免循环 import，使用 duck-typing）"""
        super().__init__()

        # ── 共享 Backbone ──
        self.backbone = SharedBackbone(pretrained=cfg.pretrained_backbone)

        # ── 墙体分割头（Section 2.3）──
        self.seg_head = SegmentationHead(
            fpn_out_channels=cfg.fpn_out_channels,
            num_classes=cfg.seg_num_classes,
        )

        # ── 门窗检测头（Section 2.2）──
        self.det_fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=cfg.fpn_out_channels,
        )

        rpn_anchor_generator = AnchorGenerator(
            sizes=((16,), (32,), (64,), (128,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 4,
        )
        rpn_head = RPNHead(
            cfg.fpn_out_channels,
            rpn_anchor_generator.num_anchors_per_location()[0],
        )
        self.rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            fg_iou_thresh=0.7, bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n={'training': 2000, 'testing': 1000},
            post_nms_top_n={'training': 2000, 'testing': 300},
            nms_thresh=0.7,
        )

        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2,
        )
        box_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cfg.fpn_out_channels * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
        )
        box_predictor = FastRCNNPredictor(1024, cfg.det_num_classes)

        self.roi_heads = RoIHeads(
            box_roi_pool, box_head, box_predictor,
            fg_iou_thresh=0.5, bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=100,
        )

    def forward(
        self,
        images:  torch.Tensor,
        targets: Optional[List[dict]] = None,
    ) -> dict:
        """
        images  : (B, 3, H, W)
        targets : list of {'boxes': (N,4), 'labels': (N,)} 训练时传入
        """
        input_size = images.shape[-2:]

        features     = self.backbone(images)
        seg_logits   = self.seg_head(features, input_size)
        det_features = self.det_fpn(features)

        image_sizes = [images.shape[-2:]] * images.shape[0]
        img_list    = ImageList(images, image_sizes)

        if self.training and targets is not None:
            proposals, rpn_losses = self.rpn(img_list, det_features, targets)
            _, roi_losses = self.roi_heads(det_features, proposals, image_sizes, targets)
            return {'seg_logits': seg_logits, 'det_losses': {**rpn_losses, **roi_losses}}
        else:
            proposals, _ = self.rpn(img_list, det_features, None)
            det_outputs, _ = self.roi_heads(det_features, proposals, image_sizes, None)
            return {'seg_logits': seg_logits, 'det_outputs': det_outputs}


# ══════════════════════════════════════════════════════════════
# 损失函数（Section 2.3）
# ══════════════════════════════════════════════════════════════

class AffinityFieldLoss(nn.Module):
    """
    Affinity Field Loss（Section 2.3，基于 Ke et al. ECCV 2018）
    邻域一致性损失，改善墙体边界清晰度
    """

    def __init__(self, neighborhood: int = 5):
        super().__init__()
        self.n = neighborhood

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pred.shape
        n = self.n

        pred_prob = F.softmax(pred, dim=1)
        target_oh = F.one_hot(target.clamp(0, C - 1), C).permute(0, 3, 1, 2).float()

        loss  = torch.tensor(0.0, device=pred.device)
        count = 0
        for dy in range(-n, n + 1):
            for dx in range(-n, n + 1):
                if dy == 0 and dx == 0:
                    continue
                sp       = torch.roll(pred_prob,  (dy, dx), (2, 3))
                st       = torch.roll(target_oh,  (dy, dx), (2, 3))
                same     = (target_oh * st).sum(1, keepdim=True)
                aff_pred = (pred_prob  * sp).sum(1, keepdim=True)
                loss    += F.mse_loss(aff_pred, same)
                count   += 1
        return loss / max(count, 1)


class KendallWeighting(nn.Module):
    """
    Kendall 不确定性自动加权（Section 2.3，基于 Kendall et al. CVPR 2018）
    自动学习 BCE Loss 和 Affinity Loss 的权重
    """

    def __init__(self):
        super().__init__()
        self.log_var_bce = nn.Parameter(torch.tensor(0.0))
        self.log_var_aff = nn.Parameter(torch.tensor(0.0))

    def forward(self, loss_bce: torch.Tensor, loss_aff: torch.Tensor) -> torch.Tensor:
        w_bce = torch.exp(-self.log_var_bce)
        w_aff = torch.exp(-self.log_var_aff)
        return (w_bce * loss_bce + self.log_var_bce +
                w_aff * loss_aff + self.log_var_aff)


class SegmentationLoss(nn.Module):
    """
    墙体分割损失（Section 2.3）：
    L_seg = Kendall(L_BCE, L_Affinity)
    """

    def __init__(self, cfg):
        super().__init__()
        w = torch.tensor([1.0, cfg.wall_class_weight])
        self.ce       = nn.CrossEntropyLoss(weight=w, ignore_index=255)
        self.affinity = AffinityFieldLoss(cfg.affinity_neighborhood)
        self.kendall  = KendallWeighting()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> dict:
        loss_bce = self.ce(logits, target)
        loss_aff = self.affinity(logits, target)
        loss_seg = self.kendall(loss_bce, loss_aff)
        return {
            'loss_seg': loss_seg,
            'loss_bce': loss_bce.item(),
            'loss_aff': loss_aff.item(),
        }


def build_model(cfg, device: str = 'cpu') -> FloorplanModel:
    """工厂函数：构建模型并移到指定设备"""
    model = FloorplanModel(cfg).to(device)
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'模型创建完成  总参数={total/1e6:.1f}M  可训练={trainable/1e6:.1f}M')
    return model
