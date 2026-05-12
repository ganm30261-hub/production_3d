# model/model.py
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.detection.rpn import (
    AnchorGenerator, RPNHead, RegionProposalNetwork,
)
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.image_list import ImageList

from config import logger, PseudoLabelConfig
from model.encoder import DINOv2Encoder, inject_lora


# ══════════════════════════════════════════════════════════════
# 分割头
# ══════════════════════════════════════════════════════════════

class SegHead(nn.Module):
    """
    FPN + 轻量 decoder → wall mask logits。

    FPN 把四个深度层的特征对齐到同一分辨率，
    decoder 用三层卷积输出每像素的 wall/background 概率。
    最后 bilinear 上采样回输入分辨率。
    """

    def __init__(
        self,
        in_ch_list: List[int],
        fpn_ch:     int = 256,
        num_classes: int = 2,
    ):
        super().__init__()
        self.fpn     = FeaturePyramidNetwork(in_ch_list, fpn_ch)
        self.decoder = nn.Sequential(
            nn.Conv2d(fpn_ch, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128,  64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(
        self,
        features:   dict,
        input_size: tuple,
    ) -> torch.Tensor:
        """
        features  : OrderedDict，来自 DINOv2Encoder.forward()
        input_size: (H, W)，目标上采样分辨率（即原图大小）
        返回: (B, num_classes, H, W)
        """
        fpn_out     = self.fpn(features)
        target_size = fpn_out['0'].shape[-2:]

        # 把所有 FPN 层加权融合到最大分辨率
        fused = fpn_out['0']
        for k in ['1', '2', '3']:
            if k in fpn_out:
                fused = fused + F.interpolate(
                    fpn_out[k], target_size, mode='bilinear', align_corners=False
                )

        logits = self.decoder(fused)
        return F.interpolate(logits, input_size, mode='bilinear', align_corners=False)


# ══════════════════════════════════════════════════════════════
# 完整模型
# ══════════════════════════════════════════════════════════════

class DINOv2LoRAModel(nn.Module):
    """
    DINOv2 ViT-L + LoRA + FPN 分割头 + Faster R-CNN 检测头。

    训练时 (targets 不为 None):
        返回 {'seg_logits': Tensor, 'det_losses': dict}

    推理时 (targets=None):
        返回 {'seg_logits': Tensor, 'det_outputs': List[dict]}
        det_outputs 每项: {'boxes', 'labels', 'scores'}

    分工：
        - 分割头负责 wall mask（语义分割）
        - 检测头负责门窗 bbox（目标检测）
        两个任务共享同一个 DINOv2 编码器，只有 LoRA 参数可训练。
    """

    def __init__(self, cfg: PseudoLabelConfig):
        super().__init__()

        # ── 编码器（inject_lora 内部同时完成冻结）──
        self.encoder  = DINOv2Encoder(cfg.dinov2_model)
        self.encoder  = inject_lora(self.encoder, cfg)
        in_ch         = self.encoder.out_channels   # [1024, 1024, 1024, 1024]

        # ── 分割头 ──
        self.seg_head = SegHead(in_ch, cfg.fpn_out_channels, cfg.seg_num_classes)

        # ── 检测头：独立 FPN + RPN + RoI ──
        fpn_ch = cfg.fpn_out_channels
        self.det_fpn = FeaturePyramidNetwork(in_ch, fpn_ch)

        rpn_anchor = AnchorGenerator(
            sizes        = ((16,), (32,), (64,), (128,)),
            aspect_ratios = ((0.5, 1.0, 2.0),) * 4,
        )
        self.rpn = RegionProposalNetwork(
            anchor_generator      = rpn_anchor,
            head                  = RPNHead(fpn_ch, rpn_anchor.num_anchors_per_location()[0]),
            fg_iou_thresh         = 0.7,
            bg_iou_thresh         = 0.3,
            batch_size_per_image  = 256,
            positive_fraction     = 0.5,
            pre_nms_top_n         = {'training': 2000, 'testing': 1000},
            post_nms_top_n        = {'training': 2000, 'testing': 300},
            nms_thresh            = 0.7,
        )
        self.roi_heads = RoIHeads(
            box_roi_pool        = MultiScaleRoIAlign(['0', '1', '2', '3'], 7, 2),
            box_head            = nn.Sequential(
                nn.Flatten(),
                nn.Linear(fpn_ch * 49, 1024), nn.ReLU(True),
                nn.Linear(1024, 1024),         nn.ReLU(True),
            ),
            box_predictor       = FastRCNNPredictor(1024, cfg.det_num_classes),
            fg_iou_thresh       = 0.5,
            bg_iou_thresh       = 0.5,
            batch_size_per_image = 512,
            positive_fraction   = 0.25,
            bbox_reg_weights    = None,
            score_thresh        = 0.05,
            nms_thresh          = 0.5,
            detections_per_img  = 100,
        )

    def forward(
        self,
        images:  torch.Tensor,
        targets: Optional[List[dict]] = None,
    ) -> dict:
        """
        images  : (B, 3, H, W)
        targets : List[{'boxes': Tensor(N,4), 'labels': Tensor(N,)}]  训练时传入
        """
        input_size   = images.shape[-2:]
        features     = self.encoder(images)

        # ── 分割 ──
        seg_logits   = self.seg_head(features, input_size)

        # ── 检测 ──
        det_features = self.det_fpn(features)
        img_list     = ImageList(
            images,
            [images.shape[-2:]] * images.shape[0],
        )

        if self.training and targets is not None:
            proposals, rpn_losses = self.rpn(img_list, det_features, targets)
            _, roi_losses = self.roi_heads(
                det_features, proposals,
                [images.shape[-2:]] * images.shape[0],
                targets,
            )
            return {
                'seg_logits': seg_logits,
                'det_losses': {**rpn_losses, **roi_losses},
            }
        else:
            proposals, _ = self.rpn(img_list, det_features, None)
            det_out, _   = self.roi_heads(
                det_features, proposals,
                [images.shape[-2:]] * images.shape[0],
                None,
            )
            return {
                'seg_logits':  seg_logits,
                'det_outputs': det_out,
            }
