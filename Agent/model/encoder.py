# model/encoder.py
from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import logger

# ── 可选依赖检查 ──
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    logger.warning('[!] timm 未安装: pip install timm')

try:
    from peft import LoraConfig, inject_adapter_in_model
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    logger.warning('[!] peft 未安装: pip install peft')


# ══════════════════════════════════════════════════════════════
# DINOv2 多尺度特征提取器
# ══════════════════════════════════════════════════════════════

class DINOv2Encoder(nn.Module):
    """
    DINOv2 ViT-L/14 多尺度特征提取器。

    从 4 个深度层级抽取中间特征，模拟 FPN 的多尺度输入：
        ViT-L: 24 个 block，embed_dim=1024
        抽取层: [5, 11, 17, 23]  → 1/4, 2/4, 3/4, 4/4 深度

    浅层特征做 avg_pool 降分辨率，模拟真正的多尺度金字塔。
    """

    EXTRACT_LAYERS = [5, 11, 17, 23]

    def __init__(self, model_name: str = 'vit_large_patch14_dinov2'):
        super().__init__()
        if not HAS_TIMM:
            raise RuntimeError('pip install timm')

        # dynamic_img_size=True：接受任意 patch_size 倍数的输入，不固定 518×518
        self.vit = timm.create_model(
            model_name,
            pretrained       = True,
            dynamic_img_size = True,
        )
        self.embed_dim    = self.vit.embed_dim   # ViT-L: 1024
        self.out_channels = [self.embed_dim] * 4  # FPN 需要知道每层通道数
        self._features: Dict[int, torch.Tensor] = {}

        # 注册 forward hook，捕获中间层输出
        for layer_idx in self.EXTRACT_LAYERS:
            self.vit.blocks[layer_idx].register_forward_hook(
                self._make_hook(layer_idx)
            )

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            self._features[layer_idx] = output
        return hook

    def forward(self, x: torch.Tensor) -> OrderedDict:
        """
        输入: (B, 3, H, W)  H/W 必须是 14 的倍数
        输出: OrderedDict {'0':feat0, '1':feat1, '2':feat2, '3':feat3}
              每个 feat 形状 (B, embed_dim, h_i, w_i)，h/w 逐层递减
        """
        B, C, H, W = x.shape
        self._features.clear()
        _ = self.vit(x)

        patch_h = H // 14
        patch_w = W // 14
        result  = OrderedDict()

        for i, layer_idx in enumerate(self.EXTRACT_LAYERS):
            feat = self._features[layer_idx]           # (B, N+1, D)
            feat = feat[:, 1:, :].permute(0, 2, 1)    # (B, D, N)，去掉 cls token
            feat = feat.reshape(B, -1, patch_h, patch_w)

            # 浅层下采样，模拟低分辨率 FPN 层
            if i < 3:
                scale = 2 ** (3 - i)
                feat  = F.avg_pool2d(feat, kernel_size=scale, stride=scale)

            result[str(i)] = feat

        return result


# ══════════════════════════════════════════════════════════════
# LoRA 注入
# ══════════════════════════════════════════════════════════════

def inject_lora(encoder: nn.Module, cfg) -> nn.Module:
    """
    用 peft inject_adapter_in_model 注入 LoRA，同时冻结非 LoRA 参数。

    为什么不用 get_peft_model？
        get_peft_model 会包成 PeftModelForFeatureExtraction，其 forward
        强制传 input_ids / attention_mask 等 NLP 参数，和自定义
        DINOv2Encoder.forward(x) 不兼容，直接 TypeError。

    inject_adapter_in_model 是 peft 底层 API：
        - 只替换目标 Linear 层，不碰 forward，不改接口
        - 权重命名遵循 peft 标准（lora_A / lora_B）
        - 可用 peft 工具只存 LoRA 增量（几 MB 而非几百 MB）
    """
    if not HAS_PEFT:
        raise RuntimeError('pip install peft')

    lora_config = LoraConfig(
        r              = cfg.lora_r,
        lora_alpha     = cfg.lora_alpha,
        lora_dropout   = cfg.lora_dropout,
        target_modules = cfg.lora_target,   # ['qkv', 'proj']
        bias           = 'none',
    )
    encoder = inject_adapter_in_model(lora_config, encoder)

    # 冻结所有非 LoRA 参数
    for name, p in encoder.named_parameters():
        if 'lora_' not in name:
            p.requires_grad = False

    total     = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    logger.info(
        f'LoRA 注入完成: {trainable / 1e6:.2f}M / {total / 1e6:.1f}M '
        f'可训练 ({trainable / total * 100:.1f}%)'
    )
    return encoder


# ══════════════════════════════════════════════════════════════
# LoRA 权重 IO
# ══════════════════════════════════════════════════════════════

def save_lora_weights(encoder: nn.Module, save_path: str) -> None:
    """
    只保存 LoRA 增量权重（几 MB），不保存冻结的原始权重。
    文件可以直接用 load_lora_weights 恢复，原始 DINOv2 权重从 timm 重新加载。
    """
    lora_state = {k: v for k, v in encoder.state_dict().items() if 'lora_' in k}
    torch.save(lora_state, save_path)

    size_mb = sum(v.numel() * v.element_size() for v in lora_state.values()) / 1e6
    logger.info(
        f'LoRA 增量已保存: {save_path}  '
        f'({size_mb:.1f} MB  {len(lora_state)} 个张量)'
    )


def load_lora_weights(encoder: nn.Module, load_path: str) -> nn.Module:
    """
    加载 LoRA 增量权重。
    strict=False 忽略冻结的原始权重（它们不在增量文件里），
    只加载 lora_A / lora_B，如有缺失会发出 warning。
    """
    lora_state          = torch.load(load_path, map_location='cpu')
    missing, unexpected = encoder.load_state_dict(lora_state, strict=False)

    lora_missing = [k for k in missing if 'lora_' in k]
    if lora_missing:
        logger.warning(f'以下 LoRA 权重未加载: {lora_missing}')
    if unexpected:
        logger.warning(f'以下权重在模型中不存在: {unexpected}')

    logger.info(f'LoRA 权重加载完成: {load_path}')
    return encoder
