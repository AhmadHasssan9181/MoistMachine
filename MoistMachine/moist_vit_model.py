"""
MoistMachine - MoistViT: Vision Transformer Flood & Disaster Segmentation
Architecture:
  - Backbone: EfficientViT-B2 (Linear Multi-Head Self-Attention Vision Transformer)
  - Capacity: ~16.8M Parameters (10.7x larger capacity than MobileNetV3-Small)
  - Multi-Scale Feature Pyramid Network (FPN) with 128-dim channels
  - Tri-Receptive Depthwise Convolutions:
      d=1 (local building cracks / car outlines)
      d=3 (submerged roads / neighborhood footprints)
      d=6 (wide-area floodwater flow / river corridors)
  - Global Context Squeeze-and-Excitation (SE) Modulation for valley shadow suppression
  - Dual Output Heads:
      1. Dense 10-Class Segmentation Logits [B, 10, H, W]
      2. Scene-Level Flood Inundation Index [B, 1] (0.0 = Dry, 1.0 = Submerged)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class TriReceptiveBlock(nn.Module):
    """
    Captures micro-level details (d=1, vehicles/walls), mid-level infrastructure (d=3, roads),
    and wide-area floodwaters (d=6, river spreads) simultaneously.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.dw_local = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, dilation=1,
            groups=channels, bias=False
        )
        self.dw_mid = nn.Conv2d(
            channels, channels, kernel_size=3, padding=3, dilation=3,
            groups=channels, bias=False
        )
        self.dw_wide = nn.Conv2d(
            channels, channels, kernel_size=3, padding=6, dilation=6,
            groups=channels, bias=False
        )
        self.pw = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_local = self.dw_local(x)
        feat_mid   = self.dw_mid(x)
        feat_wide  = self.dw_wide(x)
        cat = torch.cat([feat_local, feat_mid, feat_wide], dim=1)
        return self.act(self.bn(self.pw(cat))) + x

class GlobalContextModulation(nn.Module):
    """
    Global Context Module: Squeezes whole-image statistics to re-weight feature maps,
    preventing localized hallucinations in deep mountain shadows or dark asphalt.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w

class MoistViT(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        backbone_name: str = "efficientvit_b2",
        fpn_dim: int = 128,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.fpn_dim = fpn_dim

        # 1. Linear Multi-Head Self-Attention Vision Transformer Backbone
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
        )

        in_channels = self.encoder.feature_info.channels()
        # For efficientvit_b2: [48, 96, 192, 384]

        # 2. Lateral 1x1 Projections
        self.lat0 = nn.Sequential(
            nn.Conv2d(in_channels[0], fpn_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.SiLU(inplace=True),
        )
        self.lat1 = nn.Sequential(
            nn.Conv2d(in_channels[1], fpn_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.SiLU(inplace=True),
        )
        self.lat2 = nn.Sequential(
            nn.Conv2d(in_channels[2], fpn_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.SiLU(inplace=True),
        )
        self.lat3 = nn.Sequential(
            nn.Conv2d(in_channels[3], fpn_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.SiLU(inplace=True),
        )

        # 3. Top-Down Pyramid Fusion with Tri-Receptive Field
        self.smooth3 = TriReceptiveBlock(fpn_dim)
        self.smooth2 = TriReceptiveBlock(fpn_dim)
        self.smooth1 = TriReceptiveBlock(fpn_dim)
        self.smooth0 = TriReceptiveBlock(fpn_dim)

        # 4. Multi-Scale Aggregation Head
        self.agg_conv = nn.Sequential(
            nn.Conv2d(fpn_dim * 4, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.SiLU(inplace=True),
        )
        self.global_context = GlobalContextModulation(fpn_dim)

        # 5. Output Heads
        # Head 1: Dense 10-Class Semantic Segmentation
        self.seg_head = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(fpn_dim // 2, num_classes, kernel_size=1),
        )

        # Head 2: Scene-Level Inundation Threat Score (0.0 to 1.0)
        self.threat_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(fpn_dim, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        input_size = x.shape[2:]

        # Encoder features
        c0, c1, c2, c3 = self.encoder(x)

        # Lateral projections
        p3 = self.lat3(c3)
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[2:], mode="bilinear", align_corners=False)
        p1 = self.lat1(c1) + F.interpolate(p2, size=c1.shape[2:], mode="bilinear", align_corners=False)
        p0 = self.lat0(c0) + F.interpolate(p1, size=c0.shape[2:], mode="bilinear", align_corners=False)

        # Anti-Tunnel-Vision smoothing
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)
        p1 = self.smooth1(p1)
        p0 = self.smooth0(p0)

        # Multi-scale aggregation to highest-res feature grid (c0)
        u3 = F.interpolate(p3, size=c0.shape[2:], mode="bilinear", align_corners=False)
        u2 = F.interpolate(p2, size=c0.shape[2:], mode="bilinear", align_corners=False)
        u1 = F.interpolate(p1, size=c0.shape[2:], mode="bilinear", align_corners=False)
        u0 = p0

        fused = self.agg_conv(torch.cat([u0, u1, u2, u3], dim=1))
        fused = self.global_context(fused)

        # Dense Segmentation Logits
        seg_logits = self.seg_head(fused)
        seg_logits = F.interpolate(seg_logits, size=input_size, mode="bilinear", align_corners=False)

        # Scene Inundation Threat Score
        threat_score = self.threat_head(fused)

        return seg_logits, threat_score

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoistViT(num_classes=10, backbone_name="efficientvit_b2", fpn_dim=128, pretrained=False).to(device)
    dummy = torch.randn(2, 3, 512, 512, device=device)
    seg, threat = model(dummy)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"MoistViT successfully instantiated!")
    print(f"  Params      : {params:.2f} Million")
    print(f"  Seg Shape   : {seg.shape}")
    print(f"  Threat Shape: {threat.shape}")
