import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class MultiHeadFloodNetLite(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, k_clusters=0):
        super().__init__()
        # FIX: Handle deprecated "pretrained" arg gracefully
        weights = "IMAGENET1K_V1" if pretrained else None
        mbv3 = models.mobilenet_v3_small(weights=weights)
        self.encoder = mbv3.features
        self.num_classes = num_classes
        self.k_clusters = k_clusters

        # --- FIX: CORRECT LAYERS FOR MOBILENET-V3-SMALL ---
        # The standard encoder only goes up to index 12.
        # Layer 0: Stride 2 (H/2)  -> 16 ch
        # Layer 3: Stride 4 (H/4)  -> 24 ch
        # Layer 8: Stride 8 (H/8)  -> 48 ch
        # Layer 12: Stride 16 (H/16) -> 576 ch (Bottleneck)
        self.skip_ids = [0, 3, 8, 12]

        # Decoder Configuration
        # 1. Bottleneck reduction (Input 576 from Layer 12)
        self.reduce_bot = nn.Conv2d(576, 128, 1, bias=False)

        # 2. Decode H/16 -> H/8
        self.up1 = nn.ConvTranspose2d(128, 96, 4, stride=2, padding=1)
        # Cat with Layer 8 (48 ch). Input to conv1 is 96 + 48 = 144
        self.conv1 = UpConv(96 + 48, 96)

        # 3. Decode H/8 -> H/4
        self.up2 = nn.ConvTranspose2d(96, 64, 4, stride=2, padding=1)
        # Cat with Layer 3 (24 ch). Input to conv2 is 64 + 24 = 88
        self.conv2 = UpConv(64 + 24, 64)

        # 4. Decode H/4 -> H/2
        self.up3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        # Cat with Layer 0 (16 ch). Input to conv3 is 32 + 16 = 48
        self.conv3 = UpConv(32 + 16, 32)

        # 5. Decode H/2 -> H (Final Resolution)
        self.up4 = nn.ConvTranspose2d(32, 24, 4, stride=2, padding=1)
        self.conv4 = UpConv(24, 24)

        self.seg_head = nn.Conv2d(24, num_classes, 1)

        if k_clusters > 0:
            self.cluster_head = nn.Conv2d(24, 16, 1)
            self.cluster_proj = nn.Conv2d(16, k_clusters, 1)
        else:
            self.cluster_head = None
            self.cluster_proj = None

    def forward(self, x):
        skips = {}
        cur = x
        # Run Encoder and save skip connections
        for i, layer in enumerate(self.encoder):
            cur = layer(cur)
            if i in self.skip_ids:
                skips[i] = cur

        # Bottleneck (Layer 12, H/16)
        bot = skips[12]
        x = self.reduce_bot(bot)

        # Up 1 (H/16 -> H/8)
        x = F.relu(self.up1(x))
        # Concatenate with Layer 8
        if x.shape[2:] != skips[8].shape[2:]:
            x = F.interpolate(x, size=skips[8].shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skips[8]], dim=1)
        x = self.conv1(x)

        # Up 2 (H/8 -> H/4)
        x = F.relu(self.up2(x))
        # Concatenate with Layer 3
        if x.shape[2:] != skips[3].shape[2:]:
            x = F.interpolate(x, size=skips[3].shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skips[3]], dim=1)
        x = self.conv2(x)

        # Up 3 (H/4 -> H/2)
        x = F.relu(self.up3(x))
        # Concatenate with Layer 0
        if x.shape[2:] != skips[0].shape[2:]:
            x = F.interpolate(x, size=skips[0].shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skips[0]], dim=1)
        x = self.conv3(x)

        # Up 4 (H/2 -> H)
        x = F.relu(self.up4(x))
        x = self.conv4(x)

        seg_logits = self.seg_head(x)

        cluster_feat = None
        if self.cluster_head is not None:
            cluster_feat = self.cluster_head(x)

        probs = torch.softmax(seg_logits, dim=1)
        pooled = probs.mean(dim=[2, 3])

        return seg_logits, cluster_feat, pooled