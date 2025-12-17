import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch)
        )
    def forward(self, x):
        return self.conv(x)

class MultiHeadFloodNet(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, bottleneck_channels=1280, k_clusters=8):
        super().__init__()
        self.encoder_backbone = models.mobilenet_v2(pretrained=pretrained).features
        self.num_classes = num_classes
        self.k_clusters = k_clusters

        self.reduce_bottleneck = nn.Conv2d(1280, 256, kernel_size=1)
        self.up1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.conv1 = UpConv(128+96, 128)  # skip with prior
        self.up2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.conv2 = UpConv(64+32, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.conv3 = UpConv(32+24, 32)
        self.up4 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.conv4 = UpConv(16, 16)
        # semantic head
        self.seg_head = nn.Conv2d(16, self.num_classes, kernel_size=1)
        # unsupervised clustering head: produce low-dim features per pixel (for kmeans/pseudo-labeling)
        self.cluster_head = nn.Conv2d(16, 32, kernel_size=1)
        # projection for pseudo cluster logits (registered param, unlike the old per-call layer)
        self.cluster_proj = nn.Conv2d(32, k_clusters, kernel_size=1)

    def forward(self, x):
        # run encoder capturing skips
        skips = []
        cur = x
        for i, layer in enumerate(self.encoder_backbone):
            cur = layer(cur)
            if i in {1, 3, 6, 13, 16}:  # empirically chosen indices
                skips.append(cur)
        bottleneck = skips[-1]
        x = self.reduce_bottleneck(bottleneck)   # [B,256,h,w]
        x = F.relu(self.up1(x))                   # [B,128,2h,2w]
        s3 = skips[-2] if len(skips) >= 2 else None
        if s3 is not None:
            x = torch.cat([x, F.interpolate(s3, size=x.shape[2:], mode='bilinear', align_corners=False)], dim=1)
        x = self.conv1(x)
        x = F.relu(self.up2(x))
        s2 = skips[-3] if len(skips) >= 3 else None
        if s2 is not None:
            x = torch.cat([x, F.interpolate(s2, size=x.shape[2:], mode='bilinear', align_corners=False)], dim=1)
        x = self.conv2(x)
        x = F.relu(self.up3(x))
        s1 = skips[-4] if len(skips) >= 4 else None
        if s1 is not None:
            x = torch.cat([x, F.interpolate(s1, size=x.shape[2:], mode='bilinear', align_corners=False)], dim=1)
        x = self.conv3(x)
        x = F.relu(self.up4(x))
        x = self.conv4(x)

        seg_logits = self.seg_head(x)       # [B, num_classes, H, W]
        cluster_feat = self.cluster_head(x) # [B, 32, H, W]

        # feature for GRU: global average per class over seg logits -> per-frame vector
        probs = torch.softmax(seg_logits, dim=1)
        pooled = probs.mean(dim=[2, 3])  # [B, num_classes]

        return seg_logits, cluster_feat, pooled