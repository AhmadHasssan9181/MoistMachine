"""
MoistMachine - MoistViT Training Pipeline
Trains the Vision Transformer (EfficientViT-B2 + Tri-Receptive FPN)
on the Multi-Domain Flood Dataset (FloodNet + LoveDA Rural) using the RTX 3090.

Features:
  - 10-Class Semantic Segmentation + Global Inundation Index.
  - Turbidity-Invariant Augmentations & Mountain Shadow Invariance.
  - Multi-Class Focal + Dice Loss penalizing misclassification on flooded buildings/roads.
  - Mixed Precision AMP for high GPU utilization.
  - Auto-exports PyTorch Mobile Lite (.ptl) and ONNX for the Waffle Android app.
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moist_vit_model import MoistViT
from moist_dataset import MoistDataset

NUM_CLASSES = 10
CLASS_NAMES = [
    "Background", "Build-Flood", "Build-Dry", "Road-Flood", "Road-Dry",
    "Water", "Tree", "Vehicle", "Pool", "Grass"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
EXPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exports")

# Class weights emphasizing critical hazard classes (Flooded Buildings, Flooded Roads, Water)
CLASS_WEIGHTS = torch.tensor([
    0.5,  # 0: Background
    3.5,  # 1: Building Flooded
    1.5,  # 2: Building Dry
    3.5,  # 3: Road Flooded
    1.5,  # 4: Road Dry
    2.5,  # 5: Water
    0.8,  # 6: Tree
    2.5,  # 7: Vehicle
    1.5,  # 8: Pool
    0.8,  # 9: Grass
], dtype=torch.float32)

class MoistFloodLoss(nn.Module):
    def __init__(self, weights, gamma=2.0):
        super().__init__()
        self.weights = weights.to(DEVICE)
        self.gamma = gamma

    def forward(self, seg_logits, threat_pred, targets, threat_targets):
        # 1. Focal Cross Entropy
        ce = F.cross_entropy(seg_logits, targets, weight=self.weights, reduction='none')
        pt = torch.exp(-ce)
        focal_ce = (((1.0 - pt) ** self.gamma) * ce).mean()

        # 2. Multi-Class Dice Loss
        probs = torch.softmax(seg_logits, dim=1)
        targets_onehot = F.one_hot(targets.clamp(0, NUM_CLASSES - 1), num_classes=NUM_CLASSES).permute(0, 3, 1, 2).float()
        intersection = (probs * targets_onehot).sum(dim=(2, 3))
        cardinality  = probs.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))
        dice_scores  = (2.0 * intersection + 1e-5) / (cardinality + 1e-5)

        # Focus dice loss on hazard classes (1: Build-Flood, 3: Road-Flood, 5: Water)
        dice_weights = torch.tensor([0.2, 0.4, 0.2, 0.4, 0.2, 0.3, 0.1, 0.3, 0.2, 0.1], device=DEVICE).unsqueeze(0)
        dice_loss = ((1.0 - dice_scores) * dice_weights).sum(dim=1).mean()

        # 3. Scene Threat Score Loss
        threat_loss = F.smooth_l1_loss(threat_pred, threat_targets)

        total = focal_ce + 1.2 * dice_loss + 0.1 * threat_loss
        return total, focal_ce.detach(), dice_loss.detach()

def compute_confusion_matrix(preds, targets, num_classes=NUM_CLASSES):
    mask = (targets >= 0) & (targets < num_classes)
    return np.bincount(
        num_classes * targets[mask] + preds[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)

def compute_miou(hist):
    ious = []
    for c in range(hist.shape[0]):
        intersection = hist[c, c]
        union = hist[c, :].sum() + hist[:, c].sum() - intersection
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    valid = [x for x in ious if not np.isnan(x)]
    miou = np.mean(valid) if valid else 0.0
    return miou, ious

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for images, targets, _ in tqdm(val_loader, desc="Validating", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        targets_np = targets.numpy()

        with autocast(device_type="cuda"):
            seg_logits, _ = model(images)
            preds = torch.argmax(seg_logits, dim=1).cpu().numpy()

        for b in range(preds.shape[0]):
            hist += compute_confusion_matrix(preds[b], targets_np[b])

    miou, per_class_ious = compute_miou(hist)
    return miou, per_class_ious

def export_models(model, img_size=512):
    os.makedirs(EXPORT_DIR, exist_ok=True)
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=DEVICE)

    # 1. TorchScript
    ts_path = os.path.join(EXPORT_DIR, f"moist_moistvit_{img_size}.pt")
    scripted = torch.jit.trace(model, dummy)
    scripted.save(ts_path)
    print(f"  [OK] TorchScript: {ts_path} ({os.path.getsize(ts_path)/(1024*1024):.2f} MB)")

    # 2. PyTorch Mobile Lite (for Waffle Android app)
    ptl_path = os.path.join(EXPORT_DIR, f"moist_moistvit_{img_size}.ptl")
    try:
        from torch.utils.mobile_optimizer import optimize_for_mobile
        opt_mobile = optimize_for_mobile(scripted)
        opt_mobile._save_for_lite_interpreter(ptl_path)
        print(f"  [OK] PyTorch Mobile Lite (Waffle Android): {ptl_path} ({os.path.getsize(ptl_path)/(1024*1024):.2f} MB)")
    except Exception as e:
        print(f"  [WARN] PyTorch Mobile export error: {e}")

    # 3. ONNX
    onnx_path = os.path.join(EXPORT_DIR, f"moist_moistvit_{img_size}.onnx")
    try:
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=["input"],
            output_names=["segmentation", "flood_threat"],
            opset_version=14,
            dynamic_axes={"input": {0: "batch"}, "segmentation": {0: "batch"}, "flood_threat": {0: "batch"}},
        )
        print(f"  [OK] ONNX Model: {onnx_path} ({os.path.getsize(onnx_path)/(1024*1024):.2f} MB)")
    except Exception as e:
        print(f"  [WARN] ONNX export error: {e}")

def train(epochs=30, batch_size=16, lr=4e-4, img_size=512):
    os.makedirs(SAVE_DIR, exist_ok=True)
    best_ckpt = os.path.join(SAVE_DIR, "best_moist_vit.pth")

    print("=" * 75)
    print(f">> Training MoistViT (EfficientViT-B2) on Multi-Domain Flood Dataset")
    print(f"   Epochs: {epochs} | Batch: {batch_size} | Size: {img_size}px | Device: {DEVICE}")
    print("=" * 75)

    # 1. Datasets & Loaders
    train_ds = MoistDataset(split="train", img_size=img_size)
    val_ds = MoistDataset(split="val", img_size=img_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    print(f"Loaded: {len(train_ds)} train samples, {len(val_ds)} val samples.")

    # 2. Model & Optimizer
    model = MoistViT(num_classes=NUM_CLASSES, backbone_name="efficientvit_b2", fpn_dim=128, pretrained=True).to(DEVICE)
    criterion = MoistFloodLoss(weights=CLASS_WEIGHTS)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler("cuda")

    best_miou = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_focal, total_dice = 0.0, 0.0, 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs}")
        for images, targets, threat_targets in pbar:
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            threat_targets = threat_targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                seg_logits, threat_pred = model(images)
                loss, focal, dice = criterion(seg_logits, threat_pred, targets, threat_targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_focal += focal.item()
            total_dice += dice.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Dice": f"{dice.item():.4f}"})

        scheduler.step()

        # Validation
        val_miou, per_class = evaluate(model, val_loader)
        avg_loss = total_loss / len(train_loader)

        # Log class breakdown
        water_iou = per_class[5] if not np.isnan(per_class[5]) else 0.0
        flood_bld_iou = per_class[1] if not np.isnan(per_class[1]) else 0.0
        flood_rd_iou = per_class[3] if not np.isnan(per_class[3]) else 0.0

        print(f"\n[Epoch {epoch:02d}] Train Loss: {avg_loss:.4f} | Val mIoU: {val_miou*100:5.2f}%")
        print(f"  Water IoU: {water_iou*100:5.2f}% | Flooded Building IoU: {flood_bld_iou*100:5.2f}% | Flooded Road IoU: {flood_rd_iou*100:5.2f}%")

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), best_ckpt)
            print(f"  [BEST] New Best MoistViT Saved: {best_ckpt} (mIoU: {best_miou*100:.2f}%)")

    print("\n" + "=" * 75)
    print(f"[DONE] Training Finished! Best Validation mIoU: {best_miou*100:.2f}%")
    print("=" * 75)

    # Load best weights for mobile export
    print(">> Exporting Best MoistViT to Mobile Formats (Android Waffle & ONNX)...")
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
    export_models(model, img_size=img_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--img_size", type=int, default=512)
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch, lr=args.lr, img_size=args.img_size)
