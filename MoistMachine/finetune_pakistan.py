"""
MoistMachine - Pakistan Domain Adaptation Fine-Tuning
Calibrates MoistViT on real Pakistan 2022 flood imagery to eliminate brown-water bias.
Uses:
  - Base checkpoint: checkpoints/best_moist_vit.pth (70.24% mIoU)
  - Heavy augmentation (rotations, flips, zoom, noise) on Pakistan calibration masks
  - Replay regularization from LoveDA/FloodNet to prevent forgetting
  - Low learning rate (1e-4) with Cosine Annealing
"""

import os
import sys
import glob
import argparse
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moist_vit_model import MoistViT
from train_moist_vit import CLASS_WEIGHTS, MoistFloodLoss, export_models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_CKPT = "checkpoints/best_moist_vit.pth"
SAVE_CKPT = "checkpoints/best_moist_vit_pakistan_adapted.pth"

# New Real Flood Calibration Pack, produced by
# MoistMachine/real_floods_pipeline/. Falls back to the legacy 4-image set.
DEFAULT_PAK_DIR = "data/MoistMachine/real_floods"
LEGACY_PAK_DIR = "data/MoistMachine/pakistan_calibration"


def resolve_pak_dir(preferred):
    """Use the scraped pack when it has content, else the legacy hand-drawn set."""
    p = sorted(glob.glob(os.path.join(preferred, "Images", "*.png")))
    if p:
        return preferred
    print(f"[warn] no images under {preferred!r}, falling back to {LEGACY_PAK_DIR!r}")
    return LEGACY_PAK_DIR


class PakistanCalibrationDataset(Dataset):
    def __init__(self, repeats=30, img_size=512, pak_dir=None, replay_dir=None, replay=40):
        self.img_size = img_size
        pak_dir = pak_dir or resolve_pak_dir(DEFAULT_PAK_DIR)
        pak_img_paths = sorted(glob.glob(os.path.join(pak_dir, "Images", "*.png")))
        pak_mask_paths = sorted(glob.glob(os.path.join(pak_dir, "Masks", "*.png")))

        if not pak_img_paths:
            raise FileNotFoundError(f"no calibration images under {pak_dir!r}")

        # Pair by stem; sorted-zip silently mismatches if one file is missing.
        pak_by = {os.path.splitext(os.path.basename(p))[0]: p for p in pak_img_paths}
        msk_by = {os.path.splitext(os.path.basename(p))[0]: p for p in pak_mask_paths}
        stems = sorted(set(pak_by) & set(msk_by))
        if len(stems) != len(pak_img_paths):
            print(f"[warn] {len(pak_img_paths) - len(stems)} unpaired calibration "
                  f"file(s) dropped from {pak_dir}")

        # Replay samples from base dataset
        replay_dir = replay_dir or "data/MoistMachine/processed"
        base_img_paths = sorted(glob.glob(os.path.join(replay_dir, "Images", "*.png")))
        base_mask_paths = sorted(glob.glob(os.path.join(replay_dir, "Masks", "*.png")))
        base_by = {os.path.splitext(os.path.basename(p))[0]: p for p in base_img_paths}
        bmsk_by = {os.path.splitext(os.path.basename(p))[0]: p for p in base_mask_paths}
        bstems = sorted(set(base_by) & set(bmsk_by))[:replay]

        # Oversample Pakistan images to give strong domain gradient
        self.pairs = []
        for _ in range(repeats):
            for s in stems:
                self.pairs.append((pak_by[s], msk_by[s], True))

        for s in bstems:
            self.pairs.append((base_by[s], bmsk_by[s], False))

        self.n_pak = len(stems) * repeats
        print(f"[data] {pak_dir}: {len(stems)} frames x{repeats} + "
              f"{len(bstems)} replay = {len(self.pairs)} samples")

        self.aug = A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.6, 1.0), ratio=(0.85, 1.15)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.08, p=0.6),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp, is_pak = self.pairs[idx]
        img = cv2.imread(ip)
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.aug(image=rgb, mask=mask)

        flood_pixels = (res['mask'] == 1) | (res['mask'] == 3) | (res['mask'] == 5)
        threat = torch.tensor([float((flood_pixels > 0).float().mean())], dtype=torch.float32)

        return res['image'], res['mask'].long(), threat

def run_fine_tuning(epochs=12, batch_size=8, lr=1.2e-4, pak_dir=None,
                    base_ckpt=BASE_CKPT, save_ckpt=SAVE_CKPT, repeats=30,
                    replay=40, export=True):
    print("=" * 70)
    print(">> Pakistan Flood Domain Calibration & Fine-Tuning")
    print(f"   Base Model: {base_ckpt}")
    print(f"   Target    : {save_ckpt}")
    print("=" * 70)

    ds = PakistanCalibrationDataset(repeats=repeats, img_size=512,
                                    pak_dir=pak_dir, replay=replay)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    print(f"Calibration batch size: {len(ds)} samples per epoch.")

    model = MoistViT(num_classes=10, backbone_name="efficientvit_b2", fpn_dim=128, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(base_ckpt, map_location=DEVICE))

    # Boost water weight for Pakistan silt adaptation
    weights = CLASS_WEIGHTS.clone()
    weights[5] = 4.0 # Water weight boost
    weights[1] = 4.0 # Flooded building boost
    criterion = MoistFloodLoss(weights=weights)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler("cuda")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for imgs, targets, threats in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            threats = threats.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                seg_logits, threat_pred = model(imgs)
                loss, focal, dice = criterion(seg_logits, threat_pred, targets, threats)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f"  [Epoch {epoch:02d}/{epochs:02d}] Calibration Loss: {avg_loss:.4f}")

    # Save calibrated weights
    torch.save(model.state_dict(), save_ckpt)
    print(f"\n[OK] Pakistan-Calibrated Model Saved: {save_ckpt}")

    if export:
        # Also export mobile versions
        export_models(model, img_size=512)


def build_parser():
    p = argparse.ArgumentParser(
        description="Fine-tune MoistViT on the real-world flood calibration pack.",
    )
    p.add_argument("--data_dir", default=DEFAULT_PAK_DIR,
                   help="calibration pack root containing Images/ and Masks/ "
                        f"(default: {DEFAULT_PAK_DIR})")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1.2e-4)
    p.add_argument("--repeats", type=int, default=30,
                   help="oversample factor for the calibration frames")
    p.add_argument("--replay", type=int, default=40,
                   help="number of base-domain frames mixed in to limit forgetting")
    p.add_argument("--base_ckpt", default=BASE_CKPT)
    p.add_argument("--save_ckpt", default=SAVE_CKPT)
    p.add_argument("--no_export", action="store_true",
                   help="skip the mobile export step")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_fine_tuning(
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        pak_dir=args.data_dir,
        base_ckpt=args.base_ckpt,
        save_ckpt=args.save_ckpt,
        repeats=args.repeats,
        replay=args.replay,
        export=not args.no_export,
    )
