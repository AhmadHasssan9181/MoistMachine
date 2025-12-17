import os, torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn, torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

from dataset import FloodNetDataset
from model_multihead_mbv3 import MultiHeadFloodNetLite
from utils import compute_confusion_matrix, compute_miou

# --- FINAL CONFIGURATION ---
NUM_CLASSES = 10
IMG_SIZE = 256
BATCH = 32  # Increased for 3060 Ti (was 8)
EPOCHS = 50
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IGNORE_INDEX = 255
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
USE_AMP = True

# --- NEW HDD PATHS ---
DATA_ROOT = r"H:\WAFFLE_Optimized_512"
LAB_IMG = os.path.join(DATA_ROOT, "images", "train")
LAB_MASK = os.path.join(DATA_ROOT, "masks", "train")
VAL_IMG = os.path.join(DATA_ROOT, "images", "val")
VAL_MASK = os.path.join(DATA_ROOT, "masks", "val")
UNLAB_IMG = os.path.join(DATA_ROOT, "images", "unlabeled")


# ---------------------------

def get_loaders():
    print("Setting up datasets...")

    # 1. Labeled Training Data
    # Note: If you want the "Photocopier Trick" (Oversampling),
    # you need to modify dataset.py's __init__ logic directly.
    # For now, the WEIGHTED LOSS below does the heavy lifting.
    ds_labeled = FloodNetDataset(LAB_IMG, LAB_MASK, img_size=IMG_SIZE, ignore_index=IGNORE_INDEX)

    # 2. Unlabeled Data (Semi-Supervised)
    if os.path.exists(UNLAB_IMG) and len(os.listdir(UNLAB_IMG)) > 0:
        print(f"Found Unlabeled data. Merging into training...")
        ds_unlab = FloodNetDataset(UNLAB_IMG, masks_dir=None, img_size=IMG_SIZE, ignore_index=IGNORE_INDEX)
        full_train_ds = ConcatDataset([ds_labeled, ds_unlab])
    else:
        full_train_ds = ds_labeled

    # HDD OPTIMIZATION: num_workers=8 to pre-load data faster
    train_loader = DataLoader(
        full_train_ds,
        batch_size=BATCH,
        shuffle=True,
        num_workers=4,  # Higher for HDD and lower for ssd i dont maybe im suffering from ram
        pin_memory=True,
        persistent_workers=True
    )

    # 3. Validation Data
    val_ds = FloodNetDataset(VAL_IMG, VAL_MASK, img_size=IMG_SIZE, ignore_index=IGNORE_INDEX)
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader


# --- THE PUNISHMENT TRICK (Weighted Loss) ---
# 0:Back, 1:Build-Flood, 2:Build-Dry, 3:Road-Flood, 4:Road-Dry
# 5:Water, 6:Tree, 7:Vehicle, 8:Pool, 9:Grass
class_weights = torch.tensor([
    1.0,  # 0: Back
    10.0,  # 1: Build-Flood (CRITICAL)
    2.0,  # 2: Build-Dry
    10.0,  # 3: Road-Flood (CRITICAL)
    2.0,  # 4: Road-Dry
    5.0,  # 5: Water (Important)
    1.0,  # 6: Tree
    2.0,  # 7: Vehicle
    1.0,  # 8: Pool
    1.0  # 9: Grass
]).to(DEVICE)

model = MultiHeadFloodNetLite(num_classes=NUM_CLASSES, pretrained=True, k_clusters=0).to(DEVICE)

# Apply Weights here
criterion_sup = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = GradScaler(enabled=USE_AMP)


def evaluate(model, loader):
    model.eval()
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    try:
        with torch.no_grad():
            for imgs, masks in loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)
                seg_logits, _, _ = model(imgs)
                preds = torch.argmax(seg_logits, dim=1).cpu().numpy()
                labs = masks.cpu().numpy()
                hist += compute_confusion_matrix(preds, labs, NUM_CLASSES, ignore_index=IGNORE_INDEX)
        miou, per_class = compute_miou(hist)
        return miou, per_class
    except Exception as e:
        print(f"\n[Warning] Validation crashed: {e}. Skipping this epoch's eval.")
        return 0.0, []


def train():
    train_loader, val_loader = get_loaders()
    best_miou = 0.0

    print(f"Starting training on {DEVICE} for {EPOCHS} epochs...")
    print(f"Batch Size: {BATCH} | Class Weights Active: YES")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        running_loss = 0.0

        for imgs, masks in pbar:
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP):
                seg_logits, _, _ = model(imgs)
                loss = criterion_sup(seg_logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        scheduler.step()

        # Save checkpoint every epoch just in case
        latest_path = os.path.join(SAVE_DIR, "latest_model.pth")
        torch.save(model.state_dict(), latest_path)

        miou, per_class = evaluate(model, val_loader)
        print(f"Epoch {epoch} | Val mIoU: {miou:.4f}")

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_fp32.pth"))
            print(">>> Saved New Best Model!")


if __name__ == "__main__":
    train()