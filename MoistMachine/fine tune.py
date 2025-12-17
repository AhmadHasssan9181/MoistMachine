import os, torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn, torch.optim as optim
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm


from dataset import FloodNetDataset
from model_multihead_mbv3 import MultiHeadFloodNetLite
from utils import compute_confusion_matrix, compute_miou

# --- FINE-TUNING CONFIGURATION ---
NUM_CLASSES = 10
IMG_SIZE = 256
BATCH = 32  # Kept high for stability
EPOCHS = 10  # Short run (just polishing)
LR = 1e-3  # LOW LR: Critical for fine-tuning (don't break the weights!)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IGNORE_INDEX = 255
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
USE_AMP = True


DATA_ROOT = r"H:\WAFFLE_Optimized_512"


PRETRAINED_PATH = r"C:\Users\Ahmad\PycharmProjects\PythonProject\MoistMachine\checkpoints\megamodel.pth"

LAB_IMG = os.path.join(DATA_ROOT, "images", "train")
LAB_MASK = os.path.join(DATA_ROOT, "masks", "train")
VAL_IMG = os.path.join(DATA_ROOT, "images", "val")
VAL_MASK = os.path.join(DATA_ROOT, "masks", "val")


# ---------------------------

def get_loaders():
    print("Setting up datasets for Fine-Tuning...")

    # 1. Labeled Training Data (FloodNet Only)
    ds_labeled = FloodNetDataset(LAB_IMG, LAB_MASK, img_size=IMG_SIZE, ignore_index=IGNORE_INDEX)

    print(f"Dataset Size: {len(ds_labeled)} images")

    train_loader = DataLoader(
        ds_labeled,
        batch_size=BATCH,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # 2. Validation Data
    val_ds = FloodNetDataset(VAL_IMG, VAL_MASK, img_size=IMG_SIZE, ignore_index=IGNORE_INDEX)
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader


# --- WEIGHTED LOSS (Keep the punishment!) ---
class_weights = torch.tensor([
    1.0,  # 0: Back
    8.0,  # 1: Build-Flood (Slightly relaxed from 10.0)
    2.0,  # 2: Build-Dry
    8.0,  # 3: Road-Flood
    2.0,  # 4: Road-Dry
    5.0,  # 5: Water
    1.0,  # 6: Tree
    2.0,  # 7: Vehicle
    1.0,  # 8: Pool
    1.0  # 9: Grass
]).to(DEVICE)


def train():

    model = MultiHeadFloodNetLite(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)


    if os.path.exists(PRETRAINED_PATH):
        print(f"🚀Loading Mega-Model from: {PRETRAINED_PATH}")
        state_dict = torch.load(PRETRAINED_PATH, map_location=DEVICE)

        # Load weights, ignoring mismatches if strict=False (though yours should match perfectly)
        try:
            model.load_state_dict(state_dict)
            print(" Weights loaded successfully!")
        except Exception as e:
            print(f"⚠ Error loading weights: {e}")
            return
    else:
        print(f" ERROR: Weights not found at {PRETRAINED_PATH}")
        return

    # 3. Setup Optimizer
    criterion_sup = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()

    train_loader, val_loader = get_loaders()
    best_miou = 0.0

    print(f"Starting Fine-Tuning on {DEVICE} for {EPOCHS} epochs...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Fine-Tune {epoch}/{EPOCHS}")

        for imgs, masks in pbar:
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=USE_AMP):
                seg_logits, _, _ = model(imgs)
                loss = criterion_sup(seg_logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()

        # Save latest
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "finetuned_latest.pth"))

        # Evaluate
        miou, _ = evaluate(model, val_loader)
        print(f"Epoch {epoch} | Val mIoU: {miou:.4f}")

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_finetuned.pth"))
            print(">>> Saved New Best Fine-Tuned Model!")


def evaluate(model, loader):
    model.eval()
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            seg_logits, _, _ = model(imgs)
            preds = torch.argmax(seg_logits, dim=1).cpu().numpy()
            labs = masks.cpu().numpy()
            hist += compute_confusion_matrix(preds, labs, NUM_CLASSES, ignore_index=IGNORE_INDEX)
    miou, per_class = compute_miou(hist)
    return miou, per_class


if __name__ == "__main__":
    train()