import os, torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn, torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

from dataset import FloodNetDataset
from model_multihead_mbv3 import MultiHeadFloodNetLite
from utils import compute_confusion_matrix, compute_miou

# --- CONFIGURATION ---
NUM_CLASSES = 10
IMG_SIZE = 256
BATCH = 40  # Increased for RAM speed
EPOCHS = 100  # Early Stopping will handle the end
LR = 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IGNORE_INDEX = 255
SAVE_DIR = "checkpoints_refined"
os.makedirs(SAVE_DIR, exist_ok=True)
USE_AMP = True
PATIENCE = 10

# --- PATHS ---
DATA_ROOT = r"H:\WAFFLE_FINAL_REAL"
LAB_IMG = os.path.join(DATA_ROOT, "images", "train")
LAB_MASK = os.path.join(DATA_ROOT, "masks", "train")
VAL_IMG = os.path.join(DATA_ROOT, "images", "val")
VAL_MASK = os.path.join(DATA_ROOT, "masks", "val")
UNLAB_IMG = os.path.join(DATA_ROOT, "images", "unlabeled")

# --- SWEET SPOT WEIGHTS ---
class_weights = torch.tensor([
    1.0, 3.0, 1.5, 3.0, 1.5, 3.0, 1.0, 2.0, 1.0, 1.0
]).to(DEVICE)


# --- EARLY STOPPING ---
class EarlyStopping:
    def __init__(self, patience=7, delta=0.001):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_miou, model, path):
        score = val_miou
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f' EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model, path)
            self.counter = 0

    def save_checkpoint(self, score, model, path):
        print(f' Validation mIoU improved ({self.best_score:.4f} --> {score:.4f}). Saving model...')
        torch.save(model.state_dict(), path)


def entropy_loss(prob):
    b = prob * torch.log(prob + 1e-10)
    b = -1.0 * b.sum(dim=1)
    return b.mean()


def get_loaders():
    print("--- Loading Datasets into RAM (Wait for it...) ---")

    # Cache Mode is ON
    ds_labeled = FloodNetDataset(LAB_IMG, LAB_MASK, img_size=IMG_SIZE, ignore_index=IGNORE_INDEX, cache_mode=True)

    if os.path.exists(UNLAB_IMG) and len(os.listdir(UNLAB_IMG)) > 0:
        ds_unlab = FloodNetDataset(UNLAB_IMG, masks_dir=None, img_size=IMG_SIZE, ignore_index=IGNORE_INDEX,
                                   cache_mode=True)
        full_ds = ConcatDataset([ds_labeled, ds_unlab])
        print(f"   Combined: {len(ds_labeled)} Labeled + {len(ds_unlab)} Unlabeled")
    else:
        full_ds = ds_labeled

    # num_workers=0 is FASTEST for RAM cached data
    train_loader = DataLoader(full_ds, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True)
    val_ds = FloodNetDataset(VAL_IMG, VAL_MASK, img_size=IMG_SIZE, ignore_index=IGNORE_INDEX, cache_mode=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader


def train():
    train_loader, val_loader = get_loaders()

    model = MultiHeadFloodNetLite(num_classes=NUM_CLASSES, pretrained=True, k_clusters=10).to(DEVICE)

    criterion_sup = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)

    # FIXED: Removed 'verbose=True'
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    scaler = GradScaler(enabled=USE_AMP)
    early_stopper = EarlyStopping(patience=PATIENCE)

    print(f" Starting Refined Semi-Supervised Training...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for imgs, masks in pbar:
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=USE_AMP):
                seg_logits, _, _ = model(imgs)
                probs = torch.softmax(seg_logits, dim=1)

                loss_sup = criterion_sup(seg_logits, masks)
                loss_ent = entropy_loss(probs)

                loss = loss_sup + (0.1 * loss_ent)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({'Sup': f"{loss_sup.item():.2f}", 'Ent': f"{loss_ent.item():.2f}"})

        miou, _ = evaluate(model, val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} | Val mIoU: {miou:.4f} | LR: {current_lr:.2e}")

        scheduler.step(miou)
        early_stopper(miou, model, os.path.join(SAVE_DIR, "best_balanced_model.pth"))

        if early_stopper.early_stop:
            print(" Early stopping triggered. Training finished.")
            break


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
    miou, _ = compute_miou(hist)
    return miou, []


if __name__ == "__main__":
    train()