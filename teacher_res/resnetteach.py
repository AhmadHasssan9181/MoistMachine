import os
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- IMPORT ---
from flood_dataset import FloodNetDataset

# --- CONFIGURATION ---
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = 10
ACTIVATION = None
DEVICE = 'cuda'

# --- THE FIX: 512px ---
IMG_SIZE = 512
BATCH_SIZE = 8  # Lowered to 8 to handle 512px
EPOCHS = 50  # 50 is plenty for fine-tuning
LR = 0.0001

# --- PATHS ---
DATA_ROOT = r"H:\WAFFLE_FINAL_REAL"
LAB_IMG = os.path.join(DATA_ROOT, "images", "train")
LAB_MASK = os.path.join(DATA_ROOT, "masks", "train")
VAL_IMG = os.path.join(DATA_ROOT, "images", "val")
VAL_MASK = os.path.join(DATA_ROOT, "masks", "val")


# --- AUGMENTATIONS (512px) ---
def get_training_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=15, shift_limit=0.1, p=0.7, border_mode=0),
        # Ensure 512x512
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, p=1.0, border_mode=0),
        A.RandomCrop(height=IMG_SIZE, width=IMG_SIZE, p=1.0),

        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),

        A.OneOf([
            A.CLAHE(p=1),
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1),
        ], p=0.5),

        A.Normalize(),
        ToTensorV2()
    ])


def get_validation_augmentation():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(),
        ToTensorV2()
    ])


def train():
    print(f"--- Loading Data (ResNet @ {IMG_SIZE}x{IMG_SIZE}) ---")

    # cache_mode=True is usually fine for 400 images even at 512px
    train_dataset = FloodNetDataset(
        LAB_IMG, LAB_MASK,
        img_size=IMG_SIZE,
        transforms=get_training_augmentation(),
        cache_mode=True
    )

    valid_dataset = FloodNetDataset(
        VAL_IMG, VAL_MASK,
        img_size=IMG_SIZE,
        transforms=get_validation_augmentation(),
        cache_mode=True
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"--- Building Teacher Model ({ENCODER}) ---")
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=CLASSES,
        activation=ACTIVATION
    ).to(DEVICE)

    # Losses
    dice_loss_fn = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
    focal_loss_fn = smp.losses.FocalLoss(mode='multiclass', ignore_index=255)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()

    best_loss = 100.0

    print(" Starting ResNet-512 Training...")

    for i in range(EPOCHS):
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {i + 1}/{EPOCHS}")
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            with autocast(enabled=True):
                mask_pred = model(x)
                d_loss = dice_loss_fn(mask_pred, y)
                f_loss = focal_loss_fn(mask_pred, y)
                loss = d_loss + f_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                mask_pred = model(x)
                d_loss = dice_loss_fn(mask_pred, y)
                f_loss = focal_loss_fn(mask_pred, y)
                val_loss += (d_loss + f_loss).item()

        avg_val_loss = val_loss / len(valid_loader)
        print(f'Train Loss: {train_loss / len(train_loader):.4f} | Valid Loss: {avg_val_loss:.4f}')

        scheduler.step()

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            # Save as a distinct 512 version
            torch.save(model.state_dict(), 'best_teacher_resnet_512.pth')
            print('>>> Saved Best ResNet-512 Model')


if __name__ == '__main__':
    train()