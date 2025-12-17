import os, cv2, numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import pathlib


class FloodNetDataset(Dataset):
    def __init__(self, images_dir, masks_dir=None, img_size=256, transforms=None, classes=None, ignore_index=255,
                 cache_mode=True):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.classes = classes
        self.ignore_index = ignore_index
        self.cache_mode = cache_mode
        self.cached_data = []

        # Auto-detect validation
        is_val = False
        if images_dir and ('val' in str(images_dir).lower() or 'test' in str(images_dir).lower()):
            is_val = True

        self.transforms = transforms or self.get_transforms(img_size, is_val)

        # 1. Load file paths
        exts = ('.png', '.jpg', '.jpeg')
        if images_dir and os.path.exists(images_dir):
            self.image_paths = sorted([p for p in pathlib.Path(images_dir).iterdir() if p.suffix.lower() in exts])
        else:
            self.image_paths = []

        if masks_dir and os.path.exists(masks_dir):
            self.mask_paths = sorted([p for p in pathlib.Path(masks_dir).iterdir() if p.suffix.lower() in exts])
        else:
            self.mask_paths = []

        # 2. CACHE EVERYTHING INTO RAM (RESIZED)
        if self.cache_mode and len(self.image_paths) > 0:
            print(
                f" Caching {len(self.image_paths)} images from {os.path.basename(str(images_dir))} to RAM (Resized to {img_size}px)...")
            for i in tqdm(range(len(self.image_paths))):
                # We load AND RESIZE immediately here to save RAM
                img, mask = self.load_and_resize(i, self.img_size)
                self.cached_data.append((img, mask))
            print(" Caching Complete. RAM usage optimized.")

    def get_transforms(self, img_size, is_val=False):
        if is_val:
            return A.Compose([
                # Images are already resized in cache, but this ensures safety
                A.Resize(height=img_size, width=img_size),
                A.Normalize(),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                # Note: Since we pre-resized to 256, random scaling is less effective
                # but essential for texture variation.
                A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, value=0,
                              mask_value=self.ignore_index),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussNoise(p=0.2),
                A.Normalize(),
                ToTensorV2()
            ])

    def load_and_resize(self, idx, target_size):
        # 1. Read Image
        img_path = str(self.image_paths[idx])
        img = cv2.imread(img_path)
        if img is None:
            return None, None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. IMMEDIATE RESIZE (The RAM Saver)
        # We resize to target_size immediately so we don't store 4K images
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        # 3. Read Mask
        mask = None
        if idx < len(self.mask_paths):
            mask_path = str(self.mask_paths[idx])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # Resize mask to match
                mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
                mask = self.remap_mask(mask)

        if mask is None:
            mask = np.full((target_size, target_size), fill_value=self.ignore_index, dtype=np.uint8)

        return img, mask

    def load_file_raw(self, idx):
        # Fallback for non-cached mode (loads full res)
        img_path = str(self.image_paths[idx])
        img = cv2.imread(img_path)
        if img is None: return None, None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = None
        if idx < len(self.mask_paths):
            mask_path = str(self.mask_paths[idx])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if mask.shape[:2] != img.shape[:2]:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask = self.remap_mask(mask)
        if mask is None:
            mask = np.full((img.shape[0], img.shape[1]), fill_value=self.ignore_index, dtype=np.uint8)
        return img, mask

    def remap_mask(self, mask):
        mask = mask.astype(np.uint8)
        if self.classes is None: return mask
        out = np.full_like(mask, fill_value=self.ignore_index, dtype=np.uint8)
        for i, cls in enumerate(self.classes):
            out[mask == int(cls)] = i
        return out

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.cache_mode:
            img, mask = self.cached_data[idx]
            if img is None: return self.__getitem__(0)
        else:
            # Fallback to slow disk load
            img, mask = self.load_file_raw(idx)
            if img is None: return self.__getitem__(idx - 1 if idx > 0 else 0)

        # Apply transforms
        aug = self.transforms(image=img, mask=mask)
        return aug['image'], aug['mask'].long()