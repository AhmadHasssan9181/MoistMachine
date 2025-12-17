import os
import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
DEST_ROOT = Path(r"D:\big aahhh datasets\WAFFLE_Final_Ready")

RESCUENET_PAIRS = [
    # (Image Folder, Mask Folder)
    (r"D:\big aahhh datasets\rescuenet\RescueNet\train\train-org-img",
     r"D:\big aahhh datasets\rescuenet\RescueNet\train\train-label-img"),
    (r"D:\big aahhh datasets\rescuenet\RescueNet\val\val-org-img",
     r"D:\big aahhh datasets\rescuenet\RescueNet\val\val-label-img"),
    (r"D:\big aahhh datasets\rescuenet\RescueNet\test\test-org-img",
     r"D:\big aahhh datasets\rescuenet\RescueNet\test\test-label-img")
]

LOVEDA_PAIRS = [
    (r"D:\big aahhh datasets\loveda\Train\Train\Rural\images_png",
     r"D:\big aahhh datasets\loveda\Train\Train\Rural\masks_png"),
    (r"D:\big aahhh datasets\loveda\Train\Train\Urban\images_png",
     r"D:\big aahhh datasets\loveda\Train\Train\Urban\masks_png"),
    (r"D:\big aahhh datasets\loveda\Val\Val\Rural\images_png", r"D:\big aahhh datasets\loveda\Val\Val\Rural\masks_png"),
    (r"D:\big aahhh datasets\loveda\Val\Val\Urban\images_png", r"D:\big aahhh datasets\loveda\Val\Val\Urban\masks_png"),
    # Skipping Test because they usually don't have masks, well some had em but skipped
]


# --- MAPPING FUNCTIONS ---
def map_rescuenet(mask):
    # RescueNet: 1-Water, 2-NoDam, 3-Minor, 4-Major, 5-Destr, 6-Road, 7-Blocked, 8-Car, 9-Tree, 10-Pool
    new_mask = np.zeros_like(mask)
    new_mask[mask == 1] = 5  # Water
    new_mask[mask == 2] = 2  # Build-Dry
    new_mask[mask == 3] = 1  # Build-Flood
    new_mask[mask == 4] = 1  # Build-Flood
    new_mask[mask == 5] = 1  # Build-Flood
    new_mask[mask == 6] = 4  # Road-Dry
    new_mask[mask == 7] = 3  # Road-Flood
    new_mask[mask == 8] = 7  # Vehicle
    new_mask[mask == 9] = 6  # Tree
    new_mask[mask == 10] = 8  # Pool
    return new_mask


def map_loveda(mask):
    # LoveDA: 1-Back, 2-Build, 3-Road, 4-Water, 5-Barren, 6-Forest, 7-Agri
    new_mask = np.zeros_like(mask)
    new_mask[mask == 2] = 2  # Build-Dry
    new_mask[mask == 3] = 4  # Road-Dry
    new_mask[mask == 4] = 5  # Water
    new_mask[mask == 5] = 0  # Barren -> Back
    new_mask[mask == 6] = 6  # Forest -> Tree
    new_mask[mask == 7] = 9  # Agri -> Grass
    return new_mask


def process_pairs(name, pairs, map_func):
    print(f"\n--- Processing {name} ---")
    total_copied = 0

    for img_dir_str, mask_dir_str in pairs:
        img_dir = Path(img_dir_str)
        mask_dir = Path(mask_dir_str)

        if not img_dir.exists():
            print(f" ERROR: Missing Folder: {img_dir}")
            continue

        # Grab all valid image types
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))

        if len(images) == 0:
            print(f" Folder exists but is EMPTY: {img_dir}")
            continue

        print(f" Found {len(images)} images in {img_dir.name}")

        for img_path in tqdm(images, desc=img_dir.name):
            # Try to find the matching mask
            # RescueNet usually has .png masks for .jpg images
            mask_path = mask_dir / (img_path.stem + ".png")

            if not mask_path.exists():
                # Trying .jpg
                mask_path = mask_dir / (img_path.stem + ".jpg")
                if not mask_path.exists():
                    # Tryin original extension
                    mask_path = mask_dir / img_path.name

            if not mask_path.exists():
                # Skip silently (or uncomment print to debug)
                # print(f"Missing mask for {img_path.name}")
                continue

            # Read and Map
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None: continue

            mapped_mask = map_func(mask)

            # Destination Filename (Prevent overwriting)
            # e.g., rescue_train_image123.jpg
            prefix = f"{name}_{img_dir.parent.name}_{img_path.stem}"
            new_img_name = prefix + img_path.suffix
            new_mask_name = prefix + ".png"

            # Copy/Save
            shutil.copy(img_path, DEST_ROOT / "images/train" / new_img_name)
            cv2.imwrite(str(DEST_ROOT / "masks/train" / new_mask_name), mapped_mask)
            total_copied += 1

    print(f" Finished {name}: {total_copied} pairs added.")


if __name__ == "__main__":
    # Create Dirs
    (DEST_ROOT / "images/train").mkdir(parents=True, exist_ok=True)
    (DEST_ROOT / "masks/train").mkdir(parents=True, exist_ok=True)

    process_pairs("RescueNet", RESCUENET_PAIRS, map_rescuenet)
    process_pairs("LoveDA", LOVEDA_PAIRS, map_loveda)

    print("\n WAFFLE_Final_Ready/images/train folder count.")