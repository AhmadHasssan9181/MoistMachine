import os
import cv2
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
# Source: The Big Slow Dataset on D:
SOURCE_ROOT = Path(r"D:\big aahhh datasets\WAFFLE_Final_Ready")

# Destination: Optimized Dataset on H:
DEST_ROOT = Path(r"H:\WAFFLE_Optimized_512")
TARGET_SIZE = 512


def process_subset(subset_name):
    # Setup paths
    src_img_dir = SOURCE_ROOT / "images" / subset_name
    src_mask_dir = SOURCE_ROOT / "masks" / subset_name

    dst_img_dir = DEST_ROOT / "images" / subset_name
    dst_mask_dir = DEST_ROOT / "masks" / subset_name

    # Create Destination folders
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_mask_dir.mkdir(parents=True, exist_ok=True)

    # Get files
    if not src_img_dir.exists():
        print(f"Skipping {subset_name} (Not found)")
        return

    files = list(src_img_dir.glob("*.*"))
    print(f"🚀 Optimizing {subset_name}: {len(files)} images...")

    for f in tqdm(files):
        # 1. Read Image
        img = cv2.imread(str(f))
        if img is None: continue

        # 2. Resize Image (Linear for photos)
        # 512x512 is small enough to load fast, big enough to augment
        img_small = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)

        # 3. Save as JPG (Quality 95) - Much faster to read than PNG
        save_name = f.stem + ".jpg"
        cv2.imwrite(str(dst_img_dir / save_name), img_small, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # 4. Process Matching Mask
        mask_path = src_mask_dir / (f.stem + ".png")
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # CRITICAL: Nearest Neighbor for masks (Keep class IDs integer)
                mask_small = cv2.resize(mask, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)

                # Masks must stay PNG (Lossless)
                cv2.imwrite(str(dst_mask_dir / (f.stem + ".png")), mask_small)


if __name__ == "__main__":
    print(f"--- WAFFLE DATASET OPTIMIZER ---")
    print(f"Source (D:): {SOURCE_ROOT}")
    print(f"Target (H:): {DEST_ROOT}")
    print(f"Target Size: {TARGET_SIZE}x{TARGET_SIZE}")
    print("--------------------------------")

    process_subset("train")
    process_subset("val")
    # process_subset("unlabeled") # Uncomment if needed

    print("\n DONE! Optimized dataset is ready on H: drive.")
    print(f"New Path: {DEST_ROOT}")
