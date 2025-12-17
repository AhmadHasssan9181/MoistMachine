import os
import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm

# --- CONFIG ---
DEST_ROOT = Path(r"D:\big aahhh datasets\WAFFLE_Final_Ready")

# Only RescueNet this time
PAIRS = [
    (r"D:\big aahhh datasets\rescuenet\RescueNet\train\train-org-img",
     r"D:\big aahhh datasets\rescuenet\RescueNet\train\train-label-img"),
    (r"D:\big aahhh datasets\rescuenet\RescueNet\val\val-org-img",
     r"D:\big aahhh datasets\rescuenet\RescueNet\val\val-label-img"),
    (r"D:\big aahhh datasets\rescuenet\RescueNet\test\test-org-img",
     r"D:\big aahhh datasets\rescuenet\RescueNet\test\test-label-img")
]


def map_rescuenet(mask):
    # Same mapping logic as before
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


def process_rescue_fix():
    print("--- Fixing RescueNet ---")

    for img_dir_str, mask_dir_str in PAIRS:
        img_dir = Path(img_dir_str)
        mask_dir = Path(mask_dir_str)

        if not img_dir.exists(): continue

        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        print(f"📂 Scanning {img_dir.name} ({len(images)} files)...")

        count = 0
        for img_path in tqdm(images):
            possible_names = [
                img_path.stem + ".png",  # 1052.png
                img_path.stem + "_lab.png",  # 1052_lab.png
                img_path.stem + "_label.png",  # 1052_label.png
                img_path.stem + ".jpg",  # 1052.jpg (rare)
                img_path.name  # 1052.jpg
            ]

            mask_path = None
            for name in possible_names:
                candidate = mask_dir / name
                if candidate.exists():
                    mask_path = candidate
                    break

            if mask_path is None:

                continue

            # Process
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None: continue

            mapped_mask = map_rescuenet(mask)

            # Save with prefix
            prefix = f"rescue_{img_dir.parent.name}_{img_path.stem}"
            shutil.copy(img_path, DEST_ROOT / "images/train" / (prefix + img_path.suffix))
            cv2.imwrite(str(DEST_ROOT / "masks/train" / (prefix + ".png")), mapped_mask)
            count += 1

        print(f"Recovered {count} pairs from {img_dir.name}")


if __name__ == "__main__":
    process_rescue_fix()
    print("\n RescueNet is now fused")