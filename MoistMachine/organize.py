import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
SOURCE_ROOT = Path(r"H:\dataset\final_hopefully")
DEST_ROOT = Path(r"H:\WAFFLE_FINAL_REAL")


def clean_filename(fname):
    """
    Removes common suffixes to help match images to masks.
    e.g., 'image_01_lab.png' -> 'image_01'
    """
    stem = fname.stem
    # List of suffixes to strip if present
    suffixes = ["_lab", "_label", "_mask", "_gt"]
    for s in suffixes:
        if stem.endswith(s):
            return stem.replace(s, "")
    return stem


def organize():
    print(f"🚀 Starting Organization...")
    print(f"Source: {SOURCE_ROOT}")
    print(f"Destination: {DEST_ROOT}")

    # 1. Setup Directories
    dirs = {
        "train_img": DEST_ROOT / "images" / "train",
        "train_mask": DEST_ROOT / "masks" / "train",
        "val_img": DEST_ROOT / "images" / "val",
        "val_mask": DEST_ROOT / "masks" / "val",
        "unlab": DEST_ROOT / "images" / "unlabeled"
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # 2. Scanning Files
    print("   -> Scanning files...")
    all_files = list(SOURCE_ROOT.rglob("*"))

    images_map = {}  # {clean_name: path}
    masks_map = {}  # {clean_name: path}
    unlabeled_list = []

    # Specific folders you mentioned
    # We will treat 'Validation' as unlabeled if no mask is found,
    # but we will try to find masks just in case.

    for f in all_files:
        if f.is_file() and f.suffix.lower() in ['.jpg', '.png', '.jpeg', '.bmp']:
            parts = [p.lower() for p in f.parts]
            clean_name = clean_filename(f)

            # Explicit Unlabeled folders
            if "unlabeled" in parts or "test" in parts:
                unlabeled_list.append(f)
                continue

            # Potential Labeled Data
            # Check if it's a mask or image based on folder name or filename
            if "mask" in parts or "mask" in f.name.lower() or "_lab" in f.name.lower():
                masks_map[clean_name] = f
            else:
                # It's an image
                images_map[clean_name] = f

    # 3. Match Pairs
    # Only keep images that have a corresponding mask
    common_ids = list(set(images_map.keys()) & set(masks_map.keys()))

    # Identify orphans (Images in labeled folders but no mask found)
    # We move these to Unlabeled so we don't waste them!
    labeled_folder_images = set(images_map.keys())
    orphans = list(labeled_folder_images - set(common_ids))

    for o in orphans:
        unlabeled_list.append(images_map[o])

    print(f"✅ Found {len(common_ids)} valid Labeled Pairs.")
    print(f"📦 Found {len(unlabeled_list)} Unlabeled images (including orphans/test/val-without-mask).")

    # 4. Create Split (90% Train / 10% Val)
    random.shuffle(common_ids)
    split_idx = int(len(common_ids) * 0.90)
    train_ids = common_ids[:split_idx]
    val_ids = common_ids[split_idx:]

    print(f"   -> Training: {len(train_ids)}")
    print(f"   -> Validation: {len(val_ids)}")

    # 5. Move Files
    def copy_files(id_list, dest_img_dir, dest_mask_dir):
        for i in tqdm(id_list):
            src_img = images_map[i]
            src_mask = masks_map[i]

            # Copy Image
            shutil.copy2(src_img, dest_img_dir / src_img.name)

            # Copy Mask (Rename to match image exactly + .png)
            new_mask_name = src_img.stem + ".png"
            shutil.copy2(src_mask, dest_mask_dir / new_mask_name)

    print("   -> Copying Training Data...")
    copy_files(train_ids, dirs["train_img"], dirs["train_mask"])

    print("   -> Copying Validation Data...")
    copy_files(val_ids, dirs["val_img"], dirs["val_mask"])

    print("   -> Copying Unlabeled Data...")
    for f in tqdm(unlabeled_list):
        shutil.copy2(f, dirs["unlab"] / f.name)

    print("\n🎉 DONE! Dataset is ready at H:\\WAFFLE_FINAL_REAL")


if __name__ == "__main__":
    organize()