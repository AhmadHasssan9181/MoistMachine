import os
from pathlib import Path
from tqdm import tqdm

# Point to your SSD dataset
DATA_ROOT = Path(r"H:\WAFFLE_Optimized_512")


def clean_subset(subset_name):
    img_dir = DATA_ROOT / "images" / subset_name
    mask_dir = DATA_ROOT / "masks" / subset_name

    print(f"--- Cleaning {subset_name} ---")

    #deleted the files from the final dataset files from loveda and rescuenet
    prefixes_to_kill = ("rescue_", "loveda_")

    deleted_count = 0
    kept_count = 0

    # Check Images
    if img_dir.exists():
        files = list(img_dir.glob("*.*"))
        for f in tqdm(files):
            if f.name.startswith(prefixes_to_kill):
                os.remove(f)
                deleted_count += 1
            else:
                kept_count += 1

    # Check Masks
    if mask_dir.exists():
        files = list(mask_dir.glob("*.*"))
        for f in files:
            if f.name.startswith(prefixes_to_kill):
                os.remove(f)

    print(f"Result: Deleted {deleted_count} intruder files.")
    print(f"Remaining: {kept_count} original FloodNet files.")


if __name__ == "__main__":
    clean_subset("train")
    print("\n Dataset restored to FloodNet Only.")