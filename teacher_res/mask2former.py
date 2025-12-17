import torch, cv2, numpy as np
import os
from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor


HF_MODEL_ID = "facebook/mask2former-swin-large-cityscapes-semantic"
TEST_IMG_PATH = r"C:\Users\Ahmad\PycharmProjects\PythonProject\images\noflood.jpg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- MAPPING: Cityscapes -> FloodNet DRY ---
# We force "Building" -> "Build-Dry" (2)
# We force "Road" -> "Road-Dry" (4)
CLASS_MAP_DRY = {
    0: 4, 1: 4,  # Road/Sidewalk -> Road-Dry (4) (Blue)
    2: 2, 3: 2, 4: 2,  # Building/Wall/Fence -> Build-Dry (2) (Green)
    8: 6,  # Vegetation -> Tree (6) (Olive)
    9: 9,  # Terrain -> Grass (9) (Dark Green)
    11: 7, 12: 7, 13: 7,  # Vehicles -> Vehicle (7) (Yellow)

}


def test():
    if not os.path.exists(TEST_IMG_PATH):
        print(f" Error: Image not found at {TEST_IMG_PATH}")
        return

    print("--- Loading Foundation Model (Mask2Former) ---")
    try:
        processor = Mask2FormerImageProcessor.from_pretrained(HF_MODEL_ID)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(HF_MODEL_ID).to(DEVICE)
        model.eval()
    except Exception as e:
        print(f" Failed to download/load model: {e}")
        return

    # 1. Load Image
    print(f"Processing {os.path.basename(TEST_IMG_PATH)}...")
    image_pil = Image.open(TEST_IMG_PATH).convert("RGB")
    image_cv = np.array(image_pil)
    w, h = image_pil.size

    # 2. Predict (Get Shapes)
    with torch.no_grad():
        inputs = processor(images=image_pil, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)

        # Post-process to original size
        seg_map = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(h, w)]
        )[0].cpu().numpy()


    print("Applying 'Force Dry' Filter...")
    final_mask = np.full((h, w), 9, dtype=np.uint8)

    unique_ids = np.unique(seg_map)
    for uid in unique_ids:
        if uid in CLASS_MAP_DRY:
            target_id = CLASS_MAP_DRY[uid]
            final_mask[seg_map == uid] = target_id

    # 4. Visualize
    print(" Creating visualization...")
    # FloodNet Colors
    colors = np.array([
        [0, 0, 0],  # 0: Background
        [255, 0, 0],  # 1: Build-Flood
        [0, 255, 0],  # 2: Build-Dry (Target!)
        [128, 0, 128],  # 3: Road-Flood
        [0, 0, 255],  # 4: Road-Dry (Target!)
        [0, 255, 255],  # 5: Water
        [128, 128, 0],  # 6: Tree
        [255, 255, 0],  # 7: Vehicle
        [0, 128, 255],  # 8: Pool
        [0, 128, 0]  # 9: Grass
    ], dtype=np.uint8)

    colored_mask = colors[final_mask]
    img_bgr = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    colored_bgr = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(img_bgr, 0.6, colored_bgr, 0.4, 0)

    save_path = "test_force_dry_result.jpg"
    cv2.imwrite(save_path, overlay)
    print(f" Saved result to {save_path}")


if __name__ == "__main__":
    test()