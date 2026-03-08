import torch, cv2, os, numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- CONFIG ---
# --- CONFIG ---
CKPT_SEGFORMER = "best_teacher_segformer.pth" # <--- Must be the SegFormer file
CKPT_RESNET = "best_teacher_resnet_512.pth"   # <--- Must be the ResNet file"
# CHANGE THIS TO YOUR IMAGE PATH 👇
TEST_IMG_PATH = r"C:\Users\Ahmad\PycharmProjects\PythonProject\images\img.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# NEW TEST THRESHOLD: 0.70 (Standard for pseudo-labeling)
CONFIDENCE_THRESHOLD = 0.15

IMG_SIZE_SEG = 512
IMG_SIZE_RES = 512


def test():
    if not os.path.exists(TEST_IMG_PATH):
        print(f" Error: Image not found at {TEST_IMG_PATH}")
        return

    print(f"--- Testing Ensemble with Threshold {CONFIDENCE_THRESHOLD} ---")

    # 1. Load Teachers
    print("Loading Teacher 1 (SegFormer)...")
    model_seg = smp.Unet(
        encoder_name='mit_b3', classes=10, activation=None,
        decoder_channels=(256, 128, 64, 32, 16)
    ).to(DEVICE)
    try:
        model_seg.load_state_dict(torch.load(CKPT_SEGFORMER))
    except Exception as e:
        print(f" SegFormer checkpoint missing or error: {e}")
        return
    model_seg.eval()

    print("Loading Teacher 2 (ResNet34)...")
    model_res = smp.DeepLabV3Plus(
        encoder_name='resnet34', classes=10, activation=None
    ).to(DEVICE)
    try:
        model_res.load_state_dict(torch.load(CKPT_RESNET))
    except Exception as e:
        print(f" ResNet checkpoint missing or error: {e}")
        return
    model_res.eval()

    # 2. Load Image
    img = cv2.imread(TEST_IMG_PATH)
    if img is None:
        print(" Failed to read image.")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]

    # Transform for SegFormer (512)
    transform_seg = A.Compose([A.Resize(IMG_SIZE_SEG, IMG_SIZE_SEG), A.Normalize(), ToTensorV2()])
    x_seg = transform_seg(image=img)['image'].unsqueeze(0).to(DEVICE)

    # 3. Predict
    with torch.no_grad():
        # --- Teacher 1 (SegFormer) ---
        l1 = model_seg(x_seg)
        p1 = torch.softmax(l1, dim=1)

        # --- Teacher 2 (ResNet) ---
        x_res = torch.nn.functional.interpolate(x_seg, size=IMG_SIZE_RES, mode='bilinear', align_corners=False)
        l2 = model_res(x_res)
        p2_small = torch.softmax(l2, dim=1)
        p2 = torch.nn.functional.interpolate(p2_small, size=IMG_SIZE_SEG, mode='bilinear', align_corners=False)

        # --- Average ---
        avg_prob = (p1 + p2) / 2.0

        # Get Winner and Confidence
        max_prob, preds = torch.max(avg_prob, dim=1)

        preds = preds.squeeze().cpu().numpy().astype(np.uint8)
        confidence = max_prob.squeeze().cpu().numpy()

    # 4. Apply Gating
    preds[confidence < CONFIDENCE_THRESHOLD] = 255

    # 5. Visualization
    mask_resized = cv2.resize(preds, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Define Colors
    colors = np.array([
        [0, 0, 0],  # 0: Background
        [255, 0, 0],  # 1: Build-Flood (Red)
        [0, 255, 0],  # 2: Build-Dry (Green)
        [128, 0, 128],  # 3: Road-Flood (Purple)
        [0, 0, 255],  # 4: Road-Dry (Blue)
        [0, 255, 255],  # 5: Water (Cyan)
        [128, 128, 0],  # 6: Tree (Olive)
        [255, 255, 0],  # 7: Vehicle (Yellow)
        [0, 128, 255],  # 8: Pool (Orangeish)
        [0, 128, 0]  # 9: Grass (Dark Green)
    ], dtype=np.uint8)

    # Handle Ignore (255) -> Gray
    viz_mask = mask_resized.copy()
    viz_mask[viz_mask == 255] = 10

    # Add Gray for index 10 AND FORCE UINT8
    gray_color = np.array([180, 180, 180], dtype=np.uint8)
    colors_with_ignore = np.vstack([colors, gray_color]).astype(np.uint8)

    colored_mask = colors_with_ignore[viz_mask]

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    colored_mask_bgr = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)

    # Overlay
    overlay = cv2.addWeighted(img_bgr, 0.6, colored_mask_bgr, 0.4, 0)

    save_path = f"test_result_{CONFIDENCE_THRESHOLD}.jpg"
    cv2.imwrite(save_path, overlay)
    print(f" Saved visualization to {save_path}")
    print("Gray areas = IGNORED (Unsure pixels)")


if __name__ == "__main__":
    test()