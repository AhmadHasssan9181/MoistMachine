import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model_multihead_mbv3 import MultiHeadFloodNetLite

# --- CONFIGURATION ---
MODEL_PATH = r"C:\Users\Ahmad\PycharmProjects\PythonProject\MoistMachine\checkpoints_refined\best_balanced_model.pth"
IMAGE_PATH = r"C:\Users\Ahmad\PycharmProjects\PythonProject\images\provewajiwrong.png"
IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- FLOODNET COLOR PALETTE (BGR Format for OpenCV) ---
# We map class IDs (0-9) to colors
COLORS = np.array([
    [0, 0, 0],  # 0: Background (Black)
    [0, 0, 255],  # 1: Building Flooded (Red)
    [128, 64, 128],  # 2: Building Non-Flooded (Purple)
    [255, 0, 255],  # 3: Road Flooded (Magenta)
    [128, 128, 128],  # 4: Road Non-Flooded (Grey)
    [255, 255, 0],  # 5: Water (Cyan/Blue) <--- LOOK FOR THIS
    [0, 255, 0],  # 6: Tree (Green)
    [0, 255, 255],  # 7: Vehicle (Yellow)
    [255, 128, 0],  # 8: Pool (Orange)
    [0, 128, 0]  # 9: Grass (Dark Green)
], dtype=np.uint8)


def preprocess_image(img_path):
    # Load and resize
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Could not find image: {img_path}")

    original_image = image.copy()  # Keep for display
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Same transforms as validation
    transform = A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.Normalize(),
        ToTensorV2()
    ])

    tensor = transform(image=image)['image'].unsqueeze(0)  # Add batch dim [1, 3, H, W]
    return tensor.to(DEVICE), original_image


def visualize_prediction(orig_img, pred_mask):
    # Resize prediction back to original image size for better view
    h, w = orig_img.shape[:2]

    # Map class IDs to Colors
    # pred_mask is [256, 256]. We want [256, 256, 3]
    color_mask = COLORS[pred_mask]

    # Resize mask to match original image
    color_mask_resized = cv2.resize(color_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Blend: 60% Original Image + 40% Mask
    overlay = cv2.addWeighted(orig_img, 0.6, color_mask_resized, 0.4, 0)

    return overlay


def main():
    import os  # Ensure os is imported
    print(f"Loading model from {MODEL_PATH}...")

    # Initialize Model
    model = MultiHeadFloodNetLite(num_classes=10, pretrained=False, k_clusters=10).to(DEVICE)

    # Load Weights
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Processing image: {IMAGE_PATH}")
    input_tensor, orig_img = preprocess_image(IMAGE_PATH)

    with torch.no_grad():
        seg_logits, _, _ = model(input_tensor)
        preds = torch.argmax(seg_logits, dim=1).squeeze().cpu().numpy()

    # Create visualization
    result = visualize_prediction(orig_img, preds)

    # Stack Original and Result Side-by-Side for cool comparison
    h, w = 512, 512
    display_orig = cv2.resize(orig_img, (w, h))
    display_res = cv2.resize(result, (w, h))
    combined = np.hstack((display_orig, display_res))

    # SAVE to disk instead of showing
    output_filename = "waffle_result.jpg"
    cv2.imwrite(output_filename, combined)

    print("-" * 30)
    print(f"SUCCESS! Check your folder for '{output_filename}'")
    print(f"Full path: {os.path.abspath(output_filename)}")
    print("-" * 30)

if __name__ == "__main__":
    main()