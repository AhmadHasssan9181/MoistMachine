import torch, cv2, numpy as np
import os
from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- CONFIG ---
# 1. The Foundation (Shape Expert)
HF_MODEL_ID = "facebook/mask2former-swin-large-cityscapes-semantic"

# 2. Your Model (Status Expert)
MY_CKPT = "best_teacher_segformer.pth"

# 3. Test Image
TEST_IMG_PATH = r"C:\Users\Ahmad\PycharmProjects\PythonProject\images\7312.jpg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- FLOODNET CLASSES ---
# 0:Back, 1:Build-Flood, 2:Build-Dry, 3:Road-Flood, 4:Road-Dry
# 5:Water, 6:Tree, 7:Vehicle, 8:Pool, 9:Grass

def get_my_model():
    print("Loading Your SegFormer...")
    model = smp.Unet(
        encoder_name='mit_b3', classes=10, activation=None,
        decoder_channels=(256, 128, 64, 32, 16)
    ).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MY_CKPT))
    except Exception as e:
        print(f" Error loading your checkpoint: {e}")
        exit()
    model.eval()
    return model


def test():
    if not os.path.exists(TEST_IMG_PATH):
        print(f" Image not found: {TEST_IMG_PATH}")
        return

    # 1. Load Models
    print("Loading Foundation Model (Mask2Former)...")
    processor = Mask2FormerImageProcessor.from_pretrained(HF_MODEL_ID)
    foundation_model = Mask2FormerForUniversalSegmentation.from_pretrained(HF_MODEL_ID).to(DEVICE)
    foundation_model.eval()

    my_model = get_my_model()

    # 2. Load Image
    image_pil = Image.open(TEST_IMG_PATH).convert("RGB")
    image_cv = np.array(image_pil)
    orig_h, orig_w = image_cv.shape[:2]

    # --- STEP A: Foundation Prediction (Perfect Shapes) ---
    print("Foundation Model is analyzing shapes...")
    with torch.no_grad():
        inputs = processor(images=image_pil, return_tensors="pt").to(DEVICE)
        outputs = foundation_model(**inputs)
        # Get semantic map at original resolution
        foundation_mask = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(orig_h, orig_w)]
        )[0].cpu().numpy()

    # --- STEP B: Your Model Prediction (Wet vs Dry Status) ---
    print(" Your Model is analyzing flood status...")
    transform = A.Compose([A.Resize(512, 512), A.Normalize(), ToTensorV2()])
    x = transform(image=image_cv)['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = my_model(x)
        probs = torch.softmax(logits, dim=1)  # [1, 10, 512, 512]
        # Upscale probabilities to match original image
        probs = torch.nn.functional.interpolate(probs, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        probs = probs.squeeze(0).cpu().numpy()  # [10, H, W]

    # --- STEP C: The Fusion Logic (Refining) ---
    print(" Fusing models...")
    final_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    # Get unique objects found by Foundation model
    unique_ids = np.unique(foundation_mask)

    for cid in unique_ids:
        # Create mask for this specific object (e.g. all roads)
        region = (foundation_mask == cid)

        # Get average probabilities from YOUR model for this region
        # We assume if one part of a building is dry, the whole building is likely dry
        # (This removes "confetti" noise)
        region_probs = probs[:, region].mean(axis=1)  # Shape [10]

        # DECISION LOGIC:
        # Cityscapes IDs:
        # 0=Road, 1=Sidewalk -> FloodNet Road (3/4)
        # 2=Building, 3=Wall, 4=Fence -> FloodNet Building (1/2)
        # 8=Vegetation -> FloodNet Tree (6) or Grass (9) or Water (5)
        # 9=Terrain -> FloodNet Grass (9)
        # 11,12,13... -> Vehicles (7)

        final_class = 0

        if cid in [0, 1]:  # Road / Sidewalk
            # Force choice: Road-Flood (3) vs Road-Dry (4)
            final_class = 3 if region_probs[3] > region_probs[4] else 4

        elif cid in [2, 3, 4]:  # Building / Wall
            # Force choice: Build-Flood (1) vs Build-Dry (2)
            final_class = 1 if region_probs[1] > region_probs[2] else 2

        elif cid == 8:  # Vegetation
            # Could be Tree(6), Grass(9), or actually Water(5) (swamps)
            # Check if it looks like water first
            if region_probs[5] > 0.4:
                final_class = 5
            else:
                final_class = 6 if region_probs[6] > region_probs[9] else 9

        elif cid == 9:  # Terrain
            final_class = 9  # Grass

        elif cid in [11, 12, 13, 14, 15, 16]:  # Vehicles
            final_class = 7

        else:
            # Fallback: Just trust your model's winner
            final_class = region_probs.argmax()

        final_mask[region] = final_class

    # --- VISUALIZATION ---
    print(" Painting result...")
    colors = np.array([
        [0, 0, 0], [255, 0, 0], [0, 255, 0], [128, 0, 128], [0, 0, 255],
        [0, 255, 255], [128, 128, 0], [255, 255, 0], [0, 128, 255], [0, 128, 0]
    ], dtype=np.uint8)

    colored_mask = colors[final_mask]
    img_bgr = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    colored_bgr = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(img_bgr, 0.6, colored_bgr, 0.4, 0)
    cv2.imwrite("refined_result.jpg", overlay)
    print(" Saved 'refined_result.jpg'. Check it out!")


if __name__ == "__main__":
    test()