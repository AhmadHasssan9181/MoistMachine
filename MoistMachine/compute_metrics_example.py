import torch
from torch.utils.data import DataLoader
import numpy as np
from dataset import FloodNetDataset
from model_multihead import MultiHeadFloodNet
from utils import compute_confusion_matrix, compute_miou

NUM_CLASSES = 10
IMG_SIZE = 512
BATCH = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VAL_IMG = "data/images/val"
VAL_MASK = "data/masks/val"

def main():
    loader = DataLoader(FloodNetDataset(VAL_IMG, VAL_MASK, img_size=IMG_SIZE), batch_size=BATCH, shuffle=False, num_workers=4)
    model = MultiHeadFloodNet(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load("checkpoints/best_fp32.pth", map_location='cpu'))
    model.eval()
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(DEVICE); masks = masks.to(DEVICE)
            seg_logits, _, _ = model(imgs)
            preds = torch.argmax(seg_logits, dim=1).cpu().numpy()
            labs = masks.cpu().numpy()
            hist += compute_confusion_matrix(preds, labs, NUM_CLASSES)
    miou, per_class = compute_miou(hist)
    print("mIoU:", miou)
    for i, v in enumerate(per_class): print(i, v)

if __name__ == "__main__":
    main()