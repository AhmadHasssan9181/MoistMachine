import os, torch
from model_multihead_mbv3 import MultiHeadFloodNetLite

IMG_SIZE = 256   # or 320
NUM_CLASSES = 10
CKPT = "checkpoints/best_fp32.pth"  # well load qat one here later after all the experimentation is done
OUT_ONNX = "export/seg_mobile.onnx"

def main():
    device = "cpu"
    model = MultiHeadFloodNetLite(num_classes=NUM_CLASSES, pretrained=False, k_clusters=0).to(device)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    os.makedirs("export", exist_ok=True)

    torch.onnx.export(
        model, x, OUT_ONNX,
        opset_version=12,
        input_names=['input'],
        output_names=['seg', 'cluster', 'pooled'],
        dynamic_axes=None  # static for TFLite
    )
    print("Saved ONNX:", OUT_ONNX)

if __name__ == "__main__":
    main()