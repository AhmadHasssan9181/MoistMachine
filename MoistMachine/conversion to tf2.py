import torch
import torch.nn as nn
import os
from model_multihead_mbv3 import MultiHeadFloodNetLite

# CONFIG
IMG_SIZE = 256
NUM_CLASSES = 10
CKPT = "checkpoints/best_fp32.pth"
OUT_ONNX = "export/waffle_segmentation.onnx"


class TFLiteWrapper(nn.Module):
    """
    Wraps the model to return ONLY the segmentation map.
    TFLite hates complex dictionary outputs, so we give it a simple tensor.
    """

    def __init__(self, core_model):
        super().__init__()
        self.core = core_model

    def forward(self, x):
        # We only care about the first output (seg_logits)
        seg_logits, _, _ = self.core(x)
        return seg_logits


def main():
    device = "cpu"
    core = MultiHeadFloodNetLite(num_classes=NUM_CLASSES, pretrained=False, k_clusters=0).to(device)
    core.load_state_dict(torch.load(CKPT, map_location=device))
    core.eval()

    # 2. Wrap it (Strip extra heads)
    model = TFLiteWrapper(core)

    # 3. Dummy Input (Standard NCHW)
    # Don't worry about NHWC here. The converter tool handles it,will come up with a better solution later
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)

    # 4. Export
    os.makedirs("export", exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        OUT_ONNX,
        opset_version=12,
        input_names=['input_image'],  # Clear name for Android
        output_names=['segmentation'],  # Clear name for Android
        dynamic_axes=None  # STATIC SHAPE IS MANDATORY FOR TFLITE GPU
    )
    print(f"Ready for Conversion: {OUT_ONNX}")


if __name__ == "__main__":
    main()