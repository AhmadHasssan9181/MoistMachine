import os, torch
import torch.nn.functional as F
from model_multihead_mbv3 import MultiHeadFloodNetLite

# Export wrapper to produce NHWC outputs (good for TFLite)
class NHWCWrapper(torch.nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core
    def forward(self, x):
        # x: NCHW
        seg, cluster, pooled = self.core(x)
        # convert seg (N,C,H,W) -> (N,H,W,C) for TFLite friendliness
        seg_nhwc = seg.permute(0, 2, 3, 1).contiguous()
        # cluster may be None
        if cluster is not None:
            cluster_nhwc = cluster.permute(0, 2, 3, 1).contiguous()
        else:
            cluster_nhwc = torch.empty(1)  # placeholder
        return seg_nhwc, cluster_nhwc, pooled  # pooled is [N,C]

IMG_SIZE = 256
NUM_CLASSES = 10
CKPT = "checkpoints/best_fp32.pth"
OUT_ONNX = "export/seg_mobile_nhwc.onnx"

def main():
    device = "cpu"
    core = MultiHeadFloodNetLite(num_classes=NUM_CLASSES, pretrained=False, k_clusters=0).to(device)
    core.load_state_dict(torch.load(CKPT, map_location="cpu"))
    core.eval()
    model = NHWCWrapper(core)

    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    os.makedirs("export", exist_ok=True)

    torch.onnx.export(
        model, x, OUT_ONNX,
        opset_version=12,
        input_names=['input_nchw'],
        output_names=['seg_nhwc', 'cluster_nhwc', 'pooled'],
        dynamic_axes=None  # static for TFLite
    )
    print("Saved ONNX NHWC:", OUT_ONNX)

if __name__ == "__main__":
    main()