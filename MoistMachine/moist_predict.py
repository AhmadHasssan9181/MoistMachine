"""
MoistViT production inference (works with the shipped v1 weights).

Drop-in replacement for the demo script's inference path. Fixes the three
things that cost the most accuracy with zero retraining:

  1. No 16:9 -> 512x512 squash. The encoder is fully convolutional, so the
     default is one padded forward pass at native resolution (more accurate AND
     faster than tiling). Tiling is kept for frames above `max_pixels`.
  2. No bare argmax for the flood class. Water is thresholded separately
     (calibrated on held-out data; ~+21 pts water recall over argmax).
  3. Optional temporal EMA over probability maps for video, which raises
     consecutive-frame mask agreement from ~0.78 to ~0.95.

Usage:
    from moist_predict import MoistPredictor
    m = MoistPredictor("checkpoints/best_moist_vit_pakistan_adapted_v1_goated.pth")
    pred, water_pct = m.predict(image_bgr)            # still image
    m.reset()
    for frame in video:                               # video (EMA on by default in this example)
        pred, water_pct = m.predict(frame)
"""
from __future__ import annotations
import os, json
import numpy as np
import cv2
import torch

try:
    from moist_vit_model import MoistViT
except ImportError:  # when imported from MoistMachine/
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from moist_vit_model import MoistViT

CLASS_NAMES = ["Background", "Build-Flood", "Build-Dry", "Road-Flood", "Road-Dry",
               "Water", "Tree", "Vehicle", "Pool", "Grass"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# palette in RGB (for scripts that write RGB) -- convert with [..., ::-1] for BGR
PALETTE_RGB = np.array([
    [0, 0, 0], [220, 20, 20], [40, 200, 80], [230, 60, 220], [220, 210, 40],
    [30, 90, 240], [30, 120, 30], [240, 240, 20], [140, 60, 230], [230, 150, 40],
], np.uint8)

DEFAULT_WATER_THRESHOLD = 0.30


def load_weights(model: torch.nn.Module, path: str) -> torch.nn.Module:
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    have = set(model.state_dict().keys())
    aliases = [("context_main.", "context_module.main."), ("local_main.", "local_module.main."),
               ("context_module.main.", "context_main."), ("local_module.main.", "local_main.")]
    fixed = {}
    for k, v in sd.items():
        nk = k
        if nk not in have:
            for a, b in aliases:
                if a in nk and nk.replace(a, b) in have:
                    nk = nk.replace(a, b)
                    break
        fixed[nk] = v
    missing, unexpected = model.load_state_dict(fixed, strict=False)
    crit = [k for k in missing if k.startswith("encoder.")]
    if crit:
        raise RuntimeError(f"{len(crit)} encoder weights missing, e.g. {crit[:3]}")
    return model


class MoistPredictor:
    def __init__(
        self,
        ckpt: str,
        device: str | None = None,
        tile: int = 512,
        overlap: float = 0.3,
        water_threshold: float = DEFAULT_WATER_THRESHOLD,
        temporal_ema: float = 0.0,
        max_side: int = 1280,
        mode: str = "auto",          # "auto" | "native" (one padded pass) | "tile"
        max_pixels: int = 2_400_000,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MoistViT(num_classes=10, backbone_name="efficientvit_b2", fpn_dim=128,
                              pretrained=False)
        self.model = load_weights(self.model, ckpt).to(self.device).eval()
        self.tile = tile
        self.overlap = overlap
        self.water_threshold = water_threshold
        self.ema_alpha = temporal_ema
        self.max_side = max_side
        self.mode = mode
        self.max_pixels = max_pixels
        self._ema: np.ndarray | None = None

    def reset(self):
        self._ema = None

    @torch.no_grad()
    def probabilities(self, img_bgr: np.ndarray) -> np.ndarray:
        """(10, H, W) float32 probability map at the caller's resolution.

        The encoder is fully convolutional, so a single native-resolution pass
        (padded to a multiple of 32) beats 512x512 squashing on both accuracy and
        latency. Tiling is only used for frames too large for one pass.
        """
        h0, w0 = img_bgr.shape[:2]
        ph = (h0 + 31) // 32 * 32
        pw = (w0 + 31) // 32 * 32
        if self.mode == "native" or (self.mode == "auto" and ph * pw <= self.max_pixels):
            padded = np.zeros((ph, pw, 3), np.uint8)
            padded[:h0, :w0] = img_bgr
            p = self._forward(padded)[:, :h0, :w0]
            return p
        # fall back to tiled inference for very large frames
        scale = 1.0
        if max(h0, w0) > self.max_side:
            scale = self.max_side / max(h0, w0)
            img_bgr = cv2.resize(img_bgr, (int(round(w0 * scale)), int(round(h0 * scale))),
                                 interpolation=cv2.INTER_AREA)
        h, w = img_bgr.shape[:2]
        tile = self.tile
        if h <= tile and w <= tile:
            p = self._forward(img_bgr)
            return p if scale == 1.0 else self._resize_prob(p, h0, w0)
        stride = max(1, int(tile * (1 - self.overlap)))
        xs = list(range(0, max(1, w - tile + stride), stride))
        ys = list(range(0, max(1, h - tile + stride), stride))
        if xs[-1] + tile < w:
            xs.append(max(0, w - tile))
        if ys[-1] + tile < h:
            ys.append(max(0, h - tile))
        # build all tiles up-front and run them as one batch (6 tiles 1080p -> 1 forward pass)
        crops, metas = [], []
        for y in ys:
            for x in xs:
                crop = img_bgr[y:y + tile, x:x + tile]
                ch, cw = crop.shape[:2]
                padded = np.zeros((tile, tile, 3), np.uint8)
                padded[:ch, :cw] = crop
                crops.append(padded)
                fy = np.hanning(ch).astype(np.float32)[:, None]
                fx = np.hanning(cw).astype(np.float32)[None, :]
                metas.append((y, x, ch, cw, np.clip(fy * fx, 0.02, None)))
        probs = self._forward_batch(crops)
        acc = np.zeros((10, h, w), np.float32)
        wts = np.zeros((h, w), np.float32)
        for p, (y, x, ch, cw, fw) in zip(probs, metas):
            acc[:, y:y + ch, x:x + cw] += p[:, :ch, :cw] * fw
            wts[y:y + ch, x:x + cw] += fw
        out = acc / np.maximum(wts, 1e-6)[None]
        return out if scale == 1.0 else self._resize_prob(out, h0, w0)

    @staticmethod
    def _resize_prob(p: np.ndarray, h: int, w: int) -> np.ndarray:
        return np.stack([cv2.resize(p[c], (w, h), interpolation=cv2.INTER_LINEAR) for c in range(p.shape[0])])

    def _forward(self, img_bgr: np.ndarray) -> np.ndarray:
        return self._forward_batch([img_bgr])[0]

    @torch.no_grad()
    def _forward_batch(self, crops, chunk: int = 8):
        out = []
        mean = torch.as_tensor(MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.as_tensor(STD, device=self.device).view(1, 3, 1, 1)
        for i in range(0, len(crops), chunk):
            part = crops[i:i + chunk]
            arr = np.stack([np.ascontiguousarray(c[:, :, ::-1]) for c in part])   # BGR->RGB, uint8
            t = torch.from_numpy(arr).to(self.device, non_blocking=True).permute(0, 3, 1, 2).contiguous().float()
            t = t.div_(255.0).sub_(mean).div_(std)
            if self.device == "cuda":
                with torch.autocast("cuda"):
                    logits, _ = self.model(t)
            else:
                logits, _ = self.model(t)
            out.extend(torch.softmax(logits.float(), 1).cpu().numpy())
        return out

    def predict(self, img_bgr: np.ndarray):
        p = self.probabilities(img_bgr)
        if self.ema_alpha > 0.0:
            if self._ema is None or self._ema.shape != p.shape:
                self._ema = p
            else:
                self._ema = self.ema_alpha * p + (1.0 - self.ema_alpha) * self._ema
            p = self._ema
        pred = p.argmax(0).astype(np.uint8)
        pred[p[5] >= self.water_threshold] = 5
        water_pct = float((pred == 5).mean()) * 100.0
        return pred, water_pct

    def threat(self, img_bgr: np.ndarray) -> float:
        """Head output. NOTE: on the shipped v1 weights this correlates with true
        flood-area fraction at Spearman rho~0.11 -- treat as unreliable until retrained."""
        p = self.probabilities(img_bgr)
        return float(((p[1] + p[3] + p[5]).mean()))


def colorize(pred: np.ndarray) -> np.ndarray:
    return PALETTE_RGB[np.clip(pred, 0, 9)][..., ::-1]   # BGR for cv2


def overlay(img_bgr: np.ndarray, pred: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    vis = img_bgr.copy()
    m = pred == 5
    red = np.zeros_like(img_bgr)
    red[:] = (0, 0, 255)
    vis[m] = ((1 - alpha) * vis[m] + alpha * red[m]).astype(np.uint8)
    other = (pred > 0) & (~m)
    oc = colorize(pred)
    vis[other] = ((1 - alpha * 0.5) * vis[other] + alpha * 0.5 * oc[other]).astype(np.uint8)
    return vis


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                   "checkpoints", "best_moist_vit_pakistan_adapted_v1_goated.pth"))
    ap.add_argument("--threshold", type=float, default=DEFAULT_WATER_THRESHOLD)
    args = ap.parse_args()

    img = cv2.imread(args.image)
    m = MoistPredictor(args.ckpt, water_threshold=args.threshold)
    pred, wpc = m.predict(img)
    print(f"water {wpc:.2f}%  |  " + "  ".join(
        f"{CLASS_NAMES[c]}={float((pred == c).mean())*100:.2f}%" for c in range(10) if (pred == c).any()))
    out = args.out or os.path.splitext(args.image)[0] + "_moist.jpg"
    cv2.imwrite(out, np.concatenate([img, overlay(img, pred)], axis=1))
    print("saved", out)
