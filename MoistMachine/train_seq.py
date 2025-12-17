import os, torch, cv2, numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn, torch.optim as optim
from tqdm import tqdm

from model_multihead import MultiHeadFloodNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEG_CKPT = "checkpoints/best_fp32.pth"   # or use best_qat_int8 for inference-only
IMG_SIZE = 320
BATCH = 12
EPOCHS = 25
LR = 1e-3
SEQ_LEN = 5
IGNORE_INDEX = 255

# Sequence dataset assumes directory per sequence with frames and a labels.csv mapping seq_id->target vector
class SequenceDataset(Dataset):
    def __init__(self, seq_root, labels_csv, img_size=IMG_SIZE, seq_len=SEQ_LEN):
        self.seq_dirs = sorted([os.path.join(seq_root, d) for d in os.listdir(seq_root)
                                if os.path.isdir(os.path.join(seq_root, d))])
        self.labels = {}
        with open(labels_csv) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if parts[0].lower() == 'seq':
                    continue
                if len(parts) < 3:
                    continue
                seq, x, y = parts[:3]
                self.labels[seq] = np.array([float(x), float(y)], dtype=np.float32)
        self.img_size = img_size
        self.seq_len = seq_len

    def __len__(self): return len(self.seq_dirs)

    def __getitem__(self, idx):
        d = self.seq_dirs[idx]
        seq_id = os.path.basename(d)
        frames = sorted([os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if len(frames) == 0:
            raise RuntimeError(f"No frames found in {d}")
        if len(frames) < self.seq_len:
            frames = frames + [frames[-1]] * (self.seq_len - len(frames))
        frames = frames[:self.seq_len]
        imgs = []
        for p in frames:
            im = cv2.imread(p)
            if im is None:
                raise RuntimeError(f"Failed reading frame {p}")
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (self.img_size, self.img_size)).astype(np.float32) / 255.0
            im = (im - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            imgs.append(im.transpose(2, 0, 1))
        imgs = np.stack(imgs, axis=0).astype(np.float32)
        tgt = self.labels[seq_id]
        return torch.from_numpy(imgs), torch.from_numpy(tgt)

# load segmentation model to extract pooled per-class features
seg_model = MultiHeadFloodNet(num_classes=10, pretrained=False).to(DEVICE)
seg_model.load_state_dict(torch.load(SEG_CKPT, map_location='cpu'))
seg_model.eval()

class TinyGRU(nn.Module):
    def __init__(self, in_dim=10, hidden=64, out=2):
        super().__init__()
        self.reduce = nn.Linear(in_dim, 64)
        self.gru = nn.GRU(64, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, out)
    def forward(self, x):
        # x: [B,T,C]
        x = self.reduce(x)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

@torch.no_grad()
def extract_feats_batch(imgs):  # imgs: [B,T,C,H,W]
    B, T, C, H, W = imgs.shape
    x = imgs.view(B * T, C, H, W).to(DEVICE)
    _, _, pooled = seg_model(x)
    pooled = pooled.view(B, T, -1).cpu()
    return pooled

def train_sequence():
    SEQ_ROOT = "sequences"
    LABEL_CSV = os.path.join(SEQ_ROOT, "labels.csv")
    ds = SequenceDataset(SEQ_ROOT, LABEL_CSV, img_size=IMG_SIZE)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)

    gru = TinyGRU(in_dim=10, hidden=64, out=2).to(DEVICE)
    opt = optim.Adam(gru.parameters(), lr=LR)
    crit = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        gru.train()
        pbar = tqdm(loader, desc=f"GRU {epoch}")
        running = 0.0
        for imgs, tg in pbar:
            feats = extract_feats_batch(imgs)   # CPU
            feats = feats.to(DEVICE)
            tg = tg.to(DEVICE)
            preds = gru(feats)
            loss = crit(preds, tg)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * imgs.size(0)
            pbar.set_postfix({'loss': loss.item()})
        print(f"Epoch {epoch} loss {running/len(ds):.4f}")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(gru.state_dict(), f"checkpoints/gru_epoch{epoch}.pth")

if __name__ == "__main__":
    train_sequence()