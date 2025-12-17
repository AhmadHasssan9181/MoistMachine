# Optional: sketch for QAT with basic fusing (MobilenetV2 has fused variants; for brevity we keep a stub).
import torch
from torch.quantization import get_default_qat_qconfig, prepare_qat, convert
from model_multihead import MultiHeadFloodNet

def prepare_model_for_qat(model, backend='fbgemm'):
    model.train()
    model.qconfig = get_default_qat_qconfig(backend)
    prepare_qat(model, inplace=True)
    return model

def finalize_qat_model(model, ckpt_path="checkpoints/best_qat_int8.pth"):
    model.eval()
    convert(model, inplace=True)
    torch.jit.script(model).save(ckpt_path)
    print(f"Saved QAT quantized scripted model: {ckpt_path}")