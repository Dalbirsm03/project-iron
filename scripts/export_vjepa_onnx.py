import torch
import os
from transformers import AutoModel

# ----------------------------
# Config
# ----------------------------

MODEL_DIR = "../models/weights/vjepa2_vitl"
ONNX_PATH = "../models/onnx/vjepa2_vitl.onnx"

os.makedirs("../models/onnx", exist_ok=True)

# ----------------------------
# Load Model
# ----------------------------

print("Loading V-JEPA ViT-L model...")

model = AutoModel.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True
)

model.eval()

# ----------------------------
# Dummy Input
# ----------------------------

# (B, T, C, H, W)
dummy_video = torch.randn(1, 8, 3, 224, 224)

# ----------------------------
# Wrapper (important)
# ----------------------------

class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, video):
        output = self.model(video)

        # Handle outputs safely
        if isinstance(output, tuple):
            return output[0]
        elif isinstance(output, dict):
            return list(output.values())[0]
        else:
            return output

wrapped_model = Wrapper(model)
wrapped_model.eval()

# ----------------------------
# Export ONNX
# ----------------------------

print("Exporting ONNX...")

with torch.no_grad():
    torch.onnx.export(
        wrapped_model,
        dummy_video,
        ONNX_PATH,
        input_names=["video"],
        output_names=["features"],
        opset_version=18,
        dynamic_axes={
            "video": {0: "batch", 1: "time"},
            "features": {0: "batch"}
        },
        dynamo=False   # 🔥 use stable exporter
    )

print(f"ONNX export complete! Saved at: {ONNX_PATH}")
