import torch
import os
from transformers import DepthAnythingForDepthEstimation

MODEL_DIR = "../models/weights/depth_anything_v2_small"
ONNX_DIR = "../models/onnx"

os.makedirs(ONNX_DIR, exist_ok=True)

ONNX_PATH = os.path.join(ONNX_DIR, "depth_anything_v2_small.onnx")

print("Loading Depth Anything V2 model...")

model = DepthAnythingForDepthEstimation.from_pretrained(MODEL_DIR)
model.eval()

device = torch.device("cpu")
model.to(device)

dummy_input = torch.randn(1, 3, 224, 224).to(device)

print("Exporting ONNX...")

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["depth"],
    opset_version=17,
    dynamic_axes={
        "input": {0: "batch"},
        "depth": {0: "batch"}
    }
)

print("ONNX export complete!")
print("Saved at:", ONNX_PATH)
