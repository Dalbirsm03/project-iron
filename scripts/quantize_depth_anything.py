import numpy as np
import nncf
from openvino.runtime import Core, serialize
from nncf import Dataset

# Paths
MODEL_PATH = "../models/ir/vjepa2_vitl.xml"
OUTPUT_XML = "../models/int8/vjepa2_vitl_int8.xml"
OUTPUT_BIN = "../models/int8/vjepa2_vitl_int8.bin"

# Load model
core = Core()
model = core.read_model(MODEL_PATH)

# Get actual input name
input_name = model.inputs[0].get_any_name()
print(f"Model input name: {input_name}")

# Dummy calibration dataset (IMPORTANT: smaller size)
def transform_fn(data_item):
    return {input_name: data_item}

calibration_data = [
    np.random.randn(1,8, 3, 224, 224).astype(np.float32)
    for _ in range(10)   # keep small for big model
]

dataset = Dataset(calibration_data, transform_fn)

print("Starting INT8 quantization...")

# Quantize
quantized_model = nncf.quantize(
    model,
    dataset
)

# Save model
serialize(
    quantized_model,
    OUTPUT_XML,
    OUTPUT_BIN
)

print("Quantization complete!")
