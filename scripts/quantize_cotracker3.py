import os
import numpy as np
import openvino as ov
import nncf

os.makedirs("../models/int8", exist_ok=True)

core  = ov.Core()
model = core.read_model("../models/ir/cotracker3.xml")

print("Inputs:")
for inp in model.inputs:
    print(f"  {inp.get_any_name()} : {inp.partial_shape}")

# CoTracker3 IR is ~5MB — safe to quantize normally
compressed = nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_ASYM)

ov.save_model(compressed, "../models/int8/cotracker3_int8.xml")
print("✅ Done!")
