"""
quantize_vjepa.py  (compress_weights version — low RAM)
========================================================
Uses nncf.compress_weights() instead of nncf.quantize().

WHY THE CHANGE
──────────────
Your system has 7.4GB RAM, 3.3GB already used.
nncf.quantize() (PTQ) needs ~3-4x model size in RAM for activation
statistics = ~2GB extra on top of existing usage → system freeze.

nncf.compress_weights() compresses one layer at a time → peak RAM
usage is roughly 1x model size (~580MB extra) → safe on your system.

RESULT
──────
INT8 weight compression. Weights stored as INT8, computed in FP32.
Size reduction: ~580MB → ~290MB (same ratio as PTQ for weights).

⚠️  RUN IN: openvino310 (conda)
    conda activate openvino310
    cd ~/Internship/openvino_project/scripts
    python quantize_vjepa.py
"""

import os
import openvino as ov
import nncf

# ── Paths ─────────────────────────────────────
MODEL_XML  = "../models/ir/vjepa2_vitl.xml"
OUTPUT_XML = "../models/int8/vjepa2_vitl_int8.xml"
OUTPUT_BIN = "../models/int8/vjepa2_vitl_int8.bin"

os.makedirs("../models/int8", exist_ok=True)

# ── Load ──────────────────────────────────────
print("Loading V-JEPA2 IR model (~580MB)...")
core  = ov.Core()
model = core.read_model(MODEL_XML)
print(f"  Input : {model.inputs[0].get_any_name()}  "
      f"{model.inputs[0].partial_shape}")

# ── Compress weights (layer by layer — low RAM) 
print("\nCompressing weights to INT8 (layer-by-layer, no calibration needed)...")
print("Expected time: 2-5 minutes. RAM usage: safe.")

compressed = nncf.compress_weights(
    model,
    mode=nncf.CompressWeightsMode.INT8,
)

# ── Save ──────────────────────────────────────
print("\nSaving compressed model...")
ov.save_model(compressed, OUTPUT_XML)

assert os.path.exists(OUTPUT_XML), "ERROR: .xml not written!"
assert os.path.exists(OUTPUT_BIN), "ERROR: .bin not written!"

orig_mb = os.path.getsize(MODEL_XML.replace(".xml", ".bin")) / 1e6
xml_mb  = os.path.getsize(OUTPUT_XML) / 1e6
bin_mb  = os.path.getsize(OUTPUT_BIN) / 1e6

print(f"\n✅ Compression complete!")
print(f"   Original  : {orig_mb:.1f} MB")
print(f"   Compressed: {bin_mb:.1f} MB  (.bin)")
print(f"   Reduction : {100*(1 - bin_mb/orig_mb):.1f}%")
