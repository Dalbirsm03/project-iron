"""
export_cotracker3_onnx_fixed.py  (v9 — patch F.grid_sample at the source)
==========================================================================
ROOT CAUSE (fully understood):
  bilinear_sampler in model_utils.py handles both 4D and 5D inputs.
  For 5D input [B,C,T,H,W] it reorders coords then calls F.grid_sample
  with a 5D tensor. OpenVINO only supports 4D grid_sample.

THE CORRECT FIX:
  Patch F.grid_sample so that when called with 5D input:
    - Flatten T into B: [B,C,T,H,W] → [B*T, C, H, W]
    - coords are [B, Ho, Wo, D, 3] in xyt order after bilinear_sampler's
      reorder. We need [B*T, Ho, Wo, D, 2] (xy only, tiled T times)
    
  BUT: bilinear_sampler already does the coord reorder to xyt BEFORE
  calling grid_sample. So grid_sample receives coords in [x,y,t] order.
  For 5D grid_sample, coords shape is [B, Ho, Wo, Do, 3].
  
  To flatten T into B:
    - inp4: [B*T, C, H, W]  
    - take only xy from coords (first 2 dims): [B, Ho, Wo, Do, 2]
    - tile to [B*T, Ho, Wo, Do, 2] ... but Do=1 here
    - flatten: [B*T, Ho, Wo*Do, 2] = [B*T, Ho, Wo, 2]
    - call 4D grid_sample → [B*T, C, Ho, Wo]
    - reshape → [B, C, T, Ho, Wo] ... but output must be [B,C,R1,R2,1]
    
  ACTUAL shapes confirmed from debug:
    Call 1 (sample_features5d):
      input  [B,C,T,H,W]  = [1, 128, 4, 56, 56]
      coords [B,R1,R2,1,3]= [1, 49, 100, 1, 3]  ← after bilinear reorder→xyt
      output must be      [B,C,R1,R2,1] = [1,128,4,49,100]? No...
      
    Actually bilinear_sampler receives [B,C,T,H,W] input and
    coords [B,R1,R2,1,3]. It does coords[...,[1,2,0]] then normalizes
    then calls F.grid_sample([B,C,T,H,W], normalized_coords[B,R1,R2,1,3])
    F.grid_sample 5D: input[B,C,D,H,W], grid[B,Ho,Wo,Do,3] → [B,C,Ho,Wo,Do]
    So output = [B, C, R1, R2, 1]  ✓ matches what sample_features5d expects

    Call 2 (get_correlation_feat):  
      input  [B*T, D, 1, H_, W_]  ← 5D with T_dim=1
      coords [B*T, N, r, r, 3]    ← after reorder: xyt
      output [B*T, D, N, r, r]    ← used in .view(B,T,D,N,r,r)

  UNIFIED PATCH for F.grid_sample when input is 5D:
    input [B, C, D, H, W], grid [B, Ho, Wo, Do, 3]
    → treat D as part of spatial: flatten [B,C,D,H,W]→[B*D,C,H,W]
      but grid coords have t component... 
      
    SIMPLER: since T (or D) is always 1 in call 2, and in call 1 we
    want to sample across T using t-coord:
    
    Use the t-coordinate to index into T dimension via gather,
    then do 4D xy grid_sample. But t is continuous...
    
    SIMPLEST CORRECT APPROACH:
    When input is 5D [B,C,T,H,W]:
      - reshape to [B, C*T, H, W] ... no, grid_sample won't know
      
    ACTUAL SIMPLEST: just call the 5D grid_sample via torch directly,
    but reimplement it manually using 4D calls by looping over T.
    Since T is small (1 or 4) this is fine for export — the loop
    gets unrolled by TorchScript into static 4D ops.

⚠️  RUN IN: openvino_env
    source ~/Internship/openvino_project/openvino_env/bin/activate
    cd ~/Internship/openvino_project/scripts
    python export_cotracker3_onnx.py
"""

import os
import torch
import torch.nn.functional as F
import torch.nn as nn

WEIGHTS_PATH = "../models/weights/cotracker3/scaled_offline.pth"
ONNX_PATH    = "../models/onnx/cotracker3_ovfix.onnx"
os.makedirs("../models/onnx", exist_ok=True)

_orig_grid_sample = F.grid_sample

def _grid_sample_ov(input, grid, mode='bilinear',
                    padding_mode='border', align_corners=True):
    """
    OpenVINO-compatible grid_sample.
    
    4D input → pass straight through (OpenVINO supports this).
    5D input [B,C,T,H,W] with grid [B,Ho,Wo,To,3] (coords in x,y,t order):
      → loop over T frames, do 4D grid_sample for each, stack result.
      
    This unrolls cleanly in TorchScript and produces only 4D grid_sample
    ops in the ONNX graph.
    """
    if input.dim() == 4:
        return _orig_grid_sample(input, grid, mode=mode,
                                 padding_mode=padding_mode,
                                 align_corners=align_corners)

    if input.dim() == 5:
        B, C, T, H, W = input.shape
        # grid: [B, Ho, Wo, To, 3] where last dim is (x, y, t) after reorder
        # To is the output time dim (usually 1)
        Ho, Wo, To = grid.shape[1], grid.shape[2], grid.shape[3]

        results = []
        for t_idx in range(T):
            # frame: [B, C, H, W]
            frame = input[:, :, t_idx, :, :]

            # For this frame, use the t_idx-th slice of the t-coordinate
            # or just use xy coords (first 2) since we're iterating over T
            # grid[..., :2] gives xy, grid[..., 2] gives t-coord
            # We want to sample frame t_idx using xy coords
            # Weight this frame's contribution by how close t-coord is to t_idx
            
            # xy grid for 4D: [B, Ho, Wo*To, 2]
            xy = grid[..., :2]                          # [B, Ho, Wo, To, 2]
            xy_4d = xy.reshape(B, Ho, Wo * To, 2)       # [B, Ho, Wo*To, 2]

            # t coordinate normalized to [0, T-1] range (already done by bilinear_sampler)
            # grid[...,2] is t in [-1,1] range after normalization
            # Convert back: t_norm in [-1,1] → t_idx_float in [0, T-1]
            t_coord = grid[..., 2]                      # [B, Ho, Wo, To]
            t_float = (t_coord + 1) / 2 * max(T - 1, 1)  # [0, T-1]

            # Weight for this frame: max(0, 1 - |t_float - t_idx|)
            weight = (1.0 - (t_float - t_idx).abs()).clamp(min=0)  # [B, Ho, Wo, To]
            weight_4d = weight.reshape(B, Ho, Wo * To)              # [B, Ho, Wo*To]

            out_frame = _orig_grid_sample(frame, xy_4d, mode=mode,
                                          padding_mode=padding_mode,
                                          align_corners=align_corners)
            # out_frame: [B, C, Ho, Wo*To]

            # Apply time weight
            weight_exp = weight_4d.unsqueeze(1)         # [B, 1, Ho, Wo*To]
            results.append(out_frame * weight_exp)

        # Sum weighted frames → [B, C, Ho, Wo*To]
        out = torch.stack(results, dim=0).sum(dim=0)

        # Reshape to [B, C, Ho, Wo, To]
        out = out.reshape(B, C, Ho, Wo, To)
        # Permute to match expected [B, C, To, Ho, Wo] ... 
        # Actually F.grid_sample 5D returns [B, C, Ho, Wo, To] directly
        # so keep as [B, C, Ho, Wo, To]
        return out

    raise ValueError(f"_grid_sample_ov: unsupported input.dim={input.dim()}")


# Patch F.grid_sample globally
F.grid_sample = _grid_sample_ov

# Also patch it in model_utils module namespace
import cotracker.models.core.model_utils as mu
mu.F.grid_sample = _grid_sample_ov

# ── Load model ────────────────────────────────────────────
print("Loading CoTracker3 model...")
from cotracker.predictor import CoTrackerPredictor

predictor = CoTrackerPredictor(
    checkpoint=WEIGHTS_PATH, offline=True, window_len=60)
cotracker = predictor.model
cotracker.eval()

class CoTrackerExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, video, queries):
        outputs = self.model(video, queries)
        if isinstance(outputs, (list, tuple)):
            return outputs[0]
        return outputs

wrapped = CoTrackerExportWrapper(cotracker)
wrapped.eval()

B, T, C, H, W = 1, 4, 3, 224, 224
video   = torch.randn(B, T, C, H, W)
queries = torch.randn(B, 100, 3)

print(f"video  : {list(video.shape)}")
print(f"queries: {list(queries.shape)}")
print("\nRunning sanity forward pass...")

with torch.no_grad():
    out = wrapped(video, queries)
print(f"✅ Sanity pass output: {out.shape}")

print("\nExporting to ONNX...")
with torch.no_grad():
    torch.onnx.export(
        wrapped,
        (video, queries),
        ONNX_PATH,
        input_names=["video", "queries"],
        output_names=["tracks"],
        opset_version=17,
        dynamic_axes={
            "video"  : {0: "batch"},
            "queries": {0: "batch"},
            "tracks" : {0: "batch"},
        },
        dynamo=False,
    )

F.grid_sample = _orig_grid_sample

size_mb = os.path.getsize(ONNX_PATH) / 1e6
print(f"\n✅ ONNX export complete!")
print(f"   Saved: {ONNX_PATH}  ({size_mb:.1f} MB)")
print()
print("── Next: IR conversion (openvino_env, from project root) ──")
print("cd ~/Internship/openvino_project")
print("ovc models/onnx/cotracker3_ovfix.onnx --output_model models/ir/cotracker3.xml")
