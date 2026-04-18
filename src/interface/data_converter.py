import pandas as pd
import numpy as np
import json
from pathlib import Path

def convert_parquet_to_json(parquet_path: str, output_json_path: str, fps: int = 30):
    try:
        df = pd.read_parquet(parquet_path)
    except FileNotFoundError:
        print(f"ERROR: {parquet_path} not found. Run the pipeline first.")
        return

    # Task 9.2: Semantic Coloring Preparation
    # Normalize the top 3 PCA components to [0, 1] for RGB mapping.
    # If Radhe hasn't pushed the PCA columns yet, this failsafe generates mock clustering colors.
    for i, color in enumerate(['r', 'g', 'b']):
        pca_col = f'pca_{i}'
        if pca_col in df.columns:
            min_val = df[pca_col].min()
            max_val = df[pca_col].max()
            # Prevent division by zero if all values are identical
            range_val = (max_val - min_val) if (max_val - min_val) > 0 else 1.0
            df[color] = (df[pca_col] - min_val) / range_val
        else:
            # Failsafe: Generate a random static color per track_id if PCA is missing
            df[color] = df.groupby('track_id')['track_id'].transform(lambda x: np.random.rand())

    tracks_data = []
    
    for track_id, group in df.groupby('track_id'):
        path = []
        for _, row in group.iterrows():
            path.append({
                "f": int(row['frame_idx']),
                "x": float(row['x']),
                "y": float(row['y']),
                "z": float(row['z']),
                "r": float(row['r']),
                "g": float(row['g']),
                "b": float(row['b'])
            })
        
        path = sorted(path, key=lambda p: p['f'])
        tracks_data.append({"id": int(track_id), "path": path})
        
    final_data = {"fps": fps, "tracks": tracks_data}
    
    out_path = Path(output_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(final_data, f)
        
    print(f"SUCCESS: Semantic trajectory data exported to {output_json_path}")

if __name__ == "__main__":
    convert_parquet_to_json("output/results.parquet", "src/interface/ui/trajectory_data.json")