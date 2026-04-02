import pandas as pd # type: ignore
import json
from pathlib import Path

def convert_parquet_to_json(parquet_path: str, output_json_path: str, fps: int = 30):
    try:
        df = pd.read_parquet(parquet_path)
    except FileNotFoundError:
        print(f"ERROR: {parquet_path} not found. Ensure the ML pipeline has run.")
        return

    tracks_data = []
    # Group by the unique point IDs
    for track_id, group in df.groupby('track_id'):
        path = []
        for _, row in group.iterrows():
            path.append({
                "f": int(row['frame_idx']),
                "x": float(row['x']),
                "y": float(row['y']),
                "z": float(row['z'])
            })
        
        # Sort the trajectory chronologically
        path = sorted(path, key=lambda p: p['f'])
        tracks_data.append({"id": int(track_id), "path": path})
        
    final_data = {"fps": fps, "tracks": tracks_data}
    
    out_path = Path(output_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(final_data, f)
        
    print(f"SUCCESS: Trajectory data exported to {output_json_path}")

if __name__ == "__main__":
    # Converts the backend output into a frontend-readable JSON
    convert_parquet_to_json("output/results.parquet", "src/interface/ui/trajectory_data.json")