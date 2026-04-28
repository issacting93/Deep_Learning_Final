import os
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath("."))

from src.config import FMA_METADATA_DIR, DATA_DIR

def main():
    # 1. Paths
    ids_file = DATA_DIR / "processed" / "fma_2000_track_ids.txt"
    output_dir = DATA_DIR / "fma_2000_metadata"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not ids_file.exists():
        print(f"Error: Track IDs file not found at {ids_file}")
        return
        
    # 2. Load Track IDs
    with open(ids_file, "r") as f:
        track_ids = [int(line.strip()) for line in f if line.strip()]
    
    print(f"Loaded {len(track_ids)} track IDs.")

    # 3. Filter tracks.csv
    print("Filtering tracks.csv...")
    tracks_path = FMA_METADATA_DIR / "tracks.csv"
    # Load with multi-index
    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])
    
    # Filter by IDs present in the subset
    # Note: tracks.index is track_id
    filtered_tracks = tracks[tracks.index.isin(track_ids)]
    
    output_tracks_path = output_dir / "tracks.csv"
    filtered_tracks.to_csv(output_tracks_path)
    print(f"Saved {len(filtered_tracks)} tracks to {output_tracks_path}")

    # 4. Filter echonest.csv
    echonest_path = FMA_METADATA_DIR / "echonest.csv"
    if echonest_path.exists():
        print("Filtering echonest.csv...")
        # Load with multi-index
        echonest = pd.read_csv(echonest_path, index_col=0, header=[0, 1, 2])
        filtered_echonest = echonest[echonest.index.isin(track_ids)]
        
        output_echonest_path = output_dir / "echonest.csv"
        filtered_echonest.to_csv(output_echonest_path)
        print(f"Saved {len(filtered_echonest)} echonest records to {output_echonest_path}")
    else:
        print("echonest.csv not found, skipping.")

    print("\nMetadata extraction complete.")

if __name__ == "__main__":
    main()
