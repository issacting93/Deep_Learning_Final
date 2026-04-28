import os
import sys
import numpy as np
import torch
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath("."))

from src.config import PROCESSED_DIR

def calculate_overlap(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return len(set1.intersection(set2)) / len(set1) if len(set1) > 0 else 0

def main():
    print("--- CLAP vs SBERT Overlap Analysis ---")
    
    try:
        clap_ids = np.load(PROCESSED_DIR / "clap_track_ids.npy").tolist()
        sbert_ids = np.load(PROCESSED_DIR / "sbert_track_ids.npy").tolist()
    except Exception as e:
        print(f"Error loading IDs: {e}")
        return

    common_ids = list(set(clap_ids).intersection(set(sbert_ids)))
    print(f"Common tracks between CLAP and SBERT: {len(common_ids)}")
    
    if len(common_ids) == 0:
        print("No overlapping tracks found. Cannot compare.")
        return

if __name__ == "__main__":
    main()
