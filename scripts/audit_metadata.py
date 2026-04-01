"""Audit FMA metadata and cross-reference with audio files on disk."""
import argparse
import json
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR, FMA_SMALL_DIR
from src.metadata import load_tracks, load_genres, get_small_subset_ids
from src.audio_utils import discover_valid_tracks


def audit(save_path=None):
    tracks = load_tracks()
    genres = load_genres()
    small_ids = get_small_subset_ids(tracks)
    valid_ids = discover_valid_tracks(small_ids)

    results = {
        "total_tracks": len(tracks),
        "small_subset_tracks": len(small_ids),
        "audio_files_on_disk": len(valid_ids),
        "missing_audio_files": len(small_ids) - len(valid_ids),
        "total_genres": len(genres),
    }

    # Subset distribution
    if ("set", "subset") in tracks.columns:
        results["subset_distribution"] = tracks[("set", "subset")].value_counts().to_dict()

    # Duration stats
    if ("track", "duration") in tracks.columns:
        durations = tracks[("track", "duration")]
        results["duration_mean_s"] = round(float(durations.mean()), 2)
        results["duration_median_s"] = round(float(durations.median()), 2)
        results["duration_min_s"] = round(float(durations.min()), 2)
        results["duration_max_s"] = round(float(durations.max()), 2)

    # Print
    print("\n--- FMA Metadata Audit ---")
    for k, v in results.items():
        print(f"  {k}: {v}")

    # Save
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nSaved audit to {save_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Audit FMA metadata.")
    parser.add_argument("--save", action="store_true",
                        help="Save results to data/processed/audit_results.json")
    args = parser.parse_args()

    save_path = PROCESSED_DIR / "audit_results.json" if args.save else None
    audit(save_path)


if __name__ == "__main__":
    main()
