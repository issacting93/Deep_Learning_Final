"""Generate CLAP audio embeddings for the FMA small subset."""
import argparse
import logging
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR, CLAP_BATCH_SIZE
from src.metadata import load_tracks, get_small_subset_ids
from src.embeddings.clap import CLAPEmbeddingGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Generate CLAP embeddings for FMA tracks.")
    parser.add_argument("--batch-size", type=int, default=CLAP_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N tracks (for testing).")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, ignore previous checkpoints.")
    args = parser.parse_args()

    tracks = load_tracks()
    track_ids = get_small_subset_ids(tracks)
    if args.limit:
        track_ids = track_ids[:args.limit]

    logging.info(f"Processing {len(track_ids)} tracks (batch_size={args.batch_size}).")

    generator = CLAPEmbeddingGenerator()
    embeddings, valid_ids = generator.generate(
        track_ids,
        output_dir=PROCESSED_DIR,
        batch_size=args.batch_size,
        resume=not args.no_resume,
    )
    logging.info(f"Done. {len(valid_ids)} embeddings saved to {PROCESSED_DIR}.")


if __name__ == "__main__":
    main()
