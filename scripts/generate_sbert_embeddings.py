import argparse
import logging
from pathlib import Path

from src.config import PROCESSED_DIR
from src.metadata import load_tracks, get_small_subset_ids
from src.embeddings.sbert import SentenceBERTEmbeddingGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate Sentence-BERT embeddings for FMA tracks.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tracks to process.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for SBERT encoding.")
    parser.add_argument("--output-dir", type=str, default=str(PROCESSED_DIR), help="Directory to save embeddings.")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="SBERT model name.")
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    # 1. Load metadata and get track IDs
    logger.info("Loading metadata...")
    tracks = load_tracks()
    track_ids = get_small_subset_ids(tracks)
    
    if args.limit:
        track_ids = track_ids[:args.limit]
        logger.info(f"Limited to first {args.limit} tracks.")

    # 2. Initialize generator
    generator = SentenceBERTEmbeddingGenerator(model_name=args.model)

    # 3. Generate embeddings
    logger.info(f"Generating embeddings for {len(track_ids)} tracks...")
    embeddings, valid_ids = generator.generate(
        track_ids=track_ids,
        output_dir=output_dir,
        batch_size=args.batch_size
    )

    logger.info("Embedding generation complete.")
    logger.info(f"Final shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
