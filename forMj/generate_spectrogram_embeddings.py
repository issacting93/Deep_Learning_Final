"""Generate spectrogram-based embeddings for the canonical 2,000-track subset.

Pipeline: audio → mel spectrogram → ResNet18 backbone → 512-d embedding

Usage:
    python scripts/generate_spectrogram_embeddings.py
    python scripts/generate_spectrogram_embeddings.py --batch-size 16  # if memory-constrained
"""

import os
import sys
import logging
import argparse

import numpy as np

sys.path.append(os.path.abspath("."))

from src.config import PROCESSED_DIR
from src.embeddings.spectrogram import SpectrogramEmbeddingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate spectrogram embeddings")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    # Load the canonical 2,000-track subset (shared across all views)
    track_ids = np.load(PROCESSED_DIR / "openl3_track_ids.npy").astype(int).tolist()
    logger.info(f"Target: {len(track_ids)} tracks from canonical subset")

    generator = SpectrogramEmbeddingGenerator()
    embeddings, valid_ids = generator.generate(
        track_ids, PROCESSED_DIR, batch_size=args.batch_size,
    )

    logger.info(f"Done: {len(valid_ids)}/{len(track_ids)} tracks embedded, shape {embeddings.shape}")

    # Quick sanity check
    canonical = set(track_ids)
    assert set(valid_ids).issubset(canonical), "IDs outside canonical subset!"
    assert np.isfinite(embeddings).all(), "Embeddings contain NaN/Inf!"
    logger.info("Sanity checks passed.")


if __name__ == "__main__":
    main()
