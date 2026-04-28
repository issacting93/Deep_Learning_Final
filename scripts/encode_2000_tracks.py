import os
import sys
import logging
import numpy as np
import pandas as pd
import torch

# Force single-threaded execution to avoid threading locks on Mac
os.environ["TORCH_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
sys.path.append(os.path.abspath("."))

from src.config import PROCESSED_DIR, FMA_METADATA_DIR
from src.embeddings.sbert import SentenceBERTEmbeddingGenerator
from src.indexing.faiss_index import FaissIndex
from src.metadata import load_tracks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 1. Load the specific 2000 track IDs
    ids_file = PROCESSED_DIR / "fma_2000_track_ids.txt"
    if not ids_file.exists():
        logger.error(f"Track IDs file not found: {ids_file}")
        return
        
    with open(ids_file, "r") as f:
        track_ids = [int(line.strip()) for line in f if line.strip()]
    
    logger.info(f"Loaded {len(track_ids)} track IDs from {ids_file}")

    # 2. Initialize SBERT generator
    generator = SentenceBERTEmbeddingGenerator()
    
    # 3. Generate embeddings
    # We use a custom call to ensure we only target these 2000 IDs
    print(f"Generating SBERT embeddings for {len(track_ids)} tracks...")
    embeddings, valid_ids = generator.generate(track_ids, output_dir=PROCESSED_DIR)
    
    if len(valid_ids) == 0:
        logger.error("No valid embeddings generated.")
        return

    # 4. Save with standardized names as per README.md
    np.save(PROCESSED_DIR / "sbert_embeddings.npy", embeddings)
    np.save(PROCESSED_DIR / "sbert_track_ids.npy", np.array(valid_ids))
    logger.info(f"Saved {len(valid_ids)} embeddings to {PROCESSED_DIR}")

    # 5. Build and save FAISS index
    logger.info("Building FAISS index...")
    index = FaissIndex(dimension=generator.dim, metric="cosine")
    index.build(embeddings, valid_ids)
    
    index_path = PROCESSED_DIR / "sbert_faiss.index"
    index.save(index_path)
    logger.info(f"FAISS index saved to {index_path}")
    
    # 6. Sanity check query
    query = "sad melancholic piano"
    logger.info(f"Running sanity check query: '{query}'")
    q_emb = generator.embed_text([query])
    results = index.query(q_emb, k=5)
    
    # Load metadata for display
    tracks_meta = load_tracks(FMA_METADATA_DIR)
    print(f"\nResults for '{query}':")
    for tid, score in results:
        try:
            row = tracks_meta.loc[tid]
            title = row[('track', 'title')]
            artist = row[('artist', 'name')]
            print(f"- {tid}: {title} by {artist} (Score: {score:.4f})")
        except:
            print(f"- {tid}: (Metadata missing) (Score: {score:.4f})")

if __name__ == "__main__":
    main()
