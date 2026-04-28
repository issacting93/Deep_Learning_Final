import logging
from pathlib import Path
import numpy as np
from src.config import PROCESSED_DIR, FMA_METADATA_DIR
from src.metadata import load_tracks, get_small_subset_ids
from src.embeddings.sbert import SentenceBERTEmbeddingGenerator
from src.indexing.faiss_index import FaissIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 1. Load tracks
    tracks = load_tracks(FMA_METADATA_DIR)
    small_ids = get_small_subset_ids(tracks)
    
    # 2. Generate embeddings
    generator = SentenceBERTEmbeddingGenerator()
    embeddings, valid_ids = generator.generate(small_ids, output_dir=PROCESSED_DIR)
    
    if len(valid_ids) == 0:
        logger.error("No valid embeddings generated. Exiting.")
        return

    # 3. Build FAISS index
    logger.info(f"Building FAISS index for {len(valid_ids)} vectors...")
    index = FaissIndex(dimension=generator.dim, metric="cosine")
    index.build(embeddings, valid_ids)
    
    # 4. Save index
    index_path = PROCESSED_DIR / "sbert_faiss.index"
    index.save(index_path)
    logger.info(f"FAISS index saved to {index_path}")

if __name__ == "__main__":
    main()
