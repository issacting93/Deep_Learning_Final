"""Build a FAISS index from precomputed embeddings."""
import argparse
import logging
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
from src.config import PROCESSED_DIR
from src.indexing.faiss_index import FaissIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from embeddings.")
    parser.add_argument("--embedding-type", default="clap",
                        choices=["clap"],
                        help="Which embedding type to index.")
    parser.add_argument("--metric", default="cosine", choices=["cosine", "l2"])
    args = parser.parse_args()

    emb_path = PROCESSED_DIR / f"{args.embedding_type}_embeddings.npy"
    ids_path = PROCESSED_DIR / f"{args.embedding_type}_track_ids.npy"

    logging.info(f"Loading embeddings from {emb_path}")
    embeddings = np.load(emb_path)
    track_ids = np.load(ids_path).tolist()

    logging.info(f"Building {args.metric} index: {embeddings.shape[0]} vectors, dim={embeddings.shape[1]}")
    index = FaissIndex(dimension=embeddings.shape[1], metric=args.metric)
    index.build(embeddings, track_ids)

    out_path = PROCESSED_DIR / f"{args.embedding_type}_faiss.index"
    index.save(out_path)
    logging.info(f"Index saved to {out_path}")


if __name__ == "__main__":
    main()
