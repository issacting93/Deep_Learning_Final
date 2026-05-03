import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import PROCESSED_DIR, SBERT_BATCH_SIZE
from src.embeddings.base import EmbeddingGenerator
from src.metadata_builder import build_metadata_strings

logger = logging.getLogger(__name__)


class SentenceBERTEmbeddingGenerator(EmbeddingGenerator):
    """Generate text embeddings using Sentence-BERT (all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        logger.info(f"Loading Sentence-BERT model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Sentence-BERT model ready (dim={self.dim}).")

    def generate(
        self,
        track_ids: list,
        output_dir: Path = PROCESSED_DIR,
        batch_size: int = SBERT_BATCH_SIZE,
        resume: bool = True,
    ) -> tuple:
        """Generate embeddings for given tracks."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = output_dir / "metadata_texts.csv"

        if resume and metadata_path.exists():
            logger.info("Loading existing metadata strings...")
            df_meta = pd.read_csv(metadata_path, index_col=0)
        else:
            logger.info("Building metadata strings...")
            df_meta = build_metadata_strings()

        # Filter to requested track IDs
        available_ids = [tid for tid in track_ids if tid in df_meta.index]
        text_data = df_meta.loc[available_ids, "metadata_text"].tolist()

        if not text_data:
            logger.warning("No metadata found for requested track IDs.")
            return np.array([]), []

        logger.info(f"Encoding {len(text_data)} strings with SBERT...")
        # normalize_embeddings=True ensures Cosine Similarity is equivalent to Dot Product
        embeddings = self.model.encode(
            text_data,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # Validate L2 norms
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=0.01):
            logger.warning(
                f"SBERT embeddings not normalized (range {norms.min():.4f}–{norms.max():.4f}), "
                "renormalizing."
            )
            embeddings = embeddings / (norms[:, np.newaxis] + 1e-8)

        # Save final outputs
        np.save(output_dir / "sbert_embeddings.npy", embeddings)
        np.save(output_dir / "sbert_track_ids.npy", np.array(available_ids))

        logger.info(f"Saved {len(available_ids)} SBERT embeddings to {output_dir}")
        return embeddings, available_ids

    def load_embeddings(self, output_dir: Path = PROCESSED_DIR) -> tuple:
        output_dir = Path(output_dir)
        embeddings = np.load(output_dir / "sbert_embeddings.npy")
        track_ids = np.load(output_dir / "sbert_track_ids.npy").tolist()
        return embeddings, track_ids

    def embed_text(self, queries: list) -> np.ndarray:
        """Embed text queries for retrieval."""
        return self.model.encode(
            queries, normalize_embeddings=True, convert_to_numpy=True
        )
