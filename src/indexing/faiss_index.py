import logging
from pathlib import Path

import numpy as np
import faiss

logger = logging.getLogger(__name__)


class FaissIndex:
    """Wrapper around FAISS for embedding retrieval."""

    def __init__(self, dimension: int, metric: str = "cosine"):
        if metric == "cosine":
            self.index = faiss.IndexFlatIP(dimension)
        elif metric == "l2":
            self.index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        self.track_ids = []
        self.metric = metric

    def build(self, embeddings: np.ndarray, track_ids: list):
        """Build index from embeddings. Normalizes for cosine similarity."""
        embeddings = embeddings.astype(np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.track_ids = list(track_ids)

    def query(self, query_embedding: np.ndarray, k: int = 10) -> list:
        """Query the index. Returns list of (track_id, score) tuples."""
        q = query_embedding.astype(np.float32).reshape(1, -1)
        if self.metric == "cosine":
            faiss.normalize_L2(q)
        distances, indices = self.index.search(q, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if not (0 <= idx < len(self.track_ids)):
                logger.warning(f"FAISS returned invalid index {idx}, skipping")
                continue
            results.append((self.track_ids[idx], float(dist)))
        return results

    def save(self, path: Path):
        """Save index and track IDs atomically (write to temp, then rename)."""
        path = Path(path)
        ids_path = path.with_suffix(".ids.npy")

        # Write index to temp file, then atomically rename
        tmp_index = path.parent / (path.name + ".tmp")
        faiss.write_index(self.index, str(tmp_index))
        tmp_index.replace(path)

        # np.save auto-appends .npy, so use a name without .npy suffix
        tmp_ids = path.parent / (ids_path.stem + ".tmp")
        np.save(tmp_ids, np.array(self.track_ids))
        # np.save created tmp_ids.npy — rename to final destination
        Path(str(tmp_ids) + ".npy").replace(ids_path)

    @classmethod
    def load(cls, path: Path, metric: str = "cosine"):
        path = Path(path)
        index = faiss.read_index(str(path))
        ids_path = path.with_suffix(".ids.npy")
        track_ids = np.load(ids_path).tolist()
        obj = cls.__new__(cls)
        obj.index = index
        obj.track_ids = track_ids
        obj.metric = metric
        return obj
