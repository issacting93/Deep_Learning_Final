from pathlib import Path
import numpy as np
import faiss


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
            if idx < len(self.track_ids):
                results.append((self.track_ids[idx], float(dist)))
        return results

    def save(self, path: Path):
        path = Path(path)
        faiss.write_index(self.index, str(path))
        ids_path = path.with_suffix(".ids.npy")
        np.save(ids_path, np.array(self.track_ids))

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
