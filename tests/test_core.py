"""
Unit tests for core components: FAISS indexing, embedding validation, and config.
Run with: python -m pytest tests/ -v
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.indexing.faiss_index import FaissIndex
from src.config import (
    RRF_K, MAX_CO_GENRE_EDGES, CLAP_BATCH_SIZE, SBERT_BATCH_SIZE,
    GNN_HIDDEN_DIM, GNN_OUT_DIM, PROCESSED_DIR, DATA_DIR,
)


# ── FAISS Index Tests ────────────────────────────────────────────────────────

class TestFaissIndex:
    """Tests for the FAISS index wrapper."""

    def _make_normalized(self, n: int, dim: int) -> np.ndarray:
        rng = np.random.default_rng(42)
        embs = rng.standard_normal((n, dim)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        return embs

    def test_build_and_query_returns_correct_top1(self):
        dim = 128
        embs = self._make_normalized(100, dim)
        track_ids = list(range(100))

        idx = FaissIndex(dim, metric="cosine")
        idx.build(embs, track_ids)

        results = idx.query(embs[0:1], k=5)
        assert results[0][0] == 0, "Top result should be the query itself"
        assert len(results) == 5

    def test_query_scores_are_bounded(self):
        dim = 64
        embs = self._make_normalized(50, dim)
        idx = FaissIndex(dim, metric="cosine")
        idx.build(embs, list(range(50)))

        results = idx.query(embs[0:1], k=10)
        for _, score in results:
            assert -1.01 <= score <= 1.01, f"Cosine score out of range: {score}"

    def test_save_load_roundtrip(self):
        dim = 64
        embs = self._make_normalized(30, dim)
        track_ids = [100 + i for i in range(30)]

        idx = FaissIndex(dim, metric="cosine")
        idx.build(embs, track_ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.faiss"
            idx.save(path)

            loaded = FaissIndex.load(path, metric="cosine")
            assert loaded.track_ids == track_ids

            results_orig = idx.query(embs[0:1], k=5)
            results_loaded = loaded.query(embs[0:1], k=5)
            assert results_orig[0][0] == results_loaded[0][0]

    def test_l2_metric(self):
        dim = 32
        embs = np.random.default_rng(0).standard_normal((20, dim)).astype(np.float32)
        idx = FaissIndex(dim, metric="l2")
        idx.build(embs, list(range(20)))

        results = idx.query(embs[0:1], k=3)
        assert results[0][0] == 0
        assert results[0][1] == pytest.approx(0.0, abs=1e-5), "L2 distance to self should be 0"

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            FaissIndex(64, metric="hamming")


# ── Embedding Norm Validation Tests ──────────────────────────────────────────

class TestEmbeddingNorms:
    """Tests verifying that saved embeddings (if present) are L2-normalized."""

    @pytest.mark.skipif(
        not (PROCESSED_DIR / "clap_embeddings.npy").exists(),
        reason="CLAP embeddings not generated yet",
    )
    def test_clap_embeddings_normalized(self):
        embs = np.load(PROCESSED_DIR / "clap_embeddings.npy")
        norms = np.linalg.norm(embs, axis=1)
        assert np.allclose(norms, 1.0, atol=0.05), (
            f"CLAP norms out of range: {norms.min():.4f}–{norms.max():.4f}"
        )

    @pytest.mark.skipif(
        not (PROCESSED_DIR / "sbert_embeddings.npy").exists(),
        reason="SBERT embeddings not generated yet",
    )
    def test_sbert_embeddings_normalized(self):
        embs = np.load(PROCESSED_DIR / "sbert_embeddings.npy")
        norms = np.linalg.norm(embs, axis=1)
        assert np.allclose(norms, 1.0, atol=0.01), (
            f"SBERT norms out of range: {norms.min():.4f}–{norms.max():.4f}"
        )

    @pytest.mark.skipif(
        not (PROCESSED_DIR / "openl3_embeddings.npy").exists(),
        reason="OpenL3 embeddings not generated yet",
    )
    def test_openl3_embeddings_finite(self):
        """OpenL3 outputs raw (unnormalized) embeddings; app.py normalizes on load.
        Here we just verify they are finite and non-zero."""
        embs = np.load(PROCESSED_DIR / "openl3_embeddings.npy")
        assert np.all(np.isfinite(embs)), "OpenL3 embeddings contain NaN/Inf"
        norms = np.linalg.norm(embs, axis=1)
        assert (norms > 0).all(), "OpenL3 embeddings contain zero vectors"


# ── Config Tests ─────────────────────────────────────────────────────────────

class TestConfig:
    """Verify config constants are sensible."""

    def test_rrf_k_positive(self):
        assert RRF_K > 0

    def test_batch_sizes_positive(self):
        assert CLAP_BATCH_SIZE > 0
        assert SBERT_BATCH_SIZE > 0

    def test_gnn_dims_positive(self):
        assert GNN_HIDDEN_DIM > 0
        assert GNN_OUT_DIM > 0

    def test_max_co_genre_edges_reasonable(self):
        assert 10_000 <= MAX_CO_GENRE_EDGES <= 1_000_000

    def test_data_dir_exists(self):
        assert DATA_DIR.exists(), f"DATA_DIR not found: {DATA_DIR}"
