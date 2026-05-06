"""
plot_genre_retrieval.py — Evaluate and plot Genre Retrieval Accuracy

Loads all four embedding models, computes top-1 genre retrieval accuracy,
then generates a horizontal bar chart.

Run from Finals/:
    python scripts/plot_genre_retrieval.py
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.evaluate_genre_retrieval import (
    evaluate_top1_genre_accuracy,
    l2_normalize,
    load_clap_dict_embeddings,
    load_id_embedding_pair,
    load_metadata,
    validate_model_data,
    ModelData,
)
from src.config import FMA_METADATA_DIR


def evaluate_rrf_fast(
    models: list,
    id_to_genre: Dict[int, str],
    num_samples: int,
    seed: int,
    rrf_k: int = 60,
    top_k: int = 10,
) -> Tuple[float, int, int]:
    """Optimized RRF: precompute similarity matrices instead of per-query."""
    common_ids_set = set(int(x) for x in models[0].ids)
    for model in models[1:]:
        common_ids_set &= set(int(x) for x in model.ids)
    common_ids = sorted(list(common_ids_set))

    def align(model, common):
        id_map = {int(tid): i for i, tid in enumerate(model.ids)}
        indices = np.array([id_map[int(tid)] for tid in common])
        return l2_normalize(model.emb[indices])

    aligned_embs = [align(model, common_ids) for model in models]
    aligned_ids = np.array(common_ids, dtype=np.int64)

    mask = np.array([int(tid) in id_to_genre for tid in aligned_ids], dtype=bool)
    ids = aligned_ids[mask]
    embs = [emb[mask] for emb in aligned_embs]

    n = len(ids)
    eval_n = n if num_samples <= 0 else min(num_samples, n)
    rng = np.random.default_rng(seed)
    query_idx = np.arange(n) if eval_n == n else rng.choice(n, size=eval_n, replace=False)

    # Precompute all similarity matrices and top-K indices once
    all_topk = []
    for emb in embs:
        sim = emb @ emb.T
        np.fill_diagonal(sim, -np.inf)
        # Get top-K for every row at once
        topk_indices = np.argpartition(-sim, top_k, axis=1)[:, :top_k]
        # Sort within top-K
        for i in range(sim.shape[0]):
            order = np.argsort(-sim[i, topk_indices[i]])
            topk_indices[i] = topk_indices[i][order]
        all_topk.append(topk_indices)

    # Now compute RRF scores using precomputed top-K
    correct = 0
    for qi in query_idx:
        rrf_scores: Dict[int, float] = {}
        for topk_indices in all_topk:
            for rank, neighbor_idx in enumerate(topk_indices[qi]):
                neighbor_idx = int(neighbor_idx)
                rrf_scores[neighbor_idx] = rrf_scores.get(neighbor_idx, 0.0) + 1.0 / (rrf_k + rank + 1)

        nn_idx = max(rrf_scores.items(), key=lambda x: x[1])[0]
        if id_to_genre[int(ids[qi])] == id_to_genre[int(ids[nn_idx])]:
            correct += 1

    total = len(query_idx)
    return correct / total, correct, total


# ── Load metadata ──────────────────────────────────────────────
csv_path = FMA_METADATA_DIR / "tracks.csv"
csv_ids, genres, id_to_genre = load_metadata(csv_path)
csv_id_set = set(int(x) for x in csv_ids.tolist())

# ── Load models ────────────────────────────────────────────────
eval_dir = ROOT / "evaluation"

models = {
    "CLAP · Audio": load_clap_dict_embeddings(
        eval_dir / "CLAP" / "clap_audio_embeddings.npy", "CLAP(audio)"
    ),
    "CLAP · Text": load_id_embedding_pair(
        eval_dir / "CLAP" / "clap_text_embeddings_new.npy",
        eval_dir / "CLAP" / "clap_text_track_ids.npy",
        "CLAP(text,new)",
    ),
    "OpenL3": load_id_embedding_pair(
        eval_dir / "OpenL3" / "openl3_embeddings.npy",
        eval_dir / "OpenL3" / "openl3_track_ids.npy",
        "OpenL3",
    ),
    "SBERT": load_id_embedding_pair(
        eval_dir / "SBERT" / "sbert_lyrics_embeddings.npy",
        eval_dir / "SBERT" / "sbert_lyrics_faiss.ids.npy",
        "SBERT(lyrics)",
    ),
}

SEED = 42
NUM_SAMPLES = 500  # matches default in evaluate_genre_retrieval.py

# ── Evaluate each model ───────────────────────────────────────
print("Evaluating individual models...")
results = {}
for label, model in models.items():
    validate_model_data(model, csv_id_set)
    acc, correct, total = evaluate_top1_genre_accuracy(
        model=model, id_to_genre=id_to_genre, num_samples=NUM_SAMPLES, seed=SEED
    )
    results[label] = acc * 100
    print(f"  {label}: {acc:.4f} ({correct}/{total})")

# ── Evaluate fused (RRF) ──────────────────────────────────────
print("Evaluating fused (RRF)...")
acc_rrf, correct_rrf, total_rrf = evaluate_rrf_fast(
    models=list(models.values()),
    id_to_genre=id_to_genre,
    num_samples=NUM_SAMPLES,
    seed=SEED,
)
results["Fused (RRF)"] = acc_rrf * 100
print(f"  Fused (RRF): {acc_rrf:.4f} ({correct_rrf}/{total_rrf})")

# ── Sort by accuracy ──────────────────────────────────────────
sorted_items = sorted(results.items(), key=lambda x: x[1])
methods = [k for k, _ in sorted_items]
accuracy = [v for _, v in sorted_items]

# ── Plot ──────────────────────────────────────────────────────
random_baseline = 100.0 / len(set(id_to_genre.values()))

fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#fd7f6f" if m == "Fused (RRF)" else "#7eb0d5" for m in methods]
bars = ax.barh(methods, accuracy, color=colors, edgecolor="white", height=0.6)

ax.axvline(random_baseline, color="gray", linestyle="--", linewidth=1,
           label=f"Random baseline ({random_baseline:.1f}%)")
ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=10)
ax.set_xlim(0, max(accuracy) + 15)
ax.set_xlabel("Top-1 Genre Retrieval Accuracy (%)")
ax.set_title("Genre Retrieval Accuracy — 2,000 tracks, 8 balanced genres")
ax.legend(loc="lower right")
plt.tight_layout()

out_path = ROOT / "reports" / "genre_retrieval_accuracy.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved to {out_path}")
plt.show()
