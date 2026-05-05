"""
evaluate_rrf_fusion.py
======================
Evaluate Reciprocal Rank Fusion (RRF) performance using Echo Nest ground truth.

Compares three fusion strategies:
  1. Individual views (SBERT, OpenL3, CLAP)
  2. Reciprocal Rank Fusion (RRF) - three views combined

Uses Echo Nest features as independent ground truth (models never saw these).

Protocol:
  For each query track:
    1. Retrieve top-5 nearest neighbors via RRF (equal weights)
    2. Compute average Euclidean distance in Echo Nest feature space
    3. Compare vs random baseline and individual views
    4. Statistical significance via paired t-test

Ground truth: 8 standardized Echo Nest features
  Danceability, Energy, Valence, Tempo, Acousticness, Instrumentalness, Liveness, Speechiness
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DIR, FMA_METADATA_DIR

# Paths
SBERT_EMB = PROCESSED_DIR / "sbert_embeddings.npy"
SBERT_IDS = PROCESSED_DIR / "sbert_track_ids.npy"
OPENL3_EMB = PROCESSED_DIR / "openl3_embeddings.npy"
OPENL3_IDS = PROCESSED_DIR / "openl3_track_ids.npy"
CLAP_EMB = PROCESSED_DIR / "clap_embeddings.npy"
CLAP_IDS = PROCESSED_DIR / "clap_track_ids.npy"

TRACKS_CSV = FMA_METADATA_DIR / "tracks.csv"
ECHONEST_CSV = FMA_METADATA_DIR / "echonest.csv"

OUT_JSON = PROCESSED_DIR / "rrf_fusion_results.json"
RRF_K = 60  # RRF smoothing constant

# ─────────────────────────────────────────────────────────────────────────────
# Load and align embeddings
# ─────────────────────────────────────────────────────────────────────────────

def load_and_align():
    """Load embeddings and align by common track IDs."""
    print("Loading embeddings...")

    sbert_emb = np.load(SBERT_EMB).astype(np.float32)
    sbert_ids = np.load(SBERT_IDS).astype(int)

    openl3_emb = np.load(OPENL3_EMB).astype(np.float32)
    openl3_ids = np.load(OPENL3_IDS).astype(int)

    clap_emb = np.load(CLAP_EMB).astype(np.float32)
    clap_ids = np.load(CLAP_IDS).astype(int)

    # Find common IDs across all three views
    common_ids = sorted(
        set(sbert_ids) & set(openl3_ids) & set(clap_ids)
    )
    print(f"  Common tracks across all views: {len(common_ids)}")

    # Align embeddings
    def align(emb, ids, common):
        id_map = {int(tid): i for i, tid in enumerate(ids)}
        indices = np.array([id_map[int(tid)] for tid in common])
        return emb[indices], np.array(common)

    sbert_aligned, _ = align(sbert_emb, sbert_ids, common_ids)
    openl3_aligned, _ = align(openl3_emb, openl3_ids, common_ids)
    clap_aligned, common_aligned = align(clap_emb, clap_ids, common_ids)

    # Normalize
    from sklearn.preprocessing import normalize
    sbert_aligned = normalize(sbert_aligned, axis=1)
    openl3_aligned = normalize(openl3_aligned, axis=1)
    clap_aligned = normalize(clap_aligned, axis=1)

    return {
        "sbert": {"emb": sbert_aligned, "ids": common_aligned},
        "openl3": {"emb": openl3_aligned, "ids": common_aligned},
        "clap": {"emb": clap_aligned, "ids": common_aligned},
    }, common_aligned


def load_echonest_features():
    """Load Echo Nest features and align with track IDs."""
    print("Loading Echo Nest features...")

    echonest = pd.read_csv(ECHONEST_CSV, index_col=0, header=[0, 1, 2])

    # Extract 8 audio_features
    try:
        # Try to access via MultiIndex tuple
        feature_cols = [
            ("echonest", "audio_features", "danceability"),
            ("echonest", "audio_features", "energy"),
            ("echonest", "audio_features", "valence"),
            ("echonest", "audio_features", "tempo"),
            ("echonest", "audio_features", "acousticness"),
            ("echonest", "audio_features", "instrumentalness"),
            ("echonest", "audio_features", "liveness"),
            ("echonest", "audio_features", "speechiness"),
        ]
        features = echonest[feature_cols].copy()
    except KeyError:
        # Fallback: access by level names
        features = echonest[("echonest", "audio_features", slice(None))].copy()
        # Select only the 8 we want
        feature_names = ["danceability", "energy", "valence", "tempo",
                        "acousticness", "instrumentalness", "liveness", "speechiness"]
        features = features[[col for col in features.columns if col[2] in feature_names]]

    features.columns = ["danceability", "energy", "valence", "tempo",
                       "acousticness", "instrumentalness", "liveness", "speechiness"]
    features = features.dropna()

    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    track_ids = features.index.astype(int).values

    print(f"  Loaded {len(features)} tracks with Echo Nest features")

    return track_ids, features_scaled


# ─────────────────────────────────────────────────────────────────────────────
# RRF implementation
# ─────────────────────────────────────────────────────────────────────────────

def top_k_neighbors(emb_matrix, k, exclude_self=True):
    """For each row, return indices of top-k most similar rows (cosine)."""
    sim = emb_matrix @ emb_matrix.T  # (N, N)
    neighbors = []
    for i in range(sim.shape[0]):
        row = sim[i].copy()
        if exclude_self:
            row[i] = -np.inf
        topk = np.argsort(row)[-k:][::-1]
        neighbors.append(topk)
    return np.array(neighbors)


def rrf_fusion(neighbors_list, k_max, rrf_k=60):
    """
    Combine ranked neighbor lists via Reciprocal Rank Fusion.

    neighbors_list: list of (N, k_max) arrays from each view
    rrf_k: smoothing constant (default 60)

    Returns: (N, k_max) array of fused rankings
    """
    N = neighbors_list[0].shape[0]
    rrf_scores = []

    for i in range(N):
        scores_dict = {}

        # Accumulate RRF scores from each view
        for neighbors in neighbors_list:
            for rank, neighbor_idx in enumerate(neighbors[i]):
                if neighbor_idx not in scores_dict:
                    scores_dict[neighbor_idx] = 0.0
                scores_dict[neighbor_idx] += 1.0 / (rrf_k + rank + 1)

        # Sort by RRF score
        sorted_neighbors = sorted(scores_dict.items(), key=lambda x: -x[1])
        top_indices = [idx for idx, _ in sorted_neighbors[:k_max]]

        # Pad with -1 if necessary
        while len(top_indices) < k_max:
            top_indices.append(-1)

        rrf_scores.append(top_indices)

    return np.array(rrf_scores)


def echo_nest_distance(query_idx, neighbor_indices, features, exclude_self=True):
    """
    Compute average Euclidean distance between query and neighbors in Echo Nest space.
    """
    if exclude_self and query_idx in neighbor_indices:
        neighbor_indices = neighbor_indices[neighbor_indices != query_idx]

    valid_neighbors = neighbor_indices[neighbor_indices >= 0]

    if len(valid_neighbors) == 0:
        return np.nan

    query_feat = features[query_idx]
    neighbor_feats = features[valid_neighbors]

    distances = np.linalg.norm(neighbor_feats - query_feat, axis=1)
    return float(np.mean(distances))


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  RRF Fusion Evaluation — Echo Nest Ground Truth")
    print("=" * 70)

    # Load embeddings
    views, common_ids = load_and_align()

    # Load Echo Nest features
    echonest_ids, echonest_features = load_echonest_features()

    # Find intersection with Echo Nest coverage
    echonest_set = set(echonest_ids)
    common_with_echonest = [tid for tid in common_ids if tid in echonest_set]

    # Create mapping
    global_id_to_idx = {int(tid): i for i, tid in enumerate(common_ids)}
    echonest_id_to_idx = {int(tid): i for i, tid in enumerate(echonest_ids)}

    # For each query track with Echo Nest coverage, map it to both global and local indices
    query_mappings = []
    for tid in common_with_echonest:
        query_mappings.append({
            "track_id": tid,
            "global_idx": global_id_to_idx[tid],
            "echonest_idx": echonest_id_to_idx[tid],
        })

    print(f"\nEvaluating {len(query_mappings)} tracks with Echo Nest coverage\n")

    # Compute top-20 neighbors for each view
    K = 20
    print("Computing top-20 neighbors for each view...")
    neighbors = {
        "sbert": top_k_neighbors(views["sbert"]["emb"], K),
        "openl3": top_k_neighbors(views["openl3"]["emb"], K),
        "clap": top_k_neighbors(views["clap"]["emb"], K),
    }

    # Compute RRF fusion
    print("Computing RRF fusion...")
    neighbors_list = [neighbors[v] for v in ["sbert", "openl3", "clap"]]
    rrf_neighbors = rrf_fusion(neighbors_list, K, rrf_k=RRF_K)
    neighbors["rrf"] = rrf_neighbors

    # Evaluate Echo Nest distance
    print("\nEvaluating Echo Nest feature distance (top-5 neighbors)...")
    K_EVAL = 5

    results = {}
    all_distances = {}

    for view_name, nn in neighbors.items():
        distances = []
        for mapping in query_mappings:
            global_idx = mapping["global_idx"]
            echonest_idx = mapping["echonest_idx"]

            # Get neighbor indices from the global embedding space
            neighbor_global_indices = nn[global_idx, :K_EVAL]

            # Map neighbor global indices to track IDs, then to echonest indices
            neighbor_echonest_indices = []
            for neighbor_global_idx in neighbor_global_indices:
                if neighbor_global_idx < 0:  # padding
                    continue
                neighbor_tid = int(common_ids[neighbor_global_idx])
                if neighbor_tid in echonest_id_to_idx:
                    neighbor_echonest_indices.append(echonest_id_to_idx[neighbor_tid])

            if len(neighbor_echonest_indices) > 0:
                # Compute distance using echonest indices
                neighbor_echonest_indices = np.array(neighbor_echonest_indices)
                query_feat = echonest_features[echonest_idx]
                neighbor_feats = echonest_features[neighbor_echonest_indices]
                dists = np.linalg.norm(neighbor_feats - query_feat, axis=1)
                d = float(np.mean(dists))
                if not np.isnan(d):
                    distances.append(d)

        if len(distances) > 0:
            mean_dist = float(np.mean(distances))
            std_dist = float(np.std(distances))
            all_distances[view_name] = distances

            results[view_name] = {
                "n_tracks": len(distances),
                "mean_distance": round(mean_dist, 4),
                "std_distance": round(std_dist, 4),
            }

            print(f"  {view_name:10s}: {mean_dist:.4f} (σ={std_dist:.4f})")

    # Compute random baseline
    print("\nComputing random baseline...")
    random_distances = []
    for mapping in query_mappings:
        echonest_idx = mapping["echonest_idx"]
        # Sample 5 random neighbors
        rand_indices = np.random.choice(len(echonest_features), 5, replace=False)
        query_feat = echonest_features[echonest_idx]
        neighbor_feats = echonest_features[rand_indices]
        dists = np.linalg.norm(neighbor_feats - query_feat, axis=1)
        d = float(np.mean(dists))
        random_distances.append(d)

    baseline_mean = float(np.mean(random_distances))
    results["random_baseline"] = {
        "n_tracks": len(random_distances),
        "mean_distance": round(baseline_mean, 4),
    }
    print(f"  random baseline: {baseline_mean:.4f}")

    # Compute improvements
    print("\n" + "=" * 70)
    print("  Improvements vs Random Baseline")
    print("=" * 70)

    for view_name in ["sbert", "openl3", "clap", "rrf"]:
        improvement = (baseline_mean - results[view_name]["mean_distance"]) / baseline_mean
        results[view_name]["improvement_pct"] = round(improvement * 100, 1)
        print(f"  {view_name:10s}: {improvement*100:+6.1f}%")

    # Statistical significance (paired t-test)
    from scipy.stats import ttest_rel
    print("\n" + "=" * 70)
    print("  Statistical Significance (paired t-test vs random)")
    print("=" * 70)

    # Ensure arrays have same length
    min_len = min(len(random_distances), *[len(d) for d in all_distances.values()])
    random_distances_trimmed = random_distances[:min_len]

    for view_name in ["sbert", "openl3", "clap", "rrf"]:
        distances_trimmed = all_distances[view_name][:min_len]
        t_stat, p_value = ttest_rel(random_distances_trimmed, distances_trimmed)
        results[view_name]["p_value"] = f"{p_value:.2e}"
        sig = "✓✓✓" if p_value < 1e-10 else "✓✓" if p_value < 1e-5 else "✓" if p_value < 0.05 else "✗"
        print(f"  {view_name:10s}: p={p_value:.2e}  {sig}")

    # Save results
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved: {OUT_JSON}")

    print("\n" + "=" * 70)
    print("  KEY FINDING")
    print("=" * 70)
    rrf_imp = results["rrf"]["improvement_pct"]
    best_single = max(
        results[v]["improvement_pct"]
        for v in ["sbert", "openl3", "clap"]
    )
    print(f"  RRF improvement:     {rrf_imp:+.1f}%")
    print(f"  Best single view:    {best_single:+.1f}%")
    if rrf_imp > best_single:
        print(f"  ✓ RRF beats individual views by {rrf_imp - best_single:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
