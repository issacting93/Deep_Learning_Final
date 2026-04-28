"""
Generate t-SNE cluster plots and genre-centroid heatmaps for each pipeline:
  1. CLAP  (512-d, 7997 tracks)
  2. SBERT (384-d, 2000 tracks)
  3. OpenL3 (512-d, 2000 tracks)

Outputs saved to data/processed/viz/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine

from src.config import PROCESSED_DIR, FMA_METADATA_DIR

# ── Output dir ──────────────────────────────────────────────────────────────
VIZ_DIR = PROCESSED_DIR / "viz"
VIZ_DIR.mkdir(exist_ok=True)

# ── Load genre labels ───────────────────────────────────────────────────────
print("Loading metadata...")
tracks = pd.read_csv(FMA_METADATA_DIR / "tracks.csv", index_col=0, header=[0, 1])
genre_col = tracks[("track", "genre_top")]

GENRE_COLORS = {
    "Electronic":    "#6366f1",
    "Experimental":  "#f59e0b",
    "Folk":          "#10b981",
    "Hip-Hop":       "#ef4444",
    "Instrumental":  "#8b5cf6",
    "International": "#06b6d4",
    "Pop":           "#ec4899",
    "Rock":          "#f97316",
}
GENRES_ORDERED = ["Electronic", "Experimental", "Folk", "Hip-Hop",
                  "Instrumental", "International", "Pop", "Rock"]

# ── Pipeline definitions ────────────────────────────────────────────────────
PIPELINES = {
    "CLAP": {
        "emb_file":  PROCESSED_DIR / "clap_embeddings.npy",
        "ids_file":  PROCESSED_DIR / "clap_track_ids.npy",
        "dims": 512,
        "desc": "Vibe / Text-to-Music (512-d)",
    },
    "SBERT": {
        "emb_file":  PROCESSED_DIR / "sbert_embeddings.npy",
        "ids_file":  PROCESSED_DIR / "sbert_track_ids.npy",
        "dims": 384,
        "desc": "Lyrics & Semantic Search (384-d)",
    },
    "OpenL3": {
        "emb_file":  PROCESSED_DIR / "openl3_embeddings.npy",
        "ids_file":  PROCESSED_DIR / "openl3_track_ids.npy",
        "dims": 512,
        "desc": "Acoustic Similarity (512-d, mean-centred)",
    },
}


def load_pipeline(name):
    """Load embeddings and align with genre labels."""
    cfg = PIPELINES[name]
    embs = np.load(cfg["emb_file"]).astype(np.float32)
    ids = np.load(cfg["ids_file"])
    ids = np.array([int(i) for i in ids])

    # Mean-centre ALL embeddings to remove dominant shared component for a fair visual comparison
    # (this reveals the actual genre/acoustic structure across all models)
    embs = embs - embs.mean(axis=0)

    # L2-normalise
    embs = normalize(embs, norm="l2")

    # Match with genre labels
    genres = []
    valid_mask = []
    for tid in ids:
        g = genre_col.get(tid, np.nan)
        if pd.notna(g) and g in GENRE_COLORS:
            genres.append(g)
            valid_mask.append(True)
        else:
            genres.append(None)
            valid_mask.append(False)

    valid_mask = np.array(valid_mask)
    embs_valid = embs[valid_mask]
    ids_valid = ids[valid_mask]
    genres_valid = np.array([g for g in genres if g is not None])

    print(f"  {name}: {embs_valid.shape[0]} tracks with genre labels "
          f"(of {embs.shape[0]} total)")
    return embs_valid, ids_valid, genres_valid


# ── t-SNE plot ──────────────────────────────────────────────────────────────
def plot_tsne(embs, genres, name, desc, perplexity=30, seed=42):
    """Run t-SNE and produce a scatter plot coloured by genre."""
    print(f"  Running t-SNE for {name} ({embs.shape[0]} points)...")

    # Subsample for speed if >4000 tracks
    max_points = 4000
    if embs.shape[0] > max_points:
        rng = np.random.RandomState(seed)
        idx = rng.choice(embs.shape[0], max_points, replace=False)
        embs_sub = embs[idx]
        genres_sub = genres[idx]
        sample_note = f" (random {max_points} of {embs.shape[0]})"
    else:
        embs_sub = embs
        genres_sub = genres
        sample_note = ""

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                n_iter=1000, learning_rate="auto", init="pca")
    coords = tsne.fit_transform(embs_sub)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    for genre in GENRES_ORDERED:
        mask = genres_sub == genre
        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=GENRE_COLORS[genre], label=genre,
                   s=12, alpha=0.65, edgecolors="none")

    ax.set_title(f"{name}: t-SNE Clustering by Genre\n{desc}{sample_note}",
                 color="white", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("t-SNE 1", color="#94a3b8", fontsize=11)
    ax.set_ylabel("t-SNE 2", color="#94a3b8", fontsize=11)
    ax.tick_params(colors="#64748b", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#1e293b")

    legend = ax.legend(loc="upper right", fontsize=9, framealpha=0.85,
                       facecolor="#1e293b", edgecolor="#334155",
                       labelcolor="white", markerscale=2)

    path = VIZ_DIR / f"tsne_{name.lower()}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")
    return coords


# ── PCA plot ────────────────────────────────────────────────────────────────
def plot_pca(embs, genres, name, desc):
    """Run PCA and produce a scatter plot coloured by genre."""
    print(f"  Running PCA for {name} ({embs.shape[0]} points)...")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embs)
    var_explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    for genre in GENRES_ORDERED:
        mask = genres == genre
        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=GENRE_COLORS[genre], label=genre,
                   s=12, alpha=0.65, edgecolors="none")

    ax.set_title(f"{name}: PCA Clustering by Genre\n{desc} (var: {var_explained.sum():.1%})",
                 color="white", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%})", color="#94a3b8", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%})", color="#94a3b8", fontsize=11)
    ax.tick_params(colors="#64748b", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#1e293b")

    legend = ax.legend(loc="upper right", fontsize=9, framealpha=0.85,
                       facecolor="#1e293b", edgecolor="#334155",
                       labelcolor="white", markerscale=2)

    path = VIZ_DIR / f"pca_{name.lower()}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")
    return coords


# ── Genre centroid heatmap ──────────────────────────────────────────────────
def plot_heatmap(embs, genres, name, desc):
    """Compute cosine similarity between genre centroids and plot a heatmap."""
    # Compute centroids
    centroids = {}
    for genre in GENRES_ORDERED:
        mask = genres == genre
        if mask.sum() > 0:
            centroid = embs[mask].mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            centroids[genre] = centroid

    present = [g for g in GENRES_ORDERED if g in centroids]
    n = len(present)
    sim_matrix = np.zeros((n, n))
    for i, g1 in enumerate(present):
        for j, g2 in enumerate(present):
            sim_matrix[i, j] = np.dot(centroids[g1], centroids[g2])

    fig, ax = plt.subplots(figsize=(9, 7.5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    # Adaptive color scale: use symmetric range for mean-centred embeddings
    vmin = max(sim_matrix[~np.eye(n, dtype=bool)].min() - 0.05, -1.0)
    vmax = 1.0
    im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(present, rotation=45, ha="right", color="white", fontsize=10)
    ax.set_yticklabels(present, color="white", fontsize=10)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = sim_matrix[i, j]
            text_color = "white" if val < 0.65 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=text_color)

    ax.set_title(f"{name}: Genre Centroid Cosine Similarity\n{desc}",
                 color="white", fontsize=14, fontweight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.ax.tick_params(labelcolor="white", labelsize=9)
    cbar.set_label("Cosine Similarity", color="white", fontsize=10)

    for spine in ax.spines.values():
        spine.set_color("#1e293b")

    path = VIZ_DIR / f"heatmap_{name.lower()}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")
    return sim_matrix, present


# ── Side-by-side t-SNE comparison ──────────────────────────────────────────
def plot_tsne_comparison(all_coords, all_genres, all_names):
    """3-panel t-SNE comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.patch.set_facecolor("#0f172a")

    for ax, (coords, genres_arr, name) in zip(axes, zip(all_coords, all_genres, all_names)):
        ax.set_facecolor("#0f172a")
        for genre in GENRES_ORDERED:
            mask = genres_arr == genre
            if mask.sum() == 0:
                continue
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=GENRE_COLORS[genre], label=genre,
                       s=8, alpha=0.6, edgecolors="none")
        ax.set_title(name, color="white", fontsize=14, fontweight="bold", pad=8)
        ax.set_xlabel("t-SNE 1", color="#94a3b8", fontsize=10)
        ax.set_ylabel("t-SNE 2", color="#94a3b8", fontsize=10)
        ax.tick_params(colors="#64748b", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#1e293b")

    # Shared legend
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=GENRE_COLORS[g],
                       markersize=8, linestyle="None") for g in GENRES_ORDERED]
    fig.legend(handles, GENRES_ORDERED, loc="lower center", ncol=8,
               fontsize=10, framealpha=0.85, facecolor="#1e293b",
               edgecolor="#334155", labelcolor="white",
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("t-SNE Genre Clustering: CLAP vs SBERT vs OpenL3",
                 color="white", fontsize=16, fontweight="bold", y=1.02)

    path = VIZ_DIR / "tsne_comparison_3panel.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Side-by-side PCA comparison ──────────────────────────────────────────
def plot_pca_comparison(all_coords, all_genres, all_names):
    """3-panel PCA comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.patch.set_facecolor("#0f172a")

    for ax, (coords, genres_arr, name) in zip(axes, zip(all_coords, all_genres, all_names)):
        ax.set_facecolor("#0f172a")
        for genre in GENRES_ORDERED:
            mask = genres_arr == genre
            if mask.sum() == 0:
                continue
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=GENRE_COLORS[genre], label=genre,
                       s=8, alpha=0.6, edgecolors="none")
        ax.set_title(name, color="white", fontsize=14, fontweight="bold", pad=8)
        ax.set_xlabel("PC1", color="#94a3b8", fontsize=10)
        ax.set_ylabel("PC2", color="#94a3b8", fontsize=10)
        ax.tick_params(colors="#64748b", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#1e293b")

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=GENRE_COLORS[g],
                       markersize=8, linestyle="None") for g in GENRES_ORDERED]
    fig.legend(handles, GENRES_ORDERED, loc="lower center", ncol=8,
               fontsize=10, framealpha=0.85, facecolor="#1e293b",
               edgecolor="#334155", labelcolor="white",
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("PCA Genre Clustering: CLAP vs SBERT vs OpenL3",
                 color="white", fontsize=16, fontweight="bold", y=1.02)

    path = VIZ_DIR / "pca_comparison_3panel.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Side-by-side heatmap comparison ─────────────────────────────────────────
def plot_heatmap_comparison(all_matrices, all_present, all_names):
    """3-panel heatmap comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.patch.set_facecolor("#0f172a")

    # Compute global color range across all panels
    global_min = min(m[~np.eye(len(p), dtype=bool)].min() for m, p in zip(all_matrices, all_present))
    global_vmin = max(global_min - 0.05, -1.0)

    for ax, (matrix, present, name) in zip(axes, zip(all_matrices, all_present, all_names)):
        ax.set_facecolor("#0f172a")
        n = len(present)
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=global_vmin, vmax=1.0, aspect="equal")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(present, rotation=45, ha="right", color="white", fontsize=9)
        ax.set_yticklabels(present, color="white", fontsize=9)
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                tc = "white" if val < 0.3 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7.5, fontweight="bold", color=tc)
        ax.set_title(name, color="white", fontsize=13, fontweight="bold", pad=8)
        for spine in ax.spines.values():
            spine.set_color("#1e293b")

    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.03)
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.ax.tick_params(labelcolor="white", labelsize=9)
    cbar.set_label("Cosine Similarity", color="white", fontsize=10)

    fig.suptitle("Genre Centroid Similarity: CLAP vs SBERT vs OpenL3",
                 color="white", fontsize=16, fontweight="bold", y=1.02)

    path = VIZ_DIR / "heatmap_comparison_3panel.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_coords = []
    all_pca_coords = []
    all_genres = []
    all_names = []
    all_matrices = []
    all_present = []

    for name, cfg in PIPELINES.items():
        print(f"\n{'='*60}")
        print(f"  Pipeline: {name}")
        print(f"{'='*60}")
        embs, ids, genres = load_pipeline(name)

        coords = plot_tsne(embs, genres, name, cfg["desc"])
        pca_coords = plot_pca(embs, genres, name, cfg["desc"])
        matrix, present = plot_heatmap(embs, genres, name, cfg["desc"])

        # For comparison plots, use common 2000-track subset
        # (SBERT and OpenL3 are already 2000; subsample CLAP)
        if embs.shape[0] > 2000:
            rng = np.random.RandomState(42)
            idx = rng.choice(embs.shape[0], 2000, replace=False)
            embs_sub, genres_sub = embs[idx], genres[idx]
        else:
            embs_sub, genres_sub = embs, genres

        tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                     n_iter=1000, learning_rate="auto", init="pca")
        c = tsne.fit_transform(embs_sub)
        all_coords.append(c)

        pca = PCA(n_components=2, random_state=42)
        c_pca = pca.fit_transform(embs_sub)
        all_pca_coords.append(c_pca)

        all_genres.append(genres_sub)
        all_names.append(name)
        all_matrices.append(matrix)
        all_present.append(present)

    print(f"\n{'='*60}")
    print("  Generating comparison panels...")
    print(f"{'='*60}")
    plot_tsne_comparison(all_coords, all_genres, all_names)
    plot_pca_comparison(all_pca_coords, all_genres, all_names)
    plot_heatmap_comparison(all_matrices, all_present, all_names)

    print("\nDone! All visualizations saved to data/processed/viz/")
