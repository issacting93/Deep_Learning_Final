"""
openl3_vs_sbert_overlap.py
--------------------------
Compares OpenL3 audio embeddings (Role 1) with SBERT text embeddings (Role 2)
across the 2000-track FMA subset.

Analyses performed:
  1. Rank Correlation  — Spearman ρ between OpenL3 and SBERT top-k neighbour lists
  2. Retrieval Overlap — Overlap@k for a fixed set of seed queries
  3. t-SNE Alignment  — Side-by-side visualisation coloured by genre
  4. Per-genre agreement heatmap

Outputs saved to: data/processed/
  - openl3_sbert_overlap_results.json
  - openl3_sbert_rank_corr.png
  - openl3_sbert_tsne_comparison.png
  - openl3_sbert_genre_heatmap.png
"""

import os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
PROC      = ROOT / "data" / "processed"
META_DIR  = ROOT / "data" / "fma_2000_metadata"

OPENL3_EMB  = PROC / "openl3_embeddings.npy"
OPENL3_IDS  = PROC / "openl3_track_ids.npy"
SBERT_EMB   = PROC / "sbert_embeddings.npy"
SBERT_IDS   = PROC / "sbert_track_ids.npy"
TRACKS_CSV  = META_DIR / "tracks.csv"

OUT_JSON    = PROC / "openl3_sbert_overlap_results.json"
OUT_CORR    = PROC / "openl3_sbert_rank_corr.png"
OUT_TSNE    = PROC / "openl3_sbert_tsne_comparison.png"
OUT_HEATMAP = PROC / "openl3_sbert_genre_heatmap.png"

K_VALUES    = [5, 10, 20, 50]

# ── helpers ────────────────────────────────────────────────────────────────────

def cosine_sim_matrix(A, B):
    """Return (N, M) cosine similarity matrix between row-normalised A and B."""
    A_n = normalize(A, axis=1)
    B_n = normalize(B, axis=1)
    return A_n @ B_n.T     # (N, M)


def top_k_neighbours(sim_matrix, k, exclude_self=True):
    """
    For each row i return the indices of the top-k most similar rows
    (in sim_matrix), excluding self when exclude_self=True.
    Returns shape (N, k).
    """
    N = sim_matrix.shape[1]
    neighbours = []
    for i in range(sim_matrix.shape[0]):
        row = sim_matrix[i].copy()
        if exclude_self:
            row[i] = -np.inf
        topk = np.argpartition(row, -k)[-k:]
        topk = topk[np.argsort(row[topk])[::-1]]
        neighbours.append(topk)
    return np.array(neighbours)   # (N, k)


def overlap_at_k(nn_a, nn_b, k):
    """Mean Overlap@k between two (N, K_max) neighbour arrays, using only top k."""
    overlaps = []
    for a_row, b_row in zip(nn_a, nn_b):
        a_set = set(a_row[:k].tolist())
        b_set = set(b_row[:k].tolist())
        overlaps.append(len(a_set & b_set) / k)
    return float(np.mean(overlaps))


def rank_correlation_per_track(sim_openl3, sim_sbert):
    """
    For each track compute Spearman ρ between its full similarity score
    vector in OpenL3-space vs SBERT-space (over all other tracks).
    Returns array of ρ values of shape (N,).
    """
    N = sim_openl3.shape[0]
    rhos = []
    for i in range(N):
        rho, _ = spearmanr(sim_openl3[i], sim_sbert[i])
        rhos.append(rho)
    return np.array(rhos)


# ── load data ──────────────────────────────────────────────────────────────────

def load_and_align():
    print("Loading embeddings ...")
    openl3_emb = np.load(OPENL3_EMB).astype(np.float32)
    openl3_ids = np.array([int(x) for x in np.load(OPENL3_IDS)])

    sbert_emb  = np.load(SBERT_EMB).astype(np.float32)
    sbert_ids  = np.load(SBERT_IDS).astype(int)

    # align by track ID
    common_ids = sorted(set(openl3_ids) & set(sbert_ids))
    print(f"  Common tracks: {len(common_ids)}")

    o_idx = {tid: i for i, tid in enumerate(openl3_ids)}
    s_idx = {tid: i for i, tid in enumerate(sbert_ids)}

    order    = np.array(common_ids)
    o_rows   = np.array([o_idx[t] for t in order])
    s_rows   = np.array([s_idx[t] for t in order])

    openl3_aligned = openl3_emb[o_rows]
    sbert_aligned  = sbert_emb[s_rows]

    return openl3_aligned, sbert_aligned, order


def load_genres(track_ids):
    """Load top-level genre for each track ID. Returns a Series indexed by track_id."""
    try:
        tracks = pd.read_csv(TRACKS_CSV, index_col=0, header=[0, 1])
        genre_series = tracks[("track", "genre_top")].dropna()
        genre_map = {int(idx): str(g) for idx, g in genre_series.items()}
        return [genre_map.get(tid, "Unknown") for tid in track_ids]
    except Exception as e:
        print(f"  [warn] could not load genre data: {e}")
        return ["Unknown"] * len(track_ids)


# ── analysis ───────────────────────────────────────────────────────────────────

def run_overlap_analysis(openl3, sbert):
    print("\n[1] Computing similarity matrices ...")
    sim_o = cosine_sim_matrix(openl3, openl3)   # (N, N)
    sim_s = cosine_sim_matrix(sbert,  sbert)    # (N, N)

    K_MAX = max(K_VALUES)
    print(f"[2] Computing top-{K_MAX} neighbours for each modality ...")
    nn_o = top_k_neighbours(sim_o, K_MAX, exclude_self=True)
    nn_s = top_k_neighbours(sim_s, K_MAX, exclude_self=True)

    print("[3] Overlap@k ...")
    overlap = {}
    for k in K_VALUES:
        ov = overlap_at_k(nn_o, nn_s, k)
        overlap[f"overlap_at_{k}"] = round(ov, 4)
        print(f"    Overlap@{k:>2d}: {ov:.4f}  ({ov*100:.1f}%)")

    print("[4] Per-track Spearman rank correlation ...")
    rhos = rank_correlation_per_track(sim_o, sim_s)
    mean_rho = float(np.mean(rhos))
    std_rho  = float(np.std(rhos))
    print(f"    Mean Spearman ρ: {mean_rho:.4f}  (σ={std_rho:.4f})")

    results = {
        "n_tracks": len(openl3),
        "k_values": K_VALUES,
        **overlap,
        "mean_spearman_rho": round(mean_rho, 4),
        "std_spearman_rho":  round(std_rho, 4),
        "pct_positive_rho":  round(float((rhos > 0).mean() * 100), 2),
    }
    return results, rhos, nn_o, nn_s, sim_o, sim_s


# ── plotting ───────────────────────────────────────────────────────────────────

PALETTE = {
    "bg":      "#0f1117",
    "panel":   "#1a1d27",
    "accent1": "#6c63ff",
    "accent2": "#ff6584",
    "accent3": "#43d9ad",
    "text":    "#e0e0e0",
    "subtext": "#888ea8",
}


def _apply_style():
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["panel"],
        "axes.edgecolor":    PALETTE["subtext"],
        "axes.labelcolor":   PALETTE["text"],
        "xtick.color":       PALETTE["subtext"],
        "ytick.color":       PALETTE["subtext"],
        "text.color":        PALETTE["text"],
        "grid.color":        "#2a2d3e",
        "grid.linewidth":    0.6,
        "font.family":       "DejaVu Sans",
        "font.size":         10,
    })


def plot_rank_correlation(rhos, results, genres, unique_genres):
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle(
        "OpenL3 (Audio)  vs  SBERT (Text)  —  Rank Correlation Analysis",
        fontsize=14, color=PALETTE["text"], y=1.02
    )

    # --- Histogram of Spearman ρ values
    ax = axes[0]
    ax.hist(rhos, bins=60, color=PALETTE["accent1"], alpha=0.85, edgecolor="none")
    ax.axvline(results["mean_spearman_rho"], color=PALETTE["accent2"],
               linewidth=2, linestyle="--", label=f"Mean ρ = {results['mean_spearman_rho']:.3f}")
    ax.axvline(0, color=PALETTE["subtext"], linewidth=1, linestyle=":")
    ax.set_title("Spearman ρ Distribution", color=PALETTE["text"])
    ax.set_xlabel("Spearman ρ  (per track)")
    ax.set_ylabel("Count")
    ax.legend(framealpha=0.2, labelcolor=PALETTE["text"])
    ax.grid(True, alpha=0.3)

    # --- Overlap@k bar chart
    ax = axes[1]
    ks  = [str(k) for k in K_VALUES]
    ovs = [results[f"overlap_at_{k}"] * 100 for k in K_VALUES]
    bars = ax.bar(ks, ovs, color=PALETTE["accent3"], alpha=0.85, width=0.5)
    for bar, val in zip(bars, ovs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom",
                color=PALETTE["text"], fontsize=9)
    ax.set_title("Retrieval Overlap @ k", color=PALETTE["text"])
    ax.set_xlabel("k  (neighbours)")
    ax.set_ylabel("Overlap  (%)")
    ax.set_ylim(0, max(ovs) * 1.25 + 2)
    ax.grid(True, axis="y", alpha=0.3)

    # --- ρ by genre box-plot
    ax = axes[2]
    genre_rhos = {}
    for g, rho in zip(genres, rhos):
        genre_rhos.setdefault(g, []).append(rho)

    # sort by median
    sorted_genres = sorted(genre_rhos, key=lambda g: np.median(genre_rhos[g]), reverse=True)
    data_for_bp   = [genre_rhos[g] for g in sorted_genres]
    bp = ax.boxplot(
        data_for_bp,
        vert=False,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color=PALETTE["accent2"], linewidth=2),
    )
    genre_colors = plt.cm.cool(np.linspace(0.2, 0.9, len(sorted_genres)))
    for patch, color in zip(bp["boxes"], genre_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for element in ["whiskers", "caps", "fliers"]:
        for item in bp[element]:
            item.set_color(PALETTE["subtext"])

    ax.set_yticks(range(1, len(sorted_genres) + 1))
    ax.set_yticklabels(sorted_genres, fontsize=8)
    ax.axvline(0, color=PALETTE["subtext"], linewidth=1, linestyle=":")
    ax.set_title("Spearman ρ by Genre", color=PALETTE["text"])
    ax.set_xlabel("Spearman ρ")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_CORR, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"  Saved: {OUT_CORR}")
    plt.close()


def plot_tsne(openl3, sbert, genres, unique_genres):
    _apply_style()
    print("\n[5] Running t-SNE (this may take ~30 s) ...")
    perp = min(40, len(openl3) // 10)

    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000)
    # run separately — embeddings are different dimensions (512 vs 384)
    proj_o = tsne.fit_transform(normalize(openl3, axis=1))
    proj_s = tsne.fit_transform(normalize(sbert,  axis=1))

    cmap = plt.cm.get_cmap("tab20", len(unique_genres))
    genre_color = {g: cmap(i) for i, g in enumerate(unique_genres)}
    colors = [genre_color[g] for g in genres]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("t-SNE: OpenL3 Audio  vs  SBERT Text  (coloured by genre)",
                 fontsize=14, color=PALETTE["text"], y=1.01)

    titles  = ["OpenL3  (Audio)", "SBERT  (Text)"]
    projs   = [proj_o, proj_s]

    for ax, title, proj_xy in zip(axes, titles, projs):
        for g in unique_genres:
            mask = np.array([gi == g for gi in genres])
            ax.scatter(proj_xy[mask, 0], proj_xy[mask, 1],
                       c=[genre_color[g]], label=g, s=12, alpha=0.75, linewidths=0)
        ax.set_title(title, color=PALETTE["text"], fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor(PALETTE["panel"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["subtext"])

    # shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(unique_genres), 8),
               fontsize=8, framealpha=0.2, labelcolor=PALETTE["text"],
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    fig.savefig(OUT_TSNE, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"  Saved: {OUT_TSNE}")
    plt.close()


def plot_genre_heatmap(nn_o, nn_s, genres, unique_genres, k=10):
    """
    Genre-level agreement heatmap:
    For genre G, what fraction of its OpenL3 top-k neighbours share genre H?
    Do the same for SBERT. Plot side-by-side.
    """
    _apply_style()
    genres_arr = np.array(genres)

    def genre_matrix(nn, k, unique_genres):
        n_g = len(unique_genres)
        g2i = {g: i for i, g in enumerate(unique_genres)}
        mat = np.zeros((n_g, n_g))
        counts = np.zeros(n_g)

        for i, row in enumerate(nn):
            src_g = genres_arr[i]
            counts[g2i[src_g]] += 1
            for j in row[:k]:
                mat[g2i[src_g], g2i[genres_arr[j]]] += 1

        # normalise rows
        for r in range(n_g):
            if counts[r] > 0:
                mat[r] /= (counts[r] * k)
        return mat

    mat_o = genre_matrix(nn_o, k, unique_genres)
    mat_s = genre_matrix(nn_s, k, unique_genres)
    diff  = mat_s - mat_o   # SBERT minus OpenL3

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle(f"Genre Agreement Heatmap @ k={k}  (row = source genre, col = neighbour genre)",
                 fontsize=13, color=PALETTE["text"], y=1.01)

    labels = [g[:10] for g in unique_genres]

    sns_kw = dict(annot=True, fmt=".2f", linewidths=0.3, linecolor=PALETTE["bg"],
                  xticklabels=labels, yticklabels=labels, annot_kws={"size": 7})

    for ax, mat, title, cmap in zip(
        axes,
        [mat_o, mat_s, diff],
        ["OpenL3 (Audio)", "SBERT (Text)", "ΔSBERT − OpenL3"],
        ["Blues", "Purples", "RdBu_r"],
    ):
        sns.heatmap(mat, ax=ax, cmap=cmap, vmin=(None if "Δ" not in title else -0.3),
                    vmax=(None if "Δ" not in title else 0.3), **sns_kw)
        ax.set_title(title, color=PALETTE["text"], fontsize=11)
        ax.tick_params(colors=PALETTE["subtext"], labelsize=8)
        ax.set_facecolor(PALETTE["panel"])

    plt.tight_layout()
    fig.savefig(OUT_HEATMAP, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"  Saved: {OUT_HEATMAP}")
    plt.close()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  OpenL3 vs SBERT — Overlap & Rank Correlation Analysis")
    print("=" * 60)

    openl3, sbert, track_ids = load_and_align()

    genres = load_genres(track_ids.tolist())
    unique_genres = sorted(set(genres))
    print(f"  Genres found: {unique_genres}")

    results, rhos, nn_o, nn_s, sim_o, sim_s = run_overlap_analysis(openl3, sbert)

    # save JSON
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {OUT_JSON}")

    # plots
    print("\n[Plotting] rank correlation figure ...")
    plot_rank_correlation(rhos, results, genres, unique_genres)

    print("[Plotting] genre agreement heatmap ...")
    plot_genre_heatmap(nn_o, nn_s, genres, unique_genres, k=10)

    print("[Plotting] t-SNE alignment ...")
    plot_tsne(openl3, sbert, genres, unique_genres)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k:<30} {v}")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
