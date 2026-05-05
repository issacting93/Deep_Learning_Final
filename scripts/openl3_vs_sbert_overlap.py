"""
cross_view_overlap.py
---------------------
Compares embeddings across all view pairs (OpenL3, SBERT, CLAP)
across the 2000-track FMA subset.

Analyses performed for each view pair:
  1. Rank Correlation  — Spearman ρ between view top-k neighbour lists
  2. Retrieval Overlap — Overlap@k
  3. Per-genre agreement heatmap

Outputs saved to: data/processed/
  - cross_view_overlap_results.json (summary of all pairs)
  - {view1}_{view2}_rank_corr.png
  - {view1}_{view2}_genre_heatmap.png
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
CLAP_EMB    = PROC / "clap_embeddings.npy"
CLAP_IDS    = PROC / "clap_track_ids.npy"
TRACKS_CSV  = META_DIR / "tracks.csv"

OUT_JSON    = PROC / "cross_view_overlap_results.json"

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

def load_all_views():
    """Load and align all three views by track ID."""
    print("Loading embeddings ...")

    views = {}

    # Load OpenL3
    if OPENL3_EMB.exists() and OPENL3_IDS.exists():
        views["OpenL3"] = {
            "emb": np.load(OPENL3_EMB).astype(np.float32),
            "ids": np.array([int(x) for x in np.load(OPENL3_IDS)])
        }

    # Load SBERT
    if SBERT_EMB.exists() and SBERT_IDS.exists():
        views["SBERT"] = {
            "emb": np.load(SBERT_EMB).astype(np.float32),
            "ids": np.load(SBERT_IDS).astype(int)
        }

    # Load CLAP
    if CLAP_EMB.exists() and CLAP_IDS.exists():
        views["CLAP"] = {
            "emb": np.load(CLAP_EMB).astype(np.float32),
            "ids": np.load(CLAP_IDS).astype(int)
        }

    print(f"  Loaded {len(views)} views: {list(views.keys())}")

    # Align all views by common track IDs
    all_ids = set(views[list(views.keys())[0]]["ids"])
    for view_name, data in views.items():
        all_ids &= set(data["ids"])

    common_ids = sorted(list(all_ids))
    print(f"  Common tracks across all views: {len(common_ids)}")

    # Align each view
    aligned = {}
    for view_name, data in views.items():
        id_map = {tid: i for i, tid in enumerate(data["ids"])}
        indices = np.array([id_map[t] for t in common_ids])
        aligned[view_name] = data["emb"][indices]

    return aligned, np.array(common_ids)


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

def run_overlap_analysis(emb_a, emb_b):
    """Run overlap analysis between two embeddings. Returns metrics."""
    print("\n[1] Computing similarity matrices ...")
    sim_a = cosine_sim_matrix(emb_a, emb_a)   # (N, N)
    sim_b = cosine_sim_matrix(emb_b, emb_b)   # (N, N)

    K_MAX = max(K_VALUES)
    print(f"[2] Computing top-{K_MAX} neighbours ...")
    nn_a = top_k_neighbours(sim_a, K_MAX, exclude_self=True)
    nn_b = top_k_neighbours(sim_b, K_MAX, exclude_self=True)

    print("[3] Overlap@k ...")
    overlap = {}
    for k in K_VALUES:
        ov = overlap_at_k(nn_a, nn_b, k)
        overlap[f"overlap_at_{k}"] = round(ov, 4)
        print(f"    Overlap@{k:>2d}: {ov:.4f}  ({ov*100:.1f}%)")

    print("[4] Per-track Spearman rank correlation ...")
    rhos = rank_correlation_per_track(sim_a, sim_b)
    mean_rho = float(np.mean(rhos))
    std_rho  = float(np.std(rhos))
    print(f"    Mean Spearman ρ: {mean_rho:.4f}  (σ={std_rho:.4f})")

    results = {
        "n_tracks": len(emb_a),
        "k_values": K_VALUES,
        **overlap,
        "mean_spearman_rho": round(mean_rho, 4),
        "std_spearman_rho":  round(std_rho, 4),
        "pct_positive_rho":  round(float((rhos > 0).mean() * 100), 2),
    }
    return results, rhos, nn_a, nn_b, sim_a, sim_b


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


def plot_rank_correlation(rhos, results, genres, unique_genres, view_a, view_b):
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle(
        f"{view_a}  vs  {view_b}  —  Rank Correlation Analysis",
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
    out_path = PROC / f"{view_a.lower()}_{view_b.lower()}_rank_corr.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"  Saved: {out_path}")
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


def plot_genre_heatmap(nn_a, nn_b, genres, unique_genres, view_a, view_b, k=10):
    """Genre-level agreement heatmap for two views."""
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

        for r in range(n_g):
            if counts[r] > 0:
                mat[r] /= (counts[r] * k)
        return mat

    mat_a = genre_matrix(nn_a, k, unique_genres)
    mat_b = genre_matrix(nn_b, k, unique_genres)
    diff  = mat_b - mat_a

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle(f"{view_a} vs {view_b} — Genre Agreement Heatmap @ k={k}",
                 fontsize=13, color=PALETTE["text"], y=1.01)

    labels = [g[:10] for g in unique_genres]
    sns_kw = dict(annot=True, fmt=".2f", linewidths=0.3, linecolor=PALETTE["bg"],
                  xticklabels=labels, yticklabels=labels, annot_kws={"size": 7})

    for ax, mat, title, cmap in zip(
        axes,
        [mat_a, mat_b, diff],
        [view_a, view_b, f"Δ{view_b} − {view_a}"],
        ["Blues", "Purples", "RdBu_r"],
    ):
        sns.heatmap(mat, ax=ax, cmap=cmap, vmin=(None if "Δ" not in title else -0.3),
                    vmax=(None if "Δ" not in title else 0.3), **sns_kw)
        ax.set_title(title, color=PALETTE["text"], fontsize=11)
        ax.tick_params(colors=PALETTE["subtext"], labelsize=8)
        ax.set_facecolor(PALETTE["panel"])

    plt.tight_layout()
    out_path = PROC / f"{view_a.lower()}_{view_b.lower()}_genre_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"  Saved: {out_path}")
    plt.close()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Cross-View Overlap Analysis (All Pairs)")
    print("=" * 60)

    views, track_ids = load_all_views()

    if len(views) < 2:
        print("ERROR: Need at least 2 views to compare")
        sys.exit(1)

    genres = load_genres(track_ids.tolist())
    unique_genres = sorted(set(genres))
    print(f"  Genres found: {unique_genres}\n")

    # Generate all view pairs
    view_names = sorted(views.keys())
    pairs = []
    for i, va in enumerate(view_names):
        for vb in view_names[i+1:]:
            pairs.append((va, vb))

    all_results = {}

    # Analyze each pair
    for view_a, view_b in pairs:
        print(f"\n{'=' * 60}")
        print(f"  {view_a} vs {view_b}")
        print('=' * 60)

        results, rhos, nn_a, nn_b, sim_a, sim_b = run_overlap_analysis(
            views[view_a], views[view_b]
        )

        # Store results
        pair_key = f"{view_a.lower()}_{view_b.lower()}"
        all_results[pair_key] = results

        # Plots
        print(f"\n[Plotting] rank correlation figure ...")
        plot_rank_correlation(rhos, results, genres, unique_genres, view_a, view_b)

        print(f"[Plotting] genre agreement heatmap ...")
        plot_genre_heatmap(nn_a, nn_b, genres, unique_genres, view_a, view_b, k=10)

    # Save combined JSON
    with open(OUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nAll results saved: {OUT_JSON}")

    # Print summary
    print("\n" + "=" * 70)
    print("  SUMMARY — All View Pairs")
    print("=" * 70)
    for pair_key, results in all_results.items():
        print(f"\n{pair_key.replace('_', ' ').upper()}:")
        print(f"  Mean Spearman ρ:  {results['mean_spearman_rho']:.4f}")
        print(f"  Overlap@10:       {results['overlap_at_10']*100:.1f}%")
        print(f"  Overlap@20:       {results['overlap_at_20']*100:.1f}%")
    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
