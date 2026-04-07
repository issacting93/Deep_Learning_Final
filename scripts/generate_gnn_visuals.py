"""
generate_gnn_visuals.py — Role 3: Generate all graph visualisations

Produces:
  data/processed/gnn_genre_subgraph.png    — genre network diagram
  data/processed/gnn_ego_graph.png         — 2-hop neighbourhood of a seed track
  data/processed/gnn_tsne_comparison.png  — GNN vs CLAP t-SNE side-by-side
  data/processed/gnn_degree_distribution.png — artist/track degree histograms

Run from project root (with venv active):
    python scripts/generate_gnn_visuals.py
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import Counter

sys.path.insert(0, ".")
from src.graph.build_graph import build_hetero_graph
from src.metadata import load_tracks, load_genres, get_small_subset_ids
from src.config import PROCESSED_DIR

DARK_BG  = "#0d1117"
ACCENT1  = "#58a6ff"
ACCENT2  = "#f78166"
GRID_COL = "#21262d"

# ─────────────────────────────────────────────────────────────────────────────
print("Building graph...")
data, id_maps = build_hetero_graph(verbose=False)

tracks_df  = load_tracks()
genres_df  = load_genres()
small_ids  = get_small_subset_ids(tracks_df)
tracks_s   = tracks_df.loc[tracks_df.index.isin(small_ids)].copy()

# Flatten multi-index columns for easy access
tracks_s.columns = ["_".join(c).strip("_") for c in tracks_s.columns]

print(f"Tracks in small subset: {len(tracks_s)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Genre Subgraph
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1/4] Genre subgraph...")

# Group by artist_id → collect all genres they appear in
artist_genre_map: dict = {}
for tid, row in tracks_s.iterrows():
    a_id     = row.get("artist_id", None)
    genre    = row.get("track_genre_top", None)
    if a_id is None or not isinstance(genre, str):
        continue
    a_id = int(a_id) if not np.isnan(float(a_id)) else None
    if a_id is None:
        continue
    artist_genre_map.setdefault(a_id, set()).add(genre)

genre_cooccur: Counter = Counter()
for genres_set in artist_genre_map.values():
    glist = sorted(genres_set)
    for i in range(len(glist)):
        for j in range(i + 1, len(glist)):
            genre_cooccur[tuple(sorted([glist[i], glist[j]]))] += 1

G_genre = nx.Graph()
for (g1, g2), w in genre_cooccur.items():
    if w >= 2:
        G_genre.add_edge(g1, g2, weight=w)

genre_counts = tracks_s["track_genre_top"].value_counts()
node_sizes   = [min(max(genre_counts.get(g, 1) * 5, 80), 1800) for g in G_genre.nodes()]
weights      = [G_genre[u][v]["weight"] for u, v in G_genre.edges()]
max_w        = max(weights) if weights else 1

fig, ax = plt.subplots(figsize=(16, 12))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(DARK_BG)

pos = nx.spring_layout(G_genre, seed=42, k=3.2)

# Edges
nx.draw_networkx_edges(
    G_genre, pos,
    width=[1 + 5 * w / max_w for w in weights],
    alpha=0.45,
    edge_color=ACCENT1,
    ax=ax,
)
# Nodes
nx.draw_networkx_nodes(
    G_genre, pos,
    node_size=node_sizes,
    node_color=ACCENT1,
    alpha=0.88,
    ax=ax,
)
# Labels
nx.draw_networkx_labels(
    G_genre, pos,
    font_size=8.5,
    font_color="white",
    font_weight="bold",
    ax=ax,
)

ax.set_title(
    "FMA Genre Relationship Graph\n"
    "Node size = track count  ·  Edge width = artists spanning both genres",
    color="white", fontsize=14, pad=16,
)
ax.axis("off")
plt.tight_layout()
out1 = PROCESSED_DIR / "gnn_genre_subgraph.png"
plt.savefig(out1, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"  Saved → {out1.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Ego Graph (seed = first track in small subset)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/4] Ego graph...")

SEED_ID   = small_ids[0]
seed_row  = tracks_s.loc[SEED_ID]
seed_title  = seed_row.get("track_title", str(SEED_ID))
seed_artist = seed_row.get("artist_name", "Unknown")
seed_genre  = seed_row.get("track_genre_top", "Unknown")
seed_a_id   = seed_row.get("artist_id", None)

same_artist_tracks = []
if seed_a_id is not None and not np.isnan(float(seed_a_id)):
    same_artist_tracks = tracks_s[tracks_s["artist_id"] == seed_a_id].index.tolist()

same_genre_tracks = tracks_s[tracks_s["track_genre_top"] == seed_genre].index.tolist()[:18]

G_ego = nx.Graph()
GENRE_NODE  = f"GENRE:{seed_genre}"
ARTIST_NODE = f"ARTIST:{seed_artist[:22]}"
SEED_NODE   = f"TRACK:{SEED_ID}"

G_ego.add_node(GENRE_NODE,  ntype="genre",       label=seed_genre)
G_ego.add_node(ARTIST_NODE, ntype="artist",      label=seed_artist[:22])
G_ego.add_node(SEED_NODE,   ntype="track_seed",  label=f"#{SEED_ID}\n{seed_title[:18]}")

G_ego.add_edge(SEED_NODE, ARTIST_NODE)
G_ego.add_edge(SEED_NODE, GENRE_NODE)

for tid in same_artist_tracks:
    if tid == SEED_ID:
        continue
    n = f"TRACK:{tid}"
    G_ego.add_node(n, ntype="track_artist", label=str(tid))
    G_ego.add_edge(n, ARTIST_NODE)

for tid in same_genre_tracks:
    if tid == SEED_ID:
        continue
    n = f"TRACK:{tid}"
    if n not in G_ego:
        G_ego.add_node(n, ntype="track_genre", label=str(tid))
    G_ego.add_edge(n, GENRE_NODE)

COLOR_MAP = {
    "track_seed":   "#e74c3c",
    "track_artist": "#f39c12",
    "track_genre":  "#3498db",
    "artist":       "#2ecc71",
    "genre":        "#9b59b6",
}
SIZE_MAP = {
    "track_seed": 700, "track_artist": 280,
    "track_genre": 200, "artist": 560, "genre": 560,
}

node_colors = [COLOR_MAP[G_ego.nodes[n]["ntype"]] for n in G_ego.nodes()]
node_szs    = [SIZE_MAP[G_ego.nodes[n]["ntype"]]  for n in G_ego.nodes()]
labels_d    = {n: G_ego.nodes[n]["label"] for n in G_ego.nodes()}

fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(DARK_BG)

pos_ego = nx.spring_layout(G_ego, seed=7, k=2.0)
nx.draw_networkx_nodes(G_ego, pos_ego, node_color=node_colors, node_size=node_szs, alpha=0.92, ax=ax)
nx.draw_networkx_edges(G_ego, pos_ego, alpha=0.3, edge_color="white", ax=ax)
nx.draw_networkx_labels(G_ego, pos_ego, labels=labels_d, font_size=6.5, font_color="white", ax=ax)

legend_elems = [
    mpatches.Patch(facecolor="#e74c3c", label=f"Seed ({SEED_ID})"),
    mpatches.Patch(facecolor="#f39c12", label="Same artist"),
    mpatches.Patch(facecolor="#3498db", label="Same genre"),
    mpatches.Patch(facecolor="#2ecc71", label="Artist node"),
    mpatches.Patch(facecolor="#9b59b6", label="Genre node"),
]
ax.legend(handles=legend_elems, loc="lower left", fontsize=9,
          facecolor="#1a1a2e", edgecolor="gray", labelcolor="white")
ax.set_title(
    f'Ego Graph: "{seed_title[:40]}" by {seed_artist[:30]}\n'
    f"2-hop neighbourhood  (genre: {seed_genre})",
    color="white", fontsize=12, pad=14,
)
ax.axis("off")
plt.tight_layout()
out2 = PROCESSED_DIR / "gnn_ego_graph.png"
plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"  Saved → {out2.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. t-SNE: GNN vs CLAP
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/4] t-SNE comparison (this takes ~60s)...")

gnn_embs_path = PROCESSED_DIR / "gnn_embeddings.npy"
gnn_ids_path  = PROCESSED_DIR / "gnn_track_ids.npy"
clap_embs_path = PROCESSED_DIR / "clap_embeddings.npy"
clap_ids_path  = PROCESSED_DIR / "clap_track_ids.npy"

gnn_embs      = np.load(gnn_embs_path)
gnn_track_ids = np.load(gnn_ids_path).tolist()
clap_embs     = np.load(clap_embs_path)
clap_ids      = np.load(clap_ids_path).tolist()

# Genre labels for GNN
gnn_labels = []
for tid in gnn_track_ids:
    try:
        g = tracks_s.loc[tid, "track_genre_top"]
        gnn_labels.append(g if isinstance(g, str) else "Unknown")
    except KeyError:
        gnn_labels.append("Unknown")

# Genre labels for CLAP
clap_labels = []
for tid in clap_ids:
    try:
        g = tracks_s.loc[tid, "track_genre_top"]
        clap_labels.append(g if isinstance(g, str) else "Unknown")
    except KeyError:
        clap_labels.append("Unknown")

# Colour palette
all_genres = sorted(set(gnn_labels + clap_labels) - {"Unknown"})
PALETTE    = plt.colormaps["tab20"].resampled(len(all_genres))
g2c        = {g: PALETTE(i) for i, g in enumerate(all_genres)}
g2c["Unknown"] = (0.4, 0.4, 0.4, 0.5)

def run_tsne(embs, n_pca=50):
    print(f"    PCA ({embs.shape}) ...", end=" ", flush=True)
    n_comp = min(n_pca, embs.shape[1], embs.shape[0])
    pca    = PCA(n_components=n_comp, random_state=42)
    red    = pca.fit_transform(embs)
    print(f"t-SNE ...", end=" ", flush=True)
    tsne   = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=500)
    out    = tsne.fit_transform(red)
    print("done")
    return out

gnn_2d  = run_tsne(gnn_embs)
clap_2d = run_tsne(clap_embs)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.patch.set_facecolor(DARK_BG)

def scatter_ax(ax, embs_2d, labels, title):
    ax.set_facecolor(DARK_BG)
    colors = [g2c[g] for g in labels]
    ax.scatter(embs_2d[:, 0], embs_2d[:, 1], c=colors, s=5, alpha=0.65, linewidths=0)
    ax.set_title(title, color="white", fontsize=13, pad=10)
    ax.tick_params(colors="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.set_xlabel("t-SNE dim 1", color="gray", fontsize=9)
    ax.set_ylabel("t-SNE dim 2", color="gray", fontsize=9)

scatter_ax(axes[0], gnn_2d,  gnn_labels,  "GNN Graph Embeddings — t-SNE")
scatter_ax(axes[1], clap_2d, clap_labels, "CLAP Audio Embeddings — t-SNE")

# Shared legend
legend_elems = [
    mlines.Line2D([0], [0], marker="o", color="w",
                  markerfacecolor=g2c[g], markersize=8, label=g)
    for g in all_genres
]
fig.legend(handles=legend_elems, loc="lower center", ncol=min(len(all_genres), 8),
           fontsize=9, facecolor="#1a1a2e", edgecolor="gray", labelcolor="white",
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle("GNN vs CLAP Embeddings — Genre Clustering Comparison",
             color="white", fontsize=15, y=1.01)
plt.tight_layout()
out3 = PROCESSED_DIR / "gnn_tsne_comparison.png"
plt.savefig(out3, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"  Saved → {out3.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Degree Distribution
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/4] Degree distributions...")

tg_edges = data["track", "tagged", "genre"].edge_index
ta_edges = data["track", "made_by", "artist"].edge_index

artist_degree = Counter(ta_edges[1].tolist())
track_degree  = Counter(tg_edges[0].tolist())

# Top 10 artists by track count
top10_artist_idx   = [idx for idx, _ in artist_degree.most_common(10)]
top10_artist_names = []
idx_to_artist      = {v: k for k, v in id_maps["artist_to_idx"].items()}
for a_idx in top10_artist_idx:
    a_fma_id = idx_to_artist.get(a_idx, -1)
    name_matches = tracks_s[tracks_s["artist_id"] == a_fma_id]["artist_name"]
    name = name_matches.iloc[0] if len(name_matches) else str(a_fma_id)
    top10_artist_names.append(name[:28])
top10_counts = [artist_degree[idx] for idx in top10_artist_idx]

fig, axes = plt.subplots(1, 3, figsize=(19, 6))
fig.patch.set_facecolor(DARK_BG)

# Panel A — artist degree histogram
ax = axes[0]
ax.set_facecolor(DARK_BG)
vals = list(artist_degree.values())
ax.hist(vals, bins=40, color=ACCENT1, edgecolor=DARK_BG, alpha=0.85)
ax.set_xlabel("Tracks per artist", color="white")
ax.set_ylabel("Number of artists (log scale)", color="white")
ax.set_title("Artist Degree Distribution", color="white", fontsize=12)
ax.set_yscale("log")
ax.tick_params(colors="white")
ax.spines[:].set_edgecolor(GRID_COL)
ax.set_facecolor(DARK_BG)

# Panel B — top 10 most productive artists
ax = axes[1]
ax.set_facecolor(DARK_BG)
bars = ax.barh(range(10), top10_counts[::-1], color=ACCENT2, alpha=0.85, edgecolor=DARK_BG)
ax.set_yticks(range(10))
ax.set_yticklabels(top10_artist_names[::-1], color="white", fontsize=8)
ax.set_xlabel("Number of tracks", color="white")
ax.set_title("Top 10 Artists by Track Count", color="white", fontsize=12)
ax.tick_params(axis="x", colors="white")
ax.spines[:].set_edgecolor(GRID_COL)
for bar in bars:
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            str(int(bar.get_width())), va="center", color="white", fontsize=8)

# Panel C — tracks per genre
ax = axes[2]
ax.set_facecolor(DARK_BG)
genre_track_counts = tracks_s["track_genre_top"].value_counts().head(15)
ax.barh(range(len(genre_track_counts)), genre_track_counts.values[::-1],
        color=ACCENT1, alpha=0.85, edgecolor=DARK_BG)
ax.set_yticks(range(len(genre_track_counts)))
ax.set_yticklabels([g[:22] for g in genre_track_counts.index[::-1]], color="white", fontsize=8)
ax.set_xlabel("Number of tracks", color="white")
ax.set_title("Top 15 Genres by Track Count", color="white", fontsize=12)
ax.tick_params(axis="x", colors="white")
ax.spines[:].set_edgecolor(GRID_COL)

plt.suptitle("Graph Structure Analysis — FMA Small Subset",
             color="white", fontsize=14, y=1.01)
plt.tight_layout()
out4 = PROCESSED_DIR / "gnn_degree_distribution.png"
plt.savefig(out4, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"  Saved → {out4.name}")

print("\n✓ All visuals generated:")
for p in [out1, out2, out3, out4]:
    size_kb = p.stat().st_size // 1024
    print(f"  {p.name}  ({size_kb} KB)")
