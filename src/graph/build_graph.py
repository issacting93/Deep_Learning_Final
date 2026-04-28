"""
build_graph.py — Role 3: Graph-based Recommendation (Issac)

Constructs a heterogeneous graph from FMA metadata using PyTorch Geometric's
HeteroData. The graph has three node types (track, artist, genre) and four
edge types:
  - track → artist   (each track belongs to one artist)
  - track → genre    (each track has a top genre)
  - artist → genre   (derived from track→genre)
  - track ↔ track    (co-genre: tracks sharing the same top genre)

Initial node features:
  - track nodes:  512-dim CLAP embeddings (from clap_embeddings.npy)
  - artist nodes: mean-pooled CLAP embeddings of their tracks
  - genre nodes:  one-hot vectors (163-dim)
"""

import numpy as np
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

from src.config import PROCESSED_DIR, MAX_CO_GENRE_EDGES
from src.metadata import load_tracks, load_genres, get_small_subset_ids


def build_hetero_graph(verbose: bool = True) -> tuple[HeteroData, dict]:
    """
    Build the heterogeneous graph from FMA metadata and CLAP embeddings.

    Returns:
        data    — torch_geometric HeteroData object, ready for HeteroConv
        id_maps — dict with mappings:
                    'track_to_idx'  : {fma_track_id -> node index}
                    'artist_to_idx' : {fma_artist_id -> node index}
                    'genre_to_idx'  : {genre_id -> node index}
                    'idx_to_track'  : reverse map (list of fma_track_ids)
    """
    # ------------------------------------------------------------------
    # 1. Load metadata
    # ------------------------------------------------------------------
    if verbose:
        print("Loading FMA metadata...")
    tracks = load_tracks()
    genres_df = load_genres()
    small_ids = set(get_small_subset_ids(tracks))

    # Filter to small subset only
    tracks_small = tracks.loc[tracks.index.isin(small_ids)]

    # ------------------------------------------------------------------
    # 2. Build node index mappings
    # ------------------------------------------------------------------
    track_ids_all = tracks_small.index.tolist()

    # Artist IDs (integer column in tracks.csv)
    artist_ids_raw = tracks_small[("artist", "id")].fillna(-1).astype(int).tolist()
    unique_artists = sorted(set(a for a in artist_ids_raw if a >= 0))
    artist_to_idx = {a: i for i, a in enumerate(unique_artists)}

    # Genre IDs from genres.csv index
    genre_ids_raw = genres_df.index.tolist()
    genre_to_idx = {g: i for i, g in enumerate(genre_ids_raw)}
    num_genres = len(genre_to_idx)

    # Track node index
    track_to_idx = {t: i for i, t in enumerate(track_ids_all)}

    if verbose:
        print(f"  Tracks : {len(track_to_idx):,}")
        print(f"  Artists: {len(artist_to_idx):,}")
        print(f"  Genres : {num_genres}")

    # ------------------------------------------------------------------
    # 3. Load CLAP embeddings → track node features
    # ------------------------------------------------------------------
    if verbose:
        print("Loading CLAP embeddings for track node features...")
    clap_embs = np.load(PROCESSED_DIR / "clap_embeddings.npy").astype(np.float32)
    clap_ids  = np.load(PROCESSED_DIR / "clap_track_ids.npy").tolist()
    clap_dim  = clap_embs.shape[1]  # 512

    # Map clap embedding rows → track node index
    clap_id_to_row = {tid: row for row, tid in enumerate(clap_ids)}

    # Build track feature matrix; use zero vector for the 3 missing tracks
    track_feats = np.zeros((len(track_to_idx), clap_dim), dtype=np.float32)
    found = 0
    for tid, node_idx in track_to_idx.items():
        if tid in clap_id_to_row:
            track_feats[node_idx] = clap_embs[clap_id_to_row[tid]]
            found += 1

    if verbose:
        print(f"  CLAP embeddings matched: {found}/{len(track_to_idx)} tracks")
        print(f"  ({len(track_to_idx) - found} missing tracks get zero-vector features)")

    # ------------------------------------------------------------------
    # 4. Artist node features = mean-pool of their tracks' CLAP embeddings
    # ------------------------------------------------------------------
    artist_feats = np.zeros((len(artist_to_idx), clap_dim), dtype=np.float32)
    artist_counts = np.zeros(len(artist_to_idx), dtype=np.int32)

    for tid, a_id in zip(track_ids_all, artist_ids_raw):
        if a_id < 0 or a_id not in artist_to_idx:
            continue
        a_idx = artist_to_idx[a_id]
        t_idx = track_to_idx[tid]
        artist_feats[a_idx] += track_feats[t_idx]
        artist_counts[a_idx] += 1

    # Avoid division by zero
    mask = artist_counts > 0
    artist_feats[mask] /= artist_counts[mask, np.newaxis]

    # ------------------------------------------------------------------
    # 5. Genre node features = one-hot (163-dim)
    # ------------------------------------------------------------------
    genre_feats = np.eye(num_genres, dtype=np.float32)

    # ------------------------------------------------------------------
    # 6. Build edges
    # ------------------------------------------------------------------
    if verbose:
        print("Building edges...")

    # --- 6a. track → artist ---
    ta_src, ta_dst = [], []
    for tid, a_id in zip(track_ids_all, artist_ids_raw):
        if a_id >= 0 and a_id in artist_to_idx:
            ta_src.append(track_to_idx[tid])
            ta_dst.append(artist_to_idx[a_id])

    # --- 6b. track → genre (top genre) ---
    tg_src, tg_dst = [], []
    genre_top_raw = tracks_small[("track", "genre_top")].tolist()
    for tid, g_name in zip(track_ids_all, genre_top_raw):
        if not isinstance(g_name, str):
            continue
        # genres.csv uses integer IDs; look up by genre title
        matches = genres_df[genres_df["title"] == g_name].index.tolist()
        if matches:
            g_id = matches[0]
            tg_src.append(track_to_idx[tid])
            tg_dst.append(genre_to_idx[g_id])

    # --- 6c. artist → genre (derived: if any of their tracks → genre) ---
    ag_pairs = set()
    for t_src, g_dst in zip(tg_src, tg_dst):
        # Find artist for this track
        tid = track_ids_all[t_src]
        a_raw = tracks_small.loc[tid, ("artist", "id")]
        if isinstance(a_raw, float) and np.isnan(a_raw):
            continue
        a_id = int(a_raw)
        if a_id in artist_to_idx:
            ag_pairs.add((artist_to_idx[a_id], g_dst))
    ag_src = [p[0] for p in ag_pairs]
    ag_dst = [p[1] for p in ag_pairs]

    # --- 6d. track ↔ track co-genre edges ---
    # Group tracks by their genre node index; cap edges per genre to avoid
    # memory blow-up on large genres (e.g. Rock: 1000 tracks → 1M edges).
    import random as _random
    _random.seed(42)
    # MAX_CO_GENRE_EDGES imported from src.config

    genre_to_tracks: dict[int, list] = {}
    for t, g in zip(tg_src, tg_dst):
        genre_to_tracks.setdefault(g, []).append(t)

    tt_src, tt_dst = [], []
    for g_idx, t_list in genre_to_tracks.items():
        # Build all pairs as (src, dst) tuples, then subsample if too many
        pairs = [(t_list[i], t_list[j])
                 for i in range(len(t_list))
                 for j in range(len(t_list)) if i != j]
        if len(tt_src) + len(pairs) > MAX_CO_GENRE_EDGES:
            remaining = max(0, MAX_CO_GENRE_EDGES - len(tt_src))
            pairs = _random.sample(pairs, min(remaining, len(pairs)))
        for s, d in pairs:
            tt_src.append(s)
            tt_dst.append(d)
        if len(tt_src) >= MAX_CO_GENRE_EDGES:
            break

    if verbose:
        print(f"  track→artist edges : {len(ta_src):,}")
        print(f"  track→genre edges  : {len(tg_src):,}")
        print(f"  artist→genre edges : {len(ag_src):,}")
        print(f"  track↔track edges  : {len(tt_src):,} (co-genre)")

    # ------------------------------------------------------------------
    # 7. Assemble HeteroData
    # ------------------------------------------------------------------
    data = HeteroData()

    # Node features
    data["track"].x  = torch.tensor(track_feats)
    data["artist"].x = torch.tensor(artist_feats)
    data["genre"].x  = torch.tensor(genre_feats)

    # Convenience: store node counts
    data["track"].num_nodes  = len(track_to_idx)
    data["artist"].num_nodes = len(artist_to_idx)
    data["genre"].num_nodes  = num_genres

    # Edges — PyG convention: edge_index shape [2, num_edges]
    def to_edge_index(src: list, dst: list) -> torch.Tensor:
        return torch.tensor([src, dst], dtype=torch.long)

    data["track", "made_by", "artist"].edge_index   = to_edge_index(ta_src, ta_dst)
    data["track", "tagged", "genre"].edge_index      = to_edge_index(tg_src, tg_dst)
    data["artist", "plays", "genre"].edge_index      = to_edge_index(ag_src, ag_dst)
    data["track", "co_genre", "track"].edge_index    = to_edge_index(tt_src, tt_dst)

    # Add reverse edges so message can flow in both directions
    data = T.ToUndirected()(data)

    id_maps = {
        "track_to_idx":  track_to_idx,
        "artist_to_idx": artist_to_idx,
        "genre_to_idx":  genre_to_idx,
        "idx_to_track":  track_ids_all,   # list: position i → fma_track_id
    }

    if verbose:
        print("\nHeteroData summary:")
        print(data)

    return data, id_maps


if __name__ == "__main__":
    data, id_maps = build_hetero_graph(verbose=True)
    # Quick sanity check
    print("\nSanity check:")
    print(f"  track x shape  : {data['track'].x.shape}")
    print(f"  artist x shape : {data['artist'].x.shape}")
    print(f"  genre x shape  : {data['genre'].x.shape}")
