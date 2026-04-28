"""
app.py — Role 3: Graph-based Recommendation Demo Server (Issac)

Run from the project root (Finals/) with venv active:
    python role3_graph_issac/app.py

Then open: http://localhost:5050
"""

import sys
import os

# macOS: prevent libomp.dylib conflict between PyTorch and FAISS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Make src/ importable regardless of where the script is run from
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

from src.indexing.faiss_index import FaissIndex
from src.metadata import load_tracks, load_genres, get_small_subset_ids
from src.config import PROCESSED_DIR
from src.audio_utils import get_audio_path

# ─────────────────────────────────────────────────────────────────────────────
# Startup: load everything once, with clear error messages
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED_FILES = {
    "gnn_faiss.index":    "FAISS index (run training pipeline first)",
    "gnn_embeddings.npy": "GNN embeddings (run training pipeline first)",
    "gnn_track_ids.npy":  "GNN track IDs (run training pipeline first)",
}

missing = [f for f in REQUIRED_FILES if not (PROCESSED_DIR / f).exists()]
if missing:
    print("\n[ERROR] Missing required data files in", PROCESSED_DIR)
    for f in missing:
        print(f"  ✗ {f} — {REQUIRED_FILES[f]}")
    print("\nRun the GNN training pipeline before starting the app.")
    sys.exit(1)

print("Loading GNN FAISS index...")
index    = FaissIndex.load(PROCESSED_DIR / "gnn_faiss.index")
gnn_embs = np.load(PROCESSED_DIR / "gnn_embeddings.npy")
gnn_ids  = np.load(PROCESSED_DIR / "gnn_track_ids.npy").tolist()
id_to_idx = {int(tid): i for i, tid in enumerate(gnn_ids)}

print("Loading FMA metadata...")
tracks_df = load_tracks()
small_ids = set(get_small_subset_ids(tracks_df))
tracks_s  = tracks_df[tracks_df.index.isin(small_ids)].copy()
tracks_s.columns = ["_".join(c).strip("_") for c in tracks_s.columns]

# Precompute a simple lookup dict for speed
TRACK_META = {}
for tid, row in tracks_s.iterrows():
    aid = row.get("artist_id")
    TRACK_META[int(tid)] = {
        "title":  str(row.get("track_title",   "Unknown")),
        "artist": str(row.get("artist_name",   "Unknown")),
        "genre":  str(row.get("track_genre_top","Unknown")),
        "artist_id": None if aid is None or pd.isna(aid) else int(float(aid)),
    }

print(f"Ready — {len(TRACK_META):,} tracks loaded.\n")

# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates")


@app.route("/")
def index_page():
    return render_template("index.html")


@app.route("/visuals/<path:filename>")
def serve_visual(filename):
    """Serve the generated PNG visuals."""
    visuals_dir = os.path.join(os.path.dirname(__file__))
    return send_from_directory(visuals_dir, filename)


@app.route("/api/audio/<int:track_id>")
def serve_audio(track_id):
    """Stream the MP3 file for a given track ID."""
    path = get_audio_path(track_id)
    if not path.exists():
        return jsonify({"error": "Audio file not found"}), 404
    return send_from_directory(str(path.parent), path.name, mimetype="audio/mpeg")


# ─────────────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/search")
def search():
    q = request.args.get("q", "").lower().strip()
    if len(q) < 2:
        return jsonify([])

    results = []
    for tid, meta in TRACK_META.items():
        if q in meta["title"].lower() or q in meta["artist"].lower():
            results.append({"id": tid, **meta})
        if len(results) >= 10:
            break
    return jsonify(results)


@app.route("/api/recommend/<int:track_id>")
def recommend(track_id):
    try:
        k = max(3, min(int(request.args.get("k", 10)), 20))
    except (ValueError, TypeError):
        return jsonify({"error": "k must be an integer between 3 and 20"}), 400

    if track_id not in id_to_idx:
        return jsonify({"error": "Track not found in GNN index"}), 404

    emb     = gnn_embs[id_to_idx[track_id]]
    raw     = index.query(emb, k=k + 1)   # +1 because seed itself may appear

    seed    = TRACK_META.get(track_id, {})
    seed_a  = seed.get("artist_id")
    seed_g  = seed.get("genre")

    recs = []
    for tid, score in raw:
        if int(tid) == track_id:
            continue
        meta = TRACK_META.get(int(tid))
        if not meta:
            continue

        # Why is this recommended?
        reason = []
        if seed_a and meta["artist_id"] and seed_a == meta["artist_id"]:
            reason.append("same_artist")
        if seed_g and meta["genre"] == seed_g:
            reason.append("same_genre")
        if not reason:
            reason.append("graph_proximity")

        recs.append({
            "id":     int(tid),
            "title":  meta["title"],
            "artist": meta["artist"],
            "genre":  meta["genre"],
            "score":  round(float(score), 4),
            "reason": reason,
        })

    return jsonify({
        "seed": {"id": track_id, **seed},
        "recommendations": recs[:k],
    })


@app.route("/api/graph/<int:track_id>")
def get_graph(track_id):
    """Return ego graph as {nodes, edges} JSON for D3 visualisation."""
    try:
        k = max(3, min(int(request.args.get("k", 8)), 15))
    except (ValueError, TypeError):
        return jsonify({"error": "k must be an integer between 3 and 15"}), 400

    if track_id not in id_to_idx:
        return jsonify({"error": "Track not found"}), 404

    emb  = gnn_embs[id_to_idx[track_id]]
    raw  = index.query(emb, k=k + 1)
    seed = TRACK_META.get(track_id, {})

    nodes, edges = [], []
    seen_nodes = set()

    def add_node(nid, label, ntype, genre="", artist="", score=0.0):
        if nid not in seen_nodes:
            nodes.append({"id": nid, "label": label, "type": ntype,
                          "genre": genre, "artist": artist, "score": score})
            seen_nodes.add(nid)

    # Seed track
    add_node(f"track_{track_id}", seed.get("title", str(track_id))[:28],
             "seed", seed.get("genre",""), seed.get("artist",""), 1.0)

    # Artist node
    if seed.get("artist_id"):
        anode = f"artist_{seed['artist_id']}"
        add_node(anode, (seed.get("artist","?"))[:24], "artist")
        edges.append({"source": f"track_{track_id}", "target": anode, "rel": "made_by"})

    # Genre node
    gnode = f"genre_{seed.get('genre','?')}"
    add_node(gnode, seed.get("genre","?"), "genre")
    edges.append({"source": f"track_{track_id}", "target": gnode, "rel": "tagged"})

    for tid, score in raw:
        tid = int(tid)
        if tid == track_id:
            continue
        meta = TRACK_META.get(tid)
        if not meta:
            continue

        tnode = f"track_{tid}"
        add_node(tnode, meta["title"][:25], "rec",
                 meta["genre"], meta["artist"], round(float(score), 4))

        # Connect to shared artist
        if seed.get("artist_id") and meta["artist_id"] == seed["artist_id"]:
            edges.append({"source": tnode, "target": f"artist_{seed['artist_id']}", "rel": "made_by"})

        # Connect to genre (shared or own)
        if meta["genre"] == seed.get("genre"):
            edges.append({"source": tnode, "target": gnode, "rel": "tagged"})
        else:
            tgnode = f"genre_{meta['genre']}"
            add_node(tgnode, meta["genre"], "genre")
            edges.append({"source": tnode, "target": tgnode, "rel": "tagged"})

        # Similarity edge back to seed
        edges.append({"source": f"track_{track_id}", "target": tnode,
                      "rel": "similar", "score": round(float(score), 4)})

    return jsonify({"nodes": nodes, "edges": edges})


@app.route("/api/stats")
def stats():
    genre_counts = {}
    for meta in TRACK_META.values():
        g = meta["genre"]
        genre_counts[g] = genre_counts.get(g, 0) + 1
    top_genres = sorted(genre_counts.items(), key=lambda x: -x[1])[:10]
    return jsonify({
        "total_tracks": len(TRACK_META),
        "total_artists": len(set(m["artist_id"] for m in TRACK_META.values() if m["artist_id"])),
        "total_genres": len(genre_counts),
        "top_genres": [{"genre": g, "count": c} for g, c in top_genres],
    })


if __name__ == "__main__":
    app.run(debug=False, port=5050, host="127.0.0.1")
