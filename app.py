"""
app.py — Multi-View Music Recommendation Interface

Loads three embedding views (SBERT, OpenL3, CLAP) and serves
side-by-side recommendations for any track in the shared subset.

Run from Finals/:
    python app.py

Then open: http://localhost:5001
"""

import logging
import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.config import PROCESSED_DIR, FMA_SMALL_DIR, RRF_K
from src.metadata import load_tracks, get_small_subset_ids
from src.audio_utils import get_audio_path

# ─────────────────────────────────────────────────────────────────────────────
# Load metadata
# ─────────────────────────────────────────────────────────────────────────────
print("Loading FMA metadata...")
tracks_df = load_tracks()
small_ids = set(get_small_subset_ids(tracks_df))
tracks_s = tracks_df[tracks_df.index.isin(small_ids)].copy()
tracks_s.columns = ["_".join(c).strip("_") for c in tracks_s.columns]

TRACK_META = {}
for tid, row in tracks_s.iterrows():
    TRACK_META[int(tid)] = {
        "title":  str(row.get("track_title", "Unknown") or "Unknown"),
        "artist": str(row.get("artist_name", "Unknown") or "Unknown"),
        "genre":  str(row.get("track_genre_top", "Unknown") or "Unknown"),
    }


def get_meta(tid):
    return {"track_id": int(tid), **TRACK_META.get(int(tid), {
        "title": "Unknown", "artist": "Unknown", "genre": "Unknown"
    })}


# ─────────────────────────────────────────────────────────────────────────────
# Load embedding views
# ─────────────────────────────────────────────────────────────────────────────
def load_view(name, emb_file, ids_file):
    embs = np.load(emb_file).astype(np.float32)
    ids = np.load(ids_file)
    ids = np.array([int(i) for i in ids])
    # Validate and L2-normalise for cosine similarity via dot product
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    if not np.allclose(norms, 1.0, atol=0.05):
        logger.warning(
            f"{name}: embeddings not normalized "
            f"(norm range {norms.min():.4f}–{norms.max():.4f}), renormalizing."
        )
    norms[norms == 0] = 1
    embs = embs / norms
    id_to_idx = {int(tid): i for i, tid in enumerate(ids)}
    logger.info(f"  {name}: {embs.shape[0]} tracks, {embs.shape[1]}-d")
    return {"name": name, "embeddings": embs, "ids": ids, "id_to_idx": id_to_idx}


print("Loading embeddings...")
VIEWS = {
    "sbert":  load_view("SBERT (Text)",    PROCESSED_DIR / "sbert_embeddings.npy",  PROCESSED_DIR / "sbert_track_ids.npy"),
    "openl3": load_view("OpenL3 (Audio)",   PROCESSED_DIR / "openl3_embeddings.npy", PROCESSED_DIR / "openl3_track_ids.npy"),
    "clap":   load_view("CLAP (Vibe)",      PROCESSED_DIR / "clap_embeddings.npy",   PROCESSED_DIR / "clap_track_ids.npy"),
}

# Common track IDs across all three views
common_ids = None
all_view_ids = {}
for vk, v in VIEWS.items():
    s = set(v["id_to_idx"].keys())
    all_view_ids[vk] = s
    common_ids = s if common_ids is None else common_ids & s
common_ids = sorted(common_ids)

# Log tracks dropped per view during intersection
for vk, s in all_view_ids.items():
    dropped = len(s) - len(common_ids)
    if dropped > 0:
        logger.info(f"  {vk}: {dropped} tracks excluded (not in all views)")
logger.info(f"Common tracks across all views: {len(common_ids)}")

# Precompute searchable list with metadata
TRACK_LIST = [get_meta(tid) for tid in common_ids]
COMMON_SET = set(common_ids)

# Genre colour map
GENRE_COLOURS = {
    "Electronic":   "#6366f1",
    "Experimental": "#f59e0b",
    "Folk":         "#10b981",
    "Hip-Hop":      "#ef4444",
    "Instrumental": "#8b5cf6",
    "International": "#06b6d4",
    "Pop":          "#ec4899",
    "Rock":         "#f97316",
}

print("Ready.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Recommendation engine
# ─────────────────────────────────────────────────────────────────────────────
def recommend(track_id, view_key, k=8):
    view = VIEWS[view_key]
    if track_id not in view["id_to_idx"]:
        return []
    idx = view["id_to_idx"][track_id]
    query = view["embeddings"][idx:idx + 1]
    scores = (view["embeddings"] @ query.T).flatten()

    top_indices = np.argsort(scores)[::-1]
    results = []
    for i in top_indices:
        tid = int(view["ids"][i])
        if tid == track_id:
            continue
        if tid not in COMMON_SET:
            continue
        meta = get_meta(tid)
        meta["score"] = round(float(scores[i]), 4)
        results.append(meta)
        if len(results) >= k:
            break
    return results


def cosine_score(track_a, track_b, view_key):
    """Return cosine similarity between two tracks in a given view."""
    view = VIEWS[view_key]
    if track_a not in view["id_to_idx"] or track_b not in view["id_to_idx"]:
        return 0.0
    ea = view["embeddings"][view["id_to_idx"][track_a]]
    eb = view["embeddings"][view["id_to_idx"][track_b]]
    return float(ea @ eb)


def fused_recommend(track_id, k=8, weights=None):
    """Reciprocal Rank Fusion across all views, with per-view scores."""
    if weights is None:
        weights = {key: 1.0 for key in VIEWS}
    rrf_scores = {}
    rrf_k = RRF_K
    for view_key, w in weights.items():
        recs = recommend(track_id, view_key, k=k * 3)
        for rank, rec in enumerate(recs):
            tid = rec["track_id"]
            if tid not in rrf_scores:
                rrf_scores[tid] = 0.0
            rrf_scores[tid] += w / (rrf_k + rank + 1)

    ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])
    results = []
    for tid, score in ranked:
        if tid == track_id:
            continue
        meta = get_meta(tid)
        meta["score"] = round(score, 4)
        meta["view_scores"] = {
            vk: round(cosine_score(track_id, tid, vk), 4)
            for vk in VIEWS
        }
        results.append(meta)
        if len(results) >= k:
            break
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def index_page():
    return render_template("index.html")


@app.route("/api/search")
def search():
    q = request.args.get("q", "").lower().strip()
    if len(q) < 2:
        return jsonify([])
    results = []
    for t in TRACK_LIST:
        if q in t["title"].lower() or q in t["artist"].lower() or q in t.get("genre", "").lower():
            results.append(t)
        if len(results) >= 20:
            break
    return jsonify(results)


@app.route("/api/tracks")
def list_tracks():
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))
    genre = request.args.get("genre", "").strip()
    filtered = TRACK_LIST
    if genre:
        filtered = [t for t in TRACK_LIST if t["genre"] == genre]
    start = (page - 1) * per_page
    end = start + per_page
    return jsonify({
        "tracks": filtered[start:end],
        "total": len(filtered),
        "page": page,
    })


@app.route("/api/recommend/<int:track_id>")
def get_recommendations(track_id):
    k = max(3, min(int(request.args.get("k", 8)), 20))
    result = {}
    for key, view in VIEWS.items():
        result[key] = {
            "name": view["name"],
            "results": recommend(track_id, key, k),
        }
    result["fused"] = {
        "name": "Fused (RRF)",
        "results": fused_recommend(track_id, k),
    }
    return jsonify({
        "seed": get_meta(track_id),
        "views": result,
    })


@app.route("/api/audio/<int:track_id>")
def serve_audio(track_id):
    path = get_audio_path(track_id)
    if not path.exists():
        return jsonify({"error": "Audio not found"}), 404
    return send_from_directory(str(path.parent), path.name, mimetype="audio/mpeg")


@app.route("/api/genres")
def genres():
    counts = {}
    for t in TRACK_LIST:
        g = t["genre"]
        counts[g] = counts.get(g, 0) + 1
    return jsonify(sorted(counts.items(), key=lambda x: -x[1]))


if __name__ == "__main__":
    app.run(debug=True, port=5001, host="127.0.0.1")
