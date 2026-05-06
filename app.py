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
import json

# Set environment variables BEFORE importing torch/transformers to prevent macOS ARM mutex deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import subprocess
from flask import Flask, jsonify, render_template, request, send_from_directory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.config import PROCESSED_DIR, FMA_SMALL_DIR, RRF_K
from src.metadata import load_tracks, get_small_subset_ids
from src.audio_utils import get_audio_path

# ─────────────────────────────────────────────────────────────────────────────
# Semantic search helpers
# ─────────────────────────────────────────────────────────────────────────────
def run_semantic_search_process(query):
    """Run CLAP in a separate process to avoid macOS mutex deadlocks."""
    try:
        script_path = os.path.join(PROJECT_ROOT, "scripts", "semantic_search_tool.py")
        # Use venv python to ensure CLAP and other deps are available
        venv_python = os.path.join(PROJECT_ROOT, "venv", "bin", "python")
        python_exe = venv_python if os.path.exists(venv_python) else sys.executable
        cmd = [python_exe, script_path, query]
        
        # Capture stderr to debug potential import or file issues
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True, 
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT}
        )
        # Clean output: find the first '[' to start parsing JSON
        stdout = result.stdout.strip()
        if "[" in stdout:
            stdout = stdout[stdout.index("["):]
        return json.loads(stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"Semantic search process failed (exit {e.returncode}):")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"Semantic search process failed: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Load metadata
# ─────────────────────────────────────────────────────────────────────────────
logger.info("Loading FMA metadata...")
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
    
    # Mean-center to remove DC offset before normalization
    embs = embs - embs.mean(axis=0)

    # Validate and L2-normalise for cosine similarity via dot product
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    if not np.allclose(norms, 1.0, atol=0.05):
        logger.warning(
            f"{name}: embeddings not normalized "
            f"(norm range {norms.min():.4f}–{norms.max():.4f}), renormalizing."
        )
    # Skip zero-norm vectors (degenerate embeddings) and use epsilon for safety
    zero_mask = (norms < 1e-10).flatten()
    if zero_mask.any():
        logger.warning(f"{name}: {zero_mask.sum()} zero-norm embeddings detected, skipping them.")
    embs = embs / (norms + 1e-8)
    id_to_idx = {int(tid): i for i, tid in enumerate(ids)}
    logger.info(f"  {name}: {embs.shape[0]} tracks, {embs.shape[1]}-d")
    return {"name": name, "embeddings": embs, "ids": ids, "id_to_idx": id_to_idx}


logger.info("Loading embeddings...")
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

# ─────────────────────────────────────────────────────────────────────────────
# Compute Genre Centroids
# ─────────────────────────────────────────────────────────────────────────────
logger.info("Computing genre centroids...")
GENRE_CENTROIDS = {}
for vk, v in VIEWS.items():
    GENRE_CENTROIDS[vk] = {}
    for g in GENRE_COLOURS.keys():
        # Get all tracks belonging to this genre in this view
        g_ids = [tid for tid, meta in TRACK_META.items() if meta["genre"] == g and tid in v["id_to_idx"]]
        if g_ids:
            indices = [v["id_to_idx"][tid] for tid in g_ids]
            centroid = v["embeddings"][indices].mean(axis=0)
            # L2-normalize centroid
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            GENRE_CENTROIDS[vk][g] = centroid

logger.info("Ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Recommendation engine
# ─────────────────────────────────────────────────────────────────────────────
def get_genre_score(tid, genre, view_key):
    """Return similarity of track tid to the centroid of the given genre."""
    view = VIEWS[view_key]
    if tid not in view["id_to_idx"] or genre not in GENRE_CENTROIDS[view_key]:
        return 0.0
    emb = view["embeddings"][view["id_to_idx"][tid]]
    centroid = GENRE_CENTROIDS[view_key][genre]
    return float(emb @ centroid)


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
        # Add genre score (similarity to centroid)
        meta["genre_score"] = round(get_genre_score(tid, meta["genre"], view_key), 4)
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
        # Add per-model genre scores
        meta["genre_scores"] = {
            vk: round(get_genre_score(tid, meta["genre"], vk), 4)
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


@app.route("/api/semantic_search")
def semantic_search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])

    logger.info(f"Semantic search (process isolation): '{q}'")
    
    # Run in separate process to avoid mutex deadlock
    results_raw = run_semantic_search_process(q)
    
    if results_raw is None:
        return jsonify({"error": "Semantic search process failed. This is likely due to environment constraints on macOS."}), 500
    
    # Map back to full metadata
    results = []
    for r in results_raw:
        tid = r["track_id"]
        meta = get_meta(tid)
        meta["score"] = r["score"]
        results.append(meta)
        
    return jsonify(results)


@app.errorhandler(500)
def internal_error(e):
    logger.exception("Internal server error")
    return jsonify({"error": "Internal server error"}), 500


@app.route("/api/tracks")
def list_tracks():
    try:
        page = max(1, int(request.args.get("page", 1)))
        per_page = max(1, min(int(request.args.get("per_page", 50)), 200))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid page or per_page parameter"}), 400
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
    if track_id not in COMMON_SET:
        return jsonify({"error": f"Track {track_id} not found in dataset"}), 404
    try:
        k = max(3, min(int(request.args.get("k", 8)), 20))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid k parameter"}), 400
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
    if track_id not in COMMON_SET:
        return jsonify({"error": f"Track {track_id} not found in dataset"}), 404
    path = get_audio_path(track_id)
    if not path.exists():
        return jsonify({"error": "Audio file not found on disk"}), 404
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
