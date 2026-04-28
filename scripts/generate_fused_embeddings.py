"""
Role 2 — Lyrics-enriched SBERT retrieval with OpenL3 fusion.

Builds text embeddings from FMA metadata + Genius lyrics, then optionally
fuses them with OpenL3 audio embeddings for multimodal search.

Usage:
    python text_to_text_SBERT_FMA_GENIUS_2.py              # full pipeline
    python text_to_text_SBERT_FMA_GENIUS_2.py --skip-lyrics # reuse cached lyrics
"""

import os
import argparse
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.config import FMA_METADATA_DIR, PROCESSED_DIR, PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRACK_IDS_PATH = PROCESSED_DIR / "openl3_track_ids.npy"
OPENL3_EMB_PATH = PROCESSED_DIR / "openl3_embeddings.npy"
TRACKS_CSV = FMA_METADATA_DIR / "tracks.csv"

OUTPUT_DIR = PROCESSED_DIR / "lyrics_enriched"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DF_CACHE = OUTPUT_DIR / "lyrics_df.csv"
SBERT_EMB_PATH = OUTPUT_DIR / "sbert_lyrics_embeddings.npy"
SBERT_INDEX_PATH = OUTPUT_DIR / "sbert_lyrics_faiss.index"
FUSED_EMB_PATH = OUTPUT_DIR / "fused_embeddings.npy"
FUSED_INDEX_PATH = OUTPUT_DIR / "fused_faiss.index"
FUSED_IDS_PATH = OUTPUT_DIR / "fused_track_ids.npy"


# ---------------------------------------------------------------------------
# 1. Load metadata for the 2 000-track OpenL3 subset
# ---------------------------------------------------------------------------
def load_metadata(track_ids: np.ndarray) -> pd.DataFrame:
    """Load FMA metadata and filter to the OpenL3 2 000-track subset."""
    tracks = pd.read_csv(TRACKS_CSV, index_col=0, header=[0, 1])

    int_ids = track_ids.astype(int)

    df = pd.DataFrame({
        "track_id": int_ids,
        "title": tracks.loc[int_ids, ("track", "title")].values,
        "artist": tracks.loc[int_ids, ("artist", "name")].values,
        "genres": tracks.loc[int_ids, ("track", "genre_top")].values,
        "tags": tracks.loc[int_ids, ("track", "tags")].values,
    })
    df = df.dropna(subset=["title", "artist"])
    return df


GENRE_LABELS = {"electronic", "experimental", "folk", "hip-hop", "instrumental",
                 "international", "pop", "rock"}


def strip_genre_from_tags(tags_raw: str) -> str:
    """Remove genre-like words from the tags field to prevent evaluation leakage."""
    if not isinstance(tags_raw, str) or not tags_raw.strip():
        return ""
    # Tags are often stored as "['tag1', 'tag2']"
    cleaned = tags_raw.strip("[]").replace("'", "").replace('"', "")
    tokens = [t.strip() for t in cleaned.split(",")]
    filtered = [t for t in tokens if t.lower() not in GENRE_LABELS]
    return ", ".join(filtered)


def metadata_string(row) -> str:
    tags = strip_genre_from_tags(row["tags"])
    parts = [f"{row['title']} by {row['artist']}."]
    if tags:
        parts.append(f"Tags: {tags}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# 2. Fetch lyrics from Genius (optional)
# ---------------------------------------------------------------------------
def fetch_lyrics_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'lyrics' column via the Genius API.  Requires GENIUS_API_KEY."""
    api_key = os.environ.get("GENIUS_API_KEY")
    if not api_key:
        print("WARNING: GENIUS_API_KEY not set — skipping lyrics fetch.")
        df["lyrics"] = ""
        return df

    import lyricsgenius
    genius = lyricsgenius.Genius(api_key, timeout=10)
    genius.verbose = False
    genius.remove_section_headers = True

    def _fetch(title, artist):
        try:
            song = genius.search_song(title, artist)
            if song and song.lyrics:
                # Drop the first line (song title header)
                return song.lyrics.split("\n", 1)[-1][:1000]
        except Exception:
            pass
        return ""

    print(f"Fetching lyrics for {len(df)} tracks (this may take a while)...")
    df["lyrics"] = df.apply(
        lambda r: _fetch(r["title"], r["artist"]), axis=1
    )
    return df


# ---------------------------------------------------------------------------
# 3. Build full-text strings and encode with SBERT
# ---------------------------------------------------------------------------
def build_texts(df: pd.DataFrame) -> list[str]:
    df["text"] = df.apply(metadata_string, axis=1)

    def _full(row):
        base = row["text"]
        if row.get("lyrics"):
            return f"{base} Lyrics: {row['lyrics']}"
        return base

    df["full_text"] = df.apply(_full, axis=1)
    return df["full_text"].tolist()


def encode_sbert(texts: list[str]) -> np.ndarray:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Encoding {len(texts)} strings with SBERT...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings


# ---------------------------------------------------------------------------
# 4. Multimodal fusion (SBERT 384-d + OpenL3 512-d)
# ---------------------------------------------------------------------------
def fuse_embeddings(
    sbert_emb: np.ndarray,
    openl3_emb: np.ndarray,
    sbert_ids: np.ndarray,
    openl3_ids: np.ndarray,
    weight_text: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Late fusion: L2-normalise each modality, weight, concatenate, re-normalise.
    Only fuses tracks present in both sets.
    """
    # Align on common track IDs
    sbert_map = {int(tid): i for i, tid in enumerate(sbert_ids)}
    openl3_map = {int(tid): i for i, tid in enumerate(openl3_ids)}
    common = sorted(set(sbert_map) & set(openl3_map))
    print(f"Fusing {len(common)} tracks common to SBERT and OpenL3...")

    s_idx = [sbert_map[tid] for tid in common]
    o_idx = [openl3_map[tid] for tid in common]

    s = sbert_emb[s_idx].astype(np.float32)
    o = openl3_emb[o_idx].astype(np.float32)

    # Mean-center to remove the DC offset from both spaces
    s = s - s.mean(axis=0)
    o = o - o.mean(axis=0)

    # L2-normalise each modality
    faiss.normalize_L2(s)
    faiss.normalize_L2(o)

    # Weighted concatenation → (384 + 512 = 896)-d
    weight_audio = 1.0 - weight_text
    fused = np.hstack([s * weight_text, o * weight_audio])

    # Final L2-normalisation for cosine search
    faiss.normalize_L2(fused)

    return fused, np.array(common)


# ---------------------------------------------------------------------------
# 5. Build & save FAISS indices
# ---------------------------------------------------------------------------
def build_and_save_index(embeddings: np.ndarray, track_ids: np.ndarray,
                         index_path: Path, ids_path: Path):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on L2-normed = cosine
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(index_path))
    np.save(ids_path, track_ids)
    print(f"Saved FAISS index ({index.ntotal} vectors, {dim}-d) → {index_path}")


# ---------------------------------------------------------------------------
# 6. Search helper
# ---------------------------------------------------------------------------
def search(query: str, index_path: Path, ids_path: Path, df: pd.DataFrame,
           k: int = 5, mode: str = "text"):
    """
    Search an index with a text query.

    mode="text"  — query the text-only SBERT index (384-d).
    mode="fused" — project the query into the fused space.  The text half is
                   the SBERT encoding; the audio half is set to zero because
                   we have no audio for a free-text query.  This means the
                   fused search can only match on the text portion of the
                   stored vectors — it will *not* outperform the text-only
                   index for text queries.  Fused search is most useful for
                   track-to-track retrieval (see fuse_embeddings).
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(str(index_path))
    track_ids = np.load(ids_path)

    q_vec = model.encode([query], normalize_embeddings=True,
                         convert_to_numpy=True).astype(np.float32)

    if mode == "fused" and q_vec.shape[1] < index.d:
        # Zero-pad the audio dimensions.  Note: this means only the text
        # half of each stored vector contributes to the similarity score.
        pad = np.zeros((1, index.d - q_vec.shape[1]), dtype=np.float32)
        q_vec = np.hstack([q_vec, pad])
        faiss.normalize_L2(q_vec)

    distances, indices = index.search(q_vec, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        tid = int(track_ids[idx])
        row = df[df["track_id"] == tid]
        title = row["title"].values[0] if len(row) else "?"
        artist = row["artist"].values[0] if len(row) else "?"
        genres = row["genres"].values[0] if len(row) else "?"
        results.append({"track_id": tid, "title": title, "artist": artist,
                        "genres": genres, "score": float(dist)})
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Role 2: Lyrics-enriched SBERT + OpenL3 fusion")
    parser.add_argument("--skip-lyrics", action="store_true",
                        help="Reuse cached lyrics DataFrame instead of re-fetching")
    parser.add_argument("--weight-text", type=float, default=0.5,
                        help="Text weight in fusion (0-1). Audio weight = 1 - text weight.")
    args = parser.parse_args()

    # --- Load track IDs (all 2 000) ---
    track_ids = np.load(TRACK_IDS_PATH, allow_pickle=True)
    print(f"Loaded {len(track_ids)} track IDs from OpenL3 subset.")

    # --- Metadata ---
    if args.skip_lyrics and DF_CACHE.exists():
        print(f"Loading cached DataFrame from {DF_CACHE}")
        df = pd.read_csv(DF_CACHE)
    else:
        df = load_metadata(track_ids)
        df = fetch_lyrics_column(df)
        df.to_csv(DF_CACHE, index=False)
        print(f"Saved DataFrame ({len(df)} rows) → {DF_CACHE}")

    # --- SBERT embeddings ---
    texts = build_texts(df)
    sbert_emb = encode_sbert(texts)
    np.save(SBERT_EMB_PATH, sbert_emb)
    print(f"Saved SBERT embeddings {sbert_emb.shape} → {SBERT_EMB_PATH}")

    sbert_ids = df["track_id"].values
    build_and_save_index(
        sbert_emb, sbert_ids,
        SBERT_INDEX_PATH,
        SBERT_INDEX_PATH.with_suffix(".ids.npy"),
    )

    # --- Multimodal fusion ---
    openl3_emb = np.load(OPENL3_EMB_PATH)
    openl3_ids = np.load(TRACK_IDS_PATH, allow_pickle=True).astype(int)

    fused_emb, fused_ids = fuse_embeddings(
        sbert_emb, openl3_emb, sbert_ids, openl3_ids,
        weight_text=args.weight_text,
    )
    np.save(FUSED_EMB_PATH, fused_emb)
    np.save(FUSED_IDS_PATH, fused_ids)
    build_and_save_index(fused_emb, fused_ids, FUSED_INDEX_PATH, FUSED_IDS_PATH)

    # --- Demo queries ---
    print("\n" + "=" * 60)
    print("TEXT-ONLY SEARCH (SBERT + lyrics)")
    print("=" * 60)
    print(search("songs about freedom and escape",
                 SBERT_INDEX_PATH, SBERT_INDEX_PATH.with_suffix(".ids.npy"), df))

    print("\n" + "=" * 60)
    print("FUSED SEARCH (SBERT + OpenL3 audio)")
    print("=" * 60)
    print(search("songs about freedom and escape",
                 FUSED_INDEX_PATH, FUSED_IDS_PATH, df, mode="fused"))


if __name__ == "__main__":
    main()
