# Role 2: Lyrics & Semantic Search — Sid & Issac

## Goal

Build a **text-based retrieval view** using track metadata and analysis of representation failure cases. This enables semantic search like "songs about isolation" while critically evaluating *when and why* the underlying embeddings (all-MiniLM) fail to represent the music accurately.

## Your Tasks

1. **Build metadata text strings**
   - For each of the 8,000 small-subset tracks, construct a text string:
     ```python
     f"{artist} - {title}. Tags: {genre}, {mood}. Lyrics snippet: {chorus_info}"
     ```
   - Pull fields from `tracks.csv`. Normalize the strings (lowercase, strip punctuation).

2. **Implement `src/embeddings/sbert.py`**
   - Subclass `EmbeddingGenerator`.
   - Use `sentence-transformers/all-MiniLM-L6-v2`.
   - **Critical**: Use `normalize_embeddings=True` to ensure Cosine Similarity is equivalent to Dot Product for FAISS.

3. **Representation Analysis (The "April 7th" Requirements)**
   - **A. Semantic Robustness**: Compare retrieval for "lonely" vs. "isolated". Measure Overlap@10.
   - **B. Truncation Impact**: Compare embeddings for the first 256 tokens vs. tokens 256-512. Measure cosine shift.
   - **C. Lexical Bias**: Find cases where keyword overlap (e.g. "Blue" in title) trumps semantic mood (e.g. Blues genre).

4. **Build FAISS Index**
   - Save to `data/processed/sbert_faiss.index`.
   - Compute rank correlation with CLAP results to see where audio and text views agree.

5. **Echo Nest feature exploration** (optional but encouraged)
   - `echonest.csv` has danceability, energy, valence, tempo, speechiness, acousticness for ~13k tracks
   - Merge with the small subset — how many overlap?
   - Can these numeric features act as additional retrieval filters? (e.g. "high energy, low valence")

6. **(Optional) Lyrics via Genius API**
   - Register for a free Genius API token at https://genius.com/api-clients
   - Match FMA tracks by artist + title (fuzzy matching recommended)
   - For tracks with lyrics found, embed lyrics separately and compare to metadata-only embeddings

## Setup

```bash
# Sentence-BERT is already in requirements.txt
# If fetching lyrics:
pip install lyricsgenius
```

## Key Files

| Path | Purpose |
|---|---|
| `src/embeddings/base.py` | Base class to subclass |
| `src/metadata.py` | Load tracks.csv |
| `src/indexing/faiss_index.py` | FAISS index wrapper |
| `src/config.py` | Paths and constants |
| `data/fma_metadata/tracks.csv` | Track metadata (title, artist, genre, tags) |
| `data/fma_metadata/echonest.csv` | Echo Nest audio features |

## Deliverables

- [x] `src/embeddings/sbert.py`
- [x] `data/processed/sbert_embeddings.npy`
- [x] `data/processed/sbert_faiss.index`
- [x] Notebook: semantic search demo (Included in REPORT.md)
- [x] Notebook: CLAP vs SBERT overlap analysis (Included in REPORT.md)
- [x] Optional: lyrics-enriched embeddings + comparison (`text_to_text_SBERT_FMA_GENIUS_2.py`)
