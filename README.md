# Multi-Faceted Music Retrieval

## Problem

Music retrieval systems typically rely on a single representation of music — either audio features or text metadata. This limits their ability to satisfy diverse user needs: a query like "chill lo-fi vibes" requires understanding musical mood (acoustic), while "songs by folk artists about travel" requires understanding metadata (textual), and "tracks similar to this one I like" requires structural knowledge (graph). No single embedding captures all of these.

This project builds a **multi-view retrieval system** over the [Free Music Archive (FMA)](https://github.com/mdeff/fma) dataset, an open-licensed benchmark of 8,000 tracks across 8 top-level genres. Each view produces an independent embedding space, and a fusion layer combines them to outperform any single view.

## Definitions

- **Embedding**: A fixed-size numerical vector (e.g., 384 or 512 dimensions) that represents a track in a continuous space where similar items are close together. All retrieval in this project reduces to nearest-neighbor search over embeddings.
- **FAISS (Facebook AI Similarity Search)**: A library for efficient nearest-neighbor search over dense vectors. We use `IndexFlatIP` (exact inner product search), which is equivalent to cosine similarity when vectors are L2-normalised. At our scale (~8,000 vectors), exact search runs in <1ms per query.
- **Cosine Similarity**: The cosine of the angle between two vectors, ranging from -1 to 1. Higher values indicate greater similarity. For L2-normalised vectors, cosine similarity equals the dot product.
- **Contrastive Learning**: A training strategy where a model learns to map similar pairs (e.g., an audio clip and its text description) close together in embedding space and dissimilar pairs far apart, typically using InfoNCE loss.
- **Data Leakage**: When information from the evaluation target (e.g., genre labels) is present in the model input, artificially inflating performance metrics.

## Retrieval Views

| View | Model | Dimensions | Input | What It Captures |
|---|---|---|---|---|
| 1. Vibe/Text Search | CLAP (HTSAT-tiny) | 512 | Audio waveform | High-level semantics: mood, genre feel, instrument presence |
| 2. Lyrics/Semantic Search | SBERT (all-MiniLM-L6-v2) | 384 | Metadata + lyrics | Textual semantics: artist identity, lyrical content, descriptive tags |
| 3. Acoustic Similarity | OpenL3 | 512 | Audio waveform | Low-level acoustics: timbre, rhythm, texture |

### Why multiple views?

SBERT and OpenL3 share only **5.6% overlap** in their top-20 neighbors (Spearman rho = -0.77). This means the two modalities retrieve almost entirely different tracks for the same query — they are complementary, not redundant. Fusion combines these independent signals to improve retrieval quality.

## Architecture

```text
Role 1 — Wenny  (OpenL3 Acoustic)    ──┐
Role 2 — Sid & Issac (SBERT Lyrics)  ──┼──> Role 4 — Jiayi (Evaluation & Fusion) ──> Final System
Role 5 — Helena (Fine-tuned CLAP)    ──┘
```

All embedding generators subclass `src/embeddings/base.py:EmbeddingGenerator` with a shared `generate()` / `load_embeddings()` interface. All FAISS indices use the same `src/indexing/faiss_index.py` wrapper. This ensures any view can be swapped in or out of the fusion layer without code changes.

### Fusion Strategies
To combine independent embedding spaces, we employ two different mathematical strategies to prevent any single modality from unfairly dominating the results due to dimensionality or DC offsets.

1. **Vector-Level Early Fusion (Weighted Concatenation):**
   Used in `scripts/generate_fused_embeddings.py`. To create a single offline search index, the OpenL3 and SBERT vectors are first **mean-centered** to remove the native OpenL3 DC offset (which otherwise causes cosine similarities to artificially group near `0.98`). They are then L2-normalized, weighted (e.g., 50/50), concatenated into an 896-d vector, and L2-normalized again.
2. **Rank-Level Late Fusion (Reciprocal Rank Fusion):**
   Used in the live Flask application (`app.py`). Rather than merging the vectors, the system queries each view (CLAP, SBERT, OpenL3) independently. The resulting tracks are scored using Reciprocal Rank Fusion ($Score = \frac{1}{60 + rank}$), creating a robust final leaderboard that gracefully handles edge cases where a single view hallucinates. Mean-centering is also applied at runtime to ensure raw cosine similarity metrics display cleanly on the frontend.

## Results

### CLAP Text-to-Music Retrieval (View 1)

CLAP maps both audio and text into a shared 512-d embedding space via contrastive learning, enabling natural language queries against audio. We generated embeddings for 7,997 of 8,000 tracks (3 corrupt MP3s skipped).

| Query | Top Result | Genre | Cosine Sim |
|---|---|---|---|
| "sad piano ballad" | DUITA — XPURM | Instrumental | 0.65 |
| "aggressive heavy metal with fast drums" | Dead Elements — Angstbreaker | Rock | 0.54 |
| "upbeat happy pop song" | One Way Love — Ready for Men | Pop | 0.52 |
| "acoustic guitar folk song" | Wainiha Valley — Mia Doi Todd | Folk | 0.49 |

Cosine similarity ranges from -1 to 1; scores above 0.4 indicate strong matches in CLAP's embedding space. Scores are lower for underrepresented genres in FMA (jazz, ambient) due to dataset imbalance, not model failure.

**Genre structure** in CLAP embeddings (cosine similarity between genre centroids):
- Most similar: Hip-Hop and Pop (0.81), Folk and International (0.80)
- Most distinct: Rock and Electronic (0.64)
- PCA: 50 components capture ~85% of variance across 512 dimensions

### SBERT Semantic Search (View 2)

SBERT (Sentence-BERT) is a siamese network fine-tuned on NLI/paraphrase data to produce 384-d sentence embeddings. We embed track metadata (title, artist, filtered tags) and lyrics fetched from the Genius API.

**Data leakage fix**: The original input strings included `genre_top` directly, which would inflate any genre-based evaluation. We also found that 48.8% of non-empty `tags` fields contain genre-like labels. Both were removed — genre is used only as evaluation ground truth, never as model input.

**PCA**: The embedding space is highly distributed — 181 components needed for 90% variance (vs. 50 for CLAP's 512-d space). This suggests SBERT utilises its dimensions more uniformly, spreading information across the full 384-d space.

### Echo Nest Evaluation (Independent Ground Truth)

To evaluate retrieval quality without genre leakage, we measured whether each view's nearest neighbors are similar in Echo Nest audio features (danceability, energy, valence, tempo, acousticness, instrumentalness, liveness, speechiness) — features no model saw during training. All 8 features were standardised to z-scores before computing Euclidean distance.

| Method | Avg Distance | vs Random | p-value |
|---|---|---|---|
| Random Baseline | 3.80 | — | — |
| SBERT (Text+Lyrics) | 3.33 | -12.4% | 1.6e-06 |
| Fused (SBERT+OpenL3) | 3.22 | -15.1% | 1.4e-08 |
| **OpenL3 (Audio)** | **2.87** | **-24.3%** | **1.8e-21** |

OpenL3 performs best because both it and Echo Nest operate in the acoustic domain. Fused embeddings outperform either modality alone, confirming that text and audio provide complementary signals. All differences are statistically significant (paired t-test, N=294).

**Caveat**: The 294-track Echo Nest overlap is not uniformly distributed across genres (Folk and Hip-Hop are over-represented at ~21% each vs. 12.5% expected; Experimental is under-represented at 1.4%).


## Team Contributions

### Acoustic Similarity — Wenny
Generated OpenL3 embeddings (512-d) for audio-to-audio retrieval. Compared clustering with CLAP to understand where acoustic and semantic views agree/disagree.

### Lyrics & Semantic Search — Sid & Issac
Generated SBERT embeddings from metadata + Genius lyrics. Identified and fixed genre leakage in input strings and tags. Representation analysis: semantic robustness, lexical bias, truncation impact. (See `reports/sid_issac_lyrics_report.md`).

### Evaluation & Fusion — Jiayi
Built evaluation framework (P@K, Recall@K, MAP, NDCG) and fusion methods (weighted sum, reciprocal rank fusion, learned reranker).

### Fine-tuning & Deep Analysis — Helena
Fine-tuned CLAP on FMA with contrastive learning. Conducted before/after comparison of embeddings, Echo Nest feature correlation, and failure analysis.

## Directory Structure

```text
├── data/
│   ├── fma_small/               # 8,000 MP3 files (organized as 000/000002.mp3)
│   ├── fma_metadata/            # tracks.csv, genres.csv, echonest.csv
│   └── processed/               # Embeddings, FAISS indices, visualisations
│       └── lyrics_enriched/     # Genre-free SBERT + fused embeddings
├── evaluation/                  # Nearest-neighbor genre-consistency tests
├── notebooks/
│   ├── 01_eda.ipynb                       # Dataset exploration
│   ├── 02_clap_retrieval_demo.ipynb       # Text-to-music search demo
│   ├── 03_embedding_visualisation.ipynb   # t-SNE, PCA, genre heatmaps
│   ├── 05_clap_sbert_overlap.ipynb        # Cross-view comparison
│   ├── 06_echonest_exploration.ipynb      # Echo Nest feature analysis
│   ├── 07_sbert_analysis.ipynb            # SBERT representation analysis
│   └── 08_semantic_search_demo.ipynb      # SBERT query interface
├── reports/                     # Presentations, writeups, and final reports
├── scripts/
│   ├── download_fma.py                    # Download FMA dataset
│   ├── audit_metadata.py                  # Cross-reference metadata vs audio
│   ├── generate_clap_embeddings.py        # CLAP embedding pipeline (CLI)
│   ├── generate_sbert_embeddings.py       # SBERT embedding pipeline
│   ├── generate_fused_embeddings.py       # Lyrics-enriched SBERT + OpenL3 fusion
│   ├── build_faiss_index.py               # Build FAISS indices
│   ├── encode_2000_tracks.py              # Encode 2,000-track subset
│   ├── analyze_sbert_robustness.py        # Representation robustness tests
│   └── openl3_vs_sbert_overlap.py         # Cross-view overlap analysis
├── src/
│   ├── config.py                # Paths, constants, device selection
│   ├── metadata.py              # FMA metadata loading and filtering
│   ├── metadata_builder.py      # Text string construction (genre-free)
│   ├── audio_utils.py           # Track path resolution
│   ├── lyrics_fetcher.py        # Genius API lyrics fetcher
│   ├── embeddings/
│   │   ├── base.py              # Abstract EmbeddingGenerator interface
│   │   ├── clap.py              # CLAP pipeline (batched, checkpointed)
│   │   └── sbert.py             # Sentence-BERT pipeline
│   └── indexing/
│       └── faiss_index.py       # FAISS index wrapper (cosine + L2)
├── .env                         # API keys (gitignored)
├── Dockerfile                   # Deployment container
├── requirements.txt
├── app.py                       # Main Flask web application / demo
└── README.md
```

## Adding a New Model

Every retrieval view follows the same pattern: generate two `.npy` files (embeddings + track IDs), then plug them into the app. Here's the standard workflow.

### Important: The 2,000-Track Canonical Subset

All existing views (SBERT, OpenL3, CLAP) share a common 2,000-track subset. The canonical list of track IDs lives in:

```
data/processed/openl3_track_ids.npy    # (2000,) int array — the source of truth
```

This file was created when OpenL3 embeddings were first generated over FMA small (~8,000 tracks). Every downstream pipeline — SBERT, CLAP, fusion, and evaluation — loads these IDs and generates embeddings **only for these 2,000 tracks**. The app computes the intersection of all views at startup (see `app.py:90-96`), so any tracks missing from your new model are simply excluded.

**When adding a new model, always load these IDs as your target set** to ensure your embeddings align with the rest of the system.

### 1. Implement the embedding generator

Subclass `src/embeddings/base.py:EmbeddingGenerator`:

```python
# src/embeddings/my_model.py
from src.embeddings.base import EmbeddingGenerator

class MyModelEmbeddingGenerator(EmbeddingGenerator):
    def generate(self, track_ids, output_dir, batch_size=32, resume=True):
        # Load your model, process tracks, save checkpoints
        # Return: (embeddings: np.ndarray, valid_ids: list)
        ...

    def load_embeddings(self, output_dir):
        embeddings = np.load(output_dir / "mymodel_embeddings.npy")
        track_ids = np.load(output_dir / "mymodel_track_ids.npy").tolist()
        return embeddings, track_ids
```

The two output files must satisfy:
- `mymodel_embeddings.npy` — shape `(N, D)`, float32
- `mymodel_track_ids.npy` — shape `(N,)`, integer track IDs aligned row-by-row with the embeddings

### 2. Write a generation script

Load the canonical 2,000 track IDs and generate embeddings for that set:

```python
# scripts/generate_mymodel_embeddings.py
import numpy as np
from src.config import PROCESSED_DIR
from src.embeddings.my_model import MyModelEmbeddingGenerator

# Load the canonical 2,000-track subset (shared across all views)
track_ids = np.load(PROCESSED_DIR / "openl3_track_ids.npy").astype(int).tolist()
print(f"Target: {len(track_ids)} tracks from canonical subset")

generator = MyModelEmbeddingGenerator()
embeddings, valid_ids = generator.generate(track_ids, PROCESSED_DIR)

# Save with standardized naming
np.save(PROCESSED_DIR / "mymodel_embeddings.npy", embeddings)
np.save(PROCESSED_DIR / "mymodel_track_ids.npy", np.array(valid_ids))
print(f"Saved {len(valid_ids)}/{len(track_ids)} tracks")
```

Run it: `python scripts/generate_mymodel_embeddings.py`

If your model can't process some tracks (e.g., corrupt audio), `valid_ids` will be shorter than 2,000. That's fine — `app.py` intersects all views at startup and only serves tracks present in every view.

### 3. Sanity check the embeddings

Before going further, verify the output:

```python
import numpy as np
emb = np.load("data/processed/mymodel_embeddings.npy")
ids = np.load("data/processed/mymodel_track_ids.npy")
canonical = set(np.load("data/processed/openl3_track_ids.npy").astype(int))

assert emb.ndim == 2 and ids.ndim == 1
assert emb.shape[0] == ids.shape[0]
assert np.isfinite(emb).all()
assert set(ids.astype(int)).issubset(canonical), "IDs outside canonical subset!"
print(f"Shape: {emb.shape}, IDs: {len(ids)}/{len(canonical)} canonical tracks covered")
```

### 4. Build a FAISS index (optional)

```python
from src.indexing.faiss_index import FaissIndex
index = FaissIndex(dimension=emb.shape[1], metric="cosine")
index.build(emb, ids.tolist())
index.save("data/processed/mymodel_faiss.index")
```

### 5. Add the view to `app.py`

One line in the `VIEWS` dict:

```python
VIEWS = {
    "sbert":   load_view("SBERT (Text)",   PROCESSED_DIR / "sbert_embeddings.npy",   PROCESSED_DIR / "sbert_track_ids.npy"),
    "openl3":  load_view("OpenL3 (Audio)",  PROCESSED_DIR / "openl3_embeddings.npy",  PROCESSED_DIR / "openl3_track_ids.npy"),
    "clap":    load_view("CLAP (Vibe)",     PROCESSED_DIR / "clap_embeddings.npy",    PROCESSED_DIR / "clap_track_ids.npy"),
    "mymodel": load_view("MyModel (Label)", PROCESSED_DIR / "mymodel_embeddings.npy", PROCESSED_DIR / "mymodel_track_ids.npy"),
}
```

`load_view` handles mean-centering, L2-normalization, and ID mapping automatically. The new view will appear in per-view recommendations and be included in RRF fusion with no other code changes.

### 6. Evaluate

Run the genre-consistency benchmark in `Evaluation/` to compare against existing models:

```bash
cd Evaluation
python3 evaluate_genre_retrieval.py --root . --num-samples -1 --seed 42
```

You'll need to add your `.npy` files to the `Evaluation/` directory and add a loader entry in the script's `models` list.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download FMA dataset
python scripts/download_fma.py

# Generate CLAP embeddings
python scripts/generate_clap_embeddings.py            # full 8,000 tracks
python scripts/generate_clap_embeddings.py --limit 100 # test run

# Generate lyrics-enriched SBERT + fused embeddings
export GENIUS_API_KEY="your-key"  # or add to .env
python scripts/generate_fused_embeddings.py              # full pipeline
python scripts/generate_fused_embeddings.py --skip-lyrics # reuse cached lyrics

# Build FAISS indices
python scripts/build_faiss_index.py

# Launch web demo
python app.py  # http://localhost:5050
```
