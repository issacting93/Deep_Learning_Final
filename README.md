# Multi-Faceted Music Retrieval

A multimodal music retrieval system using the FMA (Free Music Archive) dataset. The system supports retrieval across four "views" of music:

1. **Vibe/Text Search** — CLAP audio-text embeddings (LAION-CLAP, HTSAT-tiny)
2. **Lyrics/Semantic Search** — Sentence-BERT
3. **Acoustic Similarity** — OpenL3 or MusicFM audio embeddings
4. **Graph-based Recommendation** — Heterogeneous graph (tracks, artists, genres)

## What Has Been Done

### 1. Data Acquisition & Validation

We downloaded the FMA (Free Music Archive) small subset, which contains 8,000 tracks across 163 genres totalling ~7.7 GB of MP3 audio. The dataset also includes rich metadata: `tracks.csv` (106,574 entries across all subsets with multi-level column headers covering track info, artist info, album info, and set splits), `genres.csv` (163 genre taxonomy), and `echonest.csv` (audio features like danceability, energy, and valence from the Echo Nest API).

We built an automated download-and-extract pipeline (`scripts/download_fma.py`) that fetches the zip archives and extracts them into the project directory structure. An audit script (`scripts/audit_metadata.py`) cross-references the metadata against files actually on disk, confirming all 8,000 tracks are present. The audit also reports subset distribution (small: 8,000, medium: 17,000, large: 81,574), average track duration (277.85s), and genre counts. Results are persisted to `data/processed/audit_results.json` for downstream use.

### 2. Codebase Architecture

We designed a modular `src/` package to support all four retrieval views through a shared interface:

- **`src/config.py`** centralises all file paths (relative to project root), audio constants (48kHz sample rate, 10s CLAP window), batch sizes, and automatic device selection (CUDA > CPU; MPS disabled due to incomplete op support in CLAP).
- **`src/metadata.py`** handles loading the FMA multi-index CSV files and filtering to the small subset.
- **`src/audio_utils.py`** resolves FMA's directory structure (`000/000002.mp3`) and discovers which track IDs have valid audio files on disk.
- **`src/embeddings/base.py`** defines an abstract `EmbeddingGenerator` class with `generate()` and `load_embeddings()` methods, so all future embedding pipelines (OpenL3, Sentence-BERT, etc.) follow the same interface.
- **`src/indexing/faiss_index.py`** wraps FAISS with build/save/load/query operations, supporting both cosine similarity (via L2-normalised inner product) and L2 distance metrics.

### 3. CLAP Embedding Generation (View 1: Vibe/Text Search)

We used LAION-CLAP (HTSAT-tiny backbone) to generate 512-dimensional audio embeddings for the entire small subset. CLAP is a contrastive language-audio pretraining model that maps both audio and text into a shared embedding space, enabling text-to-music retrieval.

The pipeline (`src/embeddings/clap.py`) processes tracks in batches of 32 using the `get_audio_embedding_from_filelist` API, which internally handles resampling to 48kHz, int16 quantisation, and feature extraction with random truncation for clips longer than 10 seconds. Each batch is checkpointed to `data/processed/clap_batches/batch_NNNN.npz`, with a progress file tracking completed batches. This means the pipeline can resume after interruption without reprocessing. If a batch fails (e.g. a corrupt MP3), it falls back to single-file processing, skipping only the broken tracks.

The full run processed all 8,000 tracks in ~8 minutes on CPU, producing 7,997 embeddings (3 corrupt MP3s were skipped: tracks 99134, 108925, 133297). Batch files are consolidated into `data/processed/clap_embeddings.npy` (7997 x 512 float32) and `data/processed/clap_track_ids.npy`.

### 4. FAISS Index & Retrieval

We built a FAISS `IndexFlatIP` (exact inner product search) over the L2-normalised CLAP embeddings, making inner product equivalent to cosine similarity. With only ~8,000 512-dim vectors (~16 MB), exact search is instantaneous (<1ms per query). The index is saved to `data/processed/clap_faiss.index`.

Text-to-music retrieval works by embedding a natural language query through CLAP's text encoder, then querying the FAISS index for the nearest audio embeddings. We tested with 8 diverse queries:

| Query | Top Result | Genre | Score |
|---|---|---|---|
| "sad piano ballad" | DUITA — XPURM | Instrumental | 0.6513 |
| "aggressive heavy metal with fast drums" | Dead Elements — Angstbreaker | Rock | 0.5395 |
| "upbeat happy pop song" | One Way Love — Ready for Men | Pop | 0.5248 |
| "acoustic guitar folk song" | Wainiha Valley — Mia Doi Todd | Folk | 0.4900 |
| "chill lo-fi vibes" | Apparitions Under Glass — Sarin | Rock | 0.4072 |
| "funky bass groove" | Okružen mrtvima — CROCODILE TEARS | Experimental | 0.4200 |
| "ambient electronic soundscape" | Emptiness — Alex Mason | Instrumental | 0.3482 |
| "jazz saxophone improvisation" | Slow Moving — Moon Veil | Electronic | 0.3185 |

Results are genre-coherent for well-represented genres (Instrumental, Rock, Pop, Folk). Lower scores on jazz/ambient reflect the FMA small subset's genre imbalance rather than model failure.

### 5. Embedding Visualisation & Analysis

We projected the 7,997 CLAP embeddings into 2D using both t-SNE (with PCA-50 preprocessing, perplexity=30) and PCA, coloured by top genre. Key findings:

- **t-SNE** shows clear genre clustering: Hip-Hop forms a distinct cluster in the upper-left, Rock groups in the left, Instrumental occupies the lower-right, and Electronic scatters across the right side. Folk and International overlap in the center-bottom, which is expected given their acoustic similarity.
- **Genre centroid similarity heatmap** (cosine similarity between mean embeddings of the top 8 genres) reveals that Hip-Hop and Pop are most similar (0.81), Folk and International cluster together (0.80), and Rock and Electronic are the most distinct pair (0.64).
- **PCA variance curve** shows the 512 dimensions are well-utilised — 50 components capture ~85% of variance, indicating the embeddings are rich and non-degenerate.
- **Embedding norms** are exactly 1.0 for all tracks, confirming CLAP outputs are already L2-normalised (consistent with contrastive learning objectives).

All plots are saved in `data/processed/` as PNG files.

---

## Team Roles & Next Steps

Each role produces independent outputs that feed into a shared evaluation and fusion system. All new embedding generators should subclass `src/embeddings/base.py:EmbeddingGenerator` and output to `data/processed/`.

### Role 1: Acoustic Similarity (OpenL3/MusicFM Embeddings) — Wenny

**Goal**: Build a second embedding space that captures low-level acoustic features (timbre, rhythm, texture) rather than CLAP's high-level semantic features. This enables audio-to-audio retrieval — "find songs that sound like this one".

**Tasks**:
- Create `src/embeddings/openl3.py` following the `EmbeddingGenerator` base class
- Generate OpenL3 embeddings for all 8,000 tracks (`content_type="music"`, `embedding_size=512`)
- Alternative: use MusicFM for richer music-specific representations (pretrained on large-scale music data)
- Build a second FAISS index (`data/processed/openl3_faiss.index`)
- Produce a side-by-side t-SNE comparison: do CLAP and OpenL3 cluster genres the same way? Where do they disagree? (e.g. CLAP groups Hip-Hop and Pop together at 0.81 similarity — does OpenL3 separate them better based on acoustic features?)
- Test audio-to-audio queries: pick a track, find its nearest neighbours, check if results are musically coherent

**Deliverables**: `src/embeddings/openl3.py`, FAISS index, comparison notebook

### Role 2: Lyrics & Semantic Search (Sentence-BERT) — Sid

**Goal**: Build a text-based retrieval view using track metadata. This complements the audio-based views by enabling search over artist names, track titles, genre tags, and descriptive text.

**Tasks**:
- For each track, construct a text string from metadata: `"{title} by {artist}. Genre: {genre}. Tags: {tags}."` — pull these fields from `tracks.csv`
- Embed all 8,000 text strings with Sentence-BERT (`sentence-transformers/all-MiniLM-L6-v2`, 384-dim)
- Build a third FAISS index (`data/processed/sbert_faiss.index`)
- Optionally: pull lyrics from the Genius API for tracks where available, embed those separately as a richer text signal
- Analyse overlap with CLAP: for the same text query, do CLAP (audio-based) and SBERT (metadata-based) return the same tracks? Compute rank correlation between the two result lists
- Explore the `echonest.csv` features (danceability, energy, valence, tempo) — can these be used as additional retrieval signals or filters?

**Deliverables**: `src/embeddings/sbert.py`, FAISS index, overlap analysis notebook

### Role 3: Graph-based Recommendation (PyTorch Geometric) — Issac

**Goal**: Build a heterogeneous graph from FMA metadata and learn structural embeddings that encode relationships between tracks, artists, and genres. This enables recommendation-style retrieval based on connectivity rather than content.

**Tasks**:
- Install `torch-geometric` (version must match your torch/CUDA/platform)
- Construct a heterogeneous graph from FMA metadata:
  - **Nodes**: tracks (8,000), artists, genres (163), optionally albums
  - **Edges**: track→artist, track→genre, artist→genre, tracks sharing the same genre (co-genre edges)
- Train a GNN (GraphSAGE or GAT) to produce node embeddings via link prediction or node classification
- Build a fourth FAISS index from learned track embeddings
- Analyse: which tracks are most central in the graph (highest degree/PageRank)? Do graph-based recommendations surface tracks that audio-based methods miss?
- Visualise the graph structure (e.g. genre subgraph connectivity)

**Deliverables**: `src/graph/`, trained GNN checkpoint in `models/`, recommendation notebook

### Role 4: Evaluation & Multi-View Fusion — Jiayi

**Goal**: Quantitatively evaluate each retrieval view and build a fusion system that combines all views to outperform any single one.

**Tasks**:
- Define ground truth relevance criteria:
  - **Baseline**: same top-genre = relevant (8 classes, easy)
  - **Harder**: same sub-genre = relevant (163 classes)
  - **Optional**: manual annotations for a small query set
- Implement evaluation metrics in `src/evaluation.py`: Precision@K, Recall@K, MAP, NDCG
- Evaluate each view independently — which is best at genre retrieval? At discovering cross-genre connections?
- Build a late fusion system in `src/fusion.py`:
  - **Simple**: weighted sum of cosine similarity scores from each view
  - **Rank fusion**: reciprocal rank fusion (RRF) across all views
  - **Learned**: train a small MLP that takes per-view scores as input and predicts relevance
- Show that fusion outperforms any single view across all metrics
- Create a results notebook with comparison tables, per-genre breakdowns, and statistical significance tests

**Deliverables**: `src/evaluation.py`, `src/fusion.py`, comprehensive evaluation notebook

### Role 5: Fine-tuning & Deep Analysis — Helena

**Goal**: Improve the base CLAP model by fine-tuning on FMA data, and conduct deeper analysis of the embedding space to understand model behaviour.

**Tasks**:
- **Fine-tune CLAP** on FMA using contrastive learning:
  - Positive pairs: (audio clip, "{title} by {artist}, {genre} music") text descriptions
  - Use the existing CLAP checkpoint as initialisation, fine-tune with InfoNCE loss
  - Train on the small subset, validate on a held-out split
- **Before/after comparison**: regenerate embeddings with the fine-tuned model, compare t-SNE plots and genre similarity heatmaps. Does the Hip-Hop/Pop conflation (0.81 similarity) reduce? Do retrieval scores improve?
- **Echo Nest feature correlation**: correlate CLAP PCA dimensions with Echo Nest features (danceability, energy, valence, tempo) — which embedding dimensions encode which musical attributes?
- **Failure analysis**: identify systematic retrieval failures. Which queries consistently return wrong-genre results? Are there tracks that are outliers in embedding space? What makes them unusual?
- **Ablation study**: test different CLAP backbones (HTSAT-tiny vs HTSAT-base), embedding dimensions, and t-SNE hyperparameters

**Deliverables**: fine-tuned checkpoint in `models/`, before/after comparison notebook, failure analysis notebook

---

### How the roles connect

```
Role 1 — Wenny  (OpenL3/Acoustic) ──────┐
Role 2 — Sid    (Lyrics/SBERT) ─────────┤
Role 3 — Issac  (Graph/GNN) ────────────┼──→ Role 4 — Jiayi (Evaluation & Fusion) ──→ Final System
Role 5 — Helena (Fine-tuned CLAP) ──────┘
```

Each role produces embeddings and a FAISS index. Role 4 consumes all of them to evaluate and fuse. Role 5 improves the base CLAP view that's already working. All roles can work in parallel — the shared `EmbeddingGenerator` interface and `FaissIndex` wrapper ensure consistency.

---

## Directory Structure

```
├── data/
│   ├── raw/                  # Downloaded zip files
│   ├── fma_small/            # 8,000 MP3 audio files (organized by track ID)
│   ├── fma_metadata/         # tracks.csv, genres.csv, echonest.csv, etc.
│   └── processed/            # Embeddings, FAISS indices, plots, audit results
├── src/
│   ├── config.py             # Centralized paths, constants, device selection
│   ├── metadata.py           # FMA metadata loading and filtering
│   ├── audio_utils.py        # Track path resolution, valid-track discovery
│   ├── embeddings/
│   │   ├── base.py           # Abstract EmbeddingGenerator base class
│   │   └── clap.py           # CLAP embedding pipeline (batched, checkpointed)
│   └── indexing/
│       └── faiss_index.py    # FAISS index build/save/load/query
├── scripts/
│   ├── download_fma.py       # Download and extract FMA dataset
│   ├── audit_metadata.py     # Audit metadata and cross-reference audio files
│   ├── generate_clap_embeddings.py   # Generate CLAP embeddings (CLI)
│   └── build_faiss_index.py  # Build FAISS index from embeddings
├── notebooks/
│   ├── 01_eda.ipynb                    # Dataset exploration
│   ├── 02_clap_retrieval_demo.ipynb    # Interactive text-to-music search
│   └── 03_embedding_visualisation.ipynb # t-SNE, PCA, similarity heatmaps
├── models/                   # Pretrained model checkpoints
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download and extract data:
   ```bash
   python scripts/download_fma.py
   ```
4. Generate CLAP embeddings:
   ```bash
   python scripts/generate_clap_embeddings.py            # full run (8,000 tracks)
   python scripts/generate_clap_embeddings.py --limit 100 # test run
   ```
5. Build FAISS index:
   ```bash
   python scripts/build_faiss_index.py
   ```
6. Open the retrieval demo:
   ```bash
   jupyter notebook notebooks/02_clap_retrieval_demo.ipynb
   ```
