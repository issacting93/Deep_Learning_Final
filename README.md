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
| 4. Graph Recommendation | HeteroGNN (SAGEConv) | 256 | Track-artist-genre graph | Structural connectivity: co-genre relationships, artist similarity |

### Why multiple views?

SBERT and OpenL3 share only **5.6% overlap** in their top-20 neighbors (Spearman rho = -0.77). This means the two modalities retrieve almost entirely different tracks for the same query — they are complementary, not redundant. Fusion combines these independent signals to improve retrieval quality.

## Architecture

```text
Role 1 — Wenny  (OpenL3 Acoustic)    ──┐
Role 2 — Sid & Issac (SBERT Lyrics)  ──┤
Role 3 — Archive (GNN Graph)         ──┼──> Role 4 — Jiayi (Evaluation & Fusion) ──> Final System
Role 5 — Helena (Fine-tuned CLAP)    ──┘
```

All embedding generators subclass `src/embeddings/base.py:EmbeddingGenerator` with a shared `generate()` / `load_embeddings()` interface. All FAISS indices use the same `src/indexing/faiss_index.py` wrapper. This ensures any view can be swapped in or out of the fusion layer without code changes.

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

### GNN Graph Recommendation (View 4)

A 2-layer heterogeneous GNN (SAGEConv) trained on a track-artist-genre graph via link prediction. Produces 256-d track embeddings capturing structural connectivity. A Flask web demo (`role3_graph_archive/app.py`) provides interactive search at `localhost:5050`.

## Team Roles

### Role 1: Acoustic Similarity — Wenny
Generate OpenL3 embeddings (512-d) for audio-to-audio retrieval. Compare clustering with CLAP to understand where acoustic and semantic views agree/disagree.

### Role 2: Lyrics & Semantic Search — Sid & Issac
SBERT embeddings from metadata + Genius lyrics. Identified and fixed genre leakage in input strings and tags. Representation analysis: semantic robustness (10% overlap between "lonely" vs "isolated"), lexical bias, truncation impact. See `role2_lyrics_sid_issac/REPORT.md`.

### Role 3: Graph Recommendation — Archive
Heterogeneous GNN over track-artist-genre graph. Complete with FAISS index and web demo. See `role3_graph_archive/`.

### Role 4: Evaluation & Fusion — Jiayi
Evaluation framework (P@K, Recall@K, MAP, NDCG) and fusion methods (weighted sum, reciprocal rank fusion, learned reranker). Note: genre-based ground truth must account for the leakage fix — genre is excluded from SBERT input, so genre-based evaluation is now fair.

### Role 5: Fine-tuning & Deep Analysis — Helena
Fine-tune CLAP on FMA with contrastive learning. Before/after comparison of embeddings, Echo Nest feature correlation, failure analysis.

## Directory Structure

```text
├── data/
│   ├── fma_small/               # 8,000 MP3 files (organized as 000/000002.mp3)
│   ├── fma_metadata/            # tracks.csv, genres.csv, echonest.csv
│   └── processed/               # Embeddings, FAISS indices, visualisations
│       └── lyrics_enriched/     # Genre-free SBERT + fused embeddings
├── models/                      # Checkpoints (gnn_checkpoint.pt)
├── notebooks/
│   ├── 01_eda.ipynb                       # Dataset exploration
│   ├── 02_clap_retrieval_demo.ipynb       # Text-to-music search demo
│   ├── 03_embedding_visualisation.ipynb   # t-SNE, PCA, genre heatmaps
│   ├── 04_graph_construction_viz.ipynb    # GNN graph structure
│   ├── sbert_analysis.ipynb               # SBERT representation analysis
│   ├── semantic_search_demo.ipynb         # SBERT query interface
│   ├── clap_sbert_overlap.ipynb           # Cross-view comparison
│   └── echonest_exploration.ipynb         # Echo Nest feature analysis
├── role1_acoustic_wenny/        # OpenL3 acoustic embedding tasks
├── role2_lyrics_sid_issac/      # SBERT semantic search + REPORT.md
├── role3_graph_archive/         # GNN graph recommendation + Flask app
├── role4_evaluation_jiayi/      # Evaluation & fusion framework
├── role5_finetune_helena/       # CLAP fine-tuning tasks
├── scripts/
│   ├── download_fma.py                    # Download FMA dataset
│   ├── audit_metadata.py                  # Cross-reference metadata vs audio
│   ├── generate_clap_embeddings.py        # CLAP embedding pipeline (CLI)
│   ├── generate_sbert_embeddings.py       # SBERT embedding pipeline
│   ├── build_faiss_index.py               # Build FAISS indices
│   ├── encode_2000_tracks.py              # Encode 2,000-track subset
│   ├── analyze_sbert_robustness.py        # Representation robustness tests
│   ├── openl3_vs_sbert_overlap.py         # Cross-view overlap analysis
│   └── generate_gnn_visuals.py            # GNN graph visualisations
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
│   ├── graph/
│   │   ├── build_graph.py       # Heterogeneous graph construction
│   │   ├── gnn_model.py         # 2-layer HeteroGNN (SAGEConv)
│   │   └── train.py             # Link prediction training
│   └── indexing/
│       └── faiss_index.py       # FAISS index wrapper (cosine + L2)
├── text_to_text_SBERT_FMA_GENIUS_2.py   # Lyrics-enriched SBERT + OpenL3 fusion
├── .env                         # API keys (gitignored)
├── Dockerfile                   # Deployment container
├── requirements.txt
└── README.md
```

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
python text_to_text_SBERT_FMA_GENIUS_2.py              # full pipeline
python text_to_text_SBERT_FMA_GENIUS_2.py --skip-lyrics # reuse cached lyrics

# Build FAISS indices
python scripts/build_faiss_index.py

# Launch GNN web demo
python role3_graph_archive/app.py  # http://localhost:5050
```
