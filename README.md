# Multi-Faceted Music Retrieval

A multimodal music retrieval system using the FMA (Free Music Archive) dataset. The system supports retrieval across four "views" of music:

1. **Vibe/Text Search** — CLAP audio-text embeddings (LAION-CLAP, HTSAT-tiny)
2. **Lyrics/Semantic Search** — Sentence-BERT
3. **Acoustic Similarity** — OpenL3 or MusicFM audio embeddings
4. **Graph-based Recommendation** — Heterogeneous graph (tracks, artists, genres)

## Current Status

### Completed

**Data Pipeline**
- Downloaded and extracted FMA small subset (8,000 tracks, 163 genres, ~7.7 GB audio)
- All 8,000 audio files verified present on disk
- 3 corrupt MP3s identified and handled gracefully (tracks 99134, 108925, 133297)
- Metadata audit saved to `data/processed/audit_results.json`

**View 1: CLAP (Vibe/Text Search)**
- Generated 512-dim CLAP audio embeddings for 7,997 tracks
- Batched processing with checkpoint/resume support (32 tracks/batch)
- Built FAISS index with cosine similarity for fast retrieval
- Text-to-music retrieval working end-to-end
- Tested with diverse queries — results are genre-coherent (e.g. "sad piano ballad" returns Instrumental piano tracks, "aggressive heavy metal" returns Rock/Metal)

**Visualisation & Analysis**
- t-SNE and PCA 2D projections of embeddings, coloured by genre
- Genre centroid cosine similarity heatmap (top 8 genres)
- PCA cumulative variance curve, embedding norm distribution
- Clear genre clustering visible in t-SNE (Hip-Hop, Rock, Instrumental separate well)
- All plots saved in `data/processed/`

### Not Yet Started
- View 2: Acoustic Similarity (OpenL3/MusicFM)
- View 3: Lyrics/Semantic Search (Sentence-BERT)
- View 4: Graph-based Recommendation (PyTorch Geometric)
- Multi-view fusion and evaluation framework

---

## Next Steps

### View 2: Acoustic Similarity (OpenL3 or MusicFM)
- Create `src/embeddings/openl3.py` following the `EmbeddingGenerator` base class
- Generate OpenL3 embeddings for all 8,000 tracks (`content_type="music"`, `embedding_size=512`)
- Alternative: use MusicFM for richer music-specific representations
- Build a second FAISS index — enables "find songs that sound like this one" queries
- Compare clustering structure against CLAP embeddings (t-SNE side by side)

### View 3: Lyrics/Semantic Search (Sentence-BERT)
- **Problem**: FMA has no lyrics data
- **Options**:
  - Pull lyrics from an external API (Genius, Musixmatch) — match by artist + title
  - Pivot to metadata-based semantic search: embed track title + artist + genre + tags with Sentence-BERT
  - Use `echonest.csv` features (danceability, energy, valence) as an alternative signal
- Embed with `sentence-transformers/all-MiniLM-L6-v2` and build a third FAISS index

### View 4: Graph-based Recommendation (PyTorch Geometric)
- Build a heterogeneous graph from FMA metadata:
  - Nodes: tracks, artists, genres (possibly albums)
  - Edges: track→artist, track→genre, artist→genre, co-genre relationships
- Train a GNN (e.g. GraphSAGE or GAT) for node embeddings
- Use learned track embeddings for recommendation-style retrieval
- Install `torch-geometric` (version-specific for your torch/platform)

### Improving Retrieval Quality
- Fine-tune CLAP on FMA data using contrastive learning (match audio with genre/title text descriptions)
- Late fusion: combine scores from multiple views with learned or heuristic weights
- Re-ranking: use a cross-encoder or metadata filter to re-rank top-K results
- Evaluation: build a ground truth set (e.g. same-genre = relevant) and compute Precision@K, MAP, NDCG

### Evaluation Framework
- Define relevance criteria: same top-genre, same sub-genre, or user annotations
- Implement Precision@K, Recall@K, MAP, NDCG in `src/evaluation.py`
- Run evaluation across all four views and the fused system
- Create a notebook with comparison tables and per-query breakdowns

### Stretch Goals
- Interactive demo (Streamlit/Gradio) with audio playback and multi-view search
- Cross-modal retrieval evaluation: how often does text query X return genre Y?
- Ablation study: embedding dimension, CLAP model size, t-SNE perplexity effects

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
