# Development Guide

## Setup

```bash
# Clone and enter the project
cd Finals/

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Python version:** 3.9+

**Platform notes:**
- macOS with Apple Silicon: MPS backend is disabled for CLAP (incomplete op support). Falls back to CPU.
- CUDA: Automatically detected and used if available.

---

## Environment Variables

Create a `.env` file in the project root (already in `.gitignore`):

```bash
GENIUS_API_KEY=your_key_here
```

Required only for lyrics fetching. All other functionality works without it.

---

## Running the App

```bash
python app.py
# Open http://localhost:5001
```

The app requires precomputed embeddings in `data/processed/`. These are included in the repo for the 2,000-track subset.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests cover:
- FAISS index operations (build, query, save/load, metrics)
- Embedding norm validation (checks saved `.npy` files are properly normalized)
- Configuration sanity (constants within expected bounds)

Tests that depend on generated embeddings are skipped if the `.npy` files don't exist.

---

## Project Layout

```
src/                    # Core library (importable modules)
  config.py             # All paths and constants
  metadata.py           # FMA metadata loading
  metadata_builder.py   # Text construction for SBERT (genre-free)
  audio_utils.py        # Track path resolution
  lyrics_fetcher.py     # Genius API with retry/backoff
  embeddings/           # Embedding generators (abstract base + implementations)
  indexing/             # FAISS index wrapper

scripts/                # CLI pipelines (data generation, analysis)
tests/                  # pytest test suite
evaluation/             # Genre retrieval benchmarks
notebooks/              # Jupyter exploration and demos
templates/              # Flask frontend (single HTML file)
```

---

## Code Conventions

### Logging

All modules use Python's `logging` module:

```python
import logging
logger = logging.getLogger(__name__)
```

Do not use `print()` for operational output. Reserve `print()` for CLI scripts that need user-facing output.

### Embeddings

All embedding files follow the naming convention:
- `{model}_embeddings.npy` — shape `(N, D)`, float32
- `{model}_track_ids.npy` — shape `(N,)`, integer track IDs (row-aligned)

Embeddings must be L2-normalized before use with cosine similarity (dot product).

### Normalization

Use epsilon-based normalization to avoid division-by-zero:

```python
norms = np.linalg.norm(embs, axis=1, keepdims=True)
embs = embs / (norms + 1e-8)
```

Or use `np.maximum`:

```python
norms = np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12)
embs = embs / norms
```

### Error Handling

- Flask routes validate inputs and return proper HTTP status codes (400, 404, 500).
- External API calls (Genius) use exponential backoff with configurable retries.
- Embedding generation scripts use checkpoint/resume to handle interruptions.

---

## Docker

Build and run:

```bash
docker build -t music-retrieval .
docker run -p 5050:5050 music-retrieval
```

The container runs Gunicorn with 4 workers on port 5050. Timeout is set to 120s to accommodate model loading. Note: the Docker image does not include audio files (`data/fma_small/`) — mount them as a volume or modify the Dockerfile for your deployment.

---

## Adding a New Embedding View

See the "Adding a New Model" section in the main README. The process is:

1. Subclass `src/embeddings/base.py:EmbeddingGenerator`
2. Write a generation script targeting the canonical 2,000 track IDs
3. Save as `{model}_embeddings.npy` + `{model}_track_ids.npy`
4. Add one line to the `VIEWS` dict in `app.py`

The view automatically participates in RRF fusion and appears in the UI.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Mean-centering before normalization | Removes DC offset that artificially inflates cosine similarities (especially in OpenL3 where raw scores cluster near 0.98) |
| RRF over learned fusion | Robust, requires no training, handles missing views gracefully |
| Genre stripped from SBERT input | Prevents evaluation leakage — genre is only used as ground truth |
| FAISS IndexFlatIP (exact search) | At 2,000 vectors, exact search is <1ms. No need for approximate indices. |
| Checkpoint/resume in CLAP | CLAP inference is slow (~2 min/batch). Checkpointing prevents re-processing after interruption. |
| Atomic file saves (tmp + rename) | Prevents corrupt index files if process is killed mid-write |
