# View 2: SBERT (Sentence-BERT Semantic Search)

## Overview

SBERT encodes track metadata and lyrics into a 384-dimensional semantic embedding space. Unlike CLAP (which operates on audio) or OpenL3 (which captures acoustic texture), SBERT captures **textual semantics**: artist identity, lyrical themes, descriptive tags, and cultural context.

In our system, SBERT answers queries like "songs by folk artists about travel" or "melancholic indie tracks."

---

## Model Details

| Property | Value |
|----------|-------|
| Model | `all-MiniLM-L6-v2` |
| Embedding dimension | 384 |
| Input | Text strings (metadata + lyrics, max 256 tokens) |
| Training objective | Siamese BERT fine-tuned on NLI + paraphrase data |
| Pretrained on | 1B+ sentence pairs (AllNLI, WikiAnswers, etc.) |
| Library | `sentence-transformers==2.2.2` |

**Architecture:** A 6-layer MiniLM distilled from BERT, with mean pooling over token embeddings. Fast inference (~5,000 sentences/sec on CPU) with strong semantic quality.

---

## Implementation

### Source Files

| File | Purpose |
|------|---------|
| `src/embeddings/sbert.py` | `SentenceBERTEmbeddingGenerator` class |
| `src/metadata_builder.py` | Constructs genre-free text strings from FMA metadata |
| `src/lyrics_fetcher.py` | Genius API lyrics fetching with exponential backoff |
| `scripts/generate_sbert_embeddings.py` | CLI embedding generation script |
| `scripts/generate_fused_embeddings.py` | Lyrics-enriched SBERT + OpenL3 fusion pipeline |
| `scripts/analyze_sbert_robustness.py` | Representation quality analysis |

### Pipeline

```
Track metadata (title, artist, tags) + Lyrics → Text string → all-MiniLM-L6-v2 → 384-d vector → L2-normalize
```

1. **Metadata string construction** (`src/metadata_builder.py`):
   - Format: `"{artist} - {title}. Tags: {filtered_tags}."`
   - Genre labels are **stripped** from both `genre_top` and tag fields to prevent evaluation leakage.
   - Text is lowercased and punctuation-cleaned via `normalize_text()`.

2. **Lyrics enrichment** (optional, `scripts/generate_fused_embeddings.py`):
   - Lyrics fetched from Genius API (first 1,000 characters).
   - Appended to metadata string: `"{metadata} Lyrics: {lyrics_text}"`
   - Cached in `data/processed/lyrics_enriched/lyrics_df.csv` to avoid re-fetching.

3. **Encoding:**
   - `normalize_embeddings=True` in SentenceTransformer produces unit-length vectors directly.
   - Batch size: 32 (configurable).

4. **Validation:** Norms checked post-generation; renormalized if not unit-length.

### Data Leakage Prevention

This is the most critical design constraint for SBERT:

| Leakage Vector | Status | Implementation |
|---|---|---|
| `genre_top` in input string | **Blocked** | Field excluded from metadata string |
| Genre words in `tags` field | **Blocked** | `strip_genre_from_tags()` filters 8 genre labels |
| Artist name as genre proxy | **Acknowledged** | Kept (legitimately useful), noted as soft leakage |

Without these filters, SBERT achieves artificially inflated genre retrieval accuracy (genre words become trivial nearest-neighbor signals).

---

## Generation

### Basic (metadata only)

```bash
# Full dataset
python scripts/generate_sbert_embeddings.py

# Limited run
python scripts/generate_sbert_embeddings.py --limit 500

# Custom model
python scripts/generate_sbert_embeddings.py --model "all-mpnet-base-v2"
```

### With lyrics (Genius API)

```bash
# Set API key
export GENIUS_API_KEY="your_key"  # or add to .env

# Full pipeline (fetch lyrics + encode + fuse with OpenL3)
python scripts/generate_fused_embeddings.py

# Skip lyrics fetch (reuse cached lyrics_df.csv)
python scripts/generate_fused_embeddings.py --skip-lyrics

# Adjust fusion weight (0 = audio only, 1 = text only)
python scripts/generate_fused_embeddings.py --weight-text 0.6
```

### Canonical 2,000-track subset

```bash
python scripts/encode_2000_tracks.py
```

**Output:**
- `data/processed/sbert_embeddings.npy` — shape `(N, 384)`, float32
- `data/processed/sbert_track_ids.npy` — shape `(N,)`, int
- `data/processed/metadata_texts.csv` — the text strings fed to SBERT (for inspection)

---

## Results

### Genre Retrieval Accuracy (Top-1)

| Variant | Accuracy (500) | Accuracy (full 2,000) |
|---------|---------------|---------------------|
| SBERT (metadata + lyrics) | 0.602 | 0.576 |

Lower than CLAP text because genre labels are intentionally removed from input. This is the correct behavior — it proves SBERT isn't "cheating" via genre words.

### Representation Analysis

**PCA variance:** 181 components needed for 90% variance (vs. 50 for CLAP's 512-d space). SBERT utilizes its dimensions more uniformly — information is distributed across the full 384-d space rather than concentrated.

**Semantic robustness** (`scripts/analyze_sbert_robustness.py`):
- Synonym queries ("lonely" vs "isolated") produce overlapping top-10 results, confirming semantic rather than lexical matching.
- Lexical bias test: "Blue music" retrieves tracks with "blue" in the title more than "blues" genre tracks — mild lexical bias exists but doesn't dominate.

**Truncation impact:** For tracks with long metadata (>128 words), the first half and second half of the text produce different embeddings (avg cosine shift ~0.6). This suggests front-loading important information (title, artist) matters.

### Cross-View Complementarity

SBERT and OpenL3 share only **5.6% overlap** in top-20 neighbors (Spearman rho = -0.77). This extreme complementarity is why fusion improves results — the two views retrieve almost entirely different tracks.

---

## Echo Nest Evaluation

| Method | Avg Echo Nest Distance | vs Random |
|--------|----------------------|-----------|
| Random Baseline | 3.80 | — |
| SBERT (Text+Lyrics) | 3.33 | -12.4% |

SBERT neighbors are 12.4% closer in acoustic features than random, despite never seeing audio. This confirms metadata and lyrics carry genuine musical information beyond genre labels.

---

## Fusion with OpenL3

The lyrics-enriched pipeline (`scripts/generate_fused_embeddings.py`) creates a fused embedding:

```
SBERT 384-d ─┐
             ├── mean-center → L2-norm → weight → concatenate → L2-norm → 896-d fused
OpenL3 512-d ─┘
```

| Method | Echo Nest Distance | vs Random |
|--------|-------------------|-----------|
| SBERT alone | 3.33 | -12.4% |
| Fused (SBERT + OpenL3) | 3.22 | -15.1% |

Fusion improves over either modality alone, confirming text and audio provide complementary signals.

---

## Genius API Integration

The lyrics fetcher (`src/lyrics_fetcher.py`) handles Genius API calls with:
- **Exponential backoff:** 3 retries with 0.5s base delay (0.5s, 1s, 2s).
- **Rate limiting:** 0.2s delay between requests (5 req/s Genius limit).
- **Title cleaning:** Removes `(feat. ...)`, `[Remastered]`, etc. before searching.
- **Caching:** Results saved to CSV; API only called once per track.

```bash
# Standalone lyrics fetch (requires GENIUS_API_TOKEN env var)
python -c "from src.lyrics_fetcher import fetch_lyrics_for_fma; fetch_lyrics_for_fma('your_token')"
```

---

## References

- Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," EMNLP 2019.
- Model: [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) via sentence-transformers
- Genius API: [genius.com/developers](https://genius.com/developers)
