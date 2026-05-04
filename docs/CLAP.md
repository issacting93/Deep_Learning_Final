# View 1: CLAP (Contrastive Language-Audio Pretraining)

## Overview

CLAP maps audio waveforms and text descriptions into a shared 512-dimensional embedding space via contrastive learning. This enables both audio-to-audio similarity search and natural language queries against audio ("find me a sad piano ballad").

In our system, CLAP captures **high-level semantics**: mood, genre feel, instrument presence, and overall vibe.

---

## Model Details

| Property | Value |
|----------|-------|
| Model | LAION-CLAP (HTSAT-tiny backbone) |
| Embedding dimension | 512 |
| Input | Raw audio waveform (48 kHz, 10-second clips) |
| Training objective | InfoNCE contrastive loss (audio-text pairs) |
| Pretrained on | LAION-Audio-630K (630K audio-text pairs from the web) |
| Library | `laion-clap==1.1.6` |

**Architecture:** CLAP uses a dual-encoder design:
- **Audio encoder:** HTSAT-tiny (Hierarchical Token-Semantic Audio Transformer) — processes mel spectrograms through a Swin Transformer variant.
- **Text encoder:** RoBERTa-based — encodes free-text descriptions.

Both encoders project to a shared 512-d space where matched audio-text pairs are close (high cosine similarity) and unmatched pairs are far apart.

---

## Implementation

### Source Files

| File | Purpose |
|------|---------|
| `src/embeddings/clap.py` | `CLAPEmbeddingGenerator` class — batched inference with checkpoint/resume |
| `scripts/generate_clap_embeddings.py` | CLI script to generate embeddings for FMA tracks |
| `src/config.py` | Constants: `CLAP_SR=48000`, `CLAP_DURATION_S=10`, `CLAP_BATCH_SIZE=32` |

### Pipeline

```
Audio file (MP3) → Resample to 48 kHz → Clip to 10s → HTSAT-tiny → 512-d vector → L2-normalize
```

1. **Input preprocessing:** Audio is loaded at 48 kHz and clipped/padded to exactly 10 seconds (480,000 samples). CLAP handles this internally.
2. **Batched inference:** Tracks are processed in batches of 32. Each batch is checkpointed to disk (`data/processed/clap_batches/batch_XXXX.npz`).
3. **Fault tolerance:** If a batch fails (corrupt audio), the generator falls back to single-file processing, skipping only the corrupt track.
4. **Consolidation:** All batch files are merged into a single `clap_embeddings.npy` and `clap_track_ids.npy`.
5. **Normalization validation:** After consolidation, norms are checked. If not unit-length, vectors are renormalized with epsilon safety (`/ (norms + 1e-8)`).

### Cross-Modal Retrieval

CLAP's unique capability is **text-to-audio search**. The `embed_text()` method encodes natural language queries into the same 512-d space:

```python
generator = CLAPEmbeddingGenerator()
query_emb = generator.embed_text(["upbeat dance music with synths"])
# query_emb can be compared directly to audio embeddings via dot product
```

This is used in notebook demos (`notebooks/02_clap_retrieval_demo.ipynb`) but not in the live Flask app (which uses track-to-track retrieval only).

---

## Generation

```bash
# Full dataset (8,000 tracks, ~2-3 hours on CPU)
python scripts/generate_clap_embeddings.py

# Test run (100 tracks)
python scripts/generate_clap_embeddings.py --limit 100

# Custom batch size (reduce if memory-constrained)
python scripts/generate_clap_embeddings.py --batch-size 16

# Start fresh (ignore checkpoints)
python scripts/generate_clap_embeddings.py --no-resume
```

**Output:**
- `data/processed/clap_embeddings.npy` — shape `(N, 512)`, float32
- `data/processed/clap_track_ids.npy` — shape `(N,)`, int

**Performance:** ~2-3 hours for 8,000 tracks on CPU. Checkpoint/resume means interruptions don't lose progress.

---

## Results

### Text-to-Audio Retrieval

| Query | Top Result | Genre | Cosine Sim |
|-------|-----------|-------|------------|
| "sad piano ballad" | DUITA — XPURM | Instrumental | 0.65 |
| "aggressive heavy metal with fast drums" | Dead Elements — Angstbreaker | Rock | 0.54 |
| "upbeat happy pop song" | One Way Love — Ready for Men | Pop | 0.52 |
| "acoustic guitar folk song" | Wainiha Valley — Mia Doi Todd | Folk | 0.49 |

Scores above 0.4 indicate strong matches in CLAP's space. Lower scores for underrepresented genres reflect FMA's imbalance, not model failure.

### Genre Retrieval Accuracy (Top-1)

| Variant | Accuracy (500 samples) | Accuracy (full 2,000) |
|---------|----------------------|---------------------|
| CLAP (audio) | 0.502 | 0.537 |
| CLAP (text, metadata-derived) | 0.676 | 0.677 |

CLAP text outperforms CLAP audio because the metadata strings (title, artist, tags) carry strong genre signal even after genre label removal.

### Embedding Space Properties

- **Genre centroid similarity:** Hip-Hop/Pop most similar (0.81), Rock/Electronic most distinct (0.64)
- **PCA variance:** 50 components capture ~85% of variance across 512 dimensions
- **Interpretation:** CLAP embeddings are relatively low-rank — semantic information is concentrated in fewer dimensions than SBERT

---

## Platform Notes

- **macOS (Apple Silicon):** MPS backend disabled for CLAP due to incomplete operator support. Falls back to CPU automatically (see `src/config.py:get_device()`).
- **CUDA:** Automatically detected and used if available. Expect ~5x speedup over CPU.
- **Memory:** Each batch of 32 requires ~2 GB RAM. Reduce batch size if constrained.

---

## References

- Wu et al., "Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation," ICASSP 2023.
- Code: [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP)
- Model checkpoint: `630k-audioset-best.pt` (downloaded automatically on first run)
