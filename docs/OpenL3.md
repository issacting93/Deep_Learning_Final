# View 3: OpenL3 (Acoustic Similarity)

## Overview

OpenL3 produces 512-dimensional audio embeddings that capture **low-level acoustic properties**: timbre, rhythm, texture, and spectral characteristics. Unlike CLAP (which learns high-level semantics via contrastive text-audio training), OpenL3 is trained purely on audio-visual correspondence — it learns what sounds "go with" visual scenes, resulting in embeddings that reflect acoustic texture rather than semantic meaning.

In our system, OpenL3 answers the question: "what tracks sound like this one?" — regardless of genre, lyrics, or metadata.

---

## Model Details

| Property | Value |
|----------|-------|
| Model | OpenL3 (music-trained variant) |
| Embedding dimension | 512 |
| Input | Raw audio waveform |
| Training objective | Audio-visual correspondence (self-supervised) |
| Pretrained on | AudioSet (2M+ YouTube clips with paired video frames) |
| Content type | Music (vs. environmental sound variant) |

**Architecture:** OpenL3 uses a VGG-like CNN that processes mel spectrograms. The network was trained to predict whether an audio clip and a video frame come from the same source video — a self-supervised proxy task that forces the model to learn rich acoustic representations without any labeled data.

**Key insight:** Because OpenL3 learns from audio-visual correspondence (not text-audio or genre labels), its embedding space reflects acoustic texture and production quality rather than semantic categories. Two songs can be in completely different genres but sound similar in timbre/rhythm and thus be close in OpenL3 space.

---

## Implementation

### Source Files

| File | Purpose |
|------|---------|
| External (OpenL3 library) | Embedding generation |
| `scripts/openl3_vs_sbert_overlap.py` | Cross-view analysis with SBERT |
| `scripts/generate_fused_embeddings.py` | Fusion with SBERT |
| `evaluation/evaluate_genre_retrieval.py` | Genre consistency evaluation |

### Embeddings

OpenL3 embeddings were generated externally and provided as:
- `data/processed/openl3_embeddings.npy` — shape `(2000, 512)`, float32
- `data/processed/openl3_track_ids.npy` — shape `(2000,)`, int

These track IDs serve as the **canonical 2,000-track subset** that all other views target.

### Normalization

OpenL3 outputs **raw (unnormalized)** embeddings. Unlike SBERT (which normalizes during encoding) or CLAP (which produces near-unit-length vectors), OpenL3 embeddings have varying norms. The app handles this at load time:

```python
# app.py: load_view()
embs = embs - embs.mean(axis=0)    # Mean-center (removes DC offset)
norms = np.linalg.norm(embs, axis=1, keepdims=True)
embs = embs / (norms + 1e-8)       # L2-normalize for cosine similarity
```

**Why mean-centering matters for OpenL3:** Raw OpenL3 embeddings have a large DC offset — all vectors point roughly the same direction in the 512-d space. This causes raw cosine similarities to cluster around 0.98 regardless of actual musical similarity. Mean-centering removes this offset, making the cosine metric discriminative.

---

## Results

### Genre Retrieval Accuracy (Top-1)

| Setting | Accuracy (500 samples) | Accuracy (full 2,000) |
|---------|----------------------|---------------------|
| OpenL3 | 0.526 | 0.553 |

Moderate genre accuracy — acoustic texture partially correlates with genre (rock tracks tend to have similar distorted guitar textures; electronic tracks share synthesizer timbres) but many acoustically similar tracks cross genre boundaries.

### Echo Nest Evaluation

| Method | Avg Echo Nest Distance | vs Random | p-value |
|--------|----------------------|-----------|---------|
| Random Baseline | 3.80 | — | — |
| **OpenL3 (Audio)** | **2.87** | **-24.3%** | **1.8e-21** |

OpenL3 achieves the **best performance** on Echo Nest feature similarity — 24.3% closer than random. This makes sense: both OpenL3 and Echo Nest operate in the acoustic domain. Tracks that OpenL3 considers similar genuinely share acoustic properties (danceability, energy, tempo, etc.).

### Cross-View Analysis (vs SBERT)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Overlap@5 | 3.2% | Almost no shared neighbors at k=5 |
| Overlap@10 | 4.8% | Still minimal overlap |
| Overlap@20 | 5.6% | Views retrieve different tracks |
| Mean Spearman rho | -0.77 | Rankings are **anti-correlated** |

The negative Spearman correlation (-0.77) means that tracks SBERT considers similar, OpenL3 considers dissimilar — and vice versa. This extreme complementarity is why fusion works: the two views provide genuinely independent retrieval signals.

**Per-genre agreement** (from `scripts/openl3_vs_sbert_overlap.py`):
- Genre pairs with highest OpenL3/SBERT disagreement often involve "cross-genre" similarities — e.g., lo-fi hip-hop and indie folk may share acoustic textures (soft, intimate production) but have completely different textual metadata.

---

## Role in Fusion

### Rank-Level Fusion (Live App)

In `app.py`, OpenL3 contributes one of three ranked lists to Reciprocal Rank Fusion:

```
RRF_score(track) = 1/(60 + rank_CLAP) + 1/(60 + rank_SBERT) + 1/(60 + rank_OpenL3)
```

OpenL3's contribution ensures acoustically similar tracks are boosted even when they don't match textually (SBERT) or semantically (CLAP).

### Vector-Level Fusion (Offline Index)

In `scripts/generate_fused_embeddings.py`, OpenL3 embeddings are concatenated with SBERT:

```
OpenL3 512-d ─┐
              ├── mean-center → L2-norm → weight (0.5) → concatenate → L2-norm → 896-d
SBERT 384-d  ─┘
```

The fused embedding (896-d) outperforms either modality alone:

| Method | Echo Nest Distance | vs Random |
|--------|-------------------|-----------|
| SBERT alone | 3.33 | -12.4% |
| OpenL3 alone | 2.87 | -24.3% |
| Fused (SBERT + OpenL3) | 3.22 | -15.1% |

Note: The fused result is between the two single-view results because it's a weighted average. The real benefit of fusion is **breadth** — it satisfies both acoustic and semantic relevance simultaneously, which neither single view can do alone.

---

## Comparison with CLAP

Both CLAP and OpenL3 process audio, but they capture different information:

| Property | CLAP | OpenL3 |
|----------|------|--------|
| Training signal | Text-audio pairs (supervised semantics) | Audio-visual correspondence (self-supervised) |
| What it captures | Mood, genre feel, instrument presence | Timbre, rhythm, texture, production quality |
| Genre accuracy | 0.537 | 0.553 |
| Echo Nest distance | Not measured directly | 2.87 (best) |
| Cross-modal queries | Yes (text-to-audio) | No (audio-to-audio only) |
| Embedding norms | Near unit-length | Variable (requires normalization) |

**When OpenL3 > CLAP:** Finding tracks with similar production quality, tempo, instrumentation texture — regardless of "what genre it feels like."

**When CLAP > OpenL3:** Finding tracks that match a mood or concept described in words, or tracks in the same "semantic neighborhood" (e.g., all chill lo-fi tracks regardless of acoustic details).

---

## Visualizations

Generated by `scripts/openl3_vs_sbert_overlap.py`:

| Output | Description |
|--------|-------------|
| `openl3_sbert_rank_corr.png` | Histogram of per-track Spearman rho + Overlap@k bar chart + per-genre box plot |
| `openl3_sbert_tsne_comparison.png` | Side-by-side t-SNE colored by genre (OpenL3 vs SBERT) |
| `openl3_sbert_genre_heatmap.png` | Genre confusion matrices for both views + difference heatmap |
| `openl3_sbert_overlap_results.json` | Numerical results (overlap, correlation, percentages) |

---

## References

- Cramer et al., "Look, Listen, and Learn More: Design Choices for Deep Audio Embeddings," ICASSP 2019.
- Code: [marl/openl3](https://github.com/marl/openl3)
- Training data: AudioSet (Gemmeke et al., 2017)
