# Evaluation Methodology

## Overview

Evaluating music retrieval is challenging because there is no single ground truth for "similarity." A song can be similar to another in genre, mood, tempo, timbre, lyrical content, or cultural context. We use three complementary evaluation strategies:

1. **Genre Retrieval Accuracy** — does the nearest neighbor share the same genre?
2. **Echo Nest Feature Distance** — are neighbors similar in acoustic properties?
3. **Cross-View Overlap Analysis** — how much do different views agree?

---

## 1. Genre Retrieval Accuracy

**Script:** `evaluation/evaluate_genre_retrieval.py`

**Metric:** Top-1 genre accuracy — for each query track, retrieve its nearest neighbor (by cosine similarity in the embedding space), and check if they share the same `genre_top` label.

**Protocol:**
1. Load embedding matrix and corresponding track IDs.
2. L2-normalize all embeddings.
3. Compute full similarity matrix (embeddings @ embeddings.T).
4. Mask the diagonal (self-similarity).
5. For each query, find argmax (nearest neighbor).
6. Compare genre labels between query and nearest neighbor.
7. Report accuracy = correct / total.

**Sampling:** By default, 500 randomly sampled queries (configurable via `--num-samples`). Setting `--num-samples -1` evaluates all tracks.

**Results (canonical 2,000-track subset):**

| Model | Top-1 Genre Accuracy |
|-------|---------------------|
| CLAP (audio) | 0.52 |
| CLAP (text, metadata-derived) | 0.48 |
| OpenL3 | 0.41 |
| SBERT (lyrics-enriched) | 0.38 |

CLAP performs best because its contrastive training objective explicitly aligns audio with semantic descriptors that often correlate with genre. OpenL3 captures acoustic texture which partially correlates with genre. SBERT performs lowest because genre labels were intentionally removed from its input to prevent leakage.

**Limitations:**
- Genre is a coarse label (8 classes). Two tracks can be highly similar yet differ in genre.
- Accuracy depends on genre distribution balance. Over-represented genres inflate scores.
- This metric rewards genre clustering, not nuanced musical similarity.

---

## 2. Echo Nest Feature Distance

**Concept:** Evaluate retrieval quality using audio features the models never saw during training.

**Features (8 dimensions):**
- Danceability, Energy, Valence, Tempo
- Acousticness, Instrumentalness, Liveness, Speechiness

All features are z-score standardized before computing Euclidean distance.

**Protocol:**
1. For each model, retrieve the top-5 nearest neighbors per query track.
2. Compute average Euclidean distance in Echo Nest feature space between query and its neighbors.
3. Compare against a random baseline (average distance between random track pairs).
4. Report statistical significance via paired t-test.

**Why this works:** If a model's nearest neighbors are also close in Echo Nest features (which the model never saw), the model has learned meaningful musical structure rather than superficial patterns.

**Sample size:** 294 tracks have both embeddings and Echo Nest features (FMA's Echo Nest subset).

---

## 3. Cross-View Overlap

**Script:** `scripts/openl3_vs_sbert_overlap.py`

**Metric:** For each track, retrieve top-20 neighbors in View A and top-20 in View B. Compute:
- **Jaccard overlap:** |A intersect B| / |A union B|
- **Spearman rank correlation:** between the two ranked lists

**Finding:** SBERT and OpenL3 share only 5.6% overlap (Spearman rho = -0.77). This confirms the views are complementary — fusion can combine genuinely independent information.

---

## Data Leakage Prevention

A critical design constraint: **genre labels are never part of model input.** They are used only as evaluation ground truth.

Potential leakage vectors we identified and blocked:
1. **Direct genre in metadata string:** The `genre_top` field was initially included in SBERT input. Removed in `src/metadata_builder.py`.
2. **Genre-like tags:** 48.8% of non-empty `tags` fields contain genre words ("rock", "electronic"). Filtered via `strip_genre_from_tags()` in both `src/metadata_builder.py` and `scripts/generate_fused_embeddings.py`.
3. **Artist name as genre proxy:** Artist names implicitly correlate with genre. We keep them as input (they're legitimately useful for retrieval) but acknowledge this as a soft leakage vector in our analysis.

---

## Reproducing Results

```bash
# Full evaluation on all tracks
cd evaluation/
python evaluate_genre_retrieval.py --root . --num-samples -1 --seed 42

# Quick test (500 samples)
python evaluate_genre_retrieval.py --root . --num-samples 500 --seed 42
```

The script expects the following files in its `--root` directory:
- `our_2000_tracks.csv` — metadata with track IDs and `genre_top`
- `CLAP/clap_audio_embeddings.npy` — CLAP audio embeddings (dict format)
- `CLAP/clap_text_embeddings_new.npy` — CLAP text embeddings (ordered)
- `OpenL3/openl3_embeddings.npy` + `openl3_track_ids.npy`
- `SBERT/sbert_lyrics_embeddings.npy` + `sbert_lyrics_faiss.ids.npy`

---

## Metrics Glossary

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| Top-1 Accuracy | correct / total | 0-1 | Fraction of queries whose nearest neighbor shares the same genre |
| Echo Nest Distance | mean L2 distance in 8-d standardized feature space | 0+ | Lower = more acoustically similar neighbors |
| Jaccard Overlap | \|A ∩ B\| / \|A ∪ B\| | 0-1 | Fraction of shared neighbors between two views |
| Spearman Rho | Rank correlation between two result lists | -1 to 1 | How much two views agree on ranking |
| RRF Score | sum(1 / (k + rank)) across views | 0+ | Fused relevance score (higher = ranked highly in more views) |
