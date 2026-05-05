# Evaluation Results — Technical Details

Detailed evaluation protocol, data validation, and genre consistency benchmark results on the 2,000-track canonical subset.

(See `../reports/EXPERIMENT_RESULTS.md` for comprehensive multi-metric summary and interpretation.)

---

## 1. Goal

Establish baseline retrieval quality metrics across three embedding views (CLAP, OpenL3, SBERT) using three complementary evaluation strategies:
1. Genre Retrieval Accuracy — coarse-grained genre consistency
2. Echo Nest Feature Distance — acoustic feature alignment
3. Cross-View Overlap — view complementarity

---

## 2. Data & Embeddings

### Metadata

**Source:** `our_2000_tracks.csv` (2,000 tracks × 8 genres)

**Fields validated:**
- `track_id` — Unique identifier
- `genre_top` — Primary genre label (8 classes: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock)
- All rows contain non-null genre

### Embeddings

| Model | Shape | Source File | Format |
|-------|-------|-------------|--------|
| CLAP (audio) | (2000, 512) | `CLAP/clap_audio_embeddings.npy` | float32 |
| CLAP (text) | (2000, 512) | `CLAP/clap_text_embeddings_new.npy` | float32 |
| OpenL3 | (2000, 512) | `OpenL3/openl3_embeddings.npy` | float32 |
| OpenL3 IDs | (2000,) | `OpenL3/openl3_track_ids.npy` | int64 |
| SBERT | (2000, 384) | `SBERT/sbert_lyrics_embeddings.npy` | float32 |
| SBERT IDs | (2000,) | `SBERT/sbert_lyrics_faiss.ids.npy` | int64 |

---

## 3. Data Validation Summary

✓ **Metadata integrity:**
- 2,000 unique track IDs
- 8 genres (balanced: each genre has 250 tracks)
- No null `genre_top` values

✓ **Embedding integrity:**
- All shapes match expected dimensions
- All values finite (no NaN, Inf, or missing values)
- ID alignment verified: embedding IDs match metadata track IDs
- L2-norms computed for all embeddings (ready for cosine similarity)

✓ **Format validation:**
- OpenL3 embeddings aligned row-by-row with canonical 2,000-track list
- SBERT embeddings aligned with metadata CSV order
- CLAP embeddings accessible and convertible to (2000, 512) ndarray

---

## 4. Genre Retrieval Accuracy Protocol

### Script
`evaluation/evaluate_genre_retrieval.py`

### Procedure

```
1. Load metadata: track_id -> genre_top mapping
2. L2-normalize embeddings: emb /= ||emb||_2 per row
3. Compute similarity: sim_matrix = emb @ emb.T  [cosine similarity]
4. Mask diagonal: sim_matrix[i, i] = -inf  [exclude self-match]
5. For each query track i:
     - Find neighbor j with max(sim_matrix[i, :])
     - Check: does genre[j] == genre[i]?
6. Accuracy = (# correct) / (# queries)
```

### Settings

- **Sampled evaluation:** `--num-samples 500 --seed 42` (random query selection)
- **Full evaluation:** `--num-samples -1 --seed 42` (all 2,000 queries)

---

## 5. Genre Accuracy Results

### Sampled (500 queries)

| Model | Accuracy | Correct / Total |
|-------|----------|-----------------|
| CLAP (audio) | 0.5020 | 251/500 |
| CLAP (text) | 0.6760 | 338/500 |
| OpenL3 | 0.5260 | 263/500 |
| SBERT | 0.6020 | 301/500 |

### Full (2,000 queries)

| Model | Accuracy | Correct / Total |
|-------|----------|-----------------|
| CLAP (audio) | 0.5365 | 1073/2000 |
| CLAP (text) | 0.6770 | 1354/2000 |
| OpenL3 | 0.5525 | 1105/2000 |
| SBERT | 0.5755 | 1151/2000 |

### Interpretation

- **CLAP (text)** achieves highest accuracy (67.7%) — semantic text descriptors align with genre
- **SBERT** achieves 57.6% — metadata + lyrics moderately genre-indicative (but genre was intentionally removed from input to prevent leakage)
- **OpenL3** achieves 55.3% — acoustic texture partially captures genre structure
- **CLAP (audio)** achieves 53.7% — high-level mood is less genre-specific than low-level acoustics might suggest

**Caveat:** Genre is a coarse label; two acoustically identical tracks may differ in genre due to artist style, cultural context, or annotation inconsistency. This metric rewards *genre clustering* but penalizes *nuanced similarity*.

---

## 6. Echo Nest Feature Distance (Supplementary)

**Sample size:** 294 tracks (intersection of embeddings + Echo Nest features)

**Features (z-score standardized):**
- Danceability, Energy, Valence, Tempo, Acousticness, Instrumentalness, Liveness, Speechiness

**Finding:** Top-5 neighbors in OpenL3 space are **24.3% closer** in Echo Nest feature space than random pairs (statistically significant, p < 1e-20).

See `../reports/EXPERIMENT_RESULTS.md` for full Echo Nest evaluation and multi-view comparison.

---

## 7. Cross-View Overlap (Supplementary)

**Sample:** Full 2,000 tracks

**Finding:** SBERT and OpenL3 top-20 neighbors share only **5.6% overlap** (Jaccard), with Spearman rank correlation ρ = -0.77 (negatively correlated).

**Interpretation:** Semantic (SBERT) and acoustic (OpenL3) views are highly complementary. Fusion can combine orthogonal information.

See `../reports/EXPERIMENT_RESULTS.md` for detailed cross-view analysis.

---

## 8. Leakage Prevention

### Critical Design Constraint

**Genre labels never appear in model inputs.** They are used *only* as evaluation ground truth.

### Identified & Blocked Vectors

1. **Direct genre in SBERT metadata string**
   - ❌ Initial implementation included `genre_top` field in text input
   - ✓ Fixed: Removed from `src/metadata_builder.py`

2. **Genre words in FMA tags**
   - ❌ 48.8% of non-empty tags contain words like "rock", "electronic"
   - ✓ Fixed: Filtered via `strip_genre_from_tags()` function

3. **Artist name as implicit genre proxy** ⚠️
   - Retained in SBERT input (legitimate for music retrieval)
   - Acknowledged as *soft leakage* but necessary for quality recommendations

### Data Integrity Checks

- ✓ All NaN/Inf values flagged and handled
- ✓ ID alignment verified across all embeddings and metadata
- ✓ Track IDs match canonical 2,000-track list

---

## 9. Reproducibility

To reproduce all results:

```bash
cd evaluation/

# Genre accuracy (all 2,000 tracks)
python evaluate_genre_retrieval.py --root . --num-samples -1 --seed 42

# Echo Nest evaluation
# (Requires Echo Nest CSV in ../data/processed/)
python analyze_echonest.py  # [if script exists]

# Cross-view overlap
cd ../scripts/
python openl3_vs_sbert_overlap.py
```

---

## 10. Known Limitations

- **Genre is coarse:** 8 classes vs. rich musical diversity
- **Echo Nest subset:** 294 tracks ≠ full 2,000 (sample bias)
- **Text input variance:** SBERT lyrics column is empty in some versions
- **No listener validation:** Metrics don't reflect human preference (no user study)
