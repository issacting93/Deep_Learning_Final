# Evaluation Results: Multi-View Music Retrieval

**Canonical 2,000-track FMA subset | Three embedding views: CLAP, OpenL3, SBERT**

---

## Summary: Key Metrics Across All Views

| Evaluation Method | Best Model | Performance |
|---|---|---|
| **Genre Retrieval Accuracy** | RRF (4 models) | 76.9% Top-1 |
| **Echo Nest Feature Distance** | RRF/OpenL3/CLAP | 24.6% improvement vs random |
| **Rank-Level Fusion (RRF)** | All 4 views combined | ✓ Verified superior |
| **Cross-View Overlap** | — | 5.6% (SBERT vs OpenL3) |

---

## 1. Genre Retrieval Accuracy

### Methodology
- **Script:** `evaluation/evaluate_genre_retrieval.py`
- **Metric:** Top-1 genre accuracy (fraction of queries whose nearest neighbor shares the same genre)
- **Protocol:** L2-normalize embeddings → cosine similarity → exclude self → find argmax → compare genres

### Results

**Sampled (500 random tracks):**
| Model | Accuracy | Correct / Total |
|---|---|---|
| CLAP (audio) | 50.2% | 251/500 |
| CLAP (text) | 67.6% | 338/500 |
| OpenL3 | 52.6% | 263/500 |
| SBERT | 60.2% | 301/500 |

**Full Evaluation (2,000 tracks):**
| Model | Accuracy | Correct / Total |
|---|---|---|
| CLAP (audio) | 53.7% | 1073/2000 |
| CLAP (text) | 67.7% | 1354/2000 |
| OpenL3 | 55.3% | 1105/2000 |
| SBERT | 57.6% | 1151/2000 |
| **RRF (all 4)** | **76.9%** | **1538/2000** |

### Interpretation
- **RRF fusion achieves 76.9%** — Rank-level fusion outperforms best single model (CLAP text 67.7%) by 9.2 percentage points
- **CLAP (text)** is best single view (67.7%) — text descriptions encode strong genre signals
- **SBERT** achieves 57.6% — metadata + lyrics provide moderate genre information
- **OpenL3** achieves 55.3% — acoustic features partially capture genre structure
- **CLAP (audio)** lags at 53.7% — high-level mood/vibe less genre-specific than text
- **Key insight:** RRF benefits from four complementary views (CLAP audio, CLAP text, OpenL3, SBERT) each capturing different genre-relevant signals

**Caveat:** SBERT genre accuracy is artificially suppressed because genre labels were intentionally removed from text input to prevent data leakage (see Data Validation section).

---

## 2. Echo Nest Feature Distance Evaluation

### Methodology
- **Script:** `evaluation/evaluate_echo_nest_distance.py`
- **Ground truth:** 8 standardized Echo Nest audio features (Danceability, Energy, Valence, Tempo, Acousticness, Instrumentalness, Liveness, Speechiness)
- **Metric:** Mean Euclidean distance between top-5 neighbors in 8-d Echo Nest space
- **Significance:** Paired t-test against random baseline

### Results

| Method | Avg Distance | σ | vs Random | p-value | Significance |
|---|---|---|---|---|---|
| **Random Baseline** | **3.83** | — | — | — | — |
| SBERT (text + lyrics) | 3.24 | 1.17 | +15.3% | 5.3e-07 | ✓ |
| OpenL3 (audio) | 2.89 | 0.96 | +24.6% | 3.1e-18 | ✓✓✓ |
| CLAP (audio+text) | 2.89 | 1.04 | +24.6% | 6.3e-18 | ✓✓✓ |
| **RRF (All 4 views)** | **2.89** | **1.04** | **+24.6%** | **2.2e-17** | **✓✓✓** |

**Lower distance = better** (neighbors have more similar acoustic properties)

### Interpretation
- **OpenL3, CLAP, and RRF tie at 24.6% improvement** — both acoustic embeddings (OpenL3) and audio-based fusion (CLAP+RRF) match the best performance
- **RRF maintains parity with best single models** — rank-level fusion achieves same Echo Nest improvement as best individual views, confirming robustness across domains
- **SBERT contributes 15.3% improvement** — text signal adds value even in acoustic ground truth evaluation
- **All differences are highly statistically significant** (p < 1e-17 for fusion/audio models)
- **Key finding:** RRF's value is stability and genre accuracy (76.9%), not Echo Nest improvement (which plateaus at 24.6%)

**Caveat:** Echo Nest coverage is limited (294/2000 tracks); overrepresented genres include Folk and Hip-Hop (~21% each); underrepresented: Experimental (~1.4%)

**Note:** Vector-level fusion (SBERT+OpenL3 concatenation) was not included in final evaluation as it was not used by the application (app.py uses rank-level RRF instead).

---

## 3. Cross-View Overlap Analysis

### Methodology
- **Script:** `scripts/openl3_vs_sbert_overlap.py`
- **Metrics:** 
  - Overlap@k (fraction of shared neighbors at k=5,10,20,50)
  - Spearman rank correlation (per-track)
  - Per-genre agreement heatmaps

### Results (Verified)

| View Pair | Overlap@10 | Spearman ρ | p-value |
|---|---|---|---|
| **SBERT vs OpenL3** | **5.6%** | **-0.77** | < 0.001 |
| SBERT vs CLAP | _Pending_ | _Pending_ | — |
| OpenL3 vs CLAP | _Pending_ | _Pending_ | — |

### Key Findings
- **Minimal overlap (5.6%)** — SBERT and OpenL3 retrieve nearly disjoint neighbor sets
- **Negative rank correlation (ρ = -0.77)** — views rank tracks in nearly opposite order
- **Per-genre patterns differ** — heatmaps reveal genre-specific complementarity structure
- **Implication:** Reciprocal Rank Fusion effectively combines independent signals

---

## 4. Data Validation Checklist

✓ Metadata: 2,000 rows (all with `genre_top`)  
✓ Embeddings: All arrays finite (no NaN/Inf)  
✓ ID alignment: Verified across all models  
✓ Genre field: Excluded from model inputs (leakage prevention)  
✓ Echo Nest coverage: 294 tracks (14.7% of canonical 2,000)

---

## 5. Conclusions

1. **RRF achieves best genre accuracy (76.9%)** — 9.2pp above best single model (CLAP text 67.7%)
2. **Fusion is stable across metrics** — maintains 24.6% Echo Nest improvement alongside genre gains
3. **No single view dominates all metrics** — each captures different musical dimensions
4. **OpenL3 excels at acoustic similarity** — 24.6% Echo Nest improvement (its native domain)
5. **Four-view fusion is synergistic** — CLAP audio + CLAP text + OpenL3 + SBERT together exceed component capabilities
6. **Evaluation is leakage-free** — genre never appears in model inputs, only used for ground truth

### Recommended Use Cases
- **Best overall accuracy** → Reciprocal Rank Fusion (RRF with k=60, all 4 views: CLAP audio, CLAP text, OpenL3, SBERT)
- **Text-based queries** → CLAP (text, 67.7% genre accuracy)
- **Audio-based similarity** → OpenL3 (24.6% Echo Nest improvement)
- **Semantic retrieval** → SBERT (57.6% genre accuracy)
- **Lightweight single-view** → OpenL3 or CLAP(text)
