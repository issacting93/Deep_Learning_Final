# Role 2: Analysis Report — SBERT Semantic Search

This report documents the implementation and evaluation of a text-based music retrieval pipeline for the 2,000-track FMA subset.

## Definitions

- **Sentence-BERT (SBERT)**: A modification of the BERT architecture that uses siamese and triplet networks to produce fixed-size sentence embeddings optimised for semantic similarity. We use the `all-MiniLM-L6-v2` variant (6-layer, 384-dimensional output), which was pre-trained on over 1 billion sentence pairs from Natural Language Inference (NLI) and paraphrase datasets.

- **FAISS (Facebook AI Similarity Search)**: A library for efficient approximate nearest-neighbor search over dense vectors. We use `IndexFlatIP` (exact inner product), which is equivalent to cosine similarity when vectors are L2-normalised.

- **OpenL3**: An audio embedding model trained via self-supervised audio-visual correspondence on AudioSet. It produces 512-dimensional vectors that capture low-level acoustic properties (timbre, texture, rhythm) without any semantic label supervision.

- **Late Fusion**: A multimodal combination strategy where each modality is embedded independently, then the resulting vectors are concatenated (optionally weighted) into a single representation. This contrasts with early fusion (raw features combined before encoding) or cross-modal fusion (joint attention/training across modalities).

- **Cosine Similarity**: The cosine of the angle between two vectors, ranging from -1 to 1. For L2-normalised vectors, cosine similarity equals the dot product, which allows FAISS inner-product search to compute cosine distances.

- **Data Leakage**: When information from the evaluation target (e.g., genre labels) is present in the model input, artificially inflating performance metrics. Identified and addressed in Section 5.

---

## 1. Visualization of Latent Space

The t-SNE visualization (perplexity=30, 1,000 iterations) of the 384-dimensional SBERT embeddings shows that metadata strings for different genres form visually separable clusters. Note that t-SNE is a stochastic algorithm sensitive to hyperparameters — cluster separation can be exaggerated. We use it here for qualitative exploration, not as evidence of model quality.

![t-SNE Clusters](plots/tsne_sbert_embeddings.png)

## 2. PCA Variance Analysis

PCA decomposition of the SBERT embedding space reveals a highly distributed representation:

| Components | Cumulative Variance |
|---|---|
| 1 | 4.1% |
| 10 | 18.6% |
| 51 | 50% |
| 112 | 75% |
| 181 | 90% |
| 223 | 95% |

The top principal component captures only 4.1% of variance, and 181 components are needed to reach 90%. This indicates the embeddings use most of their 384 dimensions meaningfully — information is spread across the space rather than concentrated in a few axes, which is desirable for retrieval (more dimensions contribute to distinguishing tracks).

![PCA Variance](plots/pca_variance_sbert.png)

## 3. Representation Analysis

We audited the SBERT model for semantic robustness and lexical bias to understand its failure modes.

| Test | Method | Result | Finding |
|---|---|---|---|
| **Semantic Robustness** | Overlap@10 between queries "lonely" and "isolated" | **10%** | The model distinguishes these refined semantic nuances — only 1 in 10 top results overlaps, despite the words being near-synonyms. |
| **Lexical Bias** | Query "Blue music" | **Title-biased** | Keyword matches in titles (e.g., "Gondola Blue") outrank semantic genre matches (Blues). The model encodes surface-level lexical overlap, not musical understanding. |
| **Truncation Impact** | Cosine shift between first 128 words vs. full text | **Minimal** | FMA metadata strings are short (typically <50 words), so truncation at the model's 256-token limit rarely activates. |

![Semantic Robustness](plots/semantic_robustness_bar.png)
![Lexical Bias Analysis](plots/lexical_bias_analysis.png)

## 4. Cross-Modal Comparison (SBERT vs OpenL3)

We compared the text-based SBERT and audio-based OpenL3 embedding spaces to understand whether they capture the same or complementary information.

For 200 randomly sampled query tracks, we retrieved the top-20 nearest neighbors from each index and measured agreement:

| Metric | Value |
|---|---|
| **Overlap@20** | **5.6%** (1.1 of 20 neighbors shared) |
| **Spearman rank correlation** | **-0.77** (mean over 200 queries) |

The two modalities retrieve almost entirely different neighbors. The negative rank correlation means that tracks SBERT considers similar, OpenL3 considers dissimilar, and vice versa. This is expected: SBERT embeds textual descriptions (artist, title, tags, lyrics), while OpenL3 embeds acoustic properties (timbre, rhythm, texture). This strong complementarity is the motivation for late fusion — each modality contributes information the other lacks.

## 5. Data Leakage Identification and Fix

### Problem

The original SBERT input strings included the `genre_top` field directly:

```
"{artist} - {title}. Tags: {genre}, {tags}. Mood: various."
```

When evaluation uses genre as ground truth (e.g., "do the top-K results share the query's genre?"), this creates **data leakage**: the model can match on the genre label itself rather than learning anything about the music.

Further investigation revealed that the `tags` field in FMA also contains genre-like labels (e.g., `['folk']`, `['pop', 'melodic']`) in **48.8%** of tracks with non-empty tags (163 out of 334). This is a second source of leakage.

### Fix

1. **Removed `genre_top`** from the input string entirely
2. **Filtered genre words** (`electronic`, `experimental`, `folk`, `hip-hop`, `instrumental`, `international`, `pop`, `rock`) from the `tags` field
3. **Added lyrics** from the Genius API (first 1,000 characters per track) as a semantic signal to replace the removed genre information

The cleaned input format is:

```
"{title} by {artist}. Tags: {filtered_tags}. Lyrics: {lyrics}"
```

Genre is retained in the DataFrame for evaluation, but is never seen by the embedding model.

## 6. Multimodal Fusion (SBERT + OpenL3)

Given the low overlap between modalities (Section 4), we implemented late fusion to combine text and audio signals.

**Method:**
1. L2-normalise each modality independently (SBERT: 384-d, OpenL3: 512-d)
2. Apply weights (default: 0.5 text / 0.5 audio) and concatenate → 896-d vector
3. L2-normalise the fused vector for cosine search via FAISS

**Limitation:** For text-to-track queries (e.g., "songs about freedom"), only the SBERT portion of the fused vector is populated — the audio portion is zero-padded. This means text queries against the fused index can only match on the text half of each stored vector, diluting the signal. The fused index is therefore most useful for **track-to-track retrieval**, where both modalities are available for the query.

## 7. Echo Nest Evaluation (Independent Ground Truth)

### Why Echo Nest

Standard genre-based evaluation (P@K, NDCG) measures whether retrieved tracks share the query track's genre. After removing genre from the input, we need an independent ground truth. Echo Nest provides 8 continuous audio features — danceability, energy, valence, tempo, acousticness, instrumentalness, liveness, speechiness — derived from audio analysis. No embedding model in this project saw these features during training.

### Method

For each of the 294 tracks present in both our subset and Echo Nest, we:
1. **Standardized** all 8 features to zero mean and unit variance (z-scores) to prevent tempo (60-200 BPM) from dominating over 0-1 scale features
2. Retrieved the **top-5 nearest neighbors** from each embedding index
3. Measured the **Euclidean distance** in standardized Echo Nest space between the query and its neighbors
4. Computed a **paired t-test** against the random baseline

### Subset representativeness

The 294 Echo Nest tracks are not uniformly distributed across genres:

| Genre | Echo Nest subset | Full dataset (N=2000) |
|---|---|---|
| Folk | 22.1% | 12.5% |
| Hip-Hop | 21.1% | 12.5% |
| Rock | 16.0% | 12.5% |
| Pop | 15.3% | 12.5% |
| Electronic | 11.6% | 12.5% |
| International | 7.5% | 12.5% |
| Instrumental | 5.1% | 12.5% |
| Experimental | 1.4% | 12.5% |

Folk and Hip-Hop are over-represented; Experimental and Instrumental are under-represented. Results may be biased toward genres with more Echo Nest coverage.

### Results

| Method | Avg Distance (z-score) | vs Random | Paired t-test |
|---|---|---|---|
| Random Baseline | 3.80 | — | — |
| SBERT (Text+Lyrics) | 3.33 | -12.4% | p = 1.6e-06 |
| Fused (SBERT+OpenL3) | 3.22 | -15.1% | p = 1.4e-08 |
| **OpenL3 (Audio)** | **2.87** | **-24.3%** | **p = 1.8e-21** |

![Echo Nest Evaluation](plots/echonest_evaluation.png)

### Interpretation

- **OpenL3 performs best** on Echo Nest features (p < 10^-21). This makes sense: OpenL3 embeds audio directly, and Echo Nest features are derived from audio analysis. Both operate in the acoustic domain.
- **Fused beats SBERT alone**, confirming that the audio signal from OpenL3 adds complementary information even when concatenated with text embeddings.
- **SBERT still significantly beats random** (p < 10^-6), showing that text metadata and lyrics do carry some information about acoustic properties — an artist's name or lyrical content correlates with musical style.
- All differences vs. random are statistically significant at p < 0.001.

---

## File Manifest

| File | Location |
|---|---|
| SBERT Embeddings (lyrics-enriched, genre-free) | `data/processed/lyrics_enriched/sbert_lyrics_embeddings.npy` |
| Fused Embeddings (SBERT+OpenL3) | `data/processed/lyrics_enriched/fused_embeddings.npy` |
| FAISS Index (text-only) | `data/processed/lyrics_enriched/sbert_lyrics_faiss.index` |
| FAISS Index (fused) | `data/processed/lyrics_enriched/fused_faiss.index` |
| Lyrics DataFrame | `data/processed/lyrics_enriched/lyrics_df.csv` |
| Lyrics Pipeline Script | `text_to_text_SBERT_FMA_GENIUS_2.py` |
| Metadata Builder | `src/metadata_builder.py` |
| SBERT Embedding Generator | `src/embeddings/sbert.py` |
| Analysis Script | `scripts/analyze_sbert_robustness.py` |

---
**Prepared by Sid & Issac**
