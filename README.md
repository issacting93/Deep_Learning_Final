# Multi-Faceted Music Retrieval

**A Multi-View Retrieval System** | Deep Learning Final Project | April 2026

Wenny, Sid, Issac, MJ, Jiayi, Yunchu (Helena)

Combining audio, text, and joint embedding into a single recommendation system that outperforms any single view.

---

## Introduction

Music retrieval systems typically rely on a single representation of music — either audio features or text metadata. This limits their ability to satisfy diverse user needs: a query like "chill lo-fi vibes" requires understanding musical mood (acoustic), while "songs by folk artists about travel" requires understanding metadata (textual), and "tracks similar to this one I like" requires structural knowledge (graph). No single embedding captures all of these.

We set out to answer: **can fusing multiple independent embedding spaces produce better music recommendations than any single view alone?** The answer is yes — our fused system retrieves neighbors that are 24% closer in acoustic features than random retrieval and 9 percentage points more genre-accurate than any single view, while maintaining semantic relevance that audio-only models miss.

This project builds a **multi-view retrieval system** over the [Free Music Archive (FMA)](https://github.com/mdeff/fma) dataset, an open-licensed benchmark of 8,000 tracks across 8 top-level genres. We use three different embedding models to search for similar music — each one captures a different dimension of similarity: sound, meaning, and vibe. Each view produces an independent embedding space, and a fusion layer combines them to give a combined score for recommendation.

## System Architecture

```text
                    ┌─────────────────────────────────────────────────────────┐
                    │                    Input: Audio Track                   │
                    └────────────┬──────────────────┬──────────────┬──────────┘
                                 │                  │              │
                                 ▼                  ▼              ▼
                    ┌────────────────┐  ┌────────────────┐  ┌──────────────┐
                    │   CLAP Audio   │  │    OpenL3      │  │   Metadata   │
                    │  (HTSAT-tiny)  │  │  (Audio CNN)   │  │  + Lyrics    │
                    └───────┬────────┘  └───────┬────────┘  └──────┬───────┘
                            │                   │                  │
                            ▼                   ▼                  ▼
                    ┌────────────────┐  ┌────────────────┐  ┌──────────────┐
                    │  512-d embed   │  │  512-d embed   │  │ 384-d embed  │
                    │ (mood, genre)  │  │ (timbre, rhythm)│  │ (semantic)  │
                    └───────┬────────┘  └───────┬────────┘  └──────┬───────┘
                            │                   │                  │
                            ▼                   ▼                  ▼
                    ┌────────────────────────────────────────────────────────┐
                    │              FAISS Nearest-Neighbor Search             │
                    │            (IndexFlatIP — cosine similarity)           │
                    └───────┬───────────────────┬──────────────────┬─────────┘
                            │                   │                  │
                            ▼                   ▼                  ▼
                    ┌────────────────────────────────────────────────────────┐
                    │           Reciprocal Rank Fusion (k=60)               │
                    │         score = Σ  1 / (60 + rank_per_view)           │
                    └───────────────────────┬────────────────────────────────┘
                                            │
                                            ▼
                                ┌──────────────────────┐
                                │   Fused Top-K Recs   │
                                └──────────────────────┘
```

The system takes an audio track as input and processes it through three independent pipelines:

| View | Model | Dimensions | Input | What It Captures |
|---|---|---|---|---|
| 1. Vibe/Text Search | CLAP (HTSAT-tiny) | 512 | Audio waveform | High-level semantics: mood, genre feel, instrument presence |
| 2. Lyrics/Semantic Search | SBERT (all-MiniLM-L6-v2) | 384 | Metadata + lyrics | Textual semantics: artist identity, lyrical content, descriptive tags |
| 3. Acoustic Similarity | OpenL3 | 512 | Audio waveform | Low-level acoustics: timbre, rhythm, texture |

All embedding generators subclass `src/embeddings/base.py:EmbeddingGenerator` with a shared `generate()` / `load_embeddings()` interface. All FAISS indices use the same `src/indexing/faiss_index.py` wrapper. This ensures any view can be swapped in or out of the fusion layer without code changes.

### Why Multiple Views?

SBERT and OpenL3 share only **3.5% Jaccard overlap** in their top-20 neighbors (Spearman ρ = +0.07). This means the two modalities retrieve almost entirely different tracks for the same query — they are complementary, not redundant. Fusion combines these independent signals to improve retrieval quality.

## Demo & Web Application

Built with Flask + Vanilla JS. The web application provides three interactive components:

1. **Browse Grid** — Search and filter the 2,000-track dataset by title/artist/genre with color-coded genre badges
2. **Per-View Rankings** — See top-10 recommendations from each embedding separately (CLAP, SBERT, OpenL3), with raw cosine similarity scores
3. **Fused Leaderboard** — Combined ranking via Reciprocal Rank Fusion (RRF) with a similarity radar chart and per-model score visualization

### Launch

```bash
python app.py  # Visit http://localhost:5001
```

Precomputed embeddings are included in `data/processed/`, so the demo works immediately without regenerating embeddings.

### Interactive Walkthrough

1. **Find a Track** — Use the search bar or genre filter buttons (250 tracks per genre)
2. **Click to Set Seed** — Triggers embedding lookup and RRF calculation across all three views
3. **Compare Views** — Toggle between CLAP/SBERT/OpenL3 tabs to see how each model ranks neighbors differently
4. **Investigate Matches** — The radar chart shows alignment between the seed track and each model; dynamic bars indicate relative contribution of each view to the fused rank

See also:
- `notebooks/02_clap_retrieval_demo.ipynb` — Text-to-music search examples
- `notebooks/08_semantic_search_demo.ipynb` — SBERT semantic query examples

## Dataset

We use the [Free Music Archive (FMA)](https://github.com/mdeff/fma) — specifically the `fma_small` subset (8,000 tracks, ~30s clips, 8 genres) and its metadata (`tracks.csv`, `genres.csv`, `echonest.csv`).

**To download:**

```bash
python scripts/download_fma.py
```

This places audio files in `data/fma_small/` (organized as `000/000002.mp3`). Total download: ~7.2 GB.

**Pre-filtered subset:** A canonical 2,000-track metadata subset (`data/fma_2000_metadata/`) is already included in the repo — all pipelines use this canonical subset by default, with 250 tracks per genre for balanced evaluation. The track IDs are stored in `data/processed/openl3_track_ids.npy`.

**Lyrics:** Fetched at runtime from the [Genius API](https://genius.com/developers). Requires a `GENIUS_API_KEY` in your `.env` file. Cached lyrics are stored in `data/processed/lyrics_enriched/lyrics_df.csv` so the API only needs to be called once. Lyrics are truncated to the first 1,000 characters per track.

**Echo Nest / Spotify audio features:** Used as independent ground truth for evaluation. Includes danceability, energy, valence, tempo, acousticness, instrumentalness, liveness, and speechiness. 294 tracks in the FMA subset have Echo Nest features available.

---

## Section 01: CLAP

**Vibe & text-to-music search — a 512-d shared space for audio and text via contrastive pretraining.**

### How CLAP Works

CLAP (Contrastive Language-Audio Pretraining) learns acoustic concepts from natural language supervision. It uses two encoders — an audio encoder (HTSAT-tiny) and a text encoder — trained with contrastive loss to map audio and text descriptions into a joint multimodal space. Matched audio/text pairs are pulled together, while unmatched pairs are pushed apart using InfoNCE loss.

This means you can search for music using natural language: the text query and audio tracks live in the same 512-d space, so cosine similarity directly measures how well a track matches a description.

### Pipeline

```
Audio file (MP3) → Resample to 48 kHz → Clip to 10s → HTSAT-tiny → 512-d vector → L2-normalize
```

We generated embeddings for 7,997 of 8,000 tracks (3 corrupt MP3s skipped). CLAP retrieves on-genre top results across all natural-language queries — even with no fine-tuning.

### CLAP Text-to-Music Retrieval Results

Natural-language queries against 2,000 audio embeddings:

| Query | Top Result | Genre | Cosine Sim |
|---|---|---|---|
| "sad piano ballad" | Blue Dot Sessions | Instrumental | 0.36 |
| "aggressive heavy metal" | Lately Kind of Yeah — 恐怖 (Terror) | Electronic | 0.42 |
| "upbeat happy pop song" | Project 5am — 5am, Wabi Sabi | Electronic | 0.28 |
| "acoustic guitar folk" | Blue Dot Sessions — Wistful | Instrumental | 0.40 |

The model correctly identifies "Instrumental" as the best fit for piano and acoustic guitar queries within this subset, though "Electronic" is surfaced for the heavy metal query, indicating a limitation in the subset's coverage of specific high-intensity genres. CLAP prioritizes acoustic texture over metadata labels.

### Embedding Space Analysis

**Genre structure** in CLAP embeddings (cosine similarity between genre centroids):
- Most similar: Pop and Rock (0.87), Folk and Pop (0.87)
- Most distinct: Experimental and Hip-Hop (0.56), Hip-Hop and Instrumental (0.58)
- PCA: 50 components capture ~85% of variance across 512 dimensions

**t-SNE clustering** — CLAP forms the tightest genre clusters among all three views:

![CLAP: t-SNE Clustering by Genre](data/processed/viz/tsne_clap.png)

![CLAP: PCA Clustering by Genre](data/processed/viz/pca_clap.png)

![CLAP: Genre Centroid Cosine Similarity](data/processed/viz/heatmap_clap.png)

---

## Section 02: SBERT

**Lyrics & semantic search — a siamese transformer mapping metadata + lyrics to 384-d embeddings.**

### The Core Idea

We build a text-based view of each song. Instead of listening, we read everything written about the song and turn that into a searchable vector. Sentence-BERT (2019) embeds each sentence independently into a fixed vector, and the `all-MiniLM-L6-v2` variant (2021) is a distilled model trained on a diverse range of multi-domain text data, producing compact 384-d embeddings.

### Two Sources of Text

SBERT draws from two complementary text sources:

1. **FMA Metadata** — Title, artist, and filtered tags from the Free Music Archive `tracks.csv`
2. **Genius API Lyrics** (optional, when available) — The largest lyrics database online with a free API. Artist name + track title are used to search the Genius endpoint, and lyrics are cleaned and appended to the metadata string

### Genius API Lyrics Cleaning

Raw lyrics from Genius require several cleaning steps before they are useful for embedding:

1. **Remove section headers** — `genius.remove_section_headers = True` strips markers like `[Verse]`, `[Chorus]`, `[Bridge]` that add noise without semantic value
2. **Drop the title line** — The first line of Genius lyrics typically repeats the song title; it is skipped via `lyrics.split('\n', 1)[-1]`
3. **Truncate to 1,000 characters** — `lyrics[:1000]` caps input length. SBERT (`all-MiniLM-L6-v2`) has a 256 word-piece token limit, and 1,000 characters comfortably fits within that while capturing the most distinctive opening verses
4. **Append to metadata** — Lyrics are concatenated as `"{metadata_string} Lyrics: {lyrics}"` only when available; tracks without lyrics fall back to metadata-only embeddings

### The Complete Pipeline

```
FMA tracks.csv → Build metadata string per track → Fetch lyrics via Genius API → Clean & append lyrics (if found)
→ SBERT (all-MiniLM-L6-v2) embeds each document → 384-dim vector per track
→ Store all 2000 vectors in FAISS index → User query → SBERT embed → FAISS search → top-K tracks → Ranked List
```

The metadata string is constructed as:
```python
def build_metadata_string(track):
    tags = strip_genre_from_tags(track['tags'])
    parts = [f"{track['title']} by {track['artist']}."]
    if tags:
        parts.append(f"Tags: {tags}.")
    return " ".join(parts)
```

### Data Leakage Fix

**Critical constraint:** Genre labels are never part of model input; used only for evaluation.

The original input strings included `genre_top` directly, which would inflate any genre-based evaluation. We also found that **48.8% of non-empty `tags` fields** contain genre-like labels (words like "rock", "electronic"). Both were removed — genre is used only as evaluation ground truth, never as model input. Tags are filtered via `strip_genre_from_tags()`.

Artist names are retained as a legitimate feature but acknowledged as soft leakage (certain artists map strongly to specific genres).

### Embedding Space Analysis

**PCA:** The SBERT embedding space is highly distributed — 163 components needed for 90% variance (vs. 50 for CLAP's 512-d space). This suggests SBERT utilises its dimensions more uniformly, spreading information across the full 384-d space.

**t-SNE clustering** — SBERT is more diffuse than CLAP (semantic similarity ≠ genre similarity):

![SBERT: t-SNE Clustering by Genre](data/processed/viz/tsne_sbert.png)

![SBERT: PCA Clustering by Genre](data/processed/viz/pca_sbert.png)

![SBERT: Genre Centroid Cosine Similarity](data/processed/viz/heatmap_sbert.png)

---

## Section 03: OpenL3

**Acoustic similarity — self-supervised audio embeddings capturing timbre, rhythm, and texture.**

### A Purely Acoustic Space — No Semantic Supervision

OpenL3 (`content_type='music'`, 512-d) is trained via self-supervised audio-visual correspondence on AudioSet — it captures low-level acoustic properties without any label supervision.

**Key difference from CLAP:**
- **CLAP** maps audio into a space shared with text — semantic
- **OpenL3** maps audio into a purely acoustic space — sounds-alike

This means two songs that sound similar are close in OpenL3 space — even across genres. The use case is "find songs that sound like this one."

**Stats:** 2,000 tracks indexed | 512 dimensions | <1ms per query

### Implementation Pipeline

```
librosa load (22,050 Hz) → openl3.get_audio_embedding() → mean pool frames → 512-d vector per track → FAISS IndexFlatIP
```

### The DC Offset Problem

**RAW:** Raw cosine similarities cluster artificially near 0.98 — a DC offset in the embedding space makes all tracks look similar.

**FIX:** Mean-center embeddings before L2 normalization — removes the offset and makes similarity scores meaningful.

**Result:** Similarity scores now spread meaningfully from 0.3–0.9 across track pairs.

### Checkpoint System

Colab sessions time out mid-run. Progress is saved to `.npy` every 100 tracks. On crash: resume from last checkpoint, skip `done_set`. No track is processed twice.

---

## Fusion

To combine independent embedding spaces, we employ two different mathematical strategies to prevent any single modality from unfairly dominating the results due to dimensionality or DC offsets.

### Rank-Level Late Fusion: Reciprocal Rank Fusion (RRF)

Used in the live Flask application (`app.py`). Rather than merging the vectors, the system queries each view (CLAP, SBERT, OpenL3) independently. The resulting tracks are scored using Reciprocal Rank Fusion:

```
score(d) = Σ 1 / (k + rank(result(q), d))
```

Where:
- `d` is a document in the result set
- `result(q)` is the ranked list of documents for query `q`
- `rank(result(q), d)` is the position of document `d` in that list
- `k` is a constant (=60) that smooths the influence of rank differences

Documents that appear high in multiple lists accumulate higher scores, while documents appearing in only one list receive lower scores. This approach **requires no training data** and is insensitive to the quality variations of individual input lists.

Mean-centering is also applied at runtime to ensure raw cosine similarity metrics display cleanly on the frontend.

### Vector-Level Early Fusion (Weighted Concatenation)

Used in `scripts/generate_fused_embeddings.py`. To create a single offline search index, the OpenL3 and SBERT vectors are first **mean-centered** to remove the native OpenL3 DC offset (which otherwise causes cosine similarities to artificially group near `0.98`). They are then L2-normalized, weighted (e.g., 50/50), concatenated into an 896-d vector, and L2-normalized again. 

---

## Evaluation

We use three complementary evaluation strategies to assess retrieval quality without genre leakage.

### 1. Genre Retrieval Accuracy

**Metric:** Top-1 genre accuracy — for each query track, retrieve its nearest neighbor (by cosine similarity) and check if they share the same `genre_top` label. Supervised ground truth over 2,000 tracks across 8 balanced genres.

**Protocol:**
1. L2-normalize all embeddings
2. Compute full similarity matrix (`emb @ emb.T`)
3. Mask the diagonal (self-similarity)
4. For each query, find the nearest neighbor (argmax)
5. Compare genre labels and report accuracy

**Results on 2,000-track subset (full evaluation):**

| Model | Top-1 Genre Accuracy | Correct / Total |
|-------|---------------------|-----------------|
| CLAP (audio) | 53.7% | 1073/2000 |
| CLAP (text) | 67.7% | 1354/2000 |
| OpenL3 | 55.3% | 1105/2000 |
| SBERT (lyrics) | 57.6% | 1151/2000 |
| **RRF (all 4 views)** | **76.9%** | **1538/2000** |

Random baseline: 12.5% (1 in 8 genres).

**Key takeaways:**
1. CLAP text embedding is the strongest single view
2. Text semantics outperform acoustic features for genre classification
3. Fused model performs significantly better than all four models alone — 9.2 percentage points above the best single view (CLAP text 67.7%)

**Limitations:** Genre is a coarse label; two acoustically similar tracks can differ in genre. This metric rewards genre clustering rather than nuanced similarity.

### 2. Echo Nest Feature Distance (Independent Ground Truth)

To evaluate without genre leakage, we measure if each view's nearest neighbors are similar in **Echo Nest audio features** — features the models never saw during training: danceability, energy, valence, tempo, acousticness, instrumentalness, liveness, speechiness. All features are z-score standardized before computing Euclidean distance.

For each track, grab its top-5 nearest neighbors from each model, then measure how far apart they are in Echo Nest feature space. Lower distance = the recommendations are more acoustically similar in ways the model didn't directly learn.

**Protocol:**
1. For each model, retrieve top-5 nearest neighbors per query
2. Compute average Euclidean distance in Echo Nest feature space
3. Compare against random baseline (average distance between random track pairs)
4. Report statistical significance via paired t-test

| Method | Avg Distance | vs Random | p-value |
|---|---|---|---|
| Random Baseline | 3.80 | — | — |
| SBERT (Text+Lyrics) | 3.33 | -12.4% | 3.1e-09 |
| OpenL3 & CLAP | 2.87 | -24.3% | 3.3e-21 |
| **Fused (SBERT+OpenL3+CLAP)** | **2.87** | **-24.3%** | **5.9e-22** |

**Key takeaways:**
1. OpenL3 & CLAP win because they have audio-related information
2. Fused keeps the same performance as OpenL3 & CLAP, but is more robust (covering more tracks)
3. SBERT score is higher (larger distance) because it does not contain acoustic information to make sense of Echo Nest audio features (danceability, energy, valence, tempo, etc.)

**Caveat:** N=294 tracks with both embeddings and Echo Nest features. The overlap is not uniformly distributed across genres (Folk and Hip-Hop over-represented at ~21% each; Experimental under-represented at 1.4%).

![Retrieval Quality: Echo Nest Audio Feature Similarity](data/processed/lyrics_enriched/echonest_evaluation.png)

### 3. Cross-View Overlap Analysis

**Metric:** For each track, retrieve top-20 neighbors in each view and compute:
- **Jaccard overlap:** |A ∩ B| / |A ∪ B|
- **Spearman rank correlation:** agreement between ranked lists

**Finding:** SBERT and OpenL3 share only **3.5% Jaccard overlap** (Spearman ρ = +0.07), confirming the views are highly complementary and fusion can combine independent signals.

### Embedding Space Comparison (All Views)

**t-SNE clustering** — CLAP forms the tightest genre clusters, SBERT is more diffuse (semantic ≠ genre), OpenL3 separates by acoustic texture:

![t-SNE Genre Clustering: CLAP vs SBERT vs OpenL3](data/processed/viz/tsne_comparison_3panel.png)

**PCA projection** — first two principal components capture 18.5% variance for CLAP but only 8.9% for SBERT, confirming SBERT distributes information more uniformly:

![PCA Genre Clustering: CLAP vs SBERT vs OpenL3](data/processed/viz/pca_comparison_3panel.png)

**Genre centroid similarity** — CLAP shows strong negative off-diagonals (genres well-separated), SBERT is flatter (text metadata doesn't cleanly separate genres), OpenL3 shows acoustic groupings (Hip-Hop/Pop cluster together):

![Genre Centroid Similarity: CLAP vs SBERT vs OpenL3](data/processed/viz/heatmap_comparison_3panel.png)

For full reproduction steps, see [docs/EVALUATION.md](docs/EVALUATION.md).

---

## Bonus Experiment: Audio-to-Spectrogram Analysis

**Can a CNN trained on photos still learn something useful from spectrograms, because they look like images?**

### Motivation

Spectrograms are visual representations of audio — time on the x-axis, frequency on the y-axis, intensity as color. Visual features learned from natural images (ImageNet) should still capture useful structure in spectrograms — harmonic stripes, drum hits, rhythmic patterns — without any audio-specific training.

### Feature-Extraction Pipeline

```
Audio (10s, 22 kHz) → Mel Spectrogram (128 bands × ~431 frames) → Resize (224 × 224 × 3 channels)
→ ResNet18 (ImageNet-pretrained, head removed) → penultimate layer → 512-d embedding → L2-normalize
```

The spectrogram is converted to log-power dB scale, resized to match ImageNet input dimensions, normalized with ImageNet statistics, and passed through a frozen ResNet18. The penultimate layer output serves as the embedding — no fine-tuning on audio data.

### Spectrogram Per Genre

Different genres produce visually distinct spectrograms:
- **Folk:** Horizontal harmonic stripes from voice + acoustic guitar
- **Hip-Hop:** Regular vertical bands from kick + snare grid
- **Rock:** Dense broadband texture from distortion + drums
- **Electronic:** Structured patterns with sharp frequency transitions
- **Instrumental:** Clear harmonic partials with sparse structure

### Where Spectrogram Is Strong

ImageNet visual priors transfer to genres with regular spectral structure. Top-1 retrieval accuracy on three genres where Spectrogram matches or beats peer views:

**Folk:** Spectrogram 54.8% (matches OpenL3 67.2%, SBERT 55.2%)
**Hip-Hop:** Spectrogram 49.6% (trails only audio-trained models)
**Rock:** Spectrogram 45.2% (beats SBERT text at 43.2%)

All three genres share strong, repeating visual patterns the CNN's ImageNet filters can already encode. The spectrogram view is competitive with purpose-built audio models in genres where spectral structure is visually distinctive.

---

## Team Contributions

### Overall Architecture & Web Demo — Issac
Introduced the project and designed the overall pipeline. Visualized the data and outputs, and implemented multi-view fusion strategy (vector-level early fusion and rank-level RRF), built the interactive Flask web application with Vanilla JS front-end.

### CLAP Embeddings & Results — Jiayi
Generated and fine-tuned CLAP embeddings (512-d) on the FMA dataset using contrastive learning. Analyzed genre structure in embedding space via t-SNE, PCA, and genre centroid similarity heatmaps. Conducted text-to-music retrieval demonstrations showing natural language query capabilities.

### Semantic Search with SBERT & Lyrics — Sid
Generated SBERT embeddings (384-d) from track metadata and lyrics fetched via Genius API. Identified and fixed critical genre leakage in metadata strings (48.8% of tags contained genre words). Conducted semantic robustness analysis including lexical bias and truncation impact studies. (See `reports/sid_issac_lyrics_report.md`).

### Acoustic Similarity with OpenL3 — Wenny
Generated OpenL3 embeddings (512-d) for audio-to-audio retrieval. Analyzed clustering patterns and cross-view complementarity by comparing with CLAP and SBERT. Investigated mean-centering effects on cosine similarity metrics.

### Evaluation Framework & Fusion Analysis — Yunchu (Helena)
Designed and implemented the three-layer evaluation methodology: Top-1 genre accuracy, Echo Nest feature distance analysis, and cross-view overlap assessment. Ensured data leakage prevention across all evaluation pipelines. Implemented and evaluated Reciprocal Rank Fusion (RRF) strategy, demonstrating that four-view fusion achieves **76.9% genre accuracy** — 9.2 percentage points above the best single view (CLAP text 67.7%) — while maintaining competitive Echo Nest performance (24.3% improvement vs random). Conducted comprehensive statistical significance testing (paired t-tests, p-values) across all metrics.

### Audio-to-Spectrograph Analysis — MJ
Conducted exploratory analysis on converting audio waveforms to spectrograms as an alternative representation. Built the ResNet18/ImageNet feature-extraction pipeline producing 512-d embeddings from mel spectrograms. Analyzed per-genre spectrogram patterns and demonstrated that ImageNet visual priors transfer effectively to genres with regular spectral structure.

---

## Glossary

| Term | Definition |
|---|---|
| **Embedding** | A fixed-size numerical vector representing a track in a continuous space where similar items are close. |
| **FAISS** | Facebook AI Similarity Search — efficient nearest-neighbor search. We use IndexFlatIP (exact inner product). |
| **Cosine Similarity** | Cosine of the angle between two vectors, [-1, 1]. Equals dot product for L2-normalised vectors. |
| **Contrastive Learning** | Learns to map similar pairs close, dissimilar pairs far. Typically uses InfoNCE loss. |
| **CLAP** | Contrastive Language-Audio Pretraining — shared audio/text space; mood, genre, instruments. |
| **SBERT** | Sentence-BERT — siamese transformer for sentence embeddings. all-MiniLM-L6-v2. |
| **OpenL3** | Self-supervised audio embeddings on AudioSet — timbre, rhythm, texture (no labels). |
| **RRF** | Reciprocal Rank Fusion — sum 1/(k+rank) across views. Robust, no weight tuning. |
| **InfoNCE** | Contrastive loss using softmax over (positive vs many negatives). Used for CLAP fine-tuning. |
| **Data Leakage** | Eval targets leaking into model inputs, inflating metrics. We removed genre from SBERT input. |
| **t-SNE** | t-distributed Stochastic Neighbor Embedding — 2D visualisation of high-dimensional vectors. |

---

## Quick Start

### Run the Demo (Fastest)

Precomputed embeddings are included in `data/processed/`:

```bash
# Install dependencies
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# Launch the web app
python app.py
```

Open `http://localhost:5001` and start exploring music similarities across three views.

### Reproduce Embeddings (If Needed)

To regenerate embeddings from scratch:

```bash
# Download FMA dataset (~7.2 GB, one-time only)
python scripts/download_fma.py

# Generate CLAP embeddings
python scripts/generate_clap_embeddings.py

# Generate SBERT embeddings + lyrics
export GENIUS_API_KEY="your-api-key"  # Add to .env instead
python scripts/generate_sbert_embeddings.py

# Fuse all views
python scripts/generate_fused_embeddings.py --skip-lyrics  # reuse cached lyrics
```

### Run Evaluations

```bash
cd evaluation
python evaluate_genre_retrieval.py --num-samples -1 --seed 42
```

Produces three metrics: genre accuracy, Echo Nest feature distance, cross-view overlap.

## Directory Structure

```text
├── data/
│   ├── fma_small/               # 8,000 MP3 files (organized as 000/000002.mp3)
│   ├── fma_2000_metadata/       # Canonical 2,000-track metadata (tracks.csv, genres.csv, echonest.csv)
│   └── processed/               # Embeddings, FAISS indices, visualisations
│       └── lyrics_enriched/     # Genre-free SBERT + fused embeddings
├── docs/                        # Project documentation and guides
│   ├── EVALUATION.md            # Detailed evaluation methodology
│   ├── CLAP.md                  # CLAP model documentation
│   ├── SBERT.md                 # SBERT model documentation
│   ├── OpenL3.md                # OpenL3 model documentation
│   ├── API.md                   # REST API reference
│   ├── DEVELOPMENT.md           # Development guide
│   └── neural_reranking.md
├── evaluation/                  # Evaluation scripts and results
│   ├── evaluate_genre_retrieval.py      # Genre accuracy evaluation
│   ├── compare_mean_center.py           # Mean-centering comparison
│   ├── results.md                       # Evaluation results summary
│   └── EXPERIMENT_RESULTS.md
├── forMj/                       # Audio spectrogram exploration
│   ├── generate_spectrogram_embeddings.py
│   ├── spectrogram.py
│   └── README.md
├── notebooks/
│   ├── 01_eda.ipynb                     # Dataset exploration
│   ├── 02_clap_retrieval_demo.ipynb     # Text-to-music search demo
│   ├── 03_embedding_visualisation.ipynb # t-SNE, PCA, genre heatmaps
│   ├── 05_clap_sbert_overlap.ipynb      # Cross-view comparison
│   ├── 06_echonest_exploration.ipynb    # Echo Nest feature analysis
│   ├── 07_sbert_analysis.ipynb          # SBERT representation analysis
│   └── 08_semantic_search_demo.ipynb    # SBERT query interface
├── reports/                     # Presentations, writeups, and final reports
│   ├── sid_issac_lyrics_report.md       # SBERT & lyrics analysis report
│   └── EXPERIMENT_RESULTS.md
├── scripts/
│   ├── download_fma.py                           # Download FMA dataset
│   ├── audit_metadata.py                        # Cross-reference metadata vs audio
│   ├── generate_clap_embeddings.py              # CLAP embedding generator
│   ├── generate_sbert_embeddings.py             # SBERT embedding generator
│   ├── generate_spectrogram_embeddings.py       # Spectrogram embedding generator
│   ├── generate_fused_embeddings.py             # Multi-view fusion embeddings
│   ├── generate_pipeline_visualizations.py      # Generate t-SNE, PCA, heatmaps
│   ├── build_faiss_index.py                     # Build FAISS indices
│   ├── build_sbert_index.py                     # Build SBERT-specific FAISS index
│   ├── encode_2000_tracks.py                    # Encode 2,000-track canonical subset
│   ├── extract_2000_metadata.py                 # Extract metadata for 2,000 tracks
│   ├── verify_2000_tracks.py                    # Verify canonical subset integrity
│   ├── analyze_sbert_robustness.py              # SBERT representation robustness
│   ├── visualize_sbert.py                       # SBERT visualization utilities
│   ├── visualize_robustness.py                  # Robustness analysis visualizations
│   ├── compare_clap_sbert.py                    # Cross-view comparison analysis
│   ├── openl3_vs_sbert_overlap.py               # Cross-view overlap metrics
│   ├── clap_embeddings.py                       # CLAP standalone pipeline
│   ├── text_to_text_SBERT_FMA_GENIUS_2.py       # SBERT standalone pipeline
│   └── (Legacy standalone files — use scripts above)
├── src/
│   ├── __init__.py
│   ├── config.py                # Paths, constants, device selection
│   ├── metadata.py              # FMA metadata loading and filtering
│   ├── metadata_builder.py      # Text string construction (genre-free)
│   ├── audio_utils.py           # Track path resolution
│   ├── lyrics_fetcher.py        # Genius API lyrics fetcher
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract EmbeddingGenerator interface
│   │   ├── clap.py              # CLAP pipeline (batched, checkpointed)
│   │   ├── sbert.py             # Sentence-BERT pipeline
│   │   └── spectrogram.py        # Spectrogram-based embedding
│   └── indexing/
│       ├── __init__.py
│       └── faiss_index.py       # FAISS index wrapper (cosine + L2)
├── tests/                       # Pytest test suite
│   ├── conftest.py
│   ├── test_core.py
│   └── __init__.py
├── .env                         # API keys (gitignored)
├── Dockerfile                   # Deployment container
├── requirements.txt             # Python dependencies
├── app.py                       # Main Flask web application / demo
└── README.md
```

## Documentation

### Model Documentation (per-view)

- **[CLAP (View 1)](docs/CLAP.md)** — Contrastive Language-Audio Pretraining. Architecture, cross-modal retrieval, text-to-audio search, generation pipeline.
- **[SBERT (View 2)](docs/SBERT.md)** — Sentence-BERT semantic search. Metadata/lyrics encoding, data leakage prevention, Genius API integration, robustness analysis.
- **[OpenL3 (View 3)](docs/OpenL3.md)** — Acoustic similarity. Self-supervised audio embeddings, mean-centering rationale, cross-view complementarity analysis.

### System Documentation

- **[API Reference](docs/API.md)** — REST endpoint specifications, request/response formats, and fusion algorithm.
- **[Development Guide](docs/DEVELOPMENT.md)** — Setup, testing, code conventions, Docker, and architecture decisions.
- **[Evaluation Methodology](docs/EVALUATION.md)** — Metrics, data leakage prevention, and how to reproduce results.

## References

### Models

- **CLAP (Contrastive Language-Audio Pretraining):** Wu et al., "Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation," ICASSP 2023. Code: [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP)
- **SBERT (Sentence-BERT):** Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," EMNLP 2019. Model: `all-MiniLM-L6-v2` via [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
- **OpenL3:** Cramer et al., "Look, Listen, and Learn More: Design Choices for Deep Audio Embeddings," ICASSP 2019. Code: [marl/openl3](https://github.com/marl/openl3)

### Dataset

- **FMA (Free Music Archive):** Defferrard et al., "FMA: A Dataset for Music Analysis," ISMIR 2017. [GitHub](https://github.com/mdeff/fma)
- **AudioSet:** Gemmeke et al., "Audio Set: An ontology and human-labeled dataset for audio events," ICASSP 2017. Training data for OpenL3 (audio-visual correspondence).
- **Genius API:** Used for lyrics retrieval. [genius.com/developers](https://genius.com/developers)

### Libraries

- **FAISS:** Johnson et al., "Billion-Scale Similarity Search with GPUs," IEEE Transactions on Big Data, 2019. [GitHub](https://github.com/facebookresearch/faiss)
- **Reciprocal Rank Fusion:** Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods," SIGIR 2009
