1.500 songs randomly out of 2000:
CLAP(audio): 0.5020
CLAP(text):0.6760
OpenL3: 0.5260
SBERT:0.6020
2.All 2000 songs
CLAP(audio): 0.5365
CLAP(text): 0.6770
OpenL3:0.5525
SBERT:0.5755

# Experiment Documentation: Genre Consistency Retrieval Benchmark

## 1. Goal

Evaluate whether nearest-neighbor retrieval results are genre-consistent across multiple embedding spaces on a 2,000-track subset.

Main metric:
- Top-1 genre accuracy = fraction of query tracks whose most similar retrieved track has the same `genre_top`.

## 2. Data and Embeddings

Metadata source:
- `our_2000_tracks.csv`

Embedding sources:
- `CLAP/clap_audio_embeddings.npy`
- `CLAP/clap_text_embeddings_new.npy`
- `OpenL3/openl3_embeddings.npy`
- `OpenL3/openl3_track_ids.npy`
- `SBERT/sbert_lyrics_embeddings.npy`
- `SBERT/sbert_lyrics_faiss.ids.npy`

## 3. Data/Format Validation Summary

Validated conditions:
- Metadata rows: 2000
- Unique track IDs: 2000
- `genre_top` available for all tracks
- OpenL3 embedding shape: `(2000, 512)`, IDs shape: `(2000,)`
- SBERT embedding shape: `(2000, 384)`, IDs shape: `(2000,)`
- CLAP audio embedding: stored as dict `{track_id: 512-d vector}`
- Old CLAP text embedding: dict format `{track_id: 512-d vector}`
- New CLAP text embedding: ndarray format `(2000, 512)` aligned to CSV order
- All evaluated arrays are finite (no NaN/Inf)
- ID alignment with metadata passed for all evaluated models

## 4. Evaluation Protocol

Script:
- `evaluate_genre_retrieval.py`

Procedure:
1. Load metadata and map `track_id -> genre_top`.
2. L2-normalize embeddings.
3. Compute cosine similarity via matrix product.
4. For each query, exclude self-match.
5. Retrieve top-1 nearest neighbor.
6. Count prediction as correct if neighbor genre equals query genre.

Settings used:
- Sampled evaluation: `--num-samples 500 --seed 42`
- Full evaluation: `--num-samples -1 --seed 42`

## 5. Results

Sampled (500 tracks):
- CLAP(audio): 0.5020 (251/500)
- CLAP(text, new): 0.6760 (338/500)
- OpenL3: 0.5260 (263/500)
- SBERT: 0.6020 (301/500)

Full (2000 tracks):
- CLAP(audio): 0.5365 (1073/2000)
- CLAP(text, new): 0.6770 (1354/2000)
- OpenL3: 0.5525 (1105/2000)
- SBERT: 0.5755 (1151/2000)

## 6. Notes and Interpretation

- CLAP text retrieval is substantially stronger than CLAP audio in this genre-consistency benchmark.
- CLAP audio and OpenL3 are relatively close in this setup, with OpenL3 slightly higher.
- SBERT is above both CLAP audio and OpenL3 here, but below CLAP text (new and old).

Important caveat:
- In `SBERT/lyrics_df.csv`, the `lyrics` column is empty for all 2000 rows in the inspected file. This suggests SBERT embeddings likely reflect other text fields (or were generated from a different source file/version), not raw lyrics content alone.
