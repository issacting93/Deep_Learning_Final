# API Reference

The Flask application (`app.py`) exposes a REST API for music search and recommendation. All responses are JSON.

**Base URL:** `http://localhost:5001`

---

## Endpoints

### `GET /`

Serves the single-page web application.

---

### `GET /api/search`

Full-text search over track metadata (title, artist, genre).

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `q`  | string | Yes | Search query (minimum 2 characters) |

**Response:** Array of track objects (max 20 results).

```json
[
  {
    "track_id": 2,
    "title": "Food",
    "artist": "AWOL",
    "genre": "Hip-Hop"
  }
]
```

Returns `[]` if query is less than 2 characters.

---

### `GET /api/tracks`

Paginated track listing with optional genre filtering.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `page` | int | No | 1 | Page number (min 1) |
| `per_page` | int | No | 50 | Items per page (1-200) |
| `genre` | string | No | — | Filter by genre (exact match) |

**Response:**

```json
{
  "tracks": [...],
  "total": 1847,
  "page": 1
}
```

**Errors:**
- `400` — Invalid `page` or `per_page` parameter (non-numeric).

---

### `GET /api/recommend/<track_id>`

Get recommendations from all three embedding views plus the fused result.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `track_id` | int | Yes (path) | — | Track ID to get recommendations for |
| `k` | int | No | 8 | Number of results per view (3-20) |

**Response:**

```json
{
  "seed": {
    "track_id": 2,
    "title": "Food",
    "artist": "AWOL",
    "genre": "Hip-Hop"
  },
  "views": {
    "sbert": {
      "name": "SBERT (Text)",
      "results": [
        {
          "track_id": 10,
          "title": "...",
          "artist": "...",
          "genre": "...",
          "score": 0.8231
        }
      ]
    },
    "openl3": { ... },
    "clap": { ... },
    "fused": {
      "name": "Fused (RRF)",
      "results": [
        {
          "track_id": 10,
          "title": "...",
          "artist": "...",
          "genre": "...",
          "score": 0.0328,
          "view_scores": {
            "sbert": 0.8231,
            "openl3": 0.6102,
            "clap": 0.7455
          }
        }
      ]
    }
  }
}
```

**Scores:**
- Per-view `score`: cosine similarity (range -1 to 1, higher = more similar).
- Fused `score`: RRF score (sum of `1 / (60 + rank)` across views). Not directly comparable to cosine scores.
- `view_scores`: per-view cosine similarities for each fused result, useful for understanding which views contributed.

**Errors:**
- `404` — Track ID not found in dataset.
- `400` — Invalid `k` parameter.

---

### `GET /api/audio/<track_id>`

Stream the audio file for a track.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `track_id` | int | Yes (path) | Track ID |

**Response:** Audio file (`audio/mpeg`).

**Errors:**
- `404` — Track not in dataset or audio file missing from disk.

---

### `GET /api/genres`

Returns genre distribution across the common track set.

**Response:** Array of `[genre, count]` pairs, sorted by count descending.

```json
[
  ["Rock", 490],
  ["Electronic", 386],
  ["Hip-Hop", 285],
  ...
]
```

---

## Recommendation Algorithm

### Per-View Retrieval

For each view (SBERT, OpenL3, CLAP):
1. Look up the seed track's embedding vector.
2. Compute dot product against all other embeddings (equivalent to cosine similarity since vectors are L2-normalized).
3. Sort by score descending, return top-k results that exist in the common track set.

### Reciprocal Rank Fusion

The fused recommendation combines rankings from all three views:

```
score(track) = sum over views: weight / (RRF_K + rank + 1)
```

Where `RRF_K = 60` (smoothing constant) and all weights default to 1.0. This produces a robust ranking that:
- Rewards tracks ranked highly across multiple views.
- Gracefully degrades if one view returns poor results.
- Does not require score calibration across views (uses ranks, not raw scores).

---

## Data Loading

At startup, the app:
1. Loads FMA metadata (`tracks.csv`) and filters to the `small` subset.
2. Loads three sets of precomputed embeddings from `data/processed/`.
3. Mean-centers each embedding matrix to remove DC offset.
4. L2-normalizes all vectors for cosine similarity via dot product.
5. Computes the intersection of track IDs across all three views (`COMMON_SET`).

Only tracks present in all three views are served via the API.
