# Role 4: Evaluation & Multi-View Fusion — Jiayi

## Goal

Quantitatively evaluate every retrieval view and build a **fusion system** that combines all views to outperform any single one. This role ties the whole project together.

## Background

Each of the other roles produces a FAISS index and embeddings. Your job is to load all of them, define a ground truth relevance criterion, compute standard IR metrics, and then build a fusion layer that combines scores from all views into a single ranked list.

## Your Tasks

### Part A: Evaluation Framework

1. **Define ground truth relevance**
   - **Baseline (easy)**: same `genre_top` = relevant (8 classes)
   - **Stricter**: same sub-genre from `genres_all` = relevant (163 classes)
   - Use `tracks.csv` to get genre labels for all 8,000 tracks

2. **Implement `src/evaluation.py`**
   - `precision_at_k(retrieved, relevant, k)` 
   - `recall_at_k(retrieved, relevant, k)`
   - `average_precision(retrieved, relevant)` → use to compute MAP
   - `ndcg_at_k(retrieved, relevant, k)`
   - `evaluate_index(faiss_index, track_ids, labels, k=10, n_queries=200)` — runs evaluation over a random sample of query tracks

3. **Evaluate each view independently**
   - Run evaluation on: CLAP, OpenL3 (Wenny), SBERT (Sid), GNN (Issac)
   - Report P@5, P@10, MAP, NDCG@10 for each view in a comparison table

### Part B: Fusion System

4. **Implement `src/fusion.py`**

   **Method 1 — Weighted score fusion:**
   ```python
   fused_score = w1 * clap_score + w2 * openl3_score + w3 * sbert_score + w4 * gnn_score
   ```
   - Normalise all scores to [0,1] before combining
   - Try equal weights first, then tune weights on a validation split

   **Method 2 — Reciprocal Rank Fusion (RRF):**
   ```python
   rrf_score(d) = sum(1 / (k + rank_i(d)) for each view i)
   ```
   - RRF is robust and requires no tuning (use k=60)

   **Method 3 — Learned reranker (stretch goal):**
   - For each (query, candidate) pair, concatenate per-view scores as features
   - Train a small logistic regression or MLP to predict relevance
   - Train on ~70% of queries, evaluate on held-out 30%

5. **Show fusion outperforms single views**
   - Report the full comparison table: each individual view + Method 1 + Method 2 + (optional) Method 3
   - Use bar charts to visualise P@10 and NDCG@10 across all methods

### Part C: Results Notebook

6. **Create comprehensive results notebook**
   - Full metric table (all views + fusion methods)
   - Per-genre breakdown: which view is best for Rock? Electronic? Folk?
   - Failure analysis: show 5 queries where no method does well — what do they have in common?
   - Example queries with side-by-side results from each view

## Key Files

| Path | Purpose |
|---|---|
| `src/indexing/faiss_index.py` | Load each view's index |
| `src/metadata.py` | Get genre labels for ground truth |
| `src/config.py` | Paths to all indexes |
| `data/processed/clap_faiss.index` | CLAP (View 1, done) |
| `data/processed/openl3_faiss.index` | Acoustic (Role 1 — Wenny) |
| `data/processed/sbert_faiss.index` | Semantic (Role 2 — Sid) |
| `data/processed/gnn_faiss.index` | Graph (Role 3 — Issac) |

## Deliverables

- [ ] `src/evaluation.py` — IR metrics
- [ ] `src/fusion.py` — weighted fusion, RRF, optional learned reranker
- [ ] Notebook: full evaluation results table and charts
- [ ] Notebook: per-genre analysis and failure cases
- [ ] Summary table ready for the report/presentation
