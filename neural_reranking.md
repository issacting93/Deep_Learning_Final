

## Neural Reranking (Learning to Rank)
When building multi-view retrieval systems, combining multiple independent models (e.g., audio, text, graph) requires a fusion strategy. 
* **Reciprocal Rank Fusion (RRF):** A basic mathematical baseline `1 / (k + rank)`. It treats all models equally and requires no training data.
* **Neural Reranking:** A deep learning approach that learns *when* to trust *which* model to predict true relevance.

### How it Works (The Pipeline)
1. **Candidate Generation (Fast):** Use a fast, scalable search (like FAISS) to grab the Top-N results from each individual model (e.g., Top-100 from SBERT, Top-100 from CLAP). This is the "retrieval" stage — optimized for recall, not precision.
2. **Feature Extraction:** For each retrieved item, gather the raw similarity scores it received from every model into a single input vector. `x = [SBERT_score, CLAP_score, OpenL3_score]`. You can also include metadata features (e.g., genre match, popularity) to give the reranker more signal.
3. **The Reranker Network:** Pass this vector through a small Multi-Layer Perceptron (MLP) — a few Dense layers with ReLU activations. The network is intentionally small because it only processes the Top-N candidates, not the entire dataset.
4. **Scoring & Sorting:** The network outputs a single float representing the "Predicted Relevance." The candidates are re-sorted by this new score and the top results are presented to the user.

### Why it Outperforms Basic Fusion
RRF (used in the project's `app.py`) assigns equal weight to every model with a fixed formula: `score += w / (k + rank + 1)`. It has no way to learn from data.

A Neural Reranker acts as a dynamic judge:
* **Model Confidence:** It can learn that OpenL3 is highly accurate for Instrumental queries but CLAP is better for Vocal queries, automatically shifting weight depending on context.
* **Non-linear Interactions:** It can learn that when SBERT and CLAP *both* rank a track highly it's almost certainly relevant, but when only one does, it's a coin flip. RRF can't express this — it just sums.
* **Score Calibration:** Raw cosine scores from different models live on different scales. The MLP implicitly learns to calibrate them.

### Training the Reranker
The reranker requires a **ground truth** relevance dataset — you need labels that say "for this query track, this result is relevant / not relevant."

**Creating Training Data:**
* **Positive pairs:** Tracks that share the same `genre_top` (or that users actually clicked/liked).
* **Negative pairs:** Tracks with different genres retrieved in the Top-N (hard negatives are more valuable than random ones).
* Each training sample is: `(feature_vector, relevance_label)` where the feature vector contains the per-model scores.

**Loss Functions:**
* **Binary Cross-Entropy (pointwise):** Treat it as binary classification — predict `1` for relevant, `0` for irrelevant. Simple but ignores relative ordering.
* **Margin Ranking Loss (pairwise):** Given a relevant and irrelevant item, ensure the relevant one scores higher by at least a margin. Better captures the ranking objective.
* **ListNet / LambdaRank (listwise):** Optimize the entire ranked list at once. Most expressive but more complex to implement.

### Comparison: RRF vs Neural Reranker

| | **RRF** | **Neural Reranker** |
|---|---|---|
| **Training data needed** | None | Yes (relevance labels) |
| **Weights** | Fixed, equal per model | Learned, context-dependent |
| **Interactions** | Additive only | Non-linear (MLP) |
| **Cold start** | Works immediately | Needs enough labeled data |
| **Adaptability** | Must manually tune `k` | Learns from data |
| **Complexity** | One line of math | Training pipeline + model |

### Practical Considerations
* **Two-stage pattern** is standard in production systems: fast retrieval (ANN/FAISS) → expensive reranking (MLP or cross-encoder). The first stage casts a wide net; the second stage applies precision.
* **Diminishing returns:** If your models already agree most of the time, a neural reranker won't add much. It helps most when models disagree and one is systematically better in certain contexts.
* **Overfitting risk:** The reranker MLP is small and the feature vectors are low-dimensional, so regularization (dropout, weight decay) matters to avoid memorizing the training set.
