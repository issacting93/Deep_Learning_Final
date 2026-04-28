import os
import sys
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath("."))

from src.config import PROCESSED_DIR, FMA_METADATA_DIR
from src.embeddings.sbert import SentenceBERTEmbeddingGenerator
from src.indexing.faiss_index import FaissIndex
from src.metadata import load_tracks

def calculate_overlap(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return len(set1.intersection(set2)) / len(set1) if len(set1) > 0 else 0

def main():
    print("--- Role 2: SBERT Representation Analysis ---")
    
    # Load model and data
    generator = SentenceBERTEmbeddingGenerator()
    index = FaissIndex.load(PROCESSED_DIR / "sbert_faiss.index")
    df_meta = pd.read_csv(PROCESSED_DIR / "metadata_texts.csv", index_col=0)
    tracks_meta = load_tracks(FMA_METADATA_DIR)
    
    # 1. Semantic Robustness: "lonely" vs "isolated"
    print("\n1. Analyzing Semantic Robustness...")
    q1 = "lonely"
    q2 = "isolated"
    
    emb1 = generator.embed_text([q1])
    emb2 = generator.embed_text([q2])
    
    res1 = [r[0] for r in index.query(emb1, k=10)]
    res2 = [r[0] for r in index.query(emb2, k=10)]
    
    overlap = calculate_overlap(res1, res2)
    print(f"Overlap@10 for '{q1}' vs '{q2}': {overlap:.2%}")
    
    # 2. Truncation Impact (Tokens 0-256 vs 256-512)
    print("\n2. Analyzing Truncation Impact...")
    # Note: all-MiniLM typically has a 256 token limit. 
    # We'll simulate this by taking words as rough token proxies.
    sample_ids = index.track_ids[:100] # Use first 100 for speed
    texts = df_meta.loc[sample_ids, "metadata_text"].tolist()
    
    # Split by whitespace to approximate tokens
    shorter_texts = [" ".join(t.split()[:128]) for t in texts]
    longer_texts = [" ".join(t.split()[128:256]) for t in texts]
    
    # Remove empty ones
    valid_indices = [i for i, t in enumerate(longer_texts) if len(t.strip()) > 10]
    if valid_indices:
        v_shorter = [shorter_texts[i] for i in valid_indices]
        v_longer = [longer_texts[i] for i in valid_indices]
        
        emb_short = generator.model.encode(v_shorter, normalize_embeddings=True)
        emb_long = generator.model.encode(v_longer, normalize_embeddings=True)
        
        # Measure cosine shift (1 - dot product since normalized)
        shifts = 1 - np.sum(emb_short * emb_long, axis=1)
        print(f"Average Cosine Shift (0-128 vs 128-256 words): {np.mean(shifts):.4f}")
    else:
        print("No tracks with sufficient length for truncation analysis.")

    # 3. Lexical Bias Check
    print("\n3. Analyzing Lexical Bias...")
    # Does "Blue" in title trump "Blues" genre if keywords are present?
    # Query: "Blue music"
    q_bias = "Blue music"
    emb_bias = generator.embed_text([q_bias])
    results = index.query(emb_bias, k=10)
    
    print(f"Top results for '{q_bias}':")
    keyword_hits = 0
    genre_hits = 0
    for tid, score in results:
        row = tracks_meta.loc[tid]
        title = str(row[('track', 'title')]).lower()
        genre = str(row[('track', 'genre_top')]).lower()
        
        has_blue = "blue" in title
        is_blues = "blues" in genre
        
        match_str = ""
        if has_blue: 
            match_str += "[Title Match] "
            keyword_hits += 1
        if is_blues: 
            match_str += "[Genre Match] "
            genre_hits += 1
            
        print(f"- {tid}: {title} | Genre: {genre} {match_str}")
        
    print(f"\nKeyword Hits in Top 10: {keyword_hits}")
    print(f"Genre Hits in Top 10: {genre_hits}")
    if keyword_hits > genre_hits:
        print("Finding: Lexical bias detected (Keyword match > Semantic genre match).")
    else:
        print("Finding: Semantic matching appears robust to lexical overlap.")

if __name__ == "__main__":
    main()
