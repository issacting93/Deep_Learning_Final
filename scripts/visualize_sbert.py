import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Add project root to path
sys.path.append(os.path.abspath("."))

from src.config import PROCESSED_DIR, FMA_METADATA_DIR
from src.metadata import load_tracks

def main():
    print("Loading data for visualization...")
    
    # Check if files exist
    embeddings_path = PROCESSED_DIR / "sbert_embeddings.npy"
    ids_path = PROCESSED_DIR / "sbert_track_ids.npy"
    
    if not (embeddings_path.exists() and ids_path.exists()):
        print(f"Error: Required files not found in {PROCESSED_DIR}")
        print("Please ensure scripts/build_sbert_index.py has finished running.")
        return

    embeddings = np.load(embeddings_path)
    track_ids = np.load(ids_path)
    
    # Load metadata for genre labeling
    tracks = load_tracks(FMA_METADATA_DIR)
    genres = tracks.loc[track_ids, ('track', 'genre_top')].fillna("Unknown")
    
    print(f"Loaded {len(embeddings)} embeddings. Reducing dimensions...")
    
    # 1. PCA Pre-reduction (to 50 dimensions)
    pca = PCA(n_components=50, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    
    # 2. t-SNE reduction
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, init='pca', learning_rate='auto')
    embeddings_2d = tsne.fit_transform(embeddings_pca)
    
    # 3. Plotting
    print("Generating plot...")
    plt.figure(figsize=(16, 10))
    sns.set_style("darkgrid")
    
    # Get top 12 genres to avoid legend clutter
    top_genres = genres.value_counts().nlargest(12).index
    plot_genres = genres.apply(lambda x: x if x in top_genres else "Other")
    
    scatter = sns.scatterplot(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hue=plot_genres,
        palette="husl",
        alpha=0.7,
        s=30,
        edgecolor=None
    )
    
    plt.title("t-SNE Visualization of SBERT Music Metadata Embeddings", fontsize=18)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Top Genres")
    
    # Save the plot
    output_path = PROCESSED_DIR / "tsne_sbert_embeddings.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Visualization saved to {output_path}")
    
    # Optional: Plot Explained Variance
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.title("PCA Cumulative Explained Variance (SBERT)")
    plt.xlabel("Number of Components")
    plt.ylabel("Variance Explained")
    plt.savefig(PROCESSED_DIR / "pca_variance_sbert.png")
    print("Variance plot saved.")

if __name__ == "__main__":
    main()
