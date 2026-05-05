import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.config import PROCESSED_DIR

def plot_robustness():
    # Data from previous analysis run
    queries = ["lonely", "isolated"]
    overlap = 0.10 # 10%
    
    # Simple Bar Chart for synonym overlap
    plt.figure(figsize=(8, 6))
    sns.set_style("darkgrid")
    
    # Overlap vs Non-overlap
    labels = ['Overlap', 'Unique to "lonely"', 'Unique to "isolated"']
    sizes = [overlap, 1 - overlap, 1 - overlap] # Normalized
    
    # Better visualization: Overlap@10 Bar
    plt.bar(['lonely vs isolated'], [overlap], color='teal', alpha=0.7)
    plt.ylim(0, 1)
    plt.title("Semantic Robustness: Retrieval Overlap@10", fontsize=14)
    plt.ylabel("Overlap Ratio")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate([overlap]):
        plt.text(i, v + 0.02, f"{v:.1%}", ha='center', fontweight='bold')

    plt.savefig(PROCESSED_DIR / "semantic_robustness_bar.png")
    print("Semantic robustness plot saved.")

def plot_lexical_bias():
    # Data from previous analysis run
    # Query: "Blue music"
    # Title match: 1, Genre match: 0
    categories = ['Title Matches (Lexical)', 'Genre Matches (Semantic)']
    counts = [1, 0]
    
    plt.figure(figsize=(8, 6))
    sns.set_style("darkgrid")
    sns.barplot(x=categories, y=counts, palette=['salmon', 'skyblue'])
    plt.title("Lexical Bias: Query 'Blue music' Hits (Top 10)", fontsize=14)
    plt.ylabel("Number of Hits")
    plt.ylim(0, 2)
    
    plt.savefig(PROCESSED_DIR / "lexical_bias_analysis.png")
    print("Lexical bias plot saved.")

if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    plot_robustness()
    plot_lexical_bias()
