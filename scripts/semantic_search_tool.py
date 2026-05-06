
import os
import sys
import json
import warnings
import numpy as np
import torch

# Suppress all warnings to keep stdout clean for JSON
warnings.filterwarnings("ignore")

from transformers import ClapModel, ClapProcessor

# Force single-threaded to be safe
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    query = sys.argv[1] if len(sys.argv) > 1 else ""
    if not query:
        print(json.dumps([]))
        return

    # Load shared data using absolute paths for robustness
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clap_embs = np.load(os.path.join(base_dir, 'data/processed/clap_embeddings.npy')).astype(np.float32)
    clap_ids = np.load(os.path.join(base_dir, 'data/processed/clap_track_ids.npy'))
    subset_ids = set(np.load(os.path.join(base_dir, 'data/processed/sbert_track_ids.npy')).tolist())

    mask = [int(tid) in subset_ids for tid in clap_ids]
    clap_2k_embs = clap_embs[mask]
    clap_2k_ids = clap_ids[mask]

    # Normalize
    clap_2k_embs = clap_2k_embs - clap_2k_embs.mean(axis=0)
    norms = np.linalg.norm(clap_2k_embs, axis=1, keepdims=True)
    clap_2k_embs = clap_2k_embs / (norms + 1e-8)

    # Load model (suppress logging to stdout)
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    
    torch.set_num_threads(1)
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    model.eval()

    # Encode query
    with torch.no_grad():
        inputs = processor(text=[query], return_tensors="pt", padding=True)
        text_emb = model.get_text_features(**inputs).numpy()
    
    text_emb = text_emb / (np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-8)

    # Search
    scores = (clap_2k_embs @ text_emb.T).flatten()
    top_indices = np.argsort(scores)[::-1][:20]

    results = []
    for idx in top_indices:
        results.append({
            "track_id": int(clap_2k_ids[idx]),
            "score": round(float(scores[idx]), 4)
        })

    # Output ONLY json to stdout
    sys.stdout.write(json.dumps(results))

if __name__ == "__main__":
    main()
