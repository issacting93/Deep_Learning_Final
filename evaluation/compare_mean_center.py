"""Compare top-1 genre retrieval accuracy with and without mean-centering."""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent

# Load genre metadata from SBERT's lyrics_df.csv (has track_id + genres for all 2000)
meta = pd.read_csv(ROOT / "SBERT" / "lyrics_df.csv")
id_to_genre = dict(zip(meta["track_id"].astype(int), meta["genres"].astype(str)))
csv_ids = meta["track_id"].astype(int).to_numpy()


def load_dict_emb(path):
    raw = np.load(path, allow_pickle=True).item()
    ids = np.array(sorted(int(k) for k in raw.keys()), dtype=np.int64)
    emb = np.stack([np.asarray(raw[int(i)], dtype=np.float32) for i in ids])
    return ids, emb


def load_pair_emb(emb_path, ids_path):
    return np.load(ids_path).astype(np.int64), np.load(emb_path).astype(np.float32)


def l2_norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def top1_accuracy(ids, emb, id_to_genre):
    mask = np.array([int(t) in id_to_genre for t in ids])
    ids, emb = ids[mask], emb[mask]
    emb = l2_norm(emb)
    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)
    nn = np.argmax(sim, axis=1)
    genres = np.array([id_to_genre[int(t)] for t in ids])
    correct = int(np.sum(genres == genres[nn]))
    return correct / len(ids), correct, len(ids)


models = {
    "CLAP(audio)": load_dict_emb(ROOT / "CLAP" / "clap_audio_embeddings.npy"),
    "CLAP(text)":  (csv_ids, np.load(ROOT / "CLAP" / "clap_text_embeddings_new.npy").astype(np.float32)),
    "OpenL3":      load_pair_emb(ROOT / "OpenL3" / "openl3_embeddings.npy", ROOT / "OpenL3" / "openl3_track_ids.npy"),
    "SBERT":       load_pair_emb(ROOT / "SBERT" / "sbert_lyrics_embeddings.npy", ROOT / "SBERT" / "sbert_lyrics_faiss.ids.npy"),
}

print(f"{'Model':<15} {'Raw':>10} {'Mean-Ctr':>10} {'Delta':>10}")
print("-" * 47)
for name, (ids, emb) in models.items():
    acc_raw, _, _ = top1_accuracy(ids, emb.copy(), id_to_genre)
    emb_mc = emb - emb.mean(axis=0)
    acc_mc, _, _ = top1_accuracy(ids, emb_mc, id_to_genre)
    delta = acc_mc - acc_raw
    print(f"{name:<15} {acc_raw:>9.4f} {acc_mc:>10.4f} {delta:>+10.4f}")
