import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class ModelData:
    name: str
    ids: np.ndarray
    emb: np.ndarray


def load_metadata(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    df = pd.read_csv(csv_path, header=[0, 1], skiprows=[2])

    track_id_col = ("Unnamed: 0_level_0", "Unnamed: 0_level_1")
    genre_col = ("track", "genre_top")

    if track_id_col not in df.columns:
        track_ids = df.iloc[:, 0].astype(int).to_numpy()
    else:
        track_ids = df[track_id_col].astype(int).to_numpy()

    if genre_col not in df.columns:
        raise ValueError("Cannot find ('track', 'genre_top') column in metadata CSV.")

    genres = df[genre_col].astype(str).to_numpy()
    id_to_genre = {int(tid): genre for tid, genre in zip(track_ids, genres)}
    return track_ids, genres, id_to_genre


def load_clap_dict_embeddings(path: Path, name: str) -> ModelData:
    raw = np.load(path, allow_pickle=True)

    if isinstance(raw, np.ndarray) and raw.ndim == 0 and raw.dtype == object:
        raw = raw.item()

    if not isinstance(raw, dict):
        raise ValueError(f"{name}: expected dict-like serialized object, got {type(raw)}")

    ids = np.array(sorted(int(k) for k in raw.keys()), dtype=np.int64)
    emb = np.stack([np.asarray(raw[int(i)], dtype=np.float32) for i in ids], axis=0)
    return ModelData(name=name, ids=ids, emb=emb)


def load_ordered_embeddings(path: Path, ids: np.ndarray, name: str) -> ModelData:
    emb = np.load(path).astype(np.float32)
    if emb.ndim != 2:
        raise ValueError(f"{name}: expected 2D embedding array, got shape {emb.shape}")
    if emb.shape[0] != len(ids):
        raise ValueError(
            f"{name}: row mismatch, emb rows {emb.shape[0]} vs metadata ids {len(ids)}"
        )
    return ModelData(name=name, ids=ids.astype(np.int64), emb=emb)


def load_id_embedding_pair(emb_path: Path, ids_path: Path, name: str) -> ModelData:
    emb = np.load(emb_path).astype(np.float32)
    ids = np.load(ids_path).astype(np.int64)
    return ModelData(name=name, ids=ids, emb=emb)


def validate_model_data(model: ModelData, csv_id_set: set) -> None:
    if model.emb.ndim != 2:
        raise ValueError(f"{model.name}: embedding must be 2D, got shape {model.emb.shape}")
    if model.ids.ndim != 1:
        raise ValueError(f"{model.name}: ids must be 1D, got shape {model.ids.shape}")
    if model.emb.shape[0] != model.ids.shape[0]:
        raise ValueError(
            f"{model.name}: row mismatch, emb rows {model.emb.shape[0]} vs ids {model.ids.shape[0]}"
        )
    if len(np.unique(model.ids)) != len(model.ids):
        raise ValueError(f"{model.name}: duplicate track IDs found")
    if not np.isfinite(model.emb).all():
        raise ValueError(f"{model.name}: embeddings contain NaN or Inf")
    not_in_csv = np.sum(~np.isin(model.ids, list(csv_id_set)))
    if not_in_csv > 0:
        raise ValueError(f"{model.name}: {not_in_csv} ids are not in metadata CSV")


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def evaluate_top1_genre_accuracy(
    model: ModelData,
    id_to_genre: Dict[int, str],
    num_samples: int,
    seed: int,
) -> Tuple[float, int, int]:
    mask = np.array([int(tid) in id_to_genre for tid in model.ids], dtype=bool)
    ids = model.ids[mask]
    emb = model.emb[mask]

    n = len(ids)
    if n < 2:
        raise ValueError(f"{model.name}: not enough valid tracks for evaluation (n={n})")

    eval_n = n if num_samples <= 0 else min(num_samples, n)
    rng = np.random.default_rng(seed)
    query_idx = np.arange(n) if eval_n == n else rng.choice(n, size=eval_n, replace=False)

    emb = l2_normalize(emb)
    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)
    nn_idx = np.argmax(sim[query_idx], axis=1)

    query_genres = np.array([id_to_genre[int(ids[i])] for i in query_idx], dtype=object)
    nn_genres = np.array([id_to_genre[int(ids[i])] for i in nn_idx], dtype=object)
    correct = int(np.sum(query_genres == nn_genres))
    total = int(len(query_idx))
    acc = correct / total
    return acc, correct, total


def evaluate_rrf_top1_genre_accuracy(
    models: list,
    id_to_genre: Dict[int, str],
    num_samples: int,
    seed: int,
    rrf_k: int = 60,
) -> Tuple[float, int, int]:
    common_ids_set = set(int(x) for x in models[0].ids)
    for model in models[1:]:
        common_ids_set &= set(int(x) for x in model.ids)
    common_ids = sorted(list(common_ids_set))

    def align(model, common):
        id_map = {int(tid): i for i, tid in enumerate(model.ids)}
        indices = np.array([id_map[int(tid)] for tid in common])
        return l2_normalize(model.emb[indices])

    aligned_embs = [align(model, common_ids) for model in models]
    aligned_ids = np.array(common_ids, dtype=np.int64)

    mask = np.array([int(tid) in id_to_genre for tid in aligned_ids], dtype=bool)
    ids = aligned_ids[mask]
    embs = [emb[mask] for emb in aligned_embs]

    n = len(ids)
    if n < 2:
        raise ValueError("RRF: not enough valid tracks for evaluation")

    eval_n = n if num_samples <= 0 else min(num_samples, n)
    rng = np.random.default_rng(seed)
    query_idx = np.arange(n) if eval_n == n else rng.choice(n, size=eval_n, replace=False)

    K = 10
    rrf_top1 = []
    for qi in query_idx:
        rrf_scores: Dict[int, float] = {}
        for emb in embs:
            sim = emb @ emb.T
            sim[qi, qi] = -np.inf
            topk = np.argsort(sim[qi])[-K:][::-1]
            for rank, neighbor_idx in enumerate(topk):
                neighbor_idx = int(neighbor_idx)
                if neighbor_idx not in rrf_scores:
                    rrf_scores[neighbor_idx] = 0.0
                rrf_scores[neighbor_idx] += 1.0 / (rrf_k + rank + 1)

        nn_idx = max(rrf_scores.items(), key=lambda x: x[1])[0] if rrf_scores else -1
        rrf_top1.append(nn_idx)

    rrf_top1 = np.array(rrf_top1)
    valid_mask = rrf_top1 >= 0

    query_genres = np.array([id_to_genre[int(ids[i])] for i in query_idx], dtype=object)
    nn_genres = np.array([id_to_genre[int(ids[i])] for i in rrf_top1[valid_mask]], dtype=object)

    correct = int(np.sum(query_genres[valid_mask] == nn_genres))
    total = int(len(query_idx))
    acc = correct / total if total > 0 else 0.0
    return acc, correct, total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate embeddings and evaluate top-1 genre retrieval accuracy."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root folder (default: current directory).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of query tracks to evaluate per model; <=0 means all tracks.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    args = parser.parse_args()

    root = args.root

    # Use unified metadata path from config
    import sys
    sys.path.insert(0, str(root))
    from src.config import FMA_METADATA_DIR

    csv_path = FMA_METADATA_DIR / "tracks.csv"

    csv_ids, genres, id_to_genre = load_metadata(csv_path)
    csv_id_set = set(int(x) for x in csv_ids.tolist())

    print("=== Metadata Check ===")
    print(f"rows: {len(csv_ids)}")
    print(f"unique track ids: {len(np.unique(csv_ids))}")
    print(f"unique genres: {len(pd.unique(genres))}")

    # Use evaluation directory embeddings
    eval_dir = root / "evaluation"
    models = [
        load_clap_dict_embeddings(eval_dir / "CLAP" / "clap_audio_embeddings.npy", "CLAP(audio)"),
        load_id_embedding_pair(
            eval_dir / "CLAP" / "clap_text_embeddings_new.npy",
            eval_dir / "CLAP" / "clap_text_track_ids.npy",
            "CLAP(text,new)",
        ),
        load_id_embedding_pair(
            eval_dir / "OpenL3" / "openl3_embeddings.npy",
            eval_dir / "OpenL3" / "openl3_track_ids.npy",
            "OpenL3",
        ),
        load_id_embedding_pair(
            eval_dir / "SBERT" / "sbert_lyrics_embeddings.npy",
            eval_dir / "SBERT" / "sbert_lyrics_faiss.ids.npy",
            "SBERT(lyrics)",
        ),
    ]

    print("\n=== Embedding Format Check ===")
    for model in models:
        validate_model_data(model, csv_id_set)
        print(
            f"{model.name}: shape={model.emb.shape}, dtype={model.emb.dtype}, ids={model.ids.shape[0]}"
        )

    print("\n=== Top-1 Genre Retrieval Accuracy ===")
    print(f"num_samples={args.num_samples}, seed={args.seed}")
    for model in models:
        acc, correct, total = evaluate_top1_genre_accuracy(
            model=model,
            id_to_genre=id_to_genre,
            num_samples=args.num_samples,
            seed=args.seed,
        )
        print(f"{model.name}: accuracy={acc:.4f} ({correct}/{total})")

    acc_rrf, correct_rrf, total_rrf = evaluate_rrf_top1_genre_accuracy(
        models=models,
        id_to_genre=id_to_genre,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    print(f"RRF(all 4 models): accuracy={acc_rrf:.4f} ({correct_rrf}/{total_rrf})")


if __name__ == "__main__":
    main()
