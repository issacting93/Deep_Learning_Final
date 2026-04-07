"""
train.py — Role 3: Graph-based Recommendation (Issac)

Training loop for the heterogeneous GNN on the FMA graph.

Task: Link prediction on track→genre edges.
  - Split track→genre edges into train (80%) / val (10%) / test (10%)
  - At each step: run GNN forward pass, score positive + negative edges,
    compute binary cross-entropy loss, backpropagate
  - Save best checkpoint (lowest val loss) to models/gnn_checkpoint.pt

After training:
  - Extract final track node embeddings
  - Save to data/processed/gnn_embeddings.npy + gnn_track_ids.npy
  - Build FAISS index → data/processed/gnn_faiss.index

Run:
    python -m src.graph.train
    python -m src.graph.train --epochs 5 --lr 0.005   (quick test)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.transforms import RandomLinkSplit

from src.config import PROCESSED_DIR, MODELS_DIR, DEVICE
from src.graph.build_graph import build_hetero_graph
from src.graph.gnn_model import HeteroGNN, LinkPredictor
from src.indexing.faiss_index import FaissIndex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def negative_sample(pos_edge_index: torch.Tensor, num_tracks: int, num_genres: int, num_neg: int) -> torch.Tensor:
    """
    Sample random (track, genre) pairs that are NOT in pos_edge_index.
    Simple uniform sampling — fast and good enough for a small graph.
    """
    pos_set = set(zip(pos_edge_index[0].tolist(), pos_edge_index[1].tolist()))
    neg_src, neg_dst = [], []
    attempts = 0
    while len(neg_src) < num_neg and attempts < num_neg * 10:
        s = torch.randint(0, num_tracks, (1,)).item()
        d = torch.randint(0, num_genres, (1,)).item()
        if (s, d) not in pos_set:
            neg_src.append(s)
            neg_dst.append(d)
        attempts += 1
    return torch.tensor([neg_src, neg_dst], dtype=torch.long)


def compute_loss(
    track_embs: torch.Tensor,
    genre_embs: torch.Tensor,
    pos_edges: torch.Tensor,
    neg_edges: torch.Tensor,
    predictor: LinkPredictor,
) -> torch.Tensor:
    """Binary cross-entropy loss on positive + negative edge scores."""
    pos_scores = predictor(track_embs, genre_embs, pos_edges)
    neg_scores = predictor(track_embs, genre_embs, neg_edges)

    pos_labels = torch.ones(pos_scores.size(0), device=pos_scores.device)
    neg_labels = torch.zeros(neg_scores.size(0), device=neg_scores.device)

    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([pos_labels, neg_labels])
    return F.binary_cross_entropy_with_logits(scores, labels)


@torch.no_grad()
def evaluate(
    model: HeteroGNN,
    predictor: LinkPredictor,
    data,
    pos_edges: torch.Tensor,
    device: str,
) -> float:
    """Compute val/test loss (no gradient)."""
    model.eval()
    predictor.eval()
    x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}

    out = model(x_dict, edge_index_dict)
    track_embs = out["track"]
    genre_embs = out["genre"]

    num_neg = pos_edges.size(1)
    neg_edges = negative_sample(pos_edges, track_embs.size(0), genre_embs.size(0), num_neg)
    neg_edges = neg_edges.to(device)

    loss = compute_loss(track_embs, genre_embs, pos_edges.to(device), neg_edges, predictor)
    return loss.item()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    epochs: int = 30,
    lr: float = 1e-3,
    hidden_dim: int = 256,
    out_dim: int = 256,
    dropout: float = 0.3,
    neg_ratio: int = 1,        # number of negatives per positive
    patience: int = 7,         # early stopping: epochs without val improvement
    save_dir: Path = MODELS_DIR,
    verbose: bool = True,
):
    device = DEVICE
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "gnn_checkpoint.pt"

    # ------------------------------------------------------------------
    # 1. Build graph
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Building heterogeneous graph...")
    data, id_maps = build_hetero_graph(verbose=verbose)

    # ------------------------------------------------------------------
    # 2. Split track→genre edges for link prediction
    # ------------------------------------------------------------------
    # We split only the ("track", "tagged", "genre") edge type.
    # 80% train / 10% val / 10% test
    all_tg_edges = data["track", "tagged", "genre"].edge_index  # [2, E]
    E = all_tg_edges.size(1)
    perm = torch.randperm(E)

    n_train = int(0.8 * E)
    n_val   = int(0.1 * E)

    train_edges = all_tg_edges[:, perm[:n_train]]
    val_edges   = all_tg_edges[:, perm[n_train:n_train + n_val]]
    test_edges  = all_tg_edges[:, perm[n_train + n_val:]]

    print(f"\nEdge split — train: {train_edges.size(1)}, val: {val_edges.size(1)}, test: {test_edges.size(1)}")

    # Move full graph data to device
    data = data.to(device)

    num_tracks = data["track"].num_nodes
    num_genres = data["genre"].num_nodes

    # ------------------------------------------------------------------
    # 3. Initialise model + optimiser
    # ------------------------------------------------------------------
    track_in_dim  = data["track"].x.size(1)
    artist_in_dim = data["artist"].x.size(1)
    genre_in_dim  = data["genre"].x.size(1)

    model = HeteroGNN(
        track_in_dim=track_in_dim,
        artist_in_dim=artist_in_dim,
        genre_in_dim=genre_in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        dropout=dropout,
    ).to(device)

    predictor = LinkPredictor().to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=lr,
        weight_decay=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    print(f"Device: {device}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ---- Forward ----
        model.train()
        predictor.train()
        optimizer.zero_grad()

        out = model(data.x_dict, data.edge_index_dict)
        track_embs = out["track"]
        genre_embs = out["genre"]

        # Sample negatives for this epoch
        num_neg = train_edges.size(1) * neg_ratio
        neg_edges = negative_sample(train_edges, num_tracks, num_genres, num_neg).to(device)

        loss = compute_loss(track_embs, genre_embs, train_edges.to(device), neg_edges, predictor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss = loss.item()

        # ---- Validation ----
        val_loss = evaluate(model, predictor, data, val_edges, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - t0
        if verbose or epoch % 5 == 0:
            print(f"Epoch {epoch:3d}/{epochs} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | {elapsed:.1f}s")

        # ---- Checkpoint ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "predictor_state_dict": predictor.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "out_dim": out_dim,
                "hidden_dim": hidden_dim,
                "track_in_dim": track_in_dim,
                "artist_in_dim": artist_in_dim,
                "genre_in_dim": genre_in_dim,
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved (val_loss: {val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"\nBest val_loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")

    # ------------------------------------------------------------------
    # 5. Evaluate on test set (load best checkpoint)
    # ------------------------------------------------------------------
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss = evaluate(model, predictor, data, test_edges, device)
    print(f"Test loss: {test_loss:.4f}")

    # ------------------------------------------------------------------
    # 6. Extract + save track embeddings
    # ------------------------------------------------------------------
    print("\nExtracting final track embeddings...")
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        gnn_embs = out["track"].cpu().numpy().astype(np.float32)

    track_ids_out = id_maps["idx_to_track"]  # list of FMA track IDs (position = node index)

    np.save(PROCESSED_DIR / "gnn_embeddings.npy",  gnn_embs)
    np.save(PROCESSED_DIR / "gnn_track_ids.npy",   np.array(track_ids_out))
    print(f"  Saved gnn_embeddings.npy  shape: {gnn_embs.shape}")
    print(f"  Saved gnn_track_ids.npy   len:   {len(track_ids_out)}")

    # ------------------------------------------------------------------
    # 7. Build FAISS index
    # ------------------------------------------------------------------
    print("\nBuilding FAISS index...")
    faiss_idx = FaissIndex(dimension=out_dim, metric="cosine")
    faiss_idx.build(gnn_embs, track_ids_out)
    faiss_idx.save(PROCESSED_DIR / "gnn_faiss.index")
    print(f"  Saved gnn_faiss.index")

    print("\nDone! ✓")
    return model, gnn_embs, track_ids_out, history


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN on FMA graph")
    parser.add_argument("--epochs",     type=int,   default=30,   help="Max training epochs")
    parser.add_argument("--lr",         type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int,   default=256,  help="GNN hidden dimension")
    parser.add_argument("--out_dim",    type=int,   default=256,  help="Output embedding dimension")
    parser.add_argument("--dropout",    type=float, default=0.3,  help="Dropout rate")
    parser.add_argument("--patience",   type=int,   default=7,    help="Early stopping patience")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        dropout=args.dropout,
        patience=args.patience,
    )
