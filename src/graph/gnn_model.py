"""
gnn_model.py — Role 3: Graph-based Recommendation (Issac)

Defines the GNN architecture using PyTorch Geometric's HeteroConv with
SAGEConv layers. The model processes the heterogeneous graph (tracks,
artists, genres) and produces a 256-dim embedding for every track node.

Architecture overview:
  Input:  track (512-dim CLAP), artist (512-dim), genre (163-dim one-hot)
  Layer 1: HeteroConv — project each node type to hidden_dim (256)
  Layer 2: HeteroConv — refine embeddings with 2-hop neighbourhood info
  Output: track node embeddings (out_dim, default 256)

The embeddings capture structural position in the graph (who made the
track, what genre it belongs to, which other tracks share those connections)
rather than raw audio content.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear


class HeteroGNN(nn.Module):
    """
    Two-layer heterogeneous Graph Neural Network.

    Each layer uses SAGEConv (GraphSAGE) per edge type, aggregated via
    HeteroConv. After each layer we apply LayerNorm + ReLU + Dropout.

    The final output is the track node embedding matrix, shape [N_tracks, out_dim].
    """

    def __init__(
        self,
        track_in_dim: int   = 512,   # CLAP embedding dim
        artist_in_dim: int  = 512,   # mean-pooled CLAP
        genre_in_dim: int   = 163,   # one-hot
        hidden_dim: int     = 256,
        out_dim: int        = 256,
        dropout: float      = 0.3,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout

        # Linear projections to bring all node types to hidden_dim
        # before the conv layers (avoids dimension mismatch in SAGEConv)
        self.proj_track  = Linear(track_in_dim,  hidden_dim)
        self.proj_artist = Linear(artist_in_dim, hidden_dim)
        self.proj_genre  = Linear(genre_in_dim,  hidden_dim)

        # Layer norms per node type (applied after each conv)
        self.norm1 = nn.ModuleDict({
            "track":  nn.LayerNorm(hidden_dim),
            "artist": nn.LayerNorm(hidden_dim),
            "genre":  nn.LayerNorm(hidden_dim),
        })
        self.norm2 = nn.ModuleDict({
            "track":  nn.LayerNorm(out_dim),
            "artist": nn.LayerNorm(out_dim),
            "genre":  nn.LayerNorm(out_dim),
        })

        # ------ Conv Layer 1: hidden_dim → hidden_dim ------
        # One SAGEConv per edge type. HeteroConv sums contributions
        # from all edge types that point to the same destination node type.
        self.conv1 = HeteroConv(
            {
                ("track",  "made_by",  "artist"): SAGEConv(hidden_dim, hidden_dim),
                ("artist", "rev_made_by", "track"): SAGEConv(hidden_dim, hidden_dim),
                ("track",  "tagged",   "genre"):  SAGEConv(hidden_dim, hidden_dim),
                ("genre",  "rev_tagged", "track"): SAGEConv(hidden_dim, hidden_dim),
                ("artist", "plays",    "genre"):  SAGEConv(hidden_dim, hidden_dim),
                ("genre",  "rev_plays", "artist"): SAGEConv(hidden_dim, hidden_dim),
                ("track",  "co_genre", "track"):  SAGEConv(hidden_dim, hidden_dim),
            },
            aggr="sum",
        )

        # ------ Conv Layer 2: hidden_dim → out_dim ------
        self.conv2 = HeteroConv(
            {
                ("track",  "made_by",  "artist"): SAGEConv(hidden_dim, out_dim),
                ("artist", "rev_made_by", "track"): SAGEConv(hidden_dim, out_dim),
                ("track",  "tagged",   "genre"):  SAGEConv(hidden_dim, out_dim),
                ("genre",  "rev_tagged", "track"): SAGEConv(hidden_dim, out_dim),
                ("artist", "plays",    "genre"):  SAGEConv(hidden_dim, out_dim),
                ("genre",  "rev_plays", "artist"): SAGEConv(hidden_dim, out_dim),
                ("track",  "co_genre", "track"):  SAGEConv(hidden_dim, out_dim),
            },
            aggr="sum",
        )

        # Final MLP on track embeddings (optional refinement head)
        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        """
        Full forward pass.

        Args:
            x_dict         — dict of node feature tensors keyed by node type
            edge_index_dict — dict of edge index tensors keyed by edge type tuple

        Returns:
            x_dict — updated feature tensors after 2 conv layers
                     (access track embeddings via x_dict["track"])
        """
        # Project all node types to hidden_dim
        x_dict = {
            "track":  F.relu(self.proj_track(x_dict["track"])),
            "artist": F.relu(self.proj_artist(x_dict["artist"])),
            "genre":  F.relu(self.proj_genre(x_dict["genre"])),
        }

        # Conv layer 1
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {
            node: self.norm1[node](F.relu(F.dropout(feat, p=self.dropout, training=self.training)))
            for node, feat in x_dict.items()
        }

        # Conv layer 2
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {
            node: self.norm2[node](F.relu(F.dropout(feat, p=self.dropout, training=self.training)))
            for node, feat in x_dict.items()
        }

        # Refine track embeddings through head MLP
        x_dict["track"] = self.head(x_dict["track"])

        return x_dict

    def get_track_embeddings(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        """Convenience: run forward and return only track embeddings."""
        out = self.forward(x_dict, edge_index_dict)
        return out["track"]


# ---------------------------------------------------------------------------
# Link prediction decoder (used during training)
# ---------------------------------------------------------------------------

class LinkPredictor(nn.Module):
    """
    Dot-product link predictor for (track, genre) edge prediction.

    During training:
      - Positive edges: real track→genre edges from the graph
      - Negative edges: randomly sampled (track, genre) pairs NOT in the graph

    Score = dot product of track embedding and genre embedding.
    Loss  = binary cross-entropy.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        track_embs: torch.Tensor,   # [N_tracks, out_dim]
        genre_embs: torch.Tensor,   # [N_genres, out_dim]
        edge_label_index: torch.Tensor,  # [2, num_edges] — first row=track idx, second=genre idx
    ) -> torch.Tensor:
        """Returns raw logits (before sigmoid) for each edge."""
        src = track_embs[edge_label_index[0]]   # [E, out_dim]
        dst = genre_embs[edge_label_index[1]]   # [E, out_dim]
        return (src * dst).sum(dim=-1)          # [E] — dot product


if __name__ == "__main__":
    # Quick architecture test (no real data needed)
    model = HeteroGNN()
    print(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total:,}")
