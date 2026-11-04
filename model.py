"""MPNN architecture and feature normalisation shared by training and scoring.

These definitions preserve the original train_simple_mpnn.py implementation.
Changes to layer shapes, feature order or normalisation require retraining.
"""

import torch
import torch.nn as nn


class SimpleMPNNLayer(nn.Module):
    """
    One round of message passing.

    For each directed edge j -> i:
        m_ji = MLP([h_j, edge_attr_ji])

    Aggregate messages at node i by summation:
        m_i = sum_j m_ji

    Then update:
        h_i_new = MLP([h_i, m_i])

    This is deliberately simple and easy to inspect.
    """

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()

        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        h:         [N, H]
        edge_index:[2, E] where edge_index[0] = source, edge_index[1] = target
        edge_attr: [E, D]
        """
        src = edge_index[0]
        dst = edge_index[1]

        h_src = h[src]                              # [E, H]
        msg_input = torch.cat([h_src, edge_attr], dim=1)
        messages = self.msg_mlp(msg_input)         # [E, H]

        agg = torch.zeros_like(h)                  # [N, H]
        agg.index_add_(0, dst, messages)           # sum incoming messages to each target node

        update_input = torch.cat([h, agg], dim=1)
        h_new = self.update_mlp(update_input)

        # residual connection helps stability
        return h + h_new


class SimpleMPNN(nn.Module):
    def __init__(self, input_dim: int, edge_dim: int, hidden_dim: int = 64, num_classes: int = 20, num_layers: int = 1):
        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList([
            SimpleMPNNLayer(hidden_dim=hidden_dim, edge_dim=edge_dim)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.node_encoder(x)

        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)

        logits = self.classifier(h)
        return logits


def normalize_edge_attr(edge_attr: torch.Tensor) -> torch.Tensor:
    """
    Current edge_attr columns are:
      0: Cα distance
      1: same_chain (0 or 1)
      2: sequence separation if same chain, else -1

    We keep it simple:
      distance      -> divide by 10
      same_chain    -> leave as is
      seq separation-> clip to 50, map -1 to 0, then divide by 50
    """
    edge_attr = edge_attr.clone()

    dist = edge_attr[:, 0:1] / 10.0
    same_chain = edge_attr[:, 1:2]

    seq_sep = edge_attr[:, 2:3]
    seq_sep = torch.clamp(seq_sep, min=0.0, max=50.0) / 50.0

    return torch.cat([dist, same_chain, seq_sep], dim=1)


