"""Train the residue-level message-passing network from a graph dataset.

The training loop and evaluation procedures are preserved from the original.
"""

import argparse
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import SimpleMPNN, normalize_edge_attr


def split_graphs(graphs: List[dict], val_fraction: float = 0.2, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    rng = random.Random(seed)
    indices = list(range(len(graphs)))
    rng.shuffle(indices)

    n_val = max(1, int(len(indices) * val_fraction))
    val_idx = set(indices[:n_val])

    train_graphs = [graphs[i] for i in range(len(graphs)) if i not in val_idx]
    val_graphs = [graphs[i] for i in range(len(graphs)) if i in val_idx]
    return train_graphs, val_graphs


def topk_accuracy(logits: torch.Tensor, y: torch.Tensor, k: int = 3) -> float:
    topk = torch.topk(logits, k=k, dim=1).indices
    correct = (topk == y.unsqueeze(1)).any(dim=1).sum().item()
    return correct / y.shape[0]


def evaluate(model, graphs, criterion, device):
    model.eval()

    total_loss = 0.0
    total_nodes = 0
    total_correct = 0
    total_top3 = 0

    with torch.no_grad():
        for g in graphs:
            x = g["x"].to(device)
            y = g["y"].to(device)
            edge_index = g["edge_index"].to(device)
            edge_attr = normalize_edge_attr(g["edge_attr"]).to(device)

            logits = model(x, edge_index, edge_attr)
            loss = criterion(logits, y)

            total_loss += loss.item() * y.shape[0]
            total_nodes += y.shape[0]
            total_correct += (torch.argmax(logits, dim=1) == y).sum().item()
            total_top3 += (torch.topk(logits, k=3, dim=1).indices == y.unsqueeze(1)).any(dim=1).sum().item()

    return (
        total_loss / total_nodes,
        total_correct / total_nodes,
        total_top3 / total_nodes,
    )


def main():
    parser = argparse.ArgumentParser(description="Train simple MPNN on graph dataset")
    parser.add_argument("data_pt", type=str, help="Input graph dataset (.pt)")
    parser.add_argument("model_out", type=str, help="Output trained model file (.pt)")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of message-passing layers")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val_fraction", type=float, default=0.2, help="Fraction of graphs for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading dataset: {args.data_pt}")
    graphs = torch.load(args.data_pt)

    if len(graphs) == 0:
        raise ValueError("Dataset is empty")

    input_dim = graphs[0]["x"].shape[1]
    edge_dim = graphs[0]["edge_attr"].shape[1]
    num_classes = int(max(int(g["y"].max().item()) for g in graphs) + 1)

    print(f"Detected input_dim   = {input_dim}")
    print(f"Detected edge_dim    = {edge_dim}")
    print(f"Detected num_classes = {num_classes}")
    print(f"Number of graphs     = {len(graphs)}")

    train_graphs, val_graphs = split_graphs(
        graphs, val_fraction=args.val_fraction, seed=args.seed
    )

    train_nodes = sum(g["x"].shape[0] for g in train_graphs)
    val_nodes = sum(g["x"].shape[0] for g in val_graphs)

    print(f"Training nodes   = {train_nodes}")
    print(f"Validation nodes = {val_nodes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleMPNN(
        input_dim=input_dim,
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_layers=args.num_layers,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_nodes = 0
        total_correct = 0
        total_top3 = 0

        # graph-by-graph training keeps this easy to understand
        random.shuffle(train_graphs)

        for g in train_graphs:
            x = g["x"].to(device)
            y = g["y"].to(device)
            edge_index = g["edge_index"].to(device)
            edge_attr = normalize_edge_attr(g["edge_attr"]).to(device)

            optimizer.zero_grad()
            logits = model(x, edge_index, edge_attr)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.shape[0]
            total_nodes += y.shape[0]
            total_correct += (torch.argmax(logits, dim=1) == y).sum().item()
            total_top3 += (torch.topk(logits, k=3, dim=1).indices == y.unsqueeze(1)).any(dim=1).sum().item()

        train_loss = total_loss / total_nodes
        train_acc = total_correct / total_nodes
        train_top3 = total_top3 / total_nodes

        val_loss, val_acc, val_top3 = evaluate(model, val_graphs, criterion, device)

        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_top3={train_top3:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_top3={val_top3:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "edge_dim": edge_dim,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "num_classes": num_classes,
                "feature_names": graphs[0].get("feature_names", None),
                "edge_feature_names": graphs[0].get("edge_feature_names", None),
                "best_val_acc": best_val_acc,
                "data_file": args.data_pt,
            }

    torch.save(best_state, args.model_out)
    print(f"Saved best model to {args.model_out}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()


