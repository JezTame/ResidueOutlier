#
# score_residues.py
#
# Test a trained MPNN model on:
#   (1) a single PDB file, or
#   (2) a preprocessed .pt graph dataset
#
# For a single PDB file, the script rebuilds the graph using structure_graph.py
# and then scores residues by:
#     score = -log P(observed amino acid)
#

import argparse
import os

import torch
import torch.nn.functional as F
from Bio.PDB import PDBParser
from structure_graph import build_graph_for_model

AA_ORDER = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL"
]


def load_graphs_from_pdb(pdb_file: str, buffer: int = 5, cutoff: float = 10.0):
    """
    Build one graph from the first model of a PDB structure.
    """
    parser = PDBParser(QUIET=True)
    pdb_name = os.path.basename(pdb_file)
    structure = parser.get_structure(pdb_name, pdb_file)

    graphs = []
    for model in structure:
        graph = build_graph_for_model(
            model=model,
            pdb_name=pdb_name,
            min_terminal_buffer=buffer,
            ca_cutoff=cutoff,
        )
        if graph is not None:
            graphs.append(graph)
        break   # only first model

    return graphs


def load_graphs(input_file: str, input_type: str, buffer: int = 5, cutoff: float = 10.0):
    if input_type == "pt":
        graphs = torch.load(input_file)
    elif input_type == "pdb":
        graphs = load_graphs_from_pdb(input_file, buffer=buffer, cutoff=cutoff)
    else:
        raise ValueError(f"Unknown input_type: {input_type}")

    if len(graphs) == 0:
        raise ValueError("No valid graphs found in input")

    return graphs


def main():
    parser = argparse.ArgumentParser(description="Score residues with a trained MPNN model")
    parser.add_argument("model_file", help="Trained MPNN checkpoint (.pt)")
    parser.add_argument("input_file", help="Single PDB file or preprocessed graph dataset (.pt)")
    parser.add_argument(
        "--input_type",
        choices=["pdb", "pt"],
        default="pdb",
        help="Type of input_file (default: pdb)"
    )
    parser.add_argument(
        "--buffer",
        type=int,
        default=5,
        help="Terminal buffer used when building graph from pdb (default: 5)"
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=10.0,
        help="C-alpha cutoff used when building graph from pdb (default: 10.0)"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=30,
        help="Number of suspicious residues to print (default: 30)"
    )
    args = parser.parse_args()

    model_data = torch.load(args.model_file)
    graphs = load_graphs(
        args.input_file,
        input_type=args.input_type,
        buffer=args.buffer,
        cutoff=args.cutoff,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # rebuild model
    from model import SimpleMPNN, normalize_edge_attr

    model = SimpleMPNN(
        input_dim=model_data["input_dim"],
        edge_dim=model_data["edge_dim"],
        hidden_dim=model_data["hidden_dim"],
        num_classes=model_data["num_classes"],
        num_layers=model_data["num_layers"],
    )

    model.load_state_dict(model_data["model_state_dict"])
    model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for g in graphs:
            x = g["x"].to(device)
            y = g["y"]
            edge_index = g["edge_index"].to(device)
            edge_attr = normalize_edge_attr(g["edge_attr"]).to(device)

            logits = model(x, edge_index, edge_attr)
            probs = F.softmax(logits, dim=1).cpu()

            for i in range(x.shape[0]):
                obs_idx = int(y[i])
                obs_aa = AA_ORDER[obs_idx]

                p_obs = probs[i, obs_idx].item()

                top3 = torch.topk(probs[i], k=3)
                top3_idx = top3.indices.tolist()
                top3_probs = top3.values.tolist()

                top3_str = " ".join(
                    f"{AA_ORDER[j]}:{p:.2f}" for j, p in zip(top3_idx, top3_probs)
                )

                pred_idx = top3_idx[0]
                pred_aa = AA_ORDER[pred_idx]

                score = -torch.log(torch.tensor(p_obs + 1e-8)).item()

                results.append((
                    score,
                    g["pdb_name"],
                    g["chain_ids"][i],
                    g["residue_ids"][i],
                    obs_aa,
                    pred_aa,
                    p_obs,
                    top3_str
                ))

    results.sort(reverse=True)

    print("\nTop suspicious residues:\n")
    print("PDB                Chain Resid  Obs  Pred  ProbObs   Score   Top3")

    for score, pdb_name, chain_id, resid, obs_aa, pred_aa, p_obs, top3_str in results[:args.top_n]:
        print(
            f"{pdb_name:<18} {chain_id:>5} {resid:>5d}   "
            f"{obs_aa}    {pred_aa}   {p_obs:7.4f}  {score:7.3f}  {top3_str}"
        )


if __name__ == "__main__":
    main()

