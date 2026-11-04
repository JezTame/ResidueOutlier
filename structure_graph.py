import math
import os
import argparse
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.PDB.vectors import calc_dihedral


AA_ORDER = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
]
AA_TO_INDEX = {aa: i for i, aa in enumerate(AA_ORDER)}

# -----------------------------
# Geometry / counting settings
# -----------------------------

HEAVY_COUNT_RADII = [5.0, 7.0, 9.0]
ATOMTYPE_RADIUS = 6.0
WATER_RADIUS = 8.0
HBOND_MAX_RADIUS = 8.0

HBOND_FULL = 2.8
HBOND_ZERO = 3.5
CHARGE_MULTIPLIER = 1.5


def residue_number(residue) -> int:
    return residue.id[1]


def insertion_code(residue) -> str:
    icode = residue.id[2]
    return icode.strip() if isinstance(icode, str) else ""


def is_consecutive(res1, res2) -> bool:
    """
    Simple continuity check for standard residues.
    Returns True only if residue numbers are consecutive and
    insertion codes are empty.
    """
    if insertion_code(res1) != "" or insertion_code(res2) != "":
        return False
    return residue_number(res2) == residue_number(res1) + 1


def angle_to_sincos(angle_rad: float) -> List[float]:
    return [math.sin(angle_rad), math.cos(angle_rad)]


def dihedral_from_atoms(a1, a2, a3, a4) -> float:
    return float(calc_dihedral(
        a1.get_vector(),
        a2.get_vector(),
        a3.get_vector(),
        a4.get_vector(),
    ))


def safe_resname_to_index(resname: str) -> Optional[int]:
    return AA_TO_INDEX.get(resname.upper(), None)


def is_standard_residue(residue) -> bool:
    if residue.id[0] != " ":
        return False
    return residue.get_resname().upper() in AA_TO_INDEX


def is_water_residue(residue) -> bool:
    if residue.id[0] == "W":
        return True
    return residue.get_resname().upper() in {"HOH", "WAT"}


def has_atoms(residue, atom_names: List[str]) -> bool:
    return all(name in residue for name in atom_names)


def extract_chain_residues(chain) -> List:
    """
    Keep only standard amino-acid residues with N, CA, C present.
    """
    residues = []
    for residue in chain:
        if not is_standard_residue(residue):
            continue
        if not has_atoms(residue, ["N", "CA", "C"]):
            continue
        residues.append(residue)
    return residues


def atom_element(atom) -> str:
    elem = atom.element.strip().upper() if hasattr(atom, "element") else ""
    if elem == "":
        # fallback from atom name
        name = atom.get_name().strip().upper()
        elem = name[0]
    return elem


def is_heavy_atom(atom) -> bool:
    return atom_element(atom) != "H"


def is_backbone_atom_name(atom_name: str) -> bool:
    return atom_name.upper() in {"N", "CA", "C", "O", "OXT"}


def is_potentially_charged_atom(resname: str, atom_name: str) -> bool:
    """
    Very simple charged-atom proxy set.
    """
    resname = resname.upper()
    atom_name = atom_name.upper()

    positive = {
        ("LYS", "NZ"),
        ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"),
        ("HIS", "ND1"), ("HIS", "NE2"),
    }

    negative = {
        ("ASP", "OD1"), ("ASP", "OD2"),
        ("GLU", "OE1"), ("GLU", "OE2"),
    }

    return (resname, atom_name) in positive or (resname, atom_name) in negative


def compute_backbone_angles(residues: List, i: int) -> Optional[Dict[str, float]]:
    if i < 1 or i >= len(residues) - 1:
        return None

    prev_res = residues[i - 1]
    res = residues[i]
    next_res = residues[i + 1]

    if not is_consecutive(prev_res, res):
        return None
    if not is_consecutive(res, next_res):
        return None

    try:
        phi = dihedral_from_atoms(
            prev_res["C"], res["N"], res["CA"], res["C"]
        )
        psi = dihedral_from_atoms(
            res["N"], res["CA"], res["C"], next_res["N"]
        )
        omega = dihedral_from_atoms(
            prev_res["CA"], prev_res["C"], res["N"], res["CA"]
        )
    except KeyError:
        return None

    return {"phi": phi, "psi": psi, "omega": omega}


def compute_ca_torsion(residues: List, i: int) -> Optional[float]:
    """
    Single CA-only torsion:
      tau = dihedral(CA[i-1], CA[i], CA[i+1], CA[i+2])

    Residue i is in position 2 of the 4-point dihedral.
    """
    if i < 1 or i > len(residues) - 3:
        return None

    r_im1 = residues[i - 1]
    r_i   = residues[i]
    r_ip1 = residues[i + 1]
    r_ip2 = residues[i + 2]

    if not (
        is_consecutive(r_im1, r_i) and
        is_consecutive(r_i, r_ip1) and
        is_consecutive(r_ip1, r_ip2)
    ):
        return None

    try:
        tau = dihedral_from_atoms(
            r_im1["CA"], r_i["CA"], r_ip1["CA"], r_ip2["CA"]
        )
    except KeyError:
        return None

    return tau


def unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (v / norm).astype(np.float32)

# =========================
# Virtual probe point
# Halfway from CA toward virtual Cbeta direction
# =========================
def virtual_point_and_axis(residue) -> Tuple[np.ndarray, np.ndarray]:
    ca = residue["CA"].coord.astype(np.float32)
    n = residue["N"].coord.astype(np.float32)
    c = residue["C"].coord.astype(np.float32)

    b = unit(n - ca)
    cvec = unit(c - ca)

    direction = unit(-b - cvec + np.cross(b, cvec))

    # very defensive fallback
    if np.linalg.norm(direction) < 1e-8:
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    virtual_atom = ca + 0.765 * direction
    return virtual_atom, direction


def build_model_atom_records(model) -> List[Dict]:
    """
    Collect non-hydrogen atoms from all standard residues and water oxygens.

    Each atom record stores:
      - atom coordinate and type
      - parent residue information
      - parent residue CA coordinate (for cheap residue-level prefilter)
      - flags that may be useful later for chemistry features
    """
    atom_records = []

    hydrophobic_resnames = {
        "ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TYR", "TRP", "PRO"
    }
    aromatic_resnames = {
        "PHE", "TYR", "TRP", "HIS"
    }

    for chain in model:
        for residue in chain:
            if is_standard_residue(residue):
                parent_ca_coord = None
                if "CA" in residue:
                    parent_ca_coord = residue["CA"].coord.astype(np.float32)

                resname = residue.get_resname().upper()

                for atom in residue:
                    if not is_heavy_atom(atom):
                        continue

                    atom_name = atom.get_name().upper()
                    elem = atom_element(atom)
                    is_backbone = is_backbone_atom_name(atom_name)

                    atom_records.append({
                        "coord": atom.coord.astype(np.float32),
                        "element": elem,
                        "atom_name": atom_name,
                        "resname": resname,
                        "residue_obj": residue,
                        "parent_ca_coord": parent_ca_coord,
                        "is_water": False,
                        "is_backbone": is_backbone,
                        "is_hydrophobic": (resname in hydrophobic_resnames) and (not is_backbone),
                        "is_aromatic": (resname in aromatic_resnames) and (not is_backbone),
                    })

            elif is_water_residue(residue):
                for atom in residue:
                    if atom_element(atom) != "O":
                        continue

                    atom_records.append({
                        "coord": atom.coord.astype(np.float32),
                        "element": "O",
                        "atom_name": atom.get_name().upper(),
                        "resname": residue.get_resname().upper(),
                        "residue_obj": residue,
                        "parent_ca_coord": None,
                        "is_water": True,
                        "is_backbone": False,
                        "is_hydrophobic": False,
                        "is_aromatic": False,
                    })

    return atom_records


def collect_hemisphere_atoms(
    residue,
    atom_records: List[Dict],
    center: np.ndarray,
    axis: np.ndarray,
    radius_max: float,
    parent_ca_cutoff: float = 20.0,
) -> List[Dict]:
    """
    Return atoms from other residues that lie within a forward hemisphere
    centered at 'center' with axis 'axis' and radius 'radius_max'.
    Records have the form of a dictionary

    A cheap residue-level prefilter is applied first:
      if the parent residue CA is more than parent_ca_cutoff away from the
      central residue CA, skip all atoms from that residue.
    """
    selected = []
    central_ca = residue["CA"].coord.astype(np.float32)

    for rec in atom_records:
        # exclude same residue atoms
        if rec["residue_obj"] is residue:
            continue

        # cheap residue-level prefilter using parent CA
        parent_ca = rec.get("parent_ca_coord", None)
        if parent_ca is not None:
            ca_dist = np.linalg.norm(parent_ca - central_ca)
            if ca_dist > parent_ca_cutoff:
                continue

        vec = rec["coord"] - center
        dist = np.linalg.norm(vec)
        if dist > radius_max:
            continue

        # hemisphere test
        if np.dot(vec, axis) < 0.0:
            continue

        selected.append({
            **rec,
            "dist": float(dist),
        })

    return selected


def weighted_hbond_score(distance: float) -> float:
    """
    1.0 below HBOND_FULL, then linear to 0.0 at HBOND_ZERO.
    """
    if distance <= HBOND_FULL:
        return 1.0
    if distance >= HBOND_ZERO:
        return 0.0
    return (HBOND_ZERO - distance) / (HBOND_ZERO - HBOND_FULL)


def central_sidechain_hbond_atoms(residue) -> List[Dict]:
    """
    Central residue atoms that may participate in side-chain hydrogen bonding.
    Keep only side-chain N/O atoms (exclude backbone atoms).
    """
    atoms = []
    for atom in residue:
        atom_name = atom.get_name().upper()
        elem = atom_element(atom)
        if is_backbone_atom_name(atom_name):
            continue
        if elem not in {"N", "O"}:
            continue
        atoms.append({
            "coord": atom.coord.astype(np.float32),
            "element": elem,
            "atom_name": atom_name,
            "resname": residue.get_resname().upper(),
        })
    return atoms


def linear_shell_weight(dist: float, cutoff: float) -> float:
    if dist <= cutoff - 1.0:
        return 1.0
    if dist >= cutoff:
        return 0.0
    return cutoff - dist


def compute_local_chemistry_features(residue, atom_records: List[Dict]) -> List[float]:
    """
    Build compact local chemistry features from one shared hemisphere atom list.

    Current features:
      heavy_weight_5A
      heavy_weight_7A
      heavy_weight_9A
      oxygen_count_6A
      nitrogen_count_6A
      water_count_8A
      aromatic_weight_7A,
      hydrophobic_weight_7A,
      hbond_weight_sum
    """
    center, axis = virtual_point_and_axis(residue)
    local_atoms = collect_hemisphere_atoms(
        residue=residue,
        atom_records=atom_records,
        center=center,
        axis=axis,
        radius_max=max(HBOND_MAX_RADIUS, max(HEAVY_COUNT_RADII)),
    )

    # Heavy atom counts at multiple radii
    heavy_counts = []
    for cutoff in HEAVY_COUNT_RADII:
        weightsum = 0.0
        for rec in local_atoms:
            if rec["is_water"]:
                continue
            weightsum += linear_shell_weight(rec["dist"], cutoff)
        heavy_counts.append(float(weightsum))

    oxygen_count = float(sum(
        1.0 for rec in local_atoms
        if rec["dist"] <= ATOMTYPE_RADIUS and rec["element"] == "O" and not rec["is_water"]
    ))

    nitrogen_count = float(sum(
        1.0 for rec in local_atoms
        if rec["dist"] <= ATOMTYPE_RADIUS and rec["element"] == "N" 
    ))

    water_count = float(sum(
        1.0 for rec in local_atoms
        if rec["dist"] <= WATER_RADIUS and rec["is_water"]
    ))

#####  hard-wired 7 Angstrom limit for distance of aromatic atoms
    aromatic_weight = 0.0
    for rec in local_atoms:
        if rec["is_aromatic"]:
            aromatic_weight += linear_shell_weight(rec["dist"], 7.0)

    hydrophobic_weight = 0.0
    for rec in local_atoms:
        if rec["is_hydrophobic"]:
            hydrophobic_weight += linear_shell_weight(rec["dist"], 7.0)
    
    # Hydrogen-bond proxy between central side-chain N/O atoms
    # and nearby N/O atoms in the hemisphere.
    center_atoms = central_sidechain_hbond_atoms(residue)
    hbond_weight_sum = 0.0

    for c_atom in center_atoms:
        elem1 = c_atom["element"]
        if elem1 not in {"N", "O"}:
                continue
        c_is_charged = is_potentially_charged_atom(
            c_atom["resname"], c_atom["atom_name"]
        )

        for rec in local_atoms:
            if rec["dist"] > HBOND_MAX_RADIUS:
                continue
            if rec["element"] not in {"N", "O"}:
                continue

            pair_dist = float(np.linalg.norm(rec["coord"] - c_atom["coord"]))
            base = weighted_hbond_score(pair_dist)
            if base <= 0.0:
                continue

            weight = base
            if c_is_charged:
                weight *= CHARGE_MULTIPLIER
            if is_potentially_charged_atom(rec["resname"], rec["atom_name"]):
                weight *= CHARGE_MULTIPLIER

#           weight down hydrogen bonds between like atoms
            if rec["element"] == elem1:
                weight *= 0.5
            hbond_weight_sum += weight

    feat = []
    feat += heavy_counts
    feat += [
        oxygen_count,
        nitrogen_count,
        water_count,
        float(hbond_weight_sum),
        float(aromatic_weight),
        float(hydrophobic_weight)
    ]
    return feat


def build_node_feature(residues: List, i: int, atom_records: List[Dict]) -> Optional[List[float]]:
    """
    7 geometric features:
      sin_phi, cos_phi
      sin_psi, cos_psi
      cos_omega
      sin_tau, cos_tau

    plus compact local chemistry features from hemisphere atom list.
    """
    backbone = compute_backbone_angles(residues, i)
    tau = compute_ca_torsion(residues, i)

    if backbone is None or tau is None:
        return None

    residue = residues[i]

    feat = []
    feat += angle_to_sincos(backbone["phi"])
    feat += angle_to_sincos(backbone["psi"])
    feat.append(math.cos(backbone["omega"]))
    feat += angle_to_sincos(tau)

    # compact chemistry layer
    feat += compute_local_chemistry_features(residue, atom_records)

    return feat


def collect_valid_nodes_from_chain(
    residues: List,
    pdb_name: str,
    chain_id: str,
    atom_records: List[Dict],
    min_terminal_buffer: int = 5,
) -> List[Dict]:
    """
    Build node records for one chain, but do not build edges yet.
    Returned nodes will later be merged across all chains in the model.
    """
    nodes = []
    n_res = len(residues)

    start = min_terminal_buffer
    end = n_res - min_terminal_buffer

    if end <= start:
        return nodes

    for i in range(start, end):
        feat = build_node_feature(residues, i, atom_records=atom_records)
        if feat is None:
            continue

        aa_idx = safe_resname_to_index(residues[i].get_resname())
        if aa_idx is None:
            continue

        node = {
            "x": feat,
            "y": aa_idx,
            "ca_coord": residues[i]["CA"].coord.astype(np.float32),
            "chain_id": chain_id,
            "residue_id": residues[i].id[1],
            "icode": residues[i].id[2].strip() if isinstance(residues[i].id[2], str) else "",
            "old_chain_index": i,
            "resname": residues[i].get_resname().upper(),
            "pdb_name": pdb_name,
        }
        nodes.append(node)

    return nodes


def build_graph_for_model(
    model,
    pdb_name: str,
    min_terminal_buffer: int = 5,
    ca_cutoff: float = 10.0,
) -> Optional[Dict]:
    """
    Build one residue graph for the entire model.
    Angle features are computed within each chain.
    Spatial edges are built across all valid residues in the model.
    """
    all_nodes = []

    # Build one shared atom list for chemistry features
    atom_records = build_model_atom_records(model)

    for chain in model:
        residues = extract_chain_residues(chain)
        if len(residues) < 2 * min_terminal_buffer + 1:
            continue

        chain_nodes = collect_valid_nodes_from_chain(
            residues=residues,
            pdb_name=pdb_name,
            chain_id=chain.id,
            atom_records=atom_records,
            min_terminal_buffer=min_terminal_buffer,
        )
        all_nodes.extend(chain_nodes)

    if len(all_nodes) < 5:
        return None

    x = torch.tensor([n["x"] for n in all_nodes], dtype=torch.float32)
    y = torch.tensor([n["y"] for n in all_nodes], dtype=torch.long)
    ca = torch.tensor(np.stack([n["ca_coord"] for n in all_nodes]), dtype=torch.float32)

    chain_ids = [n["chain_id"] for n in all_nodes]
    residue_ids = [n["residue_id"] for n in all_nodes]
    icodes = [n["icode"] for n in all_nodes]
    old_chain_indices = [n["old_chain_index"] for n in all_nodes]
    resnames = [n["resname"] for n in all_nodes]

    edge_src = []
    edge_dst = []
    edge_attr = []

    for i in range(len(ca)):
        for j in range(len(ca)):
            if i == j:
                continue

            dist = torch.norm(ca[i] - ca[j]).item()
            if dist <= ca_cutoff:
                same_chain = 1.0 if chain_ids[i] == chain_ids[j] else 0.0
                seq_sep = -1.0
                if same_chain:
                    seq_sep = float(abs(residue_ids[i] - residue_ids[j]))

                edge_src.append(i)
                edge_dst.append(j)
                edge_attr.append([dist, same_chain, seq_sep])

    if len(edge_src) == 0:
        return None

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    feature_names = [
        "sin_phi", "cos_phi",
        "sin_psi", "cos_psi",
        "cos_omega",
        "sin_tau", "cos_tau",
        "heavy_weight_5A",
        "heavy_weight_7A",
        "heavy_weight_9A",
        "oxygen_count_6A",
        "nitrogen_count_6A",
        "water_count_8A",
        "hbond_weight_sum",
        "aromatic_weight_7A",
        "hydrophobic_weight_7A",        
    ]

    graph = {
        "pdb_name": pdb_name,
        "model_id": getattr(model, "id", 0),
        "x": x,                     # [N, F]
        "y": y,                     # [N]
        "edge_index": edge_index,   # [2, E]
        "edge_attr": edge_attr,     # [E, 3] -> dist, same_chain, seq_sep
        "ca_coords": ca,            # [N, 3]
        "chain_ids": chain_ids,
        "residue_ids": residue_ids,
        "icodes": icodes,
        "old_chain_indices": old_chain_indices,
        "resnames": resnames,
        "feature_names": feature_names,
        "edge_feature_names": [
            "ca_distance",
            "same_chain",
            "sequence_separation_if_same_chain_else_minus1",
        ],
    }

    assert len(feature_names) == x.shape[1], \
        "Feature names do not match node feature dimensions"
    return graph


def preprocess_pdb_list(
    pdb_files: List[str],
    output_pt: str,
    min_terminal_buffer: int = 5,
    ca_cutoff: float = 10.0,
    min_nodes: int = 50,
    max_nodes: int = 1500,
) -> None:
    parser = PDBParser(QUIET=True)
    all_graphs = []

    for pdb_file in pdb_files:
        pdb_name = os.path.basename(pdb_file)

        try:
            structure = parser.get_structure(pdb_name, pdb_file)
        except Exception as exc:
            print(f"Skipping {pdb_file}: parse error: {exc}")
            continue

        for model in structure:
            graph = build_graph_for_model(
                model=model,
                pdb_name=pdb_name,
                min_terminal_buffer=min_terminal_buffer,
                ca_cutoff=ca_cutoff,
            )
            if graph is not None:
                n_nodes = graph["x"].shape[0]
                if min_nodes <= n_nodes <= max_nodes:
                    all_graphs.append(graph)
                else:
                    print(f"Ignoring model {pdb_name} with {n_nodes:5d} nodes.")

            # usually keep only first model from X-ray/NMR file
            break

    print(f"Built {len(all_graphs)} model graphs")
    torch.save(all_graphs, output_pt)
    print(f"Saved dataset to {output_pt}")


def read_pdb_list(list_file: str) -> List[str]:
    with open(list_file) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess PDB files into MPNN-style graph dataset"
    )

    parser.add_argument(
        "pdb_list",
        type=str,
        help="Text file containing list of PDB filenames"
    )

    parser.add_argument(
        "output_pt",
        type=str,
        help="Output .pt file"
    )

    parser.add_argument(
        "--cutoff",
        type=float,
        default=10.0,
        help="Cα distance cutoff (default: 10.0 Å)"
    )

    parser.add_argument(
        "--buffer",
        type=int,
        default=5,
        help="Residues excluded near termini (default: 5)"
    )

    parser.add_argument("--min_nodes", type=int, default=50)
    parser.add_argument("--max_nodes", type=int, default=1500)

    args = parser.parse_args()

    pdb_files = read_pdb_list(args.pdb_list)

    num_pdbfiles = len(pdb_files)
    print(f"Using {num_pdbfiles:5d} PDB models.")
    print(f"Node number cut-offs (min/max): {args.min_nodes:5d} / {args.max_nodes:5d}")

    preprocess_pdb_list(
        pdb_files=pdb_files,
        output_pt=args.output_pt,
        min_terminal_buffer=args.buffer,
        ca_cutoff=args.cutoff,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
    )


if __name__ == "__main__":
    main()

