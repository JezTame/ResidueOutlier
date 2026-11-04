# ResidueOutlier

Residue-level MPNN for highlighting amino acids assigned low probability in their structural environments. **A low probability is not a predicted stability change.**

This code uses a simple model architecture with 16 node features related to the conformation and side-chain environment of each residue in a model. The supplied model was trained with 7343 distinct PDB polypeptide chains (only one from any single PDB entry), ignoring residues within 5 places of either chain terminus, using 16 structural features per residue.  The resulting weights file `residue_outlier_16f.pt` is included in this repository. The normalisation and training loop are preserved, so that the model can be retrained on other data if the user wishes.

## Files

- `score_residues.py` — score a PDB structure or a saved graph dataset.
- `structure_graph.py` — PDB-to-graph conversion and dataset preprocessing.
- `model.py` — shared model architecture and edge normalisation.
- `train_model.py` — train or retrain a network.
- `requirements.txt` — minimum dependency names (versions to be validated).

## Installation and requirements

ResidueOutlier requires Python 3 and the following Python packages:

- PyTorch (`torch`) — neural network training and inference
- NumPy (`numpy`) — numerical calculations
- Biopython (`biopython`) — reading PDB structures and calculating backbone geometry

The program uses PyTorch tensors for graph representation and message passing. PyTorch Geometric is **not required**.

Download or clone the repository, then install the dependencies:

```bash
python3 -m pip install -r requirements.txt
```

The supplied checkpoint `residue_outlier_16f.pt` can then be used to score PDB structures without retraining the network.
A CPU-only installation of PyTorch is sufficient; a compatible GPU can accelerate training.

## Run inference

```bash
# Score a PDB structure
python3 score_residues.py residue_outlier_16f.pt protein_model.pdb

# Score previously prepared graphs
python3 score_residues.py residue_outlier_16f.pt protein_graphs.pt --input_type pt
```

The default result ranks the 30 residues with largest `-log(P(observed amino acid))`. Use `--top_n` to specify a different number of hits to be displayed. The scoring code prints amino-acid probabilities and the top three predicted residues. If a 16-feature graph has been obtained with a particular PDB file previously, then this can
be input instead of the PDB file itself using the command line switch --input_type pt. Input type defaults
to PDB.

## Prepare training graphs

Create a text file with one PDB path per line, then:

```bash
python3 structure_graph.py pdb_list.txt training_graphs.pt
python3 train_model.py training_graphs.pt new_model.pt
```

Get arguments with `python3 <script>.py --help`.

## Representation (this release)

- Node features: 7 backbone geometry features and 9 local chemistry/hemisphere features (16 total).
- Directed edges join Cα atoms within 10 Å by default. Edge features are Cα distance, same-chain flag, and within-chain sequence separation (or -1 between chains).
- Edge normalisation is defined in `model.py`.
- Standard amino-acid residues require N, CA and C atoms; the default terminal buffer excludes five residues at each end of each chain; local torsion calculation requires continuity.
- Only the first structural model of a PDB file is processed.

## Interpretation and research caveats

This is an amino-acid identity classifier, not a direct estimator of mutation-induced stability or experimental fitness. Some input chemistry features are derived from the observed central residue's own side chain, which can disclose aspects of its identity, limiting the interpretation of alternative amino-acid suggestions. Validation on proteins excluded from training is needed for predictive claims.

The original training routine selects the graph split randomly; strict efforts were not made to ensure that homologous structures are grouped in a way that prevents leakage between training and validation. In preliminary retrospective tests, the program has assigned high outlier scores to several residues for which experimentally characterised stabilising substitutions are known.

## Reproducibility

The program assigns amino-acid probabilities individually to each residue within its structural environment. When a PDB model contains multiple identical polypeptide chains, equivalent residues can be compared to assess the consistency of the predictions.

## Distribution

The MIT license permits reuse and redistribution. This repository contains ResidueOutlier v0.1, including the source code and the supplied trained model.

