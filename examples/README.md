## Example

The `examples/` directory contains a PDB structure (4S0W) and an example of the residue-scoring output.

To reproduce the results, run:

```bash
python3 score_residues.py residue_outlier_16f.pt examples/4S0W.pdb
```

Compare the resulting residue rankings and probabilities with `examples/4S0W_output.txt`.

Small numerical differences may occur between PyTorch versions or computing platforms, but the results should be substantially equivalent.
