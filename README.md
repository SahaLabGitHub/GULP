# GULP

## Learning Protein–Protein Binding Free Energies from Interface Graphs and Physicochemical Descriptors

<img width="591" height="183" alt="Screenshot 2026-06-19 at 3 40 14 PM" src="https://github.com/user-attachments/assets/0d3cc218-e925-4088-92de-de0786fcd29a" />

GULP is a graph neural network for predicting protein–protein binding free energies (ΔG) from protein complex structures. It combines residue-level interface graphs with global physicochemical descriptors. On external validation, the model achieved a mean absolute error of 2.31 kcal/mol, with Pearson \(r = 0.54\) and Spearman \(rho = 0.58\).

Link to article: https://doi.org/10.1021/acsphyschemau.6c00060

## Installation

Clone the repository and create the Conda environment:

```bash
git clone https://github.com/SahaLabGitHub/GULP.git
cd GULP
conda env create -f environment.yml
conda activate gnn-ddg
```

The supplied environment uses a CPU build of PyTorch by default.

## Input structures

Each input must be a PDB file containing a protein–protein complex with at least two interacting protein chains. Place one or more raw PDB files in an input directory.

## Preprocessing

Prepare the structures before inference:

```bash
python preprocess.py \
  --in_dir /path/to/raw_pdbs \
  --out_dir /path/to/processed_pdbs
```

## Inference

Run the trained model on the directory of processed PDB files:

```bash
python test.py \
  --pdb_dirs /path/to/processed_pdbs \
  --model_path models/model.pt \
  --out_dir /path/to/test_output
```

Predictions are written to:

```text
/path/to/test_output/test_predictions.csv
```

The predicted binding free energies are reported in kcal/mol.

### Optional benchmarking labels

To compare predictions with experimental values, provide a CSV file containing PDB identifiers and binding free energies:

```bash
python test.py \
  --pdb_dirs /path/to/processed_pdbs \
  --model_path models/model.pt \
  --out_dir /path/to/test_output \
  --labels_csv /path/to/test_labels.csv \
  --id_col PDB \
  --target_col exp_dG
```

## Feature extraction only

Features can also be extracted without running inference:

```bash
python -m GULP.extract_features \
  --pdb_dirs /path/to/processed_pdbs \
  --out_dir /path/to/feature_output \
  --nis
```

This produces `global_features.csv` and `graphs.pkl`.
