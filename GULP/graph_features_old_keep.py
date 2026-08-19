"""
graph_features.py
=================
Builds interface residue CA graphs from PDB files for use in GNN training/inference.

Graph definition:
  - Nodes  : CA atoms of interface residues
  - Edges  : CA-CA pairs within edge_cutoff (default 8.0 Å)
  - Interface: residues with ANY heavy-atom cross-chain contact <= iface_cutoff (default 5.5 Å)

Node features (23-dim):
  - 20-dim AA one-hot
  -  3-dim residue class one-hot (charged / polar / apolar)

Edge features:
  - 1-dim CA-CA distance (Å)

Usage:
    from graph_features import build_graph, build_data_list

    # Single PDB
    data = build_graph(pdb_path)

    # Batch from a DataFrame with PDB IDs
    data_list = build_data_list(df, pdb_dirs=[...])
"""

import os
import glob
import re
import numpy as np
import torch

import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance
from torch_geometric.data import Data


# Helper to parse residue numbers that may include insertion codes (e.g. '83C')
_resid_re = re.compile(r"^(-?\d+)")

def _resid_to_int(resid):
    s = str(resid).strip()
    m = _resid_re.match(s)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse residue id: {resid}")


# ============================================================
# Constants
# ============================================================

AA_LIST = [
    "ALA","CYS","ASP","GLU","PHE","GLY","HIS","ILE","LYS","LEU",
    "MET","ASN","PRO","GLN","ARG","SER","THR","VAL","TRP","TYR"
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

CHARGED = {"ASP","GLU","LYS","ARG","HIS"}
POLAR   = {"SER","THR","ASN","GLN","TYR","CYS"}
APOLAR  = {"ALA","VAL","ILE","LEU","MET","PHE","TRP","PRO","GLY"}

NODE_DIM = 23  # 20 AA + 3 class


# ============================================================
# Node Feature Helpers
# ============================================================

def one_hot(idx, K: int) -> np.ndarray:
    v = np.zeros(K, dtype=np.float32)
    if idx is not None:
        v[idx] = 1.0
    return v


def residue_class_onehot(resname: str) -> np.ndarray:
    """3-dim: [charged, polar, apolar]"""
    r = str(resname).upper()
    v = np.zeros(3, dtype=np.float32)
    if r in CHARGED:  v[0] = 1.0
    elif r in POLAR:  v[1] = 1.0
    elif r in APOLAR: v[2] = 1.0
    return v


# ============================================================
# Chain Detection
# ============================================================

def autodetect_two_protein_chains(u: mda.Universe):
    """
    Returns (chain_field, chainA, chainB).
    Prefers chainID; falls back to segid.
    """
    chainIDs = [c for c in set(u.atoms.chainIDs) if str(c).strip()]
    if len(chainIDs) >= 2:
        chain_field, labels = "chainID", sorted(chainIDs)
    else:
        segids = [s for s in set(u.atoms.segids) if str(s).strip()]
        if len(segids) >= 2:
            chain_field, labels = "segid", sorted(segids)
        else:
            raise ValueError("Could not find >=2 chains via chainID or segid.")

    protein_chains = [
        ch for ch in labels
        if len(u.select_atoms(f"protein and {chain_field} {ch}")) > 0
    ]
    if len(protein_chains) < 2:
        raise ValueError(f"Found <2 protein chains: {protein_chains}")

    return chain_field, protein_chains[0], protein_chains[1]


# ============================================================
# Interface Residue Detection
# ============================================================

def interface_resids_heavy_atom(
    u: mda.Universe,
    chain_field: str,
    chainA: str,
    chainB: str,
    cutoff: float = 5.5,
):
    """
    Find interface residues by ANY heavy-atom cross-chain contact <= cutoff Å.

    Returns:
        (residsA, residsB) — sorted lists of residue IDs on each chain
    """
    heavyA = u.select_atoms(f"protein and {chain_field} {chainA} and not name H*")
    heavyB = u.select_atoms(f"protein and {chain_field} {chainB} and not name H*")

    if len(heavyA) == 0 or len(heavyB) == 0:
        return [], []

    pairs = capped_distance(
        heavyA.positions,
        heavyB.positions,
        max_cutoff=float(cutoff),
        return_distances=False,
    )

    if pairs is None or len(pairs) == 0:
        return [], []

    residsA, residsB = set(), set()
    for iA, iB in pairs:
        residsA.add(_resid_to_int(heavyA[int(iA)].resid))
        residsB.add(_resid_to_int(heavyB[int(iB)].resid))

    return sorted(residsA), sorted(residsB)


# ============================================================
# Graph Construction: Single PDB
# ============================================================

def build_graph(
    pdb_path: str,
    edge_cutoff:  float = 8.0,
    iface_cutoff: float = 5.5,
) -> Data:
    """
    Build a PyTorch Geometric Data object for one PDB file.

    Steps:
      1. Detect two protein chains
      2. Find interface residues (heavy-atom contacts <= iface_cutoff)
      3. Select CA atoms of interface residues as nodes
      4. Connect nodes within edge_cutoff (CA-CA distance)
      5. Assign 23-dim node features: AA one-hot + residue class

    Args:
        pdb_path     : path to PDB file
        edge_cutoff  : CA-CA distance threshold for edges (Å)
        iface_cutoff : heavy-atom distance threshold for interface (Å)

    Returns:
        torch_geometric.data.Data with fields:
          x          (N, 23) node features
          pos        (N, 3)  CA coordinates
          edge_index (2, E)  edge connectivity
          edge_attr  (E, 1)  CA-CA distances
          pdb_id     str     PDB identifier
    """
    pdb_id = os.path.splitext(os.path.basename(pdb_path))[0].upper()

    # --- Step 1: Chain detection ---
    u = mda.Universe(pdb_path)
    chain_field, chainA, chainB = autodetect_two_protein_chains(u)

    # --- Step 2: Interface residues ---
    resA, resB = interface_resids_heavy_atom(
        u, chain_field, chainA, chainB, cutoff=iface_cutoff
    )

    # --- Step 3: Select CA atoms ---
    if len(resA) == 0 or len(resB) == 0:
        ca = u.select_atoms("protein and name CA")
    else:
        selA = (f"protein and name CA and {chain_field} {chainA} "
                f"and resid {' '.join(map(str, resA))}")
        selB = (f"protein and name CA and {chain_field} {chainB} "
                f"and resid {' '.join(map(str, resB))}")
        ca = u.select_atoms(selA) + u.select_atoms(selB)
        if len(ca) == 0:
            ca = u.select_atoms("protein and name CA")

    if len(ca) == 0:
        raise ValueError(f"No CA atoms found in {pdb_id}")

    pos = ca.positions.astype(np.float32)

    # --- Step 4: Build edges (CA-CA within edge_cutoff) ---
    d2   = np.sum((pos[:, None, :] - pos[None, :, :]) ** 2, axis=-1)
    mask = (d2 < edge_cutoff ** 2) & (d2 > 0)
    src, dst = np.where(mask)

    edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    edge_attr  = np.sqrt(d2[src, dst]).reshape(-1, 1).astype(np.float32)

    # --- Step 5: Node features (23-dim) ---
    x_list = []
    for atom in ca:
        resname = str(atom.resname).upper()
        aa_oh  = one_hot(AA_TO_IDX.get(resname, None), len(AA_LIST))  # 20-dim
        cls_oh = residue_class_onehot(resname)                         #  3-dim
        x_list.append(np.concatenate([aa_oh, cls_oh]))

    x = np.stack(x_list, axis=0).astype(np.float32)  # (N, 23)

    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        pos=torch.tensor(pos, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        pdb_id=pdb_id,
    )


# ============================================================
# Batch: Build Data List from DataFrame
# ============================================================

def find_pdb_path(pdb_dirs, pdb_id: str):
    """Search one or more directories for a PDB file by ID."""
    pdb_dirs = pdb_dirs if isinstance(pdb_dirs, (list, tuple)) else [pdb_dirs]
    for d in pdb_dirs:
        for candidate in [
            os.path.join(d, f"{pdb_id}.pdb"),
            os.path.join(d, f"{pdb_id.lower()}.pdb"),
        ]:
            if os.path.exists(candidate):
                return candidate
        for p in glob.glob(os.path.join(d, "*.pdb")):
            if os.path.splitext(os.path.basename(p))[0].upper() == pdb_id.upper():
                return p
    return None


def build_data_list(
    df,
    pdb_dirs,
    id_col:       str   = "PDB",
    edge_cutoff:  float = 8.0,
    iface_cutoff: float = 5.5,
    verbose:      bool  = True,
) -> list:
    """
    Build a list of PyG Data objects from a DataFrame of PDB IDs.

    Note: y (labels) and u (global features) are NOT added here.
    They are attached in the training script after merging with the
    feature CSV, keeping graph construction decoupled from training.

    Args:
        df           : DataFrame with at least an ID column
        pdb_dirs     : str or list of directories to search for PDB files
        id_col       : name of the PDB ID column in df
        edge_cutoff  : CA-CA edge threshold (Å)
        iface_cutoff : interface heavy-atom threshold (Å)
        verbose      : print per-PDB status

    Returns:
        list of torch_geometric.data.Data objects
    """
    data_list = []
    n_ok = n_fail = n_missing = 0

    for _, row in df.iterrows():
        pdb_id   = str(row[id_col]).upper()
        pdb_path = find_pdb_path(pdb_dirs, pdb_id)

        if pdb_path is None:
            n_missing += 1
            if verbose:
                print(f"[MISSING] {pdb_id}")
            continue

        try:
            data = build_graph(
                pdb_path,
                edge_cutoff=edge_cutoff,
                iface_cutoff=iface_cutoff,
            )
            data_list.append(data)
            n_ok += 1
            if verbose:
                print(f"[OK] {pdb_id}  nodes={data.x.shape[0]}  edges={data.edge_index.shape[1]}")

        except Exception as e:
            n_fail += 1
            if verbose:
                print(f"[FAIL] {pdb_id}: {e}")

    if verbose:
        print(f"\nBuilt {n_ok} graphs  |  missing={n_missing}  fail={n_fail}")

    return data_list
