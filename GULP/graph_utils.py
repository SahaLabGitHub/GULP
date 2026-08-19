"""Minimal graph utilities to replace torch_geometric dependencies.

This module provides a lightweight Data/Batch/DataLoader system and a simple GCNConv
implementation so the pipeline can run without torch_geometric.

NOTE: This is not a full replacement for torch_geometric, but it supports the
use patterns used by this repository.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

import torch


class Data:
    """Minimal graph container."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Batch(Data):
    """Batched graph container."""

    def __init__(self, data_list: List[Data]):
        # concatenate node-level tensors
        self.x = torch.cat([d.x for d in data_list], dim=0)
        self.pos = torch.cat([d.pos for d in data_list], dim=0) if hasattr(data_list[0], "pos") else None

        # build batched edge_index (adjust by offset)
        edge_index_list = []
        edge_attr_list = []
        batch_list = []
        offset = 0
        for i, d in enumerate(data_list):
            n = d.x.size(0)
            if hasattr(d, "edge_index") and d.edge_index is not None:
                edge_index_list.append(d.edge_index + offset)
                if hasattr(d, "edge_attr") and d.edge_attr is not None:
                    edge_attr_list.append(d.edge_attr)
            batch_list.append(torch.full((n,), i, dtype=torch.long))
            offset += n

        self.edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else torch.empty((2, 0), dtype=torch.long)
        self.edge_attr = torch.cat(edge_attr_list, dim=0) if edge_attr_list else None
        self.batch = torch.cat(batch_list, dim=0)

        # global features (u) and labels (y)
        if hasattr(data_list[0], "u"):
            self.u = torch.cat([d.u for d in data_list], dim=0)
        if hasattr(data_list[0], "y"):
            self.y = torch.cat([d.y for d in data_list], dim=0)

        # optional graph-level IDs
        if hasattr(data_list[0], "pdb_id"):
            self.pdb_id = [d.pdb_id for d in data_list]


class DataLoader:
    """Simple DataLoader that yields Batch objects."""

    def __init__(self, dataset: List[Data], batch_size: int = 1, shuffle: bool = False):
        self.dataset = list(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(len(self.dataset)).tolist()
            dataset = [self.dataset[i] for i in perm]
        else:
            dataset = self.dataset

        for i in range(0, len(dataset), self.batch_size):
            yield Batch(dataset[i : i + self.batch_size])

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def global_mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Average pooling by graph index."""
    assert x.size(0) == batch.size(0)
    num_graphs = int(batch.max().item()) + 1
    out = x.new_zeros((num_graphs, x.size(1)))
    count = x.new_zeros((num_graphs, 1))

    out = out.index_add(0, batch, x)
    count = count.index_add(0, batch, torch.ones((x.size(0), 1), device=x.device))
    return out / count.clamp(min=1.0)


class GCNConv(torch.nn.Module):
    """Simple GCN convolution layer (adapted for edge_index)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None):
        # Compute normalized adjacency (symmetric)
        # edge_index: [2, E]
        row, col = edge_index
        if edge_weight is None:
            edge_weight = torch.ones(row.size(0), device=x.device, dtype=x.dtype)

        deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        deg = deg.index_add(0, row, edge_weight)
        deg = deg.index_add(0, col, edge_weight)

        deg_inv_sqrt = deg.clamp(min=1e-12).pow(-0.5)
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        # Message passing
        out = torch.zeros_like(x)
        out = out.index_add(0, row, x[col] * norm.unsqueeze(-1))

        return self.lin(out)
