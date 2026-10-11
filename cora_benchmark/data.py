"""Cora loading and dataset inspection."""
from pathlib import Path

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree

ROOT = Path(__file__).resolve().parents[1]


def load_cora(root=None):
    ds = Planetoid(str(root or ROOT / "data"), "Cora", transform=T.NormalizeFeatures())
    return ds[0], ds.num_classes


def inspect(data, num_classes):
    ei = data.edge_index
    deg = degree(ei[0], data.num_nodes)
    same = (data.y[ei[0]] == data.y[ei[1]]).float().mean().item()
    raw_nnz = (data.x > 0).sum(1).float()  # words per document (features are row-normalised)
    return {
        "nodes": data.num_nodes,
        "directed_edge_entries": int(ei.shape[1]),
        "undirected_edges": int((ei[0] < ei[1]).sum()),
        "features_per_node": int(data.x.shape[1]),
        "mean_words_per_document": round(raw_nnz.mean().item(), 2),
        "classes": num_classes,
        "class_counts": torch.bincount(data.y).tolist(),
        "mean_degree": round(deg.mean().item(), 3),
        "max_degree": int(deg.max()),
        "isolated_nodes": int((deg == 0).sum()),
        "edge_homophily": round(same, 4),
        "train/val/test": [int(data.train_mask.sum()), int(data.val_mask.sum()), int(data.test_mask.sum())],
    }
