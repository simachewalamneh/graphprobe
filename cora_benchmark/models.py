"""Own model classes (MLP / GCN / GraphSAGE / GAT) built on ready-made PyG layers."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class Net(nn.Module):
    """Generic L-layer network. kind in {"mlp","gcn","sage","gat"}.
    The last layer outputs class logits; `embed` returns the input of that layer
    (the last hidden representation; logits when depth == 1)."""

    def __init__(self, kind, in_dim, hidden, out_dim, n_layers=2, dropout=0.5, heads=8):
        super().__init__()
        self.kind, self.dropout, self.n_layers = kind, dropout, n_layers
        self.layers = nn.ModuleList()
        d = in_dim
        for li in range(n_layers):
            last = li == n_layers - 1
            o = out_dim if last else hidden
            if kind == "mlp":
                self.layers.append(nn.Linear(d, o)); d = o
            elif kind == "gcn":
                self.layers.append(GCNConv(d, o)); d = o
            elif kind == "sage":
                self.layers.append(SAGEConv(d, o, aggr="mean")); d = o
            elif kind == "gat":
                if last:
                    self.layers.append(GATConv(d, o, heads=1, concat=False, dropout=dropout)); d = o
                else:
                    per_head = max(hidden // heads, 1)
                    self.layers.append(GATConv(d, per_head, heads=heads, dropout=dropout)); d = per_head * heads
            else:
                raise ValueError(kind)

    def _apply(self, layer, x, ei):
        return layer(x) if self.kind == "mlp" else layer(x, ei)

    def forward(self, x, ei, return_embedding=False):
        emb = None
        for li, layer in enumerate(self.layers):
            last = li == len(self.layers) - 1
            if last:
                emb = x if len(self.layers) > 1 else None
            x = F.dropout(x, self.dropout, self.training)
            x = self._apply(layer, x, ei)
            if not last:
                x = F.elu(x) if self.kind == "gat" else F.relu(x)
        if return_embedding:
            return x, (emb if emb is not None else x)
        return x


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
