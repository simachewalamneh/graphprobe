"""UNTRACKED BASELINE: the bare training script (what most people start with).

Limitations versus the improved pipeline (mlops/cli.py + train.py + registry.py):
  * no seed / hyper-parameter / code-version / dataset record -> results cannot be reproduced or audited
  * metrics only printed to the console and then lost; no run comparison
  * nothing saved: weights, preprocessing and label mapping are not packaged, so it cannot be served
  * unseen nodes are not excluded, so the printed accuracy may be optimistic (transductive leakage)
  * no validation gate, versioning or rollback: a worse model silently replaces the old one
  * no tests, no CI, no monitoring of graph/embedding drift
Run:  python -m mlops.baseline_untracked
"""
import torch

from mlops.bundle import row_l1_normalize
from mlops.data import load_snapshot
from scratch_gcn.core import GCN, acc, prep_graph, train_model

snap = load_snapshot("cora")
x = torch.from_numpy(row_l1_normalize(snap["x"]))
y = torch.from_numpy(snap["y"])
g = prep_graph(torch.from_numpy(snap["edge_index"]), x.shape[0])
masks = {k: torch.from_numpy(snap[f"{k}_mask"]) for k in ("train", "val", "test")}
model = GCN(x.shape[1], 16, int(y.max()) + 1)
train_model(model, x, y, g, masks)
model.eval()
print("test acc:", acc(model(x, g), y, masks["test"]))
