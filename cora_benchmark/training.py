"""Training / evaluation utilities. The test set is only touched when report_test=True."""
import copy
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from cora_benchmark.models import Net, count_params


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


@torch.no_grad()
def evaluate(model, data, ei, mask):
    model.eval()
    out = model(data.x, ei)
    pred = out.argmax(1)
    acc = (pred[mask] == data.y[mask]).float().mean().item()
    f1 = f1_score(data.y[mask].cpu(), pred[mask].cpu(), average="macro")
    loss = F.cross_entropy(out[mask], data.y[mask]).item()
    return acc, f1, loss


def fit(cfg, data, num_classes, seed, ei=None, epochs=200, patience=50, report_test=False):
    """Train one model. Checkpoint chosen by validation accuracy (ties: validation loss)."""
    ei = data.edge_index if ei is None else ei
    set_seed(seed)
    model = Net(cfg["kind"], data.x.shape[1], cfg["hidden"], num_classes,
                n_layers=cfg.get("n_layers", 2), dropout=cfg["dropout"], heads=cfg.get("heads", 8))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    best, best_state, best_ep, bad = (-1.0, 1e9), None, 0, 0
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(data.x, ei)[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()
        va, _, vl = evaluate(model, data, ei, data.val_mask)
        if (va, -vl) > (best[0], -best[1]):
            best, best_state, best_ep, bad = (va, vl), copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= patience:
                break
    train_time = time.perf_counter() - t0
    model.load_state_dict(best_state)
    res = dict(model=model, val_acc=best[0], val_loss=best[1], best_epoch=best_ep,
               train_time=train_time, params=count_params(model))
    if report_test:
        res["test_acc"], res["test_f1"], _ = evaluate(model, data, ei, data.test_mask)
    return res


@torch.no_grad()
def embedding_similarity(model, data, ei, max_nodes=2708):
    """Over-smoothing diagnostics on the last hidden representation:
    mean pairwise cosine similarity (all node pairs) and the same-class minus
    different-class cosine gap (class separability)."""
    model.eval()
    _, emb = model(data.x, ei, return_embedding=True)
    z = F.normalize(emb[:max_nodes], dim=1)
    s = z @ z.T
    y = data.y[:max_nodes]
    n = s.shape[0]
    off = ~torch.eye(n, dtype=torch.bool)
    same = (y[:, None] == y[None, :]) & off
    diff = y[:, None] != y[None, :]
    return dict(mean_cos=s[off].mean().item(), same_class_cos=s[same].mean().item(),
                diff_class_cos=s[diff].mean().item(),
                class_gap=(s[same].mean() - s[diff].mean()).item())


def drop_undirected(ei, frac, seed):
    u = ei[:, ei[0] < ei[1]]
    k = int(round(u.shape[1] * (1 - frac)))
    order = torch.randperm(u.shape[1], generator=torch.Generator().manual_seed(seed))
    keep = u[:, order[:k]]
    return torch.cat([keep, keep.flip(0)], 1)
