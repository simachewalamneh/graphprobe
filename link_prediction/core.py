"""Leakage-safe link prediction on undirected graphs: splits, negative sampling,
GNN encoder + two decoders, common-neighbour baseline, automated leakage checks."""
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.nn import GCNConv


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# splits
def unique_undirected(ei):
    """One row per undirected link (u < v), regardless of how many directions were stored."""
    a = torch.minimum(ei[0], ei[1]); b = torch.maximum(ei[0], ei[1])
    keep = a != b
    key = torch.unique(a[keep] * (int(ei.max()) + 1) + b[keep])
    m = int(ei.max()) + 1
    return torch.stack([key // m, key % m])


def split_links(ei, val_frac=0.05, test_frac=0.10, seed=0):
    u = unique_undirected(ei)
    perm = torch.randperm(u.shape[1], generator=torch.Generator().manual_seed(seed))
    n_test, n_val = int(test_frac * u.shape[1]), int(val_frac * u.shape[1])
    return {"test": u[:, perm[:n_test]], "val": u[:, perm[n_test:n_test + n_val]],
            "train": u[:, perm[n_test + n_val:]]}


def both_directions(pos):
    return torch.cat([pos, pos.flip(0)], 1)


def pair_keys(pairs, n):
    a = torch.minimum(pairs[0], pairs[1]); b = torch.maximum(pairs[0], pairs[1])
    return (a * n + b).numpy()


def observed_matrix(n, ei):
    A = torch.zeros(n, n, dtype=torch.bool)
    A[ei[0], ei[1]] = True
    A[ei[1], ei[0]] = True
    A.fill_diagonal_(True)  # self-links count as forbidden for negatives
    return A


def sample_negatives(n, forbidden, k, rng, avoid_keys=None):
    """k unique unordered non-edges (u < v): not self-links, not in `forbidden`
    (the matrix of ALL observed positives), not in `avoid_keys` (already-used pairs)."""
    forb = forbidden.numpy() if torch.is_tensor(forbidden) else forbidden
    avoid = np.asarray(sorted(avoid_keys), dtype=np.int64) if avoid_keys else np.empty(0, np.int64)
    have = np.empty(0, np.int64)
    while have.size < k:
        u = rng.integers(0, n, size=2 * k + 64)
        v = rng.integers(0, n, size=2 * k + 64)
        a, b = np.minimum(u, v), np.maximum(u, v)
        ok = (a != b) & ~forb[a, b]
        keys = np.unique(a[ok] * n + b[ok])
        keys = keys[~np.isin(keys, avoid)]
        have = np.unique(np.concatenate([have, keys]))
    rng.shuffle(have)
    have = have[:k]
    return torch.tensor(np.stack([have // n, have % n]), dtype=torch.long)


def make_eval_negatives(n, forbidden, split, seed=0):
    """Fixed validation/test negatives, disjoint from each other."""
    rng = np.random.default_rng(seed)
    val_neg = sample_negatives(n, forbidden, split["val"].shape[1], rng)
    test_neg = sample_negatives(n, forbidden, split["test"].shape[1], rng,
                                avoid_keys=set(pair_keys(val_neg, n).tolist()))
    return val_neg, test_neg


# the checks
def leakage_checks(n, mp_ei, split, val_neg, test_neg, full_ei):
    """Automated proofs that held-out links (and reverse copies) are absent from the
    message-passing graph and that negatives are valid and non-overlapping."""
    mp = set((mp_ei[0] * n + mp_ei[1]).tolist())
    mp_rev = set((mp_ei[1] * n + mp_ei[0]).tolist())
    res = {}
    for name in ("val", "test"):
        p = split[name]
        fwd = set((p[0] * n + p[1]).tolist()); rev = set((p[1] * n + p[0]).tolist())
        res[f"{name}_links_absent_from_mp_graph"] = fwd.isdisjoint(mp)
        res[f"{name}_reverse_links_absent_from_mp_graph"] = rev.isdisjoint(mp)
    res["mp_graph_is_symmetric"] = mp == mp_rev
    res["mp_graph_equals_train_links_both_directions"] = mp == (
        set((split["train"][0] * n + split["train"][1]).tolist()) |
        set((split["train"][1] * n + split["train"][0]).tolist()))
    k = {s: set(pair_keys(split[s], n).tolist()) for s in ("train", "val", "test")}
    res["positive_splits_disjoint"] = (not (k["train"] & k["val"]) and not (k["train"] & k["test"])
                                       and not (k["val"] & k["test"]))
    obs = set(pair_keys(torch.unique(torch.cat([full_ei, full_ei.flip(0)], 1), dim=1), n).tolist())
    nk = {"val": set(pair_keys(val_neg, n).tolist()), "test": set(pair_keys(test_neg, n).tolist())}
    res["negatives_have_no_self_links"] = bool((val_neg[0] != val_neg[1]).all() and (test_neg[0] != test_neg[1]).all())
    res["negatives_not_observed_positives"] = not (nk["val"] & obs) and not (nk["test"] & obs)
    res["val_test_negatives_disjoint"] = not (nk["val"] & nk["test"])
    res["negatives_unique"] = len(nk["val"]) == val_neg.shape[1] and len(nk["test"]) == test_neg.shape[1]
    return res


# the model
class Encoder(nn.Module):
    def __init__(self, i, h, o, dropout):
        super().__init__()
        self.c1, self.c2, self.dropout = GCNConv(i, h), GCNConv(h, o), dropout

    def forward(self, x, ei):
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.c1(x, ei))
        x = F.dropout(x, self.dropout, self.training)
        return self.c2(x, ei)


class DotDecoder(nn.Module):
    def forward(self, z, pairs):
        return (z[pairs[0]] * z[pairs[1]]).sum(-1)


class MLPDecoder(nn.Module):
    """Symmetric by construction: the input uses z_u*z_v and |z_u - z_v|, both
    invariant to swapping u and v, so score(u,v) == score(v,u) exactly."""

    def __init__(self, d, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, z, pairs):
        zu, zv = z[pairs[0]], z[pairs[1]]
        return self.net(torch.cat([zu * zv, (zu - zv).abs()], -1)).squeeze(-1)


class LinkModel(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout, decoder):
        super().__init__()
        self.enc = Encoder(in_dim, hidden, out_dim, dropout)
        self.dec = DotDecoder() if decoder == "dot" else MLPDecoder(out_dim)

    def forward(self, x, mp_ei, pairs):
        return self.dec(self.enc(x, mp_ei), pairs)


def metrics(pos_s, neg_s):
    y = np.r_[np.ones(len(pos_s)), np.zeros(len(neg_s))]
    s = np.r_[pos_s, neg_s]
    return roc_auc_score(y, s), average_precision_score(y, s)


@torch.no_grad()
def score_pairs(model, x, mp_ei, pairs):
    model.eval()
    return model(x, mp_ei, pairs).numpy()


def eval_split(model, x, mp_ei, pos, neg):
    return metrics(score_pairs(model, x, mp_ei, pos), score_pairs(model, x, mp_ei, neg))


def train_link(decoder, cfg, x, mp_ei, split, val_neg, test_neg, forbidden, seed, epochs=200):
    """Train on `split['train']` positives vs freshly sampled negatives. Select the epoch by
    validation AUC; test metrics are computed once at the end with that checkpoint."""
    n = x.shape[0]
    set_seed(seed)
    model = LinkModel(x.shape[1], cfg["hidden"], cfg["out"], cfg["dropout"], decoder)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("wd", 0.0))
    rng = np.random.default_rng(seed)
    avoid = set(pair_keys(val_neg, n).tolist()) | set(pair_keys(test_neg, n).tolist())
    pos = split["train"]
    best, state = -1.0, None
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        neg = sample_negatives(n, forbidden, pos.shape[1], rng, avoid_keys=avoid)
        z = model.enc(x, mp_ei)
        logits = torch.cat([model.dec(z, pos), model.dec(z, neg)])
        labels = torch.cat([torch.ones(pos.shape[1]), torch.zeros(neg.shape[1])])
        F.binary_cross_entropy_with_logits(logits, labels).backward()
        opt.step()
        v_auc, _ = eval_split(model, x, mp_ei, split["val"], val_neg)
        if v_auc > best:
            best, state = v_auc, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    return model, best


# baseline + utils
def train_adjacency(n, train_pos):
    A = torch.zeros(n, n)
    A[train_pos[0], train_pos[1]] = 1.0
    A[train_pos[1], train_pos[0]] = 1.0
    return A


def common_neighbor_scores(A, pairs):
    cn = A @ A
    return cn[pairs[0], pairs[1]].numpy()


@torch.no_grad()
def top_unobserved(model, x, mp_ei, observed, k=10, block=64):
    """Highest-scoring pairs among ALL pairs not observed in the full graph."""
    model.eval()
    n = x.shape[0]
    z = model.enc(x, mp_ei)
    cands = []
    for s in range(0, n, block):
        rows = torch.arange(s, min(s + block, n))
        u = rows.repeat_interleave(n); v = torch.arange(n).repeat(len(rows))
        keep = (u < v) & ~observed[u, v]
        u, v = u[keep], v[keep]
        sc = model.dec(z, torch.stack([u, v]))
        top = torch.topk(sc, min(k, sc.numel()))
        cands += list(zip(u[top.indices].tolist(), v[top.indices].tolist(), top.values.tolist()))
    cands.sort(key=lambda t: -t[2])
    return cands[:k]
