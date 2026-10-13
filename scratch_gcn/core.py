import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


# Prepare Dataset
def make_dataset(n=200, p_in=0.08, p_out=0.01, sep=0.6, noise=1.0, seed=0):
    """Synthetic graph where structure carries information beyond node features.

    * Labels: balanced, 2 classes (shuffled).
    * Features (3 per node): feature 1/2 = class mean (+/- sep) + N(0, noise^2);
      feature 3 is pure noise. Labels are never part of X.
    * Edges: stochastic block model, undirected. p_in > p_out -> homophilous.
    * Split: random 60/20/20 train/val/test node masks.
    """
    rng = np.random.default_rng(seed)
    y = rng.permutation(np.arange(n) % 2)
    mu = np.array([[sep, -sep, 0.0], [-sep, sep, 0.0]])
    X = mu[y] + rng.normal(0, noise, (n, 3))
    P = np.where(y[:, None] == y[None, :], p_in, p_out)
    U = np.triu(rng.random((n, n)) < P, 1)
    A = U | U.T
    src, dst = np.nonzero(A)
    ei = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    perm = rng.permutation(n)
    a, b = int(0.6 * n), int(0.8 * n)
    masks = {}
    for k, ids in zip(("train", "val", "test"), (perm[:a], perm[a:b], perm[b:])):
        m = torch.zeros(n, dtype=torch.bool)
        m[ids] = True
        masks[k] = m
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y), ei, masks


#GCN
def prep_graph(ei: torch.Tensor, n: int):
    """Add self-loops and compute symmetric degree normalisation 1/sqrt(d_i d_j)
    for every (src -> dst) message. Returns (src, dst, norm)."""
    loops = torch.arange(n)
    src = torch.cat([ei[0], loops])
    dst = torch.cat([ei[1], loops])
    deg = torch.zeros(n).index_add_(0, dst, torch.ones(dst.numel()))
    norm = deg[src].pow(-0.5) * deg[dst].pow(-0.5)
    return src, dst, norm


class GCNLayer(nn.Module):
    """H' = D^-1/2 (A+I) D^-1/2 H W + b, via explicit message passing."""

    def __init__(self, i: int, o: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(i, o))
        nn.init.xavier_uniform_(self.W)
        self.b = nn.Parameter(torch.zeros(o))

    def forward(self, x, g):
        src, dst, norm = g
        h = x @ self.W
        out = torch.zeros_like(h).index_add_(0, dst, norm[:, None] * h[src])  # aggregate
        return out + self.b


class GCN(nn.Module):
    def __init__(self, i: int, h: int, o: int, p: float = 0.5):
        super().__init__()
        self.l1, self.l2, self.p = GCNLayer(i, h), GCNLayer(h, o), p

    def embed(self, x, g):
        """Hidden representation (after layer 1 + ReLU, no dropout)."""
        return F.relu(self.l1(x, g))

    def forward(self, x, g):
        x = F.dropout(x, self.p, self.training)
        x = F.relu(self.l1(x, g))
        x = F.dropout(x, self.p, self.training)
        return self.l2(x, g)


class MLP(nn.Module):
    def __init__(self, i: int, h: int, o: int, p: float = 0.5):
        super().__init__()
        self.l1, self.l2, self.p = nn.Linear(i, h), nn.Linear(h, o), p

    def forward(self, x, g=None):
        x = F.dropout(x, self.p, self.training)
        x = F.relu(self.l1(x))
        x = F.dropout(x, self.p, self.training)
        return self.l2(x)


# training
def acc(out, y, m) -> float:
    return (out[m].argmax(1) == y[m]).float().mean().item()


def train_model(model, x, y, g, masks, epochs=200, lr=0.01, wd=5e-4):
    """Adam + cross-entropy on the train mask. Keeps the checkpoint with the best
    validation accuracy (ties broken by lower validation loss). Returns (val_acc, val_loss)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best, state = (-1.0, 1e9), None
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x, g)[masks["train"]], y[masks["train"]])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            out = model(x, g)
            vl = F.cross_entropy(out[masks["val"]], y[masks["val"]]).item()
            va = acc(out, y, masks["val"])
        if (va, -vl) > (best[0], -best[1]):
            best = (va, vl)
            state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return best


# correctness checks
def manual_layer_node(x, ei, W, b, i):
    """Independent hand calculation of one GCN layer output for node i."""
    deg = lambda j: int((ei[1] == j).sum()) + 1  # incoming neighbours + self-loop
    nbrs = set(ei[0][ei[1] == i].tolist()) | {i}
    out = torch.zeros(W.shape[1])
    for j in nbrs:
        out += (x[j] @ W) / (deg(i) * deg(j)) ** 0.5
    return out + b


def check_manual(model, x, ei, node=0):
    g = prep_graph(ei, x.shape[0])
    model.eval()
    with torch.no_grad():
        auto = model.l1(x, g)[node]
        manual = manual_layer_node(x, ei, model.l1.W, model.l1.b, node)
    return torch.allclose(auto, manual, atol=1e-5), (auto - manual).abs().max().item()


def check_permutation(model, x, ei, seed=0):
    """Consistently relabel nodes/features/edges (and shuffle edge order): the outputs
    must be the same rows, reordered."""
    n = x.shape[0]
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)  # new node k == old node perm[k]
    pos = torch.empty(n, dtype=torch.long)
    pos[perm] = torch.arange(n)
    ei2 = pos[ei][:, torch.randperm(ei.shape[1], generator=gen)]
    with torch.no_grad():
        o1 = model(x, prep_graph(ei, n))
        o2 = model(x[perm], prep_graph(ei2, n))
    return torch.allclose(o1[perm], o2, atol=1e-5), (o1[perm] - o2).abs().max().item()


# graph ablations
def drop_edges(ei, frac, seed):
    """Remove a fraction of undirected edges (both directions together)."""
    u = ei[:, ei[0] < ei[1]]
    k = int(round(u.shape[1] * (1 - frac)))
    order = torch.randperm(u.shape[1], generator=torch.Generator().manual_seed(seed))
    keep = u[:, order[:k]]
    return torch.cat([keep, keep.flip(0)], 1)


def random_graph(n, n_undirected, seed):
    """Label-blind random graph with the same number of undirected edges."""
    iu = torch.triu_indices(n, n, offset=1)
    order = torch.randperm(iu.shape[1], generator=torch.Generator().manual_seed(seed))
    e = iu[:, order[:n_undirected]]
    return torch.cat([e, e.flip(0)], 1)
