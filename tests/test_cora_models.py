import torch
from torch_geometric.data import Data

from cora_benchmark.models import Net, count_params
from cora_benchmark.training import drop_undirected, embedding_similarity, fit
from cora_benchmark.tune import sample_configs


def _toy(n=60, f=20, c=3):
    g = torch.Generator().manual_seed(0)
    y = torch.randint(0, c, (n,), generator=g)
    x = torch.rand(n, f, generator=g) + torch.nn.functional.one_hot(y, c).repeat(1, 7)[:, :f].float()
    a = torch.randint(0, n, (2, 150), generator=g)
    ei = torch.cat([a, a.flip(0)], 1)
    perm = torch.randperm(n, generator=g)
    mk = lambda ids: torch.zeros(n, dtype=torch.bool).index_fill_(0, ids, True)
    return Data(x=x, y=y, edge_index=ei, train_mask=mk(perm[:30]), val_mask=mk(perm[30:45]), test_mask=mk(perm[45:])), c


def test_all_models_forward_and_depth():
    d, c = _toy()
    for kind in ("mlp", "gcn", "sage", "gat"):
        for L in (1, 2, 4):
            m = Net(kind, d.x.shape[1], 16, c, n_layers=L, dropout=0.1, heads=4)
            out, emb = m(d.x, d.edge_index, return_embedding=True)
            assert out.shape == (60, c) and emb.shape[0] == 60 and count_params(m) > 0


def test_fit_and_test_set_only_when_requested():
    d, c = _toy()
    cfg = dict(kind="gcn", hidden=16, dropout=0.2, lr=0.02, wd=5e-4)
    r = fit(cfg, d, c, seed=0, epochs=15, patience=10)
    assert "test_acc" not in r
    r = fit(cfg, d, c, seed=0, epochs=15, patience=10, report_test=True)
    assert 0 <= r["test_acc"] <= 1
    s = embedding_similarity(r["model"], d, d.edge_index)
    assert -1.0001 <= s["mean_cos"] <= 1.0001


def test_equal_tuning_budget_and_aligned_samples():
    a, b = sample_configs("gcn", 5), sample_configs("gat", 5)
    assert len(a) == len(b) == 5
    assert all(x["hidden"] == y["hidden"] and x["lr"] == y["lr"] for x, y in zip(a, b))


def test_drop_undirected_keeps_symmetry():
    d, _ = _toy()
    e = drop_undirected(d.edge_index, 0.5, 0)
    s = set(zip(e[0].tolist(), e[1].tolist()))
    assert all((b, a) in s for a, b in s)
