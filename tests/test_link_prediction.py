import numpy as np
import torch

from link_prediction.core import (MLPDecoder, both_directions, common_neighbor_scores, leakage_checks,
                                  make_eval_negatives, observed_matrix, sample_negatives, split_links,
                                  train_adjacency, unique_undirected)


def _graph(n=80, m=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randint(0, n, (2, m), generator=g)
    a = a[:, a[0] != a[1]]
    return n, torch.cat([a, a.flip(0)], 1)


def test_unique_links_and_split_keep_both_directions_together():
    n, ei = _graph()
    u = unique_undirected(ei)
    assert (u[0] < u[1]).all() and u.shape[1] == len(set(zip(u[0].tolist(), u[1].tolist())))
    sp = split_links(ei, 0.1, 0.2, seed=0)
    assert sum(v.shape[1] for v in sp.values()) == u.shape[1]
    mp = both_directions(sp["train"])
    pairs = set(zip(mp[0].tolist(), mp[1].tolist()))
    assert all((b, a) in pairs for a, b in pairs)


def test_heldout_links_and_reverses_absent_from_message_passing_graph():
    n, ei = _graph()
    sp = split_links(ei, 0.1, 0.2, seed=0)
    forb = observed_matrix(n, ei)
    vn, tn = make_eval_negatives(n, forb, sp, seed=0)
    res = leakage_checks(n, both_directions(sp["train"]), sp, vn, tn, ei)
    assert all(res.values()), res


def test_leaky_graph_is_detected_by_the_checks():
    n, ei = _graph()
    sp = split_links(ei, 0.1, 0.2, seed=0)
    forb = observed_matrix(n, ei)
    vn, tn = make_eval_negatives(n, forb, sp, seed=0)
    leaky = both_directions(torch.cat([sp["train"], sp["val"], sp["test"]], 1))
    res = leakage_checks(n, leaky, sp, vn, tn, ei)
    assert not res["test_links_absent_from_mp_graph"] and not res["val_links_absent_from_mp_graph"]


def test_negatives_valid_unique_and_disjoint():
    n, ei = _graph()
    sp = split_links(ei, 0.1, 0.2, seed=0)
    forb = observed_matrix(n, ei)
    vn, tn = make_eval_negatives(n, forb, sp, seed=0)
    for neg in (vn, tn):
        assert (neg[0] < neg[1]).all()
        assert not forb[neg[0], neg[1]].any()
    kv = set(zip(vn[0].tolist(), vn[1].tolist())); kt = set(zip(tn[0].tolist(), tn[1].tolist()))
    assert not kv & kt
    # fixed: same seed -> identical evaluation negatives
    vn2, tn2 = make_eval_negatives(n, forb, sp, seed=0)
    assert torch.equal(vn, vn2) and torch.equal(tn, tn2)
    extra = sample_negatives(n, forb, 50, np.random.default_rng(1), avoid_keys=set((vn[0] * n + vn[1]).tolist()))
    assert not set((extra[0] * n + extra[1]).tolist()) & set((vn[0] * n + vn[1]).tolist())


def test_mlp_decoder_is_symmetric():
    torch.manual_seed(0)
    dec = MLPDecoder(16).eval()
    z = torch.randn(30, 16)
    p = torch.randint(0, 30, (2, 200))
    assert torch.allclose(dec(z, p), dec(z, p.flip(0)), atol=1e-7)


def test_common_neighbours_counts_paths():
    A = train_adjacency(4, torch.tensor([[0, 1, 0], [1, 2, 3]]))   # edges 0-1, 1-2, 0-3
    s = common_neighbor_scores(A, torch.tensor([[0, 1], [2, 3]]))
    assert s.tolist() == [1.0, 1.0]
