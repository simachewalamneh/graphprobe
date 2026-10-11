import torch

from scratch_gcn.core import (GCN, check_manual, check_permutation, drop_edges, make_dataset,
                              prep_graph, random_graph, set_seed, train_model)


def _fit():
    set_seed(0)
    x, y, ei, m = make_dataset(seed=0)
    model = GCN(3, 16, 2)
    train_model(model, x, y, prep_graph(ei, x.shape[0]), m, epochs=40)
    return model, x, y, ei, m


def test_layer_matches_manual_calculation():
    model, x, _, ei, _ = _fit()
    for node in (0, 7, 42):
        assert check_manual(model, x, ei, node=node)[0]


def test_permutation_equivariance():
    model, x, _, ei, _ = _fit()
    assert check_permutation(model, x, ei, seed=0)[0]
    assert check_permutation(model, x, ei, seed=1)[0]


def test_splits_disjoint_and_labels_not_in_features():
    x, y, _, m = make_dataset(seed=0)
    assert not (m["train"] & m["val"]).any() and not (m["train"] & m["test"]).any() and not (m["val"] & m["test"]).any()
    assert x.shape == (200, 3) and len(y.unique()) == 2
    assert not any(torch.equal(x[:, j], y.float()) for j in range(3))


def test_dataset_graph_is_homophilous_and_undirected():
    _, y, ei, _ = make_dataset(seed=0)
    assert (y[ei[0]] == y[ei[1]]).float().mean() > 0.7
    s = set(zip(ei[0].tolist(), ei[1].tolist()))
    assert all((b, a) in s for a, b in s)


def test_ablation_helpers_keep_edge_budget_and_symmetry():
    _, _, ei, _ = make_dataset(seed=0)
    d = drop_edges(ei, 0.5, 0)
    assert abs(d.shape[1] - ei.shape[1] // 2) <= 2
    r = random_graph(200, ei.shape[1] // 2, 0)
    assert r.shape[1] == ei.shape[1]
    assert drop_edges(ei, 1.0, 0).shape[1] == 0
