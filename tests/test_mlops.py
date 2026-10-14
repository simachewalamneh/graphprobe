import json
import shutil

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mlops import drift
from mlops.bundle import BundleError, InputError, load_bundle
from mlops.data import load_snapshot, make_training_graph, validation_set
from mlops.registry import GateConfig, Registry, evaluate_gate, evaluate_predictor
from mlops.service import create_app
from mlops.train import train_bundle

P = dict(hidden=8, epochs=40, n_unseen=40, seed=0)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    snap = load_snapshot("synthetic")
    root = tmp_path_factory.mktemp("w")
    m, tg, val, meta = train_bundle(snap, P, root / "b1")
    return dict(snap=snap, root=root, tg=tg, val=val, metrics=m, b1=root / "b1")


def test_unseen_nodes_and_their_edges_excluded_from_training_graph(world):
    snap, tg = world["snap"], world["tg"]
    assert not set(tg.unseen_ids.tolist()) & set(tg.orig_ids.tolist())
    touching = np.isin(snap["edge_index"][0], tg.unseen_ids) | np.isin(snap["edge_index"][1], tg.unseen_ids)
    assert tg.edge_index.shape[1] == snap["edge_index"].shape[1] - touching.sum()
    assert tg.x.shape[0] == snap["x"].shape[0] - len(tg.unseen_ids)


def test_bundle_has_all_parts_and_reload_reproduces_predictions(world):
    for f in ("weights.pt", "architecture.json", "preprocessing.json", "feature_schema.json", "label_map.json", "metadata.json"):
        assert (world["b1"] / f).exists()
    snap = world["snap"]
    a, b = load_bundle(world["b1"]), load_bundle(world["b1"])
    pa = a.predict_proba(snap["x"], snap["edge_index"].T)
    pb = b.predict_proba(snap["x"], snap["edge_index"].T)
    assert np.array_equal(pa, pb)
    assert evaluate_predictor(a, world["val"])["val_accuracy"] == pytest.approx(world["metrics"]["val_accuracy"])


def test_corrupt_or_missing_bundle_raises(world, tmp_path):
    bad = tmp_path / "bad"; shutil.copytree(world["b1"], bad)
    w = bad / "weights.pt"; w.write_bytes(w.read_bytes()[:-4] + b"xxxx")
    with pytest.raises(BundleError):
        load_bundle(bad)
    (bad / "metadata.json").unlink()
    with pytest.raises(BundleError):
        load_bundle(bad)


# ------------------------------------------------------------------------- gate
CFG = GateConfig(metric="val_macro_f1", min_improvement=0.01, tie_policy="block", first_model_floor=0.5)


def test_gate_first_model_behaviour():
    assert evaluate_gate({"val_macro_f1": 0.6}, None, CFG).allowed
    d = evaluate_gate({"val_macro_f1": 0.4}, None, CFG)
    assert not d.allowed and d.rule == "first_model"


def test_gate_improvement_regression_insufficient_and_tie():
    ap = {"val_macro_f1": 0.80}
    assert evaluate_gate({"val_macro_f1": 0.82}, ap, CFG).allowed
    assert evaluate_gate({"val_macro_f1": 0.78}, ap, CFG).rule == "regression"
    assert evaluate_gate({"val_macro_f1": 0.805}, ap, CFG).rule == "insufficient_improvement"
    t = evaluate_gate({"val_macro_f1": 0.80}, ap, CFG)
    assert not t.allowed and t.rule == "tie"
    assert evaluate_gate({"val_macro_f1": 0.80}, ap, GateConfig(tie_policy="promote", min_improvement=0.01)).allowed


def test_gate_missing_metrics_fail_closed():
    ap = {"val_macro_f1": 0.8}
    assert evaluate_gate({}, ap, CFG).rule == "missing_metric"
    assert evaluate_gate({"val_macro_f1": float("nan")}, ap, CFG).rule == "missing_metric"
    assert evaluate_gate({"val_macro_f1": 0.9}, {}, CFG).rule == "approved_metric_missing"


def test_registry_promotion_block_and_rollback(world, tmp_path):
    reg = Registry(tmp_path / "reg")
    snap, val = world["snap"], world["val"]
    v1 = reg.register(world["b1"], world["metrics"])
    _, tg, val2, _ = train_bundle(snap, {**P, "hidden": 16, "seed": 1}, tmp_path / "b2")
    v2 = reg.register(tmp_path / "b2", {})
    allow_all = GateConfig(min_improvement=-1.0, tie_policy="promote", first_model_floor=0.0)
    assert reg.promote(v1, val, allow_all).allowed and reg.approved == v1
    assert reg.promote(v2, val, GateConfig(min_improvement=1.0)).allowed is False   # impossible bar -> blocked
    assert reg.approved == v1 and reg.versions()[v2]["status"] == "rejected"
    assert reg.promote(v2, val, allow_all).allowed and reg.approved == v2
    assert reg.rollback() == v1 and reg.approved == v1
    with pytest.raises(RuntimeError):
        reg.rollback()


# -------------------------------------------------------------------------- API
@pytest.fixture(scope="module")
def api(world, tmp_path_factory):
    reg = Registry(tmp_path_factory.mktemp("apireg"))
    v = reg.register(world["b1"], world["metrics"])
    reg.promote(v, world["val"], GateConfig(first_model_floor=0.0))
    return TestClient(create_app(reg.root)), reg, world


def _payload(snap, k=60, targets=(0, 1, 2)):
    x, e = drift.induced(snap["x"], snap["edge_index"], np.arange(k))
    return dict(features=x.tolist(), edges=e.T.tolist(), target_ids=list(targets)), x, e


def test_api_matches_offline_predictions(api):
    c, reg, w = api
    body, x, e = _payload(w["snap"])
    r = c.post("/predict", json=body)
    assert r.status_code == 200 and r.json()["model_version"] == reg.approved
    off = reg.load_predictor().predict(x, e.T, body["target_ids"])
    for o, a in zip(off, r.json()["predictions"]):
        assert o["label"] == a["label"] and np.allclose(o["probabilities"], a["probabilities"], atol=1e-9)


@pytest.mark.parametrize("mutate,status,code", [
    (lambda b: b.update(target_ids=[1, 1]), 422, "duplicate_target_ids"),
    (lambda b: b.update(edges=b["edges"] + [[0, 999]]), 422, "invalid_edge_endpoint"),
    (lambda b: b.update(edges=b["edges"] + [[-1, 2]]), 422, "invalid_edge_endpoint"),
    (lambda b: b.update(features=[r[:-1] for r in b["features"]]), 422, "bad_feature_dim"),
    (lambda b: b.update(target_ids=[0, 60]), 404, "unknown_target_id"),
    (lambda b: b.update(target_ids=[]), 422, "empty_targets"),
])
def test_api_failure_cases(api, mutate, status, code):
    c, _, w = api
    body, _, _ = _payload(w["snap"])
    mutate(body)
    r = c.post("/predict", json=body)
    assert r.status_code == status and r.json()["error"]["code"] == code


def test_isolated_node_allowed_and_flagged(api):
    c, _, w = api
    body, _, _ = _payload(w["snap"])
    body["edges"] = [p for p in body["edges"] if 5 not in p]
    body["target_ids"] = [5]
    r = c.post("/predict", json=body)
    assert r.status_code == 200 and r.json()["predictions"][0]["isolated"] and r.json()["warnings"]


def test_duplicate_edges_and_self_loops_are_cleaned(api):
    c, _, w = api
    body, _, _ = _payload(w["snap"])
    base = c.post("/predict", json=body).json()["predictions"]
    body["edges"] = body["edges"] + body["edges"][:10] + [[3, 3]]
    again = c.post("/predict", json=body).json()["predictions"]
    assert all(np.allclose(a["probabilities"], b["probabilities"], atol=1e-9) for a, b in zip(base, again))


def test_model_loading_failure_returns_503(tmp_path):
    c = TestClient(create_app(tmp_path / "nothing"))
    assert c.get("/health").status_code == 503
    r = c.post("/predict", json=dict(features=[[0.0]], edges=[], target_ids=[0]))
    assert r.status_code == 503 and r.json()["error"]["code"] == "model_unavailable"


def test_unseen_node_inference_uses_supplied_context(api):
    c, reg, w = api
    snap, tg = w["snap"], w["tg"]
    pred = reg.load_predictor()
    un = tg.unseen_ids
    full = pred.predict_proba(snap["x"], snap["edge_index"].T)[un].argmax(1)
    assert (full == snap["y"][un]).mean() > 0.5   # supplying their edges lets the model classify them


# ------------------------------------------------------------------------ drift
def test_drift_detects_change_but_not_resampling(world):
    snap = world["snap"]
    pred = load_bundle(world["b1"])
    x, ei = snap["x"], snap["edge_index"]
    n = x.shape[0]
    perm = np.random.default_rng(0).permutation(n)
    A, B = perm[: n // 2], perm[n // 2:]
    th = drift.null_thresholds(pred, x, ei, n_splits=30)
    ref = drift.snapshot_stats(pred, *drift.induced(x, ei, A))
    xb, eb = drift.induced(x, ei, B)
    normal = drift.detect(ref, drift.snapshot_stats(pred, xb, eb), th)
    shifted = drift.detect(ref, drift.snapshot_stats(pred, xb, drift.drop_edges_np(eb, 0.7, 0)), th)
    assert sum(r["ALERT"] for r in normal) <= 1
    assert any(r["ALERT"] and r["statistic"] == "mean_degree_rel_change" for r in shifted)


def test_cka_is_rotation_invariant_but_raw_coordinates_are_not():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(200, 8))
    Q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    assert drift.linear_cka(E, E @ Q) == pytest.approx(1.0, abs=1e-6)
    assert np.abs(E - E @ Q).max() > 0.1
