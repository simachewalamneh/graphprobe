"""Live-demo helpers: unseen nodes, drift detection, offline/API consistency, failure handling."""
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from mlops import drift
from mlops.data import make_training_graph
from mlops.train import train_bundle


def _acc(pred, x, ei, ids, y):
    return float((pred.predict_proba(x, ei.T)[ids].argmax(1) == y[ids]).mean())


def _save(out, name, obj):
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(obj, indent=2, default=float))


# unseen nodes
def unseen_demo(reg, snap, out):
    pred = reg.load_predictor()
    tg = make_training_graph(snap, 150, 0)
    x, ei, y, un = snap["x"], snap["edge_index"], snap["y"], tg.unseen_ids
    assert not set(un.tolist()) & set(tg.orig_ids.tolist()), "unseen node leaked into the training graph"
    incident = np.isin(ei[0], un) | np.isin(ei[1], un)
    assert tg.edge_index.shape[1] == ei.shape[1] - incident.sum(), "incident edges not removed"
    seen_test = tg.orig_ids[np.nonzero(tg.test_mask)[0]]
    res = {"approved_version": pred.version, "n_unseen": int(len(un)),
           "edges_removed_from_training_graph": int(incident.sum())}
    res["existing_nodes_acc (training-graph context)"] = _acc(pred, tg.x, tg.edge_index, np.nonzero(tg.test_mask)[0], tg.y)
    res["unseen_acc, full context (features + all incident edges)"] = _acc(pred, x, ei, un, y)
    rng = np.random.default_rng(0)
    # drop ~50% of the (undirected) edges touching unseen nodes
    u = ei[:, ei[0] < ei[1]]
    touch = np.isin(u[0], un) | np.isin(u[1], un)
    kill = touch & (rng.random(u.shape[1]) < 0.5)
    ku = u[:, ~kill]
    res["unseen_acc, 50% of their edges missing"] = _acc(pred, x, np.concatenate([ku, ku[::-1]], 1), un, y)
    no_edges = ei[:, ~incident]
    res["unseen_acc, isolated (features only)"] = _acc(pred, x, no_edges, un, y)
    for k, v in res.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    _save(out, "unseen_nodes_demo.json", res)


# drift
def drift_demo(reg, snap, out):
    pred = reg.load_predictor()
    x, ei = snap["x"], snap["edge_index"]
    n = x.shape[0]
    perm = np.random.default_rng(0).permutation(n)
    A, B = perm[: n // 2], perm[n // 2: 2 * (n // 2)]
    ref = drift.snapshot_stats(pred, *drift.induced(x, ei, A))
    th = drift.null_thresholds(pred, x, ei, n_splits=100, k=3.0)
    xb, eb = drift.induced(x, ei, B)
    scenarios = {"normal (new random half of same data)": (xb, eb),
                 "graph drift (50% of edges removed)": (xb, drift.drop_edges_np(eb, 0.5, 1)),
                 "embedding drift (60% of vocabulary columns vanish)": (drift.corrupt_features(xb, 0.6, 2), eb)}
    out_rows = {"thresholds(null: mean+3*std over 100 random half-splits)": th}
    for name, (xs, es) in scenarios.items():
        rows = drift.detect(ref, drift.snapshot_stats(pred, xs, es), th)
        out_rows[name] = rows
        print(f"\n== {name}")
        for r in rows:
            print(f"  {r['statistic']:<38} value={r['value']:<9} thr={r['threshold']:<9} {'ALERT' if r['ALERT'] else 'ok'}")
    # why raw coordinates of different model versions are not comparable
    meta = pred.metadata["hyperparameters"]
    with tempfile.TemporaryDirectory() as tmp:
        train_bundle(snap, {**meta, "seed": 1}, Path(tmp) / "b")
        from mlops.bundle import load_bundle
        p2 = load_bundle(Path(tmp) / "b")
        e1, e2 = pred.embeddings(x, ei.T), p2.embeddings(x, ei.T)
    cors = [abs(np.corrcoef(e1[:, i], e2[:, i])[0, 1]) for i in range(e1.shape[1]) if e1[:, i].std() > 0 and e2[:, i].std() > 0]
    c1, c2 = e1.mean(0), e2.mean(0)
    cmp = {"mean |corr| of same-index coordinates": float(np.mean(cors)),
           "cosine of centroids (raw coordinates)": float(c1 @ c2 / (np.linalg.norm(c1) * np.linalg.norm(c2))),
           "linear CKA (rotation/permutation invariant)": drift.linear_cka(e1, e2)}
    print("\nTwo model versions (seeds 0 vs 1) on the same graph:")
    for k, v in cmp.items():
        print(f"  {k}: {v:.4f}")
    out_rows["cross_version_embedding_comparison"] = cmp
    _save(out, "drift_demo.json", out_rows)


#  consistency
def _subgraph(snap, k=300, seed=0):
    nodes = np.sort(np.random.default_rng(seed).choice(snap["x"].shape[0], k, replace=False))
    return drift.induced(snap["x"], snap["edge_index"], nodes)


def consistency_demo(reg, snap, out):
    from fastapi.testclient import TestClient
    from mlops.service import create_app
    x, ei = _subgraph(snap)
    targets = list(range(0, 40))
    offline = reg.load_predictor().predict(x, ei.T, targets)
    client = TestClient(create_app(reg.root))
    r = client.post("/predict", json={"features": x.tolist(), "edges": ei.T.tolist(), "target_ids": targets})
    api = r.json()["predictions"]
    same_label = all(o["label"] == a["label"] for o, a in zip(offline, api))
    max_diff = max(abs(p - q) for o, a in zip(offline, api) for p, q in zip(o["probabilities"], a["probabilities"]))
    res = {"status_code": r.status_code, "model_version_api": r.json()["model_version"],
           "model_version_offline": reg.approved, "labels_identical": same_label, "max_abs_prob_diff": max_diff}
    print(res)
    assert same_label and max_diff < 1e-6
    _save(out, "consistency_demo.json", res)


# failure handling
def failure_demo(reg, snap, out):
    from fastapi.testclient import TestClient
    from mlops.service import create_app
    x, ei = _subgraph(snap, k=100)
    e = ei.T.tolist()
    xl = x.tolist()
    iso = int(np.argmax(np.bincount(ei[0], minlength=100) == 0)) if (np.bincount(ei[0], minlength=100) == 0).any() else 99
    e_iso = [p for p in e if iso not in p]
    c = TestClient(create_app(reg.root))
    cases = [
        ("isolated target node", dict(features=xl, edges=e_iso, target_ids=[iso]), 200, "prediction from own features; isolated=true + warning"),
        ("duplicate target ids", dict(features=xl, edges=e, target_ids=[1, 1]), 422, "duplicate_target_ids"),
        ("invalid edge endpoint", dict(features=xl, edges=e + [[0, 5000]], target_ids=[1]), 422, "invalid_edge_endpoint"),
        ("negative edge endpoint", dict(features=xl, edges=e + [[-1, 2]], target_ids=[1]), 422, "invalid_edge_endpoint"),
        ("wrong feature dimension", dict(features=[r[:-1] for r in xl], edges=e, target_ids=[1]), 422, "bad_feature_dim"),
        ("unknown target id", dict(features=xl, edges=e, target_ids=[1, 100]), 404, "unknown_target_id"),
        ("empty target list", dict(features=xl, edges=e, target_ids=[]), 422, "empty_targets"),
        ("duplicate + self-loop edges", dict(features=xl, edges=e + e[:5] + [[3, 3]], target_ids=[3]), 200, "cleaned silently"),
    ]
    lines = ["| case | expected | observed |", "|---|---|---|"]
    ok_all = True
    for name, payload, status, expect in cases:
        r = c.post("/predict", json=payload)
        code = r.json().get("error", {}).get("code", "-")
        ok = r.status_code == status
        ok_all &= ok
        lines.append(f"| {name} | {status}: {expect} | {r.status_code}: {code} {'OK' if ok else 'MISMATCH'} |")
    with tempfile.TemporaryDirectory() as tmp:                      # model-loading failures
        empty = TestClient(create_app(Path(tmp) / "empty"))
        r = empty.post("/predict", json=cases[0][1])
        lines.append(f"| no approved model / empty registry | 503: model_unavailable | {r.status_code}: {r.json()['error']['code']} |")
        ok_all &= r.status_code == 503
        shutil.copytree(reg.root, Path(tmp) / "bad")
        bad = Path(tmp) / "bad"
        w = bad / "versions" / reg.approved / "weights.pt"
        w.write_bytes(w.read_bytes()[:-8] + b"corrupt!")
        r = TestClient(create_app(bad)).get("/health")
        lines.append(f"| corrupted weights (checksum mismatch) | 503: model_unavailable | {r.status_code}: {r.json()['error']['code']} |")
        ok_all &= r.status_code == 503
    print("\n".join(lines))
    out.mkdir(parents=True, exist_ok=True)
    (out / "failure_handling.md").write_text("\n".join(lines) + "\n")
    assert ok_all, "some failure-handling expectations did not hold"
