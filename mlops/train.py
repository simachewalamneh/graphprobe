"""Tracked training: trains the from-scratch GCN on the training graph (unseen nodes excluded),
packages a bundle, evaluates the *reloaded* bundle, and logs everything to MLflow."""
import hashlib
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from mlops.bundle import load_bundle, row_l1_normalize, save_bundle
from mlops.data import make_training_graph, validation_set
from mlops.registry import evaluate_predictor, macro_f1
from scratch_gcn.core import GCN, prep_graph, set_seed, train_model

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = dict(hidden=16, lr=0.01, wd=5e-4, dropout=0.5, epochs=200, seed=0, n_unseen=150)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def code_id() -> str:
    h = hashlib.sha256()
    for p in sorted(list((ROOT / "mlops").glob("*.py")) + [ROOT / "scratch_gcn" / "core.py"]):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def train_bundle(snap, params, out_dir):
    """Train + package. Returns (metrics, tg, val)."""
    p = {**DEFAULTS, **params}
    tg = make_training_graph(snap, p["n_unseen"], seed=0)
    n, f, c = tg.x.shape[0], tg.x.shape[1], int(snap["y"].max()) + 1
    x = torch.from_numpy(row_l1_normalize(tg.x))
    y = torch.from_numpy(tg.y)
    g = prep_graph(torch.from_numpy(tg.edge_index), n)
    masks = {k: torch.from_numpy(getattr(tg, f"{k}_mask")) for k in ("train", "val", "test")}
    set_seed(p["seed"])
    model = GCN(f, p["hidden"], c, p["dropout"])
    t0 = time.perf_counter()
    train_model(model, x, y, g, masks, epochs=p["epochs"], lr=p["lr"], wd=p["wd"])
    secs = time.perf_counter() - t0
    meta = dict(seed=p["seed"], dataset_id=tg.data_id, git_sha=git_sha(), code_id=code_id(),
                n_unseen=p["n_unseen"], hyperparameters={k: p[k] for k in ("hidden", "lr", "wd", "dropout", "epochs")})
    save_bundle(out_dir, model,
                arch=dict(type="scratch_gcn", **{"in": f}, hidden=p["hidden"], out=c, dropout=p["dropout"]),
                preprocessing=dict(type="row_l1_normalize", eps=1e-12),
                schema=dict(n_features=f, dtype="float32", description="raw binary bag-of-words per node",
                            node_ids="row index of the features matrix", edges="[[src,dst],...] undirected"),
                label_map={str(i): f"class_{i}" for i in range(c)}, metadata=meta)
    pred = load_bundle(out_dir)                       # evaluate the packaged artifact, not the live object
    val = validation_set(tg)
    m = evaluate_predictor(pred, val)
    # test accuracy on seen nodes (training-graph context)
    tid = np.nonzero(tg.test_mask)[0]
    tp = pred.predict_proba(tg.x, tg.edge_index.T)[tid].argmax(1)
    m["test_accuracy_seen"] = float((tp == tg.y[tid]).mean())
    # unseen nodes: full graph (their features + incident edges) supplied as inference-time context
    full = pred.predict_proba(snap["x"], snap["edge_index"].T)[tg.unseen_ids].argmax(1)
    m["unseen_accuracy"] = float((full == snap["y"][tg.unseen_ids]).mean())
    m["train_seconds"] = secs
    return m, tg, val, meta


def tracked_run(snap, name, params, registry=None, experiment="graphprobe-gcn"):
    """Train one run, log it to MLflow (params, seed, code/dataset ids, metrics, model bundle),
    register it in `registry` (if given) and return (version, metrics, run_id)."""
    import mlflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=name) as run:
        with tempfile.TemporaryDirectory() as tmp:
            m, tg, val, meta = train_bundle(snap, params, Path(tmp) / "bundle")
            p = {**DEFAULTS, **params}
            mlflow.log_params({**{k: p[k] for k in ("hidden", "lr", "wd", "dropout", "epochs", "seed", "n_unseen")},
                               "model": "scratch_gcn"})
            mlflow.set_tags({"git_sha": meta["git_sha"], "code_id": meta["code_id"], "dataset_id": tg.data_id,
                             "validation_data_id": val["data_id"]})
            mlflow.log_metrics({k: v for k, v in m.items() if isinstance(v, float)})
            mlflow.log_artifacts(Path(tmp) / "bundle", artifact_path="model_bundle")
            version = None
            if registry is not None:
                version = registry.register(Path(tmp) / "bundle", m, extra=dict(
                    mlflow_run_id=run.info.run_id, run_name=name, hyperparameters=meta["hyperparameters"]))
                mlflow.set_tag("registry_version", version)
    return version, m, run.info.run_id
