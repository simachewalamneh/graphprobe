"""Datasets for the MLOps pipeline: a frozen Cora snapshot (no PyG needed at serve/CI time),
a synthetic fallback for fast tests, and the 'unseen nodes' training graph."""
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets"
SNAPSHOT = ASSETS / "cora_snapshot.npz"
N_FEATURES_CORA = 1433


def build_cora_snapshot(path=SNAPSHOT):
    """One-off: export Planetoid-Cora (raw binary bag-of-words) to a small .npz."""
    from torch_geometric.datasets import Planetoid
    d = Planetoid(str(Path(__file__).resolve().parents[1] / "data"), "Cora")[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, x_packed=np.packbits(d.x.numpy().astype(np.uint8), axis=1), n_features=d.x.shape[1],
        edge_index=d.edge_index.numpy().astype(np.int32), y=d.y.numpy().astype(np.int8),
        train_mask=d.train_mask.numpy(), val_mask=d.val_mask.numpy(), test_mask=d.test_mask.numpy())
    return path


def make_synthetic(n=300, f=60, c=3, seed=0):
    """Tiny class-conditional bag-of-words graph (used by tests; no downloads)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, c, n)
    probs = np.full((c, f), 0.04)
    for k in range(c):
        probs[k, k * (f // c):(k + 1) * (f // c)] = 0.30
    x = (rng.random((n, f)) < probs[y]).astype(np.float32)
    same = y[:, None] == y[None, :]
    A = np.triu(rng.random((n, n)) < np.where(same, 0.05, 0.004), 1)
    A = A | A.T
    s, d = np.nonzero(A)
    perm = rng.permutation(n)
    masks = {k: np.zeros(n, bool) for k in ("train_mask", "val_mask", "test_mask")}
    masks["train_mask"][perm[: n // 3]] = True
    masks["val_mask"][perm[n // 3: n // 2]] = True
    masks["test_mask"][perm[n // 2:]] = True
    return dict(x=x, edge_index=np.stack([s, d]).astype(np.int64), y=y.astype(np.int64), **masks)


def load_snapshot(kind="cora"):
    if kind == "synthetic":
        return make_synthetic()
    z = np.load(SNAPSHOT)
    x = np.unpackbits(z["x_packed"], axis=1)[:, : int(z["n_features"])].astype(np.float32)
    return dict(x=x, edge_index=z["edge_index"].astype(np.int64), y=z["y"].astype(np.int64),
                train_mask=z["train_mask"], val_mask=z["val_mask"], test_mask=z["test_mask"])


def dataset_id(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()[:12]


@dataclass
class TrainGraph:
    """Training graph WITHOUT the unseen nodes (and without their incident edges).
    Node ids here are re-indexed 0..m-1; `orig_ids[i]` maps back to the snapshot id."""
    x: np.ndarray
    edge_index: np.ndarray
    y: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    orig_ids: np.ndarray
    unseen_ids: np.ndarray
    data_id: str


def make_training_graph(snap, n_unseen=150, seed=0) -> TrainGraph:
    n = snap["x"].shape[0]
    rng = np.random.default_rng(seed)
    test_ids = np.nonzero(snap["test_mask"])[0]  # unseen nodes come from the test pool (labels known)
    unseen = np.sort(rng.choice(test_ids, size=min(n_unseen, len(test_ids)), replace=False))
    keep = np.ones(n, bool); keep[unseen] = False
    remap = -np.ones(n, np.int64); remap[keep] = np.arange(keep.sum())
    ei = snap["edge_index"]
    ok = keep[ei[0]] & keep[ei[1]]            # drop every edge touching an unseen node
    new_ei = remap[ei[:, ok]]
    return TrainGraph(
        x=snap["x"][keep], edge_index=new_ei, y=snap["y"][keep],
        train_mask=snap["train_mask"][keep], val_mask=snap["val_mask"][keep],
        test_mask=snap["test_mask"][keep], orig_ids=np.nonzero(keep)[0], unseen_ids=unseen,
        data_id=dataset_id(snap["x"], snap["edge_index"], snap["y"], unseen))


def validation_set(tg: TrainGraph):
    """The fixed validation data every candidate and the approved model are scored on."""
    ids = np.nonzero(tg.val_mask)[0]
    return dict(features=tg.x, edge_index=tg.edge_index, ids=ids, y=tg.y[ids],
                data_id=dataset_id(tg.x, tg.edge_index, ids, tg.y[ids]))
