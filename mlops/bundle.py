"""Model packaging + inference. A bundle directory holds everything needed to serve:
weights, architecture, preprocessing, feature schema, label mapping, metadata (+checksum)."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scratch_gcn.core import GCN, prep_graph

FILES = ["weights.pt", "architecture.json", "preprocessing.json", "feature_schema.json",
         "label_map.json", "metadata.json"]
MAX_NODES = 50_000


class BundleError(Exception):
    """Model bundle missing/corrupt/incompatible."""


class InputError(Exception):
    def __init__(self, code: str, message: str, status: int = 422):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def row_l1_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(x.sum(1, keepdims=True), eps)


class Predictor:
    def __init__(self, model, arch, preprocessing, schema, label_map, metadata, version=None):
        self.model, self.arch, self.preprocessing = model.eval(), arch, preprocessing
        self.schema, self.label_map, self.metadata = schema, label_map, metadata
        self.version = version or metadata.get("bundle_id", "unversioned")
        self.n_features = schema["n_features"]

    # ------------------------------------------------------------- input handling
    def _clean(self, features, edges, target_ids=None):
        try:
            x = np.asarray(features, dtype=np.float64)
        except (ValueError, TypeError):
            raise InputError("bad_features", "features must be a rectangular numeric matrix")
        if x.ndim != 2 or x.shape[0] == 0:
            raise InputError("bad_features", "features must be a non-empty 2-D matrix [nodes x features]")
        if x.shape[1] != self.n_features:
            raise InputError("bad_feature_dim", f"expected {self.n_features} features per node, got {x.shape[1]}")
        if not np.isfinite(x).all():
            raise InputError("bad_features", "features contain NaN or inf")
        n = x.shape[0]
        if n > MAX_NODES:
            raise InputError("too_many_nodes", f"at most {MAX_NODES} nodes per request", 413)
        edge_list = [] if edges is None else edges
        if len(edge_list) == 0:
            e = np.empty((0, 2), np.int64)
        else:
            try:
                e = np.asarray(edge_list, dtype=np.int64)
            except (ValueError, TypeError):
                raise InputError("bad_edges", "edges must be a list of [src, dst] integer pairs")
            if e.ndim != 2 or e.shape[1] != 2:
                raise InputError("bad_edges", "edges must be a list of [src, dst] pairs")
        if e.size and (e.min() < 0 or e.max() >= n):
            raise InputError("invalid_edge_endpoint", f"edge endpoints must be in [0, {n - 1}]")
        e = e[e[:, 0] != e[:, 1]]                                    # self-loops are added internally
        e = np.concatenate([e, e[:, ::-1]]) if len(e) else e         # undirected: symmetrise
        e = np.unique(e, axis=0) if len(e) else e                    # de-duplicate
        t = None
        if target_ids is not None:
            t = list(target_ids)
            if len(t) == 0:
                raise InputError("empty_targets", "target_ids must not be empty")
            if len(set(t)) != len(t):
                raise InputError("duplicate_target_ids", "target_ids contains duplicates")
            bad = [i for i in t if not isinstance(i, (int, np.integer)) or i < 0 or i >= n]
            if bad:
                raise InputError("unknown_target_id", f"unknown target ids: {bad[:5]} (valid: 0..{n - 1})", 404)
        return x.astype(np.float32), e.T.copy(), t, n

    def _forward(self, x, ei, n, embed=False):
        xt = torch.from_numpy(row_l1_normalize(x, self.preprocessing["eps"]))
        g = prep_graph(torch.from_numpy(ei.astype(np.int64)).reshape(2, -1), n)
        with torch.no_grad():
            return self.model.embed(xt, g) if embed else self.model(xt, g)

    # ------------------------------------------------------------------- public
    def predict(self, features, edges, target_ids):
        x, ei, t, n = self._clean(features, edges, target_ids)
        prob = self._forward(x, ei, n).softmax(1).numpy()
        deg = np.bincount(ei[0], minlength=n) if ei.size else np.zeros(n, int)
        out = []
        for i in t:
            k = int(prob[i].argmax())
            out.append(dict(node_id=int(i), label=k, label_name=self.label_map[str(k)],
                            probabilities=[float(p) for p in prob[i]], isolated=bool(deg[i] == 0)))
        return out

    def predict_proba(self, features, edges):
        x, ei, _, n = self._clean(features, edges)
        return self._forward(x, ei, n).softmax(1).numpy()

    def embeddings(self, features, edges):
        x, ei, _, n = self._clean(features, edges)
        return self._forward(x, ei, n, embed=True).numpy()


def save_bundle(path, model, arch, preprocessing, schema, label_map, metadata):
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "weights.pt")
    metadata = {**metadata, "weights_sha256": sha256_file(path / "weights.pt")}
    metadata["bundle_id"] = metadata["weights_sha256"][:12]
    for name, obj in [("architecture", arch), ("preprocessing", preprocessing), ("feature_schema", schema),
                      ("label_map", label_map), ("metadata", metadata)]:
        (path / f"{name}.json").write_text(json.dumps(obj, indent=2, sort_keys=True))
    return path


def load_bundle(path, version=None) -> Predictor:
    path = Path(path)
    missing = [f for f in FILES if not (path / f).exists()]
    if missing:
        raise BundleError(f"bundle {path} is missing files: {missing}")
    try:
        arch, pre, schema, labels, meta = (json.loads((path / f"{n}.json").read_text()) for n in
                                           ("architecture", "preprocessing", "feature_schema", "label_map", "metadata"))
        if sha256_file(path / "weights.pt") != meta.get("weights_sha256"):
            raise BundleError("weights checksum mismatch (corrupt or tampered bundle)")
        if arch["type"] != "scratch_gcn":
            raise BundleError(f"unsupported architecture {arch['type']}")
        model = GCN(arch["in"], arch["hidden"], arch["out"], arch["dropout"])
        model.load_state_dict(torch.load(path / "weights.pt", map_location="cpu", weights_only=True))
    except BundleError:
        raise
    except Exception as e:  # json / torch / shape errors
        raise BundleError(f"failed to load bundle {path}: {e}") from e
    return Predictor(model, arch, pre, schema, labels, meta, version)
