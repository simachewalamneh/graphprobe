"""File-based model registry with a validation gate, promotion history and rollback.
(Kept dependency-free so it runs identically locally, in CI and in the serving image.)"""
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mlops.bundle import BundleError, load_bundle


#  metrics
def macro_f1(y_true, y_pred) -> float:
    f1s = []
    for c in np.unique(y_true):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))

def evaluate_predictor(pred, val) -> dict:
    """Score a predictor on the shared validation set (same features, graph, node ids, labels)."""
    p = pred.predict_proba(val["features"], val["edge_index"].T)[val["ids"]].argmax(1)
    return {"val_accuracy": float((p == val["y"]).mean()), "val_macro_f1": macro_f1(val["y"], p),
            "val_data_id": val["data_id"]}


#  gate
@dataclass
class GateConfig:
    metric: str = "val_macro_f1"
    min_improvement: float = 0.005     # candidate must beat the approved model by at least this
    tie_policy: str = "block"          # exact tie: "block" keeps the incumbent, "promote" accepts the candidate
    first_model_floor: float = 0.70    # no approved model yet: candidate must reach this absolute value


@dataclass
class Decision:
    allowed: bool
    rule: str
    reason: str
    metric: str
    candidate: float | None
    approved: float | None
    delta: float | None

def _bad(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))

def evaluate_gate(cand: dict, approved: dict | None, cfg: GateConfig) -> Decision:
    """Rules, evaluated in this order:
    1. candidate metric missing/NaN                -> BLOCK  (missing_metric)
    2. no approved model (first model)             -> allow iff candidate >= first_model_floor
    3. approved metric missing/NaN                 -> BLOCK  (fail closed: cannot prove no regression)
    4. exact tie (|delta| < 1e-12)                 -> tie_policy
    5. delta >= min_improvement                    -> ALLOW ; otherwise BLOCK
       (delta < 0 is a regression, 0 <= delta < min_improvement is insufficient improvement)"""
    c = cand.get(cfg.metric)
    a = None if approved is None else approved.get(cfg.metric)
    mk = lambda ok, rule, why, d=None: Decision(ok, rule, why, cfg.metric, c, a, d)
    if _bad(c):
        return mk(False, "missing_metric", f"candidate has no usable '{cfg.metric}'")
    if approved is None:
        ok = c >= cfg.first_model_floor
        return mk(ok, "first_model", f"first model: {c:.4f} {'>=' if ok else '<'} floor {cfg.first_model_floor}")
    if _bad(a):
        return mk(False, "approved_metric_missing", "approved model has no usable metric; failing closed")
    d = c - a
    if abs(d) < 1e-12:
        ok = cfg.tie_policy == "promote"
        return mk(ok, "tie", f"exact tie; tie_policy={cfg.tie_policy}", d)
    if d >= cfg.min_improvement:
        return mk(True, "improvement", f"+{d:.4f} >= required {cfg.min_improvement}", d)
    if d < 0:
        return mk(False, "regression", f"{d:.4f} is worse than the approved model", d)
    return mk(False, "insufficient_improvement", f"+{d:.4f} < required {cfg.min_improvement}", d)


#  registry
def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Registry:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.file = self.root / "registry.json"
        if not self.file.exists():
            self._save({"versions": {}, "approved": None, "history": [], "events": []})

    def _load(self):
        return json.loads(self.file.read_text())

    def _save(self, st):
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=2, sort_keys=True))
        os.replace(tmp, self.file)

    #  queries
    @property
    def approved(self):
        return self._load()["approved"]

    def versions(self):
        return self._load()["versions"]

    def bundle_path(self, version):
        return self.root / self._load()["versions"][version]["path"]

    def load_predictor(self, version=None):
        v = version or self.approved
        if v is None:
            raise BundleError("no approved model in registry")
        return load_bundle(self.bundle_path(v), version=v)

    #  mutations
    def register(self, bundle_dir, metrics, extra=None):
        st = self._load()
        v = f"v{len(st['versions']) + 1}"
        dest = self.root / "versions" / v
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(bundle_dir, dest)
        st["versions"][v] = {"path": f"versions/{v}", "metrics": metrics, "registered_at": _now(),
                             "status": "registered", **(extra or {})}
        st["events"].append({"time": _now(), "action": "register", "version": v})
        self._save(st)
        return v

    def check(self, candidate_pred, val, cfg: GateConfig) -> Decision:
        """Non-mutating gate evaluation of an in-memory candidate vs the approved model,
        both scored live on the SAME validation data."""
        cm = evaluate_predictor(candidate_pred, val)
        am = None
        if self.approved is not None:
            try:
                am = evaluate_predictor(self.load_predictor(), val)
            except Exception:                       # approved model unusable -> metric missing
                am = {cfg.metric: None}
        return evaluate_gate(cm, am, cfg)

    def promote(self, version, val, cfg: GateConfig) -> Decision:
        st = self._load()
        d = self.check(self.load_predictor(version), val, cfg)
        st["versions"][version]["gate"] = {**asdict(d), "time": _now()}
        st["events"].append({"time": _now(), "action": "promote_attempt", "version": version,
                             "allowed": d.allowed, "rule": d.rule})
        if d.allowed:
            if st["approved"] is not None:
                st["versions"][st["approved"]]["status"] = "archived"
            st["history"].append(version)
            st["approved"] = version
            st["versions"][version]["status"] = "approved"
        else:
            st["versions"][version]["status"] = "rejected"
        self._save(st)
        return d

    def rollback(self):
        """Restore the previously approved version (undo the latest promotion)."""
        st = self._load()
        if len(st["history"]) < 2:
            raise RuntimeError("nothing to roll back to (fewer than two approved versions)")
        bad = st["history"].pop()
        prev = st["history"][-1]
        st["versions"][bad]["status"] = "rolled_back"
        st["versions"][prev]["status"] = "approved"
        st["approved"] = prev
        st["events"].append({"time": _now(), "action": "rollback", "from": bad, "to": prev})
        self._save(st)
        return prev
