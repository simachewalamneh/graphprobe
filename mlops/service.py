"""FastAPI inference service. Serves the *approved* registry version.
Run:  uvicorn mlops.service:app --port 8000   (env REGISTRY_DIR overrides the registry location)"""
import os
from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from mlops.bundle import BundleError, InputError
from mlops.registry import Registry

DEFAULT_REGISTRY = Path(__file__).resolve().parent / "registry"


class PredictRequest(BaseModel):
    features: List[List[float]]       # row i = features of node i (node id == row index)
    edges: List[List[int]] = []       # [[src, dst], ...]; treated as undirected
    target_ids: List[int]             # nodes to classify


def create_app(registry_dir=None) -> FastAPI:
    app = FastAPI(title="GraphProbe node-classification service", version="1.0")
    reg_dir = Path(registry_dir or os.environ.get("REGISTRY_DIR", DEFAULT_REGISTRY))
    state = {"predictor": None, "error": None}

    def load():
        try:
            state["predictor"], state["error"] = Registry(reg_dir).load_predictor(), None
        except Exception as e:  # missing registry, no approved model, corrupt bundle, ...
            state["predictor"], state["error"] = None, f"{type(e).__name__}: {e}"

    load()

    def unavailable():
        return JSONResponse(status_code=503, content={"error": {"code": "model_unavailable", "message": state["error"]}})

    @app.get("/health")
    def health():
        if state["predictor"] is None:
            return unavailable()
        return {"status": "ok", "model_version": state["predictor"].version}

    @app.get("/model")
    def model():
        if state["predictor"] is None:
            return unavailable()
        p = state["predictor"]
        return {"model_version": p.version, "bundle_id": p.metadata["bundle_id"], "dataset_id": p.metadata["dataset_id"],
                "git_sha": p.metadata["git_sha"], "feature_schema": p.schema, "labels": p.label_map}

    @app.post("/predict")
    def predict(req: PredictRequest):
        p = state["predictor"]
        if p is None:
            return unavailable()
        try:
            preds = p.predict(req.features, req.edges, req.target_ids)
        except InputError as e:
            return JSONResponse(status_code=e.status, content={"error": {"code": e.code, "message": e.message}})
        warnings = [f"node {r['node_id']} has no edges: prediction uses its own features only" for r in preds if r["isolated"]]
        return {"model_version": p.version, "bundle_id": p.metadata["bundle_id"],
                "dataset_id": p.metadata["dataset_id"], "predictions": preds, "warnings": warnings}

    @app.post("/admin/reload")
    def reload():
        """Re-read the approved version from the registry (used after promotion / rollback)."""
        load()
        return health()

    return app


app = create_app()
