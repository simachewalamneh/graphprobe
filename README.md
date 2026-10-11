# graphprobe

**Probing when graph structure actually helps.** A from-scratch GCN, a Cora GNN benchmark, leakage-safe
link prediction, and an MLOps pipeline (tracking, packaging, API + Docker, gated promotion, rollback, drift).

Each assignment is a separate, self-contained package; shared code is limited to the scratch GCN (reused by `mlops/`)
and Cora loading (reused by `link_prediction/`).

| Folder | Assignment | Entry point |
|---|---|---|
| `scratch_gcn/` | GCN from scratch + verification + structure ablations | `python -m scratch_gcn.run` |
| `cora_benchmark/` | MLP / GCN / GraphSAGE / GAT comparison on Cora | `python -m cora_benchmark.run` |
| `link_prediction/` | Link predictor + deliberate leakage experiment | `python -m link_prediction.run` |
| `mlops/` | Tracked, packaged, served, gated, monitored GNN | `python -m mlops.cli --help` |
| `tests/` | Automated correctness / leakage / API / gate tests | `pytest -q` |
| `report/` | Technical report (all four) | `report/TECHNICAL_REPORT.md` |
| `notebooks/` | Executed notebooks with saved outputs | one per assignment |
| `results/` | Tables, plots, JSON produced by the runs | |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch                      # CPU is enough
pip install -r requirements.txt
pytest -q                              # all tests
```
Seeds: model seeds 0-4 (scratch GCN), 0-2 (Cora, link prediction), tuning seed 100, data/split seed 0.
Versions used for the shipped results: Python 3.12, torch 2.14.1, torch_geometric 2.8.0, mlflow 3.17.0.
Set `QUICK=1` for a fast smoke run of the Cora / link-prediction scripts.

## Run each assignment
```bash
python -m scratch_gcn.run
python -m cora_benchmark.run
python -m link_prediction.run

# MLOps
python -m mlops.baseline_untracked           # untracked baseline
python -m mlops.cli experiments              # >=3 MLflow-tracked runs, register, gated promotion
mlflow ui --backend-store-uri sqlite:///mlflow.db
python -m mlops.cli gate-demo                # first model / improvement / blocked regression / tie / rollback
python -m mlops.cli status
python -m mlops.cli unseen-demo              # existing vs unseen nodes
python -m mlops.cli drift-demo               # normal vs detected change + cross-version embeddings
python -m mlops.cli consistency-demo         # offline == API
python -m mlops.cli failure-demo             # isolated, duplicate, invalid, bad dim, unknown, load failure
python -m mlops.cli serve                    # http://localhost:8000/docs
python -m mlops.cli gate-check               # what CI runs
python -m mlops.cli rollback                 # restore previous approved model
docker build -t graphprobe-api . && docker run -p 8000:8000 graphprobe-api
```
Example request: `POST /predict {"features": [[...1433 numbers...], ...], "edges": [[0,1],[1,2]], "target_ids": [0,2]}`.
After `rollback` call `POST /admin/reload` so the running service re-reads the approved version.

## Design choices that make this repo different
* **Structure is tested, not assumed.** The synthetic dataset has a pure-noise feature; the GCN is compared with an MLP,
  with test-time edge removal, with a label-blind random graph, and across a homophily sweep.
* **Equal tuning budget:** every Cora model draws the *same* random hyper-parameter configs; selection uses validation only; test is read only for final reporting.
* **Leakage checks are themselves tested:** the same checker must pass on the valid graph and fail on the leaky one.
* **Gate compares live:** candidate and approved models are both scored on the identical validation set at gate time (no stale metrics); missing metrics fail closed.
* **Unseen nodes** are removed (with incident edges) from the training graph; inference needs graph context supplied in the request.
* **Drift thresholds come from data:** mean+3 std of the variation between random half-splits of the baseline.

## Credentials
No secrets are used or stored. `.env` files are git-ignored.
