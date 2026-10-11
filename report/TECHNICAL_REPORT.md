# graphprobe — technical report

All numbers below were produced by the scripts in this repo (see `results/`). Re-run them to reproduce.

## 1. GCN from scratch (`scratch_gcn/`)
**Dataset.** 200 nodes, 3 features, 2 balanced classes. Features 1–2 = class mean (±0.6) + N(0,1) noise, feature 3 = pure noise,
so features alone are only weakly informative. Edges: stochastic block model (p_in 0.08, p_out 0.01) → 897 undirected edges, 88.6% same-class.
Split 120/40/40 (train/val/test), labels never in X.
**Implementation.** Self-loops, 1/sqrt(d_i d_j) normalisation, message passing via `index_add_`, two layers, own training loop; no GNN library.
**Correctness.** One node's layer-1 output equals a hand computation (max error 1.8e-7). Relabelling nodes/features/edges and shuffling edge order gives the same rows reordered (max error 9.5e-7).

| Setting | Test acc (5 seeds) |
|---|---|
| MLP, features only | 0.840 ± 0.022 |
| GCN, true graph | 0.975 ± 0.000 |
| GCN, 50% edges removed at test time | 0.975 ± 0.018 |
| GCN, 100% removed | 0.825 |
| GCN retrained on random graph | 0.720 ± 0.108 |

**Evidence for structure use.** Gap GCN−MLP = +0.18 at homophily 0.89, +0.05 at 0.66, and −0.17/−0.20 at 0.57/0.50 (edges no longer informative). Caveat: the test set has only 40 nodes, so differences of a few points are noise.

## 2. Cora comparison (`cora_benchmark/`)
Cora: 2708 nodes, 1433 binary word features (≈18 words/doc), 7 classes, 5278 undirected edges, mean degree 3.9, edge homophily 0.81, Planetoid split 140/500/1000.
Each model got the same 12 random configs (same draws), 200 epochs, patience 50, selection by validation accuracy only; test read only for final numbers.

| Model | Test acc | Macro F1 | Train s | Params |
|---|---|---|---|---|
| MLP | 0.596 ± 0.010 | 0.575 ± 0.010 | 12.0 | 92,231 |
| GCN | 0.825 ± 0.008 | 0.811 ± 0.007 | 13.6 | 23,063 |
| GraphSAGE | 0.799 ± 0.004 | 0.790 ± 0.005 | 31.2 | 92,199 |
| GAT | 0.800 ± 0.016 | 0.790 ± 0.015 | 12.4 | 23,109 |

**Depth (test acc / mean pairwise cosine of last hidden embeddings), GCN:** 1 layer 0.780/0.36, 2 → 0.825/0.68, 4 → 0.539/0.71, 8 → 0.282/1.00. Embeddings collapse as depth grows; the MLP also degrades with depth, so part of the loss is optimisation difficulty, not only smoothing.
**Edge removal (train and test graph thinned):** GCN 0.825 → 0.559 at 100% removed (≈ MLP 0.596); GAT drops furthest (0.344), SAGE 0.608. GAT's attention has nothing to attend to and, with no self-signal beyond its own features, trains worse than the MLP.
**Errors.** Misclassification rate 0.516 on test nodes whose neighbours mostly have another class vs 0.105 otherwise. Sampled errors (`results/cora_benchmark/error_analysis.md`) include low-degree nodes whose single neighbour has another label, and a degree-44 node whose neighbourhood is 91% correct but is still predicted wrong (low-confidence, feature evidence outweighed).
**Recommendation.** GCN: highest accuracy/F1, fewest parameters, fast. Caveat: differences between GCN and SAGE/GAT are ~2.5 points on 3 seeds.

## 3. Link prediction and leakage (`link_prediction/`)
Unique undirected links split 85/5/10; both directions stay together; negatives exclude self-links and all observed links, fixed, disjoint across splits (checked in `leakage_checks.json` and `tests/`). The same checker **fails** on the leaky graph (`leakage_checks_leaky_graph.json`).
Message passing uses training links only. MLP decoder uses [z_u⊙z_v, |z_u−z_v|] so reversed pairs score identically (max diff 0.0).

| Setting | Method | Test AUC | Test AP |
|---|---|---|---|
| valid | common neighbours | 0.709 | 0.707 |
| valid | GCN + dot | 0.928 ± 0.001 | 0.934 ± 0.003 |
| valid | GCN + MLP | 0.942 ± 0.001 | 0.950 ± 0.001 |
| LEAKY (eval only) | GCN + dot / MLP | 0.974 / 0.996 | 0.972 / 0.994 |
| LEAKY (train + eval) | GCN + dot / MLP | 0.975 / 0.996 | 0.973 / 0.995 |
| LEAKY | common neighbours | 0.762 | 0.759 |

The leaky numbers are inflated because the test link is itself an edge in the graph the encoder aggregates over, so its endpoints see each other. They are never used for selection. Top-10 candidate links (`top10_candidate_links.csv`) are unobserved pairs the model scores highest; most involve high-degree nodes and share training neighbours. An unobserved link is not a confirmed negative (Cora is an incomplete citation sample), so candidates need external validation.

## 4. MLOps (`mlops/`)
Base model: the scratch GCN on Cora with 150 test nodes held out as "unseen" (removed with 1,030 incident edges from the training graph).
* **Untracked baseline:** `baseline_untracked.py`, limitations listed in its docstring.
* **Tracking:** MLflow (sqlite) with params, seed, git/code/dataset ids, metrics, bundle artifacts; three experiments (val macro-F1 0.7940, 0.7941, 0.7902).
* **Packaging:** weights, architecture, preprocessing, feature schema, label map, metadata + SHA-256; reload gives identical predictions (tested).
* **API/Docker:** FastAPI `/predict`, `/health`, `/model`, `/admin/reload`; Dockerfile provided (**not built in the authoring environment**).
* **Unseen nodes:** accuracy 0.847 with full context, 0.767 with half their edges, 0.727 as isolated nodes (features only). The service needs the features of the target and its neighbours, plus edges, in every request; degree normalisation depends on the supplied graph.
* **Gate:** metric val_macro_f1 on identical validation data (live re-scoring of both models); min improvement 0.005; exact tie blocked; missing metric blocks (fail closed); first model needs ≥0.70. Demo: weak first model promoted (0.735), better one promoted (+0.059), undertrained blocked (−0.464), tie blocked, missing metric blocked, rollback v2→v1.
* **CI:** `.github/workflows/ci.yml` runs pytest and `gate-check` (exit 1 on regression > tolerance). Locally the check passed (delta +0.0001). The workflow itself has not run on GitHub yet.
* **Drift:** thresholds = mean+3σ of random half-split variation. Normal case: no alerts. Edge removal: mean-degree, isolated-fraction and embedding-norm alerts. Vocabulary loss: embedding centroid and norm alerts.
* **Cross-version embeddings:** two seeds give same-index coordinate correlation 0.34 but linear CKA 0.93. Raw coordinates are only defined up to rotation/permutation of hidden units; use CKA or similar. (Raw centroid cosine was 0.97 — misleadingly high because ReLU embeddings are non-negative; do not use it.)
* **Failure handling:** table in `results/mlops/failure_handling.md`; offline vs API predictions identical (max diff 0).

## 5. One significant implementation problem
**Problem.** The long Cora job (tuning ≈ 30 min + depth + edge studies) stopped partway with no error and only partial CSVs. **Investigation.** The tuning and final CSVs were present but `edge_removal.csv` was missing and no process was running, so the run had been killed, not failed. **Fix.** Because tuning is deterministic (seeded sampler), I added resume logic that reuses `best_configs.json` and `depth_study.csv`, and re-ran only the remaining steps. A second, smaller problem: the first gate demo used a "weak" first model below the 0.70 floor, so nothing was approved and the rollback step crashed; I chose a weaker-but-valid config (0.735) and the demo now exercises every rule.

## Limitations
Scratch-GCN test set is 40 nodes; Cora uses 3 seeds; link prediction uses one split seed; Docker and GitHub Actions were not executed here.
