"""Assignment: link prediction on Cora + data-leakage investigation.
Run:   python -m link_prediction.run          (QUICK=1 for a smoke test)
Outputs -> results/link_prediction/. Node labels are NEVER used for training;
they are only read after the fact to describe candidate links."""
import itertools
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from cora_benchmark.data import load_cora
from link_prediction.core import (both_directions, common_neighbor_scores, eval_split, leakage_checks,
                                  make_eval_negatives, metrics, observed_matrix, score_pairs,
                                  split_links, top_unobserved, train_adjacency, train_link)

OUT = Path(__file__).resolve().parents[1] / "results" / "link_prediction"
QUICK = os.environ.get("QUICK") == "1"
EPOCHS = 20 if QUICK else 200
SEEDS = [0] if QUICK else [0, 1, 2]
GRID = list(itertools.product([0.01, 0.005], [0.0, 0.5]))  # (lr, dropout)
BASE = dict(hidden=128, out=64, wd=0.0)


def pm(v):
    return f"{np.mean(v):.4f} ± {np.std(v):.4f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, _ = load_cora()
    x, n = data.x, data.num_nodes
    labels = data.y  # post-hoc description of candidates only
    full_ei = data.edge_index
    split = split_links(full_ei, val_frac=0.05, test_frac=0.10, seed=0)
    forbidden = observed_matrix(n, full_ei)                 # ALL observed positives + diagonal
    val_neg, test_neg = make_eval_negatives(n, forbidden, split, seed=0)  # fixed once
    mp_valid = both_directions(split["train"])              # training links only
    mp_leaky = both_directions(torch.cat([split["train"], split["val"], split["test"]], 1))

    checks = leakage_checks(n, mp_valid, split, val_neg, test_neg, full_ei)
    checks["_sizes"] = {k: int(v.shape[1]) for k, v in split.items()} | {
        "val_neg": int(val_neg.shape[1]), "test_neg": int(test_neg.shape[1])}
    json.dump(checks, open(OUT / "leakage_checks.json", "w"), indent=2)
    print(checks)
    assert all(v for k, v in checks.items() if k != "_sizes"), "leakage check failed"
    # the deliberately leaky graph MUST fail the same check (proves the check can detect leaks)
    leaky_checks = leakage_checks(n, mp_leaky, split, val_neg, test_neg, full_ei)
    assert not leaky_checks["test_links_absent_from_mp_graph"]
    json.dump(leaky_checks, open(OUT / "leakage_checks_leaky_graph.json", "w"), indent=2)

    # ---- model selection: validation AUC on the valid pipeline only
    sel, tune_rows = {}, []
    for dec in ("dot", "mlp"):
        best = (-1, None)
        for lr, dp in GRID:
            cfg = {**BASE, "lr": lr, "dropout": dp}
            _, v = train_link(dec, cfg, x, mp_valid, split, val_neg, test_neg, forbidden, seed=100, epochs=EPOCHS)
            tune_rows.append(dict(decoder=dec, lr=lr, dropout=dp, val_auc=v))
            best = max(best, (v, cfg), key=lambda t: t[0])
        sel[dec] = best[1]
    pd.DataFrame(tune_rows).to_csv(OUT / "tuning.csv", index=False)
    print("selected:", sel)

    # ---- valid experiment (+ leaky experiments, never used for selection)
    A_tr = train_adjacency(n, split["train"])
    cn_auc, cn_ap = metrics(common_neighbor_scores(A_tr, split["test"]), common_neighbor_scores(A_tr, test_neg))
    A_leak = train_adjacency(n, torch.cat([split["train"], split["val"], split["test"]], 1))
    cnl_auc, cnl_ap = metrics(common_neighbor_scores(A_leak, split["test"]), common_neighbor_scores(A_leak, test_neg))
    rows = [dict(setting="valid", method="common-neighbours", seed=-1, auc=cn_auc, ap=cn_ap),
            dict(setting="LEAKY (cn over all links)", method="common-neighbours", seed=-1, auc=cnl_auc, ap=cnl_ap)]
    models = {}
    for dec in ("dot", "mlp"):
        for s in SEEDS:
            m, val_auc = train_link(dec, sel[dec], x, mp_valid, split, val_neg, test_neg, forbidden, s, EPOCHS)
            a, p = eval_split(m, x, mp_valid, split["test"], test_neg)
            rows.append(dict(setting="valid", method=f"GCN+{dec}", seed=s, auc=a, ap=p, val_auc=val_auc))
            models[(dec, s)] = (m, val_auc)
            # leaky A: trained validly, evaluated with held-out links inside message passing
            a, p = eval_split(m, x, mp_leaky, split["test"], test_neg)
            rows.append(dict(setting="LEAKY eval only", method=f"GCN+{dec}", seed=s, auc=a, ap=p))
            # leaky B: trained AND evaluated with held-out links in message passing
            ml, _ = train_link(dec, sel[dec], x, mp_leaky, split, val_neg, test_neg, forbidden, s, EPOCHS)
            a, p = eval_split(ml, x, mp_leaky, split["test"], test_neg)
            rows.append(dict(setting="LEAKY train+eval", method=f"GCN+{dec}", seed=s, auc=a, ap=p))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "all_runs.csv", index=False)
    summ = res.groupby(["setting", "method"], sort=False).agg(
        test_auc=("auc", pm), test_ap=("ap", pm), runs=("auc", "count"))
    summ.to_csv(OUT / "results_valid_vs_leaky.csv")
    print(summ)

    # ---- decoder symmetry check (reversed pairs get the same score)
    m_mlp = models[("mlp", SEEDS[0])][0]
    sc_f = score_pairs(m_mlp, x, mp_valid, split["test"])
    sc_r = score_pairs(m_mlp, x, mp_valid, split["test"].flip(0))
    sym_err = float(np.abs(sc_f - sc_r).max())
    json.dump({"mlp_decoder_max_abs_diff_on_reversed_pairs": sym_err}, open(OUT / "decoder_symmetry.json", "w"))
    assert sym_err < 1e-6

    # ---- best valid model by validation AUC -> candidates and failure cases
    best_key = max(models, key=lambda k: models[k][1])
    best_model = models[best_key][0]
    cand = top_unobserved(best_model, x, mp_valid, forbidden, k=10)
    deg = A_tr.sum(1)
    cn_mat = A_tr @ A_tr

    def describe(u, v, score):
        return dict(u=int(u), v=int(v), score=round(float(score), 4), train_common_neighbours=int(cn_mat[u, v]),
                    deg_u=int(deg[u]), deg_v=int(deg[v]), same_topic_posthoc=bool(labels[u] == labels[v]))
    pd.DataFrame([describe(*c) for c in cand]).to_csv(OUT / "top10_candidate_links.csv", index=False)

    s_pos = score_pairs(best_model, x, mp_valid, split["test"])
    s_neg = score_pairs(best_model, x, mp_valid, test_neg)
    fp = np.argsort(-s_neg)[:5]
    fn = np.argsort(s_pos)[:5]
    fail = [dict(kind="false_positive (negative scored high)", **describe(*test_neg[:, i].tolist(), s_neg[i])) for i in fp]
    fail += [dict(kind="false_negative (true link scored low)", **describe(*split["test"][:, i].tolist(), s_pos[i])) for i in fn]
    pd.DataFrame(fail).to_csv(OUT / "failure_cases.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    p = res.groupby(["setting", "method"], sort=False).auc.mean().unstack(0)
    p.plot.bar(ax=ax); ax.set_ylabel("test ROC-AUC"); ax.set_title("Valid vs leaky evaluation")
    plt.tight_layout(); plt.savefig(OUT / "valid_vs_leaky.png", dpi=150)
    print("best valid model:", best_key, "-> results in", OUT)


if __name__ == "__main__":
    main()
