"""Assignment: compare MLP / GCN / GraphSAGE / GAT on Cora.
Run:   python -m cora_benchmark.run          (full)
       QUICK=1 python -m cora_benchmark.run  (smoke test, few epochs/seeds)
Outputs -> results/cora_benchmark/"""
import json
import os
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import degree

from cora_benchmark.data import inspect, load_cora
from cora_benchmark.training import drop_undirected, embedding_similarity, fit
from cora_benchmark.tune import tune

OUT = Path(__file__).resolve().parents[1] / "results" / "cora_benchmark"
QUICK = os.environ.get("QUICK") == "1"
MODELS = ["mlp", "gcn", "sage", "gat"]
SEEDS = [0] if QUICK else [0, 1, 2]
DEPTHS = [1, 2, 4] if QUICK else [1, 2, 4, 6, 8]
EDGE_FRACS = [0.0, 1.0] if QUICK else [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
EPOCHS = 30 if QUICK else 200
PATIENCE = 10 if QUICK else 50
TRIALS = 2 if QUICK else 12


def ms(series):
    return f"{series.mean():.4f} ± {series.std(ddof=0):.4f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, C = load_cora()
    info = inspect(data, C)
    json.dump(info, open(OUT / "dataset_inspection.json", "w"), indent=2)
    print(info)
    ei = data.edge_index

    # ---------------------------------------------------------------- 1. tuning (validation only)
    best_cfg, trial_rows = {}, []
    if (OUT / "best_configs.json").exists() and not QUICK:   # resume: tuning is deterministic, reuse it
        best_cfg = json.load(open(OUT / "best_configs.json"))
    for kind in ([] if best_cfg else MODELS):
        cfg, trials = tune(kind, data, C, fit, n_trials=TRIALS, epochs=EPOCHS, patience=PATIENCE)
        best_cfg[kind] = cfg
        trial_rows += [{"model": kind, **t} for t in trials]
        print(kind, "->", cfg)
    if trial_rows:
        pd.DataFrame(trial_rows).to_csv(OUT / "tuning_trials.csv", index=False)
        json.dump(best_cfg, open(OUT / "best_configs.json", "w"), indent=2)

    # ---------------------------------------------------------------- 2. final comparison (test used now)
    rows, kept = [], {}
    for kind in MODELS:
        for s in SEEDS:
            r = fit(best_cfg[kind], data, C, s, epochs=EPOCHS, patience=PATIENCE, report_test=True)
            rows.append(dict(model=kind, seed=s, val_acc=r["val_acc"], test_acc=r["test_acc"],
                             test_macro_f1=r["test_f1"], train_time_s=r["train_time"],
                             params=r["params"], best_epoch=r["best_epoch"]))
            if s == SEEDS[0]:
                kept[kind] = r["model"]
    final = pd.DataFrame(rows)
    final.to_csv(OUT / "final_runs.csv", index=False)
    comp = final.groupby("model", sort=False).apply(lambda d: pd.Series({
        "val_acc": ms(d.val_acc), "test_acc": ms(d.test_acc), "test_macro_f1": ms(d.test_macro_f1),
        "train_time_s": f"{d.train_time_s.mean():.2f} ± {d.train_time_s.std(ddof=0):.2f}",
        "params": int(d.params.iloc[0])}), include_groups=False)
    comp.to_csv(OUT / "comparison.csv")
    print(comp)

    # ---------------------------------------------------------------- 3. depth study
    drows = []
    cached = OUT / "depth_study.csv"
    for kind in ([] if cached.exists() and not QUICK else MODELS):
        for L in DEPTHS:
            for s in SEEDS:
                cfg = {**best_cfg[kind], "n_layers": L}
                r = fit(cfg, data, C, s, epochs=EPOCHS, patience=PATIENCE, report_test=True)
                sim = embedding_similarity(r["model"], data, ei)
                drows.append(dict(model=kind, depth=L, seed=s, val_acc=r["val_acc"],
                                  test_acc=r["test_acc"], **sim))
    depth = pd.DataFrame(drows) if drows else pd.read_csv(cached)
    depth.to_csv(cached, index=False)
    print(depth.groupby(["model", "depth"])[["test_acc", "mean_cos", "class_gap"]].mean().round(4))

    # ---------------------------------------------------------------- 4. edge removal (train + test graph both thinned)
    erows = []
    for kind in MODELS:
        for f in EDGE_FRACS:
            for s in SEEDS:
                ei_d = drop_undirected(ei, f, s)
                r = fit(best_cfg[kind], data, C, s, ei=ei_d, epochs=EPOCHS, patience=PATIENCE, report_test=True)
                erows.append(dict(model=kind, frac_removed=f, seed=s, val_acc=r["val_acc"], test_acc=r["test_acc"]))
    edges = pd.DataFrame(erows)
    edges.to_csv(OUT / "edge_removal.csv", index=False)
    print(edges.groupby(["model", "frac_removed"])["test_acc"].mean().unstack().round(4))

    # ---------------------------------------------------------------- 5. error analysis (best GNN by validation)
    gnn = final[final.model != "mlp"].groupby("model").val_acc.mean().idxmax()
    model, mlp = kept[gnn], kept["mlp"]
    model.eval(); mlp.eval()
    with torch.no_grad():
        prob = model(data.x, ei).softmax(1)
        pred, conf = prob.argmax(1), prob.max(1).values
        mlp_pred = mlp(data.x, ei).argmax(1)
    deg = degree(ei[0], data.num_nodes).long()
    nbrs = [[] for _ in range(data.num_nodes)]
    for a, b in ei.t().tolist():
        nbrs[a].append(b)
    wrong = (data.test_mask & (pred != data.y)).nonzero().flatten().tolist()
    rng = np.random.default_rng(0)
    confident = sorted(wrong, key=lambda i: -conf[i].item())[:4]
    rest = [i for i in wrong if i not in confident]
    sample = confident + (rng.choice(rest, size=min(4, len(rest)), replace=False).tolist() if rest else [])

    def node_row(i):
        nb = nbrs[i]
        lab = Counter(data.y[nb].tolist()) if nb else {}
        pr = Counter(pred[nb].tolist()) if nb else {}
        hom = (data.y[nb] == data.y[i]).float().mean().item() if nb else float("nan")
        return dict(node=i, true=int(data.y[i]), pred=int(pred[i]), conf=round(conf[i].item(), 3),
                    mlp_pred=int(mlp_pred[i]), degree=int(deg[i]), words=int((data.x[i] > 0).sum()),
                    nbr_true_labels=dict(lab), nbr_pred_labels=dict(pr), nbr_homophily=round(hom, 2))
    err = pd.DataFrame([node_row(i) for i in sample])
    err.to_csv(OUT / "error_analysis_nodes.csv", index=False)
    all_err = pd.DataFrame([node_row(i) for i in data.test_mask.nonzero().flatten().tolist()])
    all_err["wrong"] = all_err.true != all_err.pred
    all_err["deg_bucket"] = pd.cut(all_err.degree, [-1, 1, 3, 5, 1000], labels=["0-1", "2-3", "4-5", "6+"])
    all_err["hom_bucket"] = pd.cut(all_err.nbr_homophily, [-0.01, 0.5, 1.0], labels=["<=0.5", ">0.5"])
    by_deg = all_err.groupby("deg_bucket", observed=True).wrong.agg(["mean", "count"]).round(3)
    by_hom = all_err.groupby("hom_bucket", observed=True).wrong.agg(["mean", "count"]).round(3)
    mlp_right = float(((err.mlp_pred == err.true).mean()))
    with open(OUT / "error_analysis.md", "w") as fh:
        fh.write(f"# Error analysis ({gnn.upper()}, seed {SEEDS[0]})\n\n")
        fh.write("Rows: 4 most confident mistakes + 4 random mistakes.\n\n")
        fh.write("| node | true | pred | conf | MLP pred | degree | words | neighbour true labels | neighbour predictions | nbr homophily |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for _, r in err.iterrows():
            fh.write(f"| {r.node} | {r.true} | {r.pred} | {r.conf} | {r.mlp_pred} | {r.degree} | {r.words} | {r.nbr_true_labels} | {r.nbr_pred_labels} | {r.nbr_homophily} |\n")
        fh.write(f"\nMLP is correct on {mlp_right:.0%} of the sampled GNN errors.\n\n")
        fh.write("Error rate by degree bucket:\n\n" + by_deg.to_string() + "\n\n")
        fh.write("Error rate by neighbourhood homophily:\n\n" + by_hom.to_string() + "\n")
    print(by_deg, by_hom)

    # ---------------------------------------------------------------- plots
    fig, ax = plt.subplots(1, 3, figsize=(17, 4))
    g = depth.groupby(["model", "depth"])[["test_acc", "mean_cos"]].mean().reset_index()
    for kind in MODELS:
        d = g[g.model == kind]
        ax[0].plot(d.depth, d.test_acc, "o-", label=kind)
        ax[1].plot(d.depth, d.mean_cos, "o-", label=kind)
    ax[0].set_title("Depth vs test accuracy"); ax[1].set_title("Depth vs mean pairwise cosine (embeddings)")
    for a in ax[:2]:
        a.set_xlabel("layers"); a.legend()
    e = edges.groupby(["model", "frac_removed"]).test_acc.mean().reset_index()
    for kind in MODELS:
        d = e[e.model == kind]
        ax[2].plot(d.frac_removed, d.test_acc, "o-", label=kind)
    ax[2].set_title("Edge removal (MLP = control)"); ax[2].set_xlabel("fraction of edges removed"); ax[2].legend()
    plt.tight_layout(); plt.savefig(OUT / "depth_and_edge_studies.png", dpi=150)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
