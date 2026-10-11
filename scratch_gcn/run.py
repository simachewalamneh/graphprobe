"""Assignment: build and verify a GCN from scratch.
Run:  python -m scratch_gcn.run
Writes tables/plots/checks to results/scratch_gcn/."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from scratch_gcn.core import (GCN, MLP, acc, check_manual, check_permutation, drop_edges,
                              make_dataset, prep_graph, random_graph, set_seed, train_model)

OUT = Path(__file__).resolve().parents[1] / "results" / "scratch_gcn"
SEEDS = [0, 1, 2, 3, 4]
HID = 16


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x, y, ei, masks = make_dataset(seed=0)
    n = x.shape[0]
    n_und = ei.shape[1] // 2
    homophily = (y[ei[0]] == y[ei[1]]).float().mean().item()
    print(f"nodes={n} undirected_edges={n_und} edge_homophily={homophily:.3f} "
          f"split={ {k: int(v.sum()) for k, v in masks.items()} }")

    rows, checks = [], {}
    for s in SEEDS:
        g = prep_graph(ei, n)
        set_seed(s)
        gcn = GCN(3, HID, 2)
        train_model(gcn, x, y, g, masks)
        set_seed(s)
        mlp = MLP(3, HID, 2)
        train_model(mlp, x, y, None, masks)
        gcn.eval(); mlp.eval()
        with torch.no_grad():
            rows.append(dict(seed=s, model="MLP", setting="features only", test=acc(mlp(x), y, masks["test"])))
            rows.append(dict(seed=s, model="GCN", setting="true graph", test=acc(gcn(x, g), y, masks["test"])))
            for f in (0.25, 0.5, 0.75, 1.0):  # test-time edge removal on the trained GCN
                gd = prep_graph(drop_edges(ei, f, s), n)
                rows.append(dict(seed=s, model="GCN", setting=f"test-time edges removed {int(f*100)}%",
                                 test=acc(gcn(x, gd), y, masks["test"])))
        # retrain on a label-blind random graph with the same number of edges
        set_seed(s)
        gr = prep_graph(random_graph(n, n_und, s), n)
        gcn_r = GCN(3, HID, 2)
        train_model(gcn_r, x, y, gr, masks)
        gcn_r.eval()
        with torch.no_grad():
            rows.append(dict(seed=s, model="GCN", setting="retrained on random graph",
                             test=acc(gcn_r(x, gr), y, masks["test"])))
        if s == SEEDS[0]:
            ok_m, err_m = check_manual(gcn, x, ei, node=0)
            ok_p, err_p = check_permutation(gcn, x, ei)
            checks = dict(manual_layer_match=ok_m, manual_max_abs_err=err_m,
                          permutation_equivariance=ok_p, permutation_max_abs_err=err_p)
            print("correctness checks:", checks)
            assert ok_m and ok_p, "correctness check failed"

    df = pd.DataFrame(rows)
    tab = df.groupby(["model", "setting"], sort=False)["test"].agg(["mean", "std"]).round(4)
    print(tab)
    tab.to_csv(OUT / "ablation_results.csv")

    # ---- structure-signal sweep: how much does the GCN-MLP gap depend on homophily?
    sweep = []
    for p_out in (0.01, 0.02, 0.04, 0.06, 0.08):
        for s in range(3):
            xs, ys, eis, ms = make_dataset(p_out=p_out, seed=100 + s)
            h = (ys[eis[0]] == ys[eis[1]]).float().mean().item()
            gs = prep_graph(eis, xs.shape[0])
            set_seed(s); a = GCN(3, HID, 2); train_model(a, xs, ys, gs, ms); a.eval()
            set_seed(s); b = MLP(3, HID, 2); train_model(b, xs, ys, None, ms); b.eval()
            with torch.no_grad():
                sweep.append(dict(p_out=p_out, seed=s, homophily=h,
                                  gcn=acc(a(xs, gs), ys, ms["test"]), mlp=acc(b(xs), ys, ms["test"])))
    sw = pd.DataFrame(sweep).groupby("p_out").mean(numeric_only=True).drop(columns="seed").round(4)
    sw["gap_gcn_minus_mlp"] = (sw["gcn"] - sw["mlp"]).round(4)
    print(sw)
    sw.to_csv(OUT / "homophily_sweep.csv")

    json.dump(dict(nodes=n, undirected_edges=n_und, edge_homophily=homophily,
                   feature_dim=3, splits={k: int(v.sum()) for k, v in masks.items()}, **checks),
              open(OUT / "dataset_and_checks.json", "w"), indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    tab["mean"].plot.barh(xerr=tab["std"], ax=ax[0])
    ax[0].set_xlabel("test accuracy (mean ± std, 5 seeds)")
    ax[0].set_title("Does the GCN use graph structure?")
    ax[1].plot(sw["homophily"], sw["gcn"], "o-", label="GCN")
    ax[1].plot(sw["homophily"], sw["mlp"], "s-", label="MLP")
    ax[1].set_xlabel("edge homophily"); ax[1].set_ylabel("test accuracy"); ax[1].legend()
    ax[1].set_title("Gap closes as edges stop being informative")
    plt.tight_layout()
    plt.savefig(OUT / "structure_evidence.png", dpi=150)


if __name__ == "__main__":
    main()
