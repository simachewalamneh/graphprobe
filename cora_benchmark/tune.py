"""Equal-budget random search: every model gets the same number of trials, epochs,
patience and seed; selection uses validation accuracy only."""
import random

SPACE = {
    "hidden": [16, 32, 64, 128],
    "dropout": [0.2, 0.5, 0.8],
    "lr": [0.02, 0.01, 0.005],
    "wd": [0.0, 5e-4, 5e-3],
}
GAT_HEADS = [4, 8]
N_TRIALS = 12
TUNE_SEED = 100


def sample_configs(kind, n_trials=N_TRIALS, seed=7):
    rng = random.Random(seed)  # same sampler seed -> identical hyper-parameter draws for every model
    cfgs = []
    for _ in range(n_trials):
        c = {k: rng.choice(v) for k, v in SPACE.items()}
        h = rng.choice(GAT_HEADS)  # always draw so streams stay aligned across models
        c["kind"] = kind
        if kind == "gat":
            c["heads"] = h
        cfgs.append(c)
    return cfgs


def tune(kind, data, num_classes, fit, n_trials=N_TRIALS, epochs=200, patience=50):
    trials = []
    for c in sample_configs(kind, n_trials):
        r = fit(c, data, num_classes, TUNE_SEED, epochs=epochs, patience=patience)
        trials.append({**c, "val_acc": r["val_acc"], "val_loss": r["val_loss"], "train_time": r["train_time"]})
    best = max(trials, key=lambda t: (t["val_acc"], -t["val_loss"]))
    cfg = {k: best[k] for k in ("kind", "hidden", "dropout", "lr", "wd")}
    if kind == "gat":
        cfg["heads"] = best["heads"]
    return cfg, trials
