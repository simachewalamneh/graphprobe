"""Drift monitoring with a FIXED model: graph-structure statistics + embedding statistics,
alert thresholds learned from a null distribution of random half-splits of the baseline data."""
import numpy as np


def induced(x, ei, nodes):
    """Induced subgraph on `nodes` (re-indexed). ei: (2,E)."""
    remap = -np.ones(x.shape[0], np.int64)
    remap[nodes] = np.arange(len(nodes))
    ok = (remap[ei[0]] >= 0) & (remap[ei[1]] >= 0)
    return x[nodes], remap[ei[:, ok]]


def drop_edges_np(ei, frac, seed):
    u = ei[:, ei[0] < ei[1]]
    k = int(round(u.shape[1] * (1 - frac)))
    keep = u[:, np.random.default_rng(seed).permutation(u.shape[1])[:k]]
    return np.concatenate([keep, keep[::-1]], 1)


def corrupt_features(x, frac_cols, seed):
    """Simulate vocabulary shift: a random subset of feature columns disappears."""
    cols = np.random.default_rng(seed).choice(x.shape[1], int(frac_cols * x.shape[1]), replace=False)
    x = x.copy(); x[:, cols] = 0
    return x


def graph_stats(ei, n):
    deg = np.bincount(ei[0], minlength=n) if ei.size else np.zeros(n)
    return {"mean_degree": float(deg.mean()), "isolated_fraction": float((deg == 0).mean())}


def snapshot_stats(pred, x, ei):
    emb = pred.embeddings(x, ei.T)
    s = graph_stats(ei, x.shape[0])
    s["_centroid"] = emb.mean(0)
    s["mean_embedding_norm"] = float(np.linalg.norm(emb, axis=1).mean())
    return s


def compare(ref, new):
    """Scalar drift statistics between two snapshots (larger = more drift)."""
    c1, c2 = ref["_centroid"], new["_centroid"]
    cos = float(c1 @ c2 / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-12))
    return {
        "mean_degree_rel_change": abs(new["mean_degree"] - ref["mean_degree"]) / max(ref["mean_degree"], 1e-12),
        "isolated_fraction_abs_change": abs(new["isolated_fraction"] - ref["isolated_fraction"]),
        "embedding_centroid_cosine_distance": 1.0 - cos,
        "mean_embedding_norm_rel_change": abs(new["mean_embedding_norm"] - ref["mean_embedding_norm"]) / max(ref["mean_embedding_norm"], 1e-12),
    }


def null_thresholds(pred, x, ei, n_splits=100, k=3.0, seed=0):
    """Random half/half node splits of the baseline give the natural (no-drift) variation of each
    statistic. Threshold = mean + k*std of that null distribution (k=3 -> ~0.1% false-alarm rate if
    roughly normal); the 99th percentile is stored for reference."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    vals = {}
    for _ in range(n_splits):
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2: 2 * (n // 2)]
        sa = snapshot_stats(pred, *induced(x, ei, a))
        sb = snapshot_stats(pred, *induced(x, ei, b))
        for key, v in compare(sa, sb).items():
            vals.setdefault(key, []).append(v)
    return {key: {"threshold": float(np.mean(v) + k * np.std(v)), "p99": float(np.percentile(v, 99)),
                  "null_mean": float(np.mean(v)), "null_std": float(np.std(v))} for key, v in vals.items()}


def detect(ref, new, thresholds):
    return [{"statistic": key, "value": round(v, 5), "threshold": round(thresholds[key]["threshold"], 5),
             "ALERT": v > thresholds[key]["threshold"]} for key, v in compare(ref, new).items()]


def linear_cka(X, Y):
    """Linear CKA: invariant to rotations/permutations/scale of feature axes, so it can compare
    embeddings from different model versions (raw coordinates cannot be compared)."""
    X = X - X.mean(0); Y = Y - Y.mean(0)
    hsic = np.linalg.norm(X.T @ Y, "fro") ** 2
    return float(hsic / (np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro") + 1e-12))
