"""Statistical comparison against random baselines.

* Repeated random simulation: R independent uniformly-random rankings over the
  exact same test draws -> null distribution of every ranking metric.
* Paired bootstrap (resampling draws) confidence intervals for differences.
* Holm-Bonferroni correction across all compared models (multiple testing).
"""
from __future__ import annotations

import numpy as np

from ..config import N_NUMBERS, PICK
from .metrics import EPS

SIM_METRICS = ["hits6", "hits10", "ap10", "ndcg10"]


def simulate_random(actual: np.ndarray, n_sims: int, seed: int) -> dict[str, np.ndarray]:
    """Null distribution of mean metrics under uniformly random rankings.

    Returns per-simulation means over the N test draws (shape (n_sims,)).
    """
    rng = np.random.default_rng(seed)
    N = actual.shape[0]
    k = np.arange(1, 11)
    disc = 1 / np.log2(k + 1)
    idcg = disc[:PICK].sum()
    out = {m: np.empty(n_sims) for m in SIM_METRICS}
    out["top6_2plus"] = np.empty(n_sims)
    out["top6_3plus"] = np.empty(n_sims)
    chunk = 250
    for s0 in range(0, n_sims, chunk):
        s1 = min(n_sims, s0 + chunk)
        S = s1 - s0
        keys = rng.random((S, N, N_NUMBERS))
        top10 = np.argpartition(keys, 10, axis=2)[:, :, :10]
        # order the top-10 by key to get ranks 1..10
        kk = np.take_along_axis(keys, top10, axis=2)
        top10 = np.take_along_axis(top10, np.argsort(kk, axis=2), axis=2)
        rel = np.take_along_axis(np.broadcast_to(actual, (S, N, N_NUMBERS)), top10, axis=2)
        h6 = rel[:, :, :6].sum(2)
        h10 = rel.sum(2)
        cum = np.cumsum(rel, axis=2)
        ap = (cum / k * rel).sum(2) / PICK
        ndcg = (rel * disc).sum(2) / idcg
        out["hits6"][s0:s1] = h6.mean(1)
        out["hits10"][s0:s1] = h10.mean(1)
        out["ap10"][s0:s1] = ap.mean(1)
        out["ndcg10"][s0:s1] = ndcg.mean(1)
        out["top6_2plus"][s0:s1] = (h6 >= 2).mean(1)
        out["top6_3plus"][s0:s1] = (h6 >= 3).mean(1)
    return out


def empirical_p(observed: float, null: np.ndarray) -> float:
    """One-sided P(null >= observed) with +1 smoothing."""
    return float((1 + np.sum(null >= observed - 1e-12)) / (1 + len(null)))


def bootstrap_mean_ci(diff: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    # two-sided bootstrap p-value for H0: mean diff = 0 (percentile method)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {"mean": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_two_sided": float(min(1.0, max(p, 1 / (n_boot + 1))))}


def per_draw_logloss(probs: np.ndarray, actual: np.ndarray) -> np.ndarray:
    p = np.clip(probs, EPS, 1 - EPS)
    return -(actual * np.log(p) + (1 - actual) * np.log(1 - p)).mean(1)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj
