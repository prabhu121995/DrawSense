"""Ranking and probability metrics for 49-number multi-label draws.

All inputs are (N_draws, 49) matrices aligned to numbers 1..49.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ..config import BASELINE_P, N_NUMBERS, PICK

EPS = 1e-6
# Theoretical expectations for a uniformly random ranking of a fair 6/49 draw
EXPECTED = {
    "hits@6": 6 * PICK / N_NUMBERS,      # 0.7347
    "hits@10": 10 * PICK / N_NUMBERS,    # 1.2245
    "precision@6": PICK / N_NUMBERS,
    "precision@10": PICK / N_NUMBERS,
    "recall@6": 6 / N_NUMBERS,
    "recall@10": 10 / N_NUMBERS,         # 0.20408
    "brier": BASELINE_P * (1 - BASELINE_P),
    "log_loss": -(BASELINE_P * np.log(BASELINE_P) + (1 - BASELINE_P) * np.log(1 - BASELINE_P)),
    "roc_auc": 0.5,
    "pr_auc": BASELINE_P,
}


def rank_order(scores: np.ndarray, states: np.ndarray, seed: int = 0) -> np.ndarray:
    """Return (N, 49) zero-based number indices sorted best-first.

    Ties (common for frequency baselines) are broken by a seeded random key so
    that ties never systematically favour low numbers.
    """
    N = scores.shape[0]
    out = np.empty((N, N_NUMBERS), dtype=np.int64)
    for i in range(N):
        tb = np.random.default_rng([seed, 104729, int(states[i])]).random(N_NUMBERS)
        out[i] = np.lexsort((tb, -scores[i]))
    return out


def hypergeom_pmf_table(k: int) -> np.ndarray:
    """P(hits = h) for h = 0..6 when a random set of k numbers is compared with 6 winners."""
    from scipy.stats import hypergeom
    return hypergeom(N_NUMBERS, PICK, k).pmf(np.arange(PICK + 1))


def per_draw_ranking(order: np.ndarray, actual: np.ndarray) -> dict[str, np.ndarray]:
    """Per-draw ranking metrics. actual: (N, 49) binary."""
    N = order.shape[0]
    rel = actual[np.arange(N)[:, None], order]  # (N, 49) relevance in ranked order
    hits6 = rel[:, :6].sum(1)
    hits10 = rel[:, :10].sum(1)
    k = np.arange(1, 11)
    cum = np.cumsum(rel[:, :10], axis=1)
    ap10 = (cum / k * rel[:, :10]).sum(1) / PICK  # min(6, 10) relevant items
    disc = 1 / np.log2(k + 1)
    dcg = (rel[:, :10] * disc).sum(1)
    idcg = disc[:PICK].sum()
    ndcg10 = dcg / idcg
    return {"hits6": hits6, "hits10": hits10, "ap10": ap10, "ndcg10": ndcg10}


def ranking_summary(pd_: dict[str, np.ndarray]) -> dict[str, float]:
    h6, h10 = pd_["hits6"], pd_["hits10"]
    out = {
        "hits@6": h6.mean(), "hits@10": h10.mean(),
        "precision@6": h6.mean() / 6, "precision@10": h10.mean() / 10,
        "recall@6": h6.mean() / PICK, "recall@10": h10.mean() / PICK,
        "map@10": pd_["ap10"].mean(), "ndcg@10": pd_["ndcg10"].mean(),
    }
    for m in (2, 3, 4, 5, 6):
        out[f"top6_{m}plus_rate"] = (h6 >= m).mean()
    out["exact_six_rate"] = (h6 == 6).mean()
    for m in (2, 3, 4):
        out[f"top10_{m}plus_rate"] = (h10 >= m).mean()
    return {k: float(v) for k, v in out.items()}


def reliability(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict:
    """Equal-mass reliability bins (probabilities cluster near 6/49, so uniform bins are useless)."""
    p = probs.ravel()
    yy = y.ravel()
    order = np.argsort(p, kind="stable")
    bins = np.array_split(order, n_bins)
    rows = []
    ece = 0.0
    mce = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        mp, fp = float(p[b].mean()), float(yy[b].mean())
        rows.append({"mean_predicted": mp, "fraction_positive": fp, "count": int(len(b)),
                     "p_min": float(p[b].min()), "p_max": float(p[b].max())})
        ece += len(b) / len(p) * abs(fp - mp)
        mce = max(mce, abs(fp - mp))
    return {"bins": rows, "ece": float(ece), "mce": float(mce)}


def probability_summary(probs: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> dict:
    p = np.clip(probs.ravel(), EPS, 1 - EPS)
    y = actual.ravel().astype(int)
    out = {
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "mean_prob": float(p.mean()),
        "mean_prob_sum_per_draw": float(probs.sum(1).mean()),
    }
    if np.unique(p).size > 1:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    else:
        out["roc_auc"] = 0.5
        out["pr_auc"] = float(y.mean())
    rel = reliability(probs, actual, n_bins)
    out["ece"] = rel["ece"]
    out["mce"] = rel["mce"]
    out["brier_skill_vs_uniform"] = 1 - out["brier"] / EXPECTED["brier"]
    return out, rel
