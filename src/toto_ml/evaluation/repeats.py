"""Do draws "repeat" old draws? Shared-number checks against what pure chance gives.

People often notice that a new draw shares 3 (or 4) numbers with some old draw and conclude the
draw is taken from old data. This module measures that on the real draws and compares it with
the exact 6/49 probabilities and with simulated, purely random draw histories of the same length.
"""
from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd

from ..config import N_NUMBERS, PICK

NUM_COLS = [f"n{i}" for i in range(1, PICK + 1)]


def p_share(k: int, n: int = N_NUMBERS, pick: int = PICK) -> float:
    """Exact probability that a random draw shares exactly k numbers with one fixed earlier draw."""
    return comb(pick, k) * comb(n - pick, pick - k) / comb(n, pick)


def p_share_at_least(k: int, n: int = N_NUMBERS, pick: int = PICK) -> float:
    return sum(p_share(j, n, pick) for j in range(k, pick + 1))


def p_any_earlier(k: int, n_earlier: int) -> float:
    """Chance that at least one of n_earlier independent past draws shares >= k numbers with a new draw."""
    return 1 - (1 - p_share_at_least(k)) ** n_earlier


def _sets(draws: pd.DataFrame) -> np.ndarray:
    """Boolean matrix (draws x 49): True where the number was one of the six main numbers."""
    m = np.zeros((len(draws), N_NUMBERS), dtype=bool)
    vals = draws[NUM_COLS].to_numpy(dtype=int) - 1
    m[np.arange(len(draws))[:, None], vals] = True
    return m


def past_matches(draws: pd.DataFrame, numbers: list[int], min_shared: int = 3) -> pd.DataFrame:
    """Every stored draw that shares at least min_shared of the given numbers, most shared first."""
    target = np.zeros(N_NUMBERS, dtype=bool)
    target[np.asarray(numbers, dtype=int) - 1] = True
    shared = (_sets(draws) & target).sum(axis=1)
    out = draws.loc[shared >= min_shared, ["draw_number", "draw_date", *NUM_COLS]].copy()
    out.insert(2, "shared", shared[shared >= min_shared])
    nums = set(int(x) for x in numbers)
    out.insert(3, "shared_numbers",
               [sorted(nums & set(int(v) for v in r)) for r in out[NUM_COLS].to_numpy()])
    return out.sort_values(["shared", "draw_date"], ascending=[False, False]).reset_index(drop=True)


def _history_counts(m: np.ndarray) -> tuple[np.ndarray, int]:
    """(best shared with any earlier draw, per draw from the 2nd on), number of draw pairs sharing >= 3."""
    ov = m.astype(np.int16) @ m.T.astype(np.int16)  # numbers shared by every pair of draws
    low = np.tril(ov, k=-1)
    best = np.array([low[i, :i].max() for i in range(1, len(m))])
    pairs3 = int((low >= 3).sum())
    return best, pairs3


def repeat_report(draws: pd.DataFrame, n_sims: int = 500, seed: int = 42) -> dict:
    """Real draws vs exact theory vs simulated random histories of the same length."""
    m = _sets(draws)
    n = len(m)
    best, pairs3 = _history_counts(m)
    rng = np.random.default_rng(seed)
    sim_pairs = np.empty(n_sims)
    sim_best = np.zeros((n_sims, PICK + 1))
    for s in range(n_sims):
        sm = np.zeros((n, N_NUMBERS), dtype=bool)
        picks = np.argsort(rng.random((n, N_NUMBERS)), axis=1)[:, :PICK]
        sm[np.arange(n)[:, None], picks] = True
        b, p3 = _history_counts(sm)
        sim_pairs[s] = p3
        sim_best[s] = np.bincount(b, minlength=PICK + 1)
    real_best = np.bincount(best, minlength=PICK + 1)
    table = pd.DataFrame({
        "most numbers shared with any earlier draw": range(PICK + 1),
        "real draws": real_best,
        "random draws (average)": sim_best.mean(axis=0).round(1),
    })
    lo, hi = np.percentile(sim_pairs, [2.5, 97.5])
    n_pairs = n * (n - 1) // 2
    return {
        "n_draws": n,
        "n_pairs": n_pairs,
        "pairs3_real": pairs3,
        "pairs3_expected": p_share_at_least(3) * n_pairs,
        "pairs3_random_95": (float(lo), float(hi)),
        "pairs3_p_value": float((np.abs(sim_pairs - sim_pairs.mean()) >= abs(pairs3 - sim_pairs.mean())).mean()),
        "share3_real_pct": 100 * float((best >= 3).mean()),
        "share3_random_pct": 100 * float(sim_best[:, 3:].sum(axis=1).mean() / (n - 1)),
        "best_table": table,
    }
