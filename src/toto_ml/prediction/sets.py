"""Rank the 49 numbers and build the next N ranked six-number sets.

Set ranking: every set is drawn from the top-``pool`` numbers. Its score is
the mean calibrated probability of its six numbers. Sets are chosen greedily
best-first while limiting how many numbers a set may share with any
higher-ranked set, so the five sets are not near-copies of each other.

In a fair draw every six-number set has the same 1 in 13,983,816 chance of
winning Group 1. The ranking reflects the model's estimates, not better odds.
"""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np

from ..config import N_NUMBERS, PICK

JACKPOT_ODDS = comb(N_NUMBERS, PICK)


def rank_numbers(probs: np.ndarray, scores: np.ndarray, seed: int = 0) -> np.ndarray:
    """Zero-based number indices, best first: calibrated probability, then raw score, then seeded tie-break."""
    tb = np.random.default_rng([seed, 104729]).random(N_NUMBERS)
    return np.lexsort((tb, -np.round(scores, 12), -np.round(probs, 12)))


def ranked_sets(probs: np.ndarray, scores: np.ndarray, n_sets: int = 5, pool: int = 12,
                max_overlap: int = 3, seed: int = 0, earlier: list | None = None) -> list[dict]:
    """``earlier``: sets already shown (e.g. from another model); new sets respect max_overlap with them too."""
    order = rank_numbers(probs, scores, seed)
    num_rank = {int(i) + 1: r + 1 for r, i in enumerate(order)}
    pool = int(np.clip(pool, PICK, N_NUMBERS))
    top = [int(i) + 1 for i in order[:pool]]
    p = {n: float(probs[n - 1]) for n in top}
    cands = sorted(combinations(top, PICK),
                   key=lambda c: (-sum(p[n] for n in c), sum(num_rank[n] for n in c)))
    earlier = [tuple(sorted(int(n) for n in e)) for e in (earlier or [])]
    chosen: list[tuple] = []
    overlap = max_overlap
    while len(chosen) < n_sets and overlap <= PICK:
        for c in cands:
            if len(chosen) >= n_sets:
                break
            if c in chosen or c in earlier:
                continue
            if all(len(set(c) & set(s)) <= overlap for s in chosen + earlier):
                chosen.append(c)
        overlap += 1  # relax only if the pool cannot supply enough distinct sets
    out = []
    for r, c in enumerate(chosen, start=1):
        nums = sorted(c)
        out.append({
            "rank": r,
            "numbers": nums,
            "number_ranks": [num_rank[n] for n in nums],
            "mean_probability": float(np.mean([probs[n - 1] for n in nums])),
            "odd": sum(n % 2 for n in nums), "even": sum(1 - n % 2 for n in nums),
            "sum": int(sum(nums)),
        })
    return out
