"""Prize value: build sets that fewer players choose, so a winning set shares its prize with fewer people.

Groups 1-4 split a fixed share of the prize fund among all winners of that group. When the drawn
numbers are popular (birthdays 1-31, favourite numbers), more people win and each share is smaller.

How popularity is learned from the results file (prize columns from the scraper):
  * prize fund proxy  = g4_share_amount x g4_winning_shares (Group 4 always pays a fixed % of the fund,
    so this product follows ticket sales, not how popular the numbers were)
  * crowd             = winners in Groups 5-7 / prize fund proxy (how many people matched, per $ of sales)
  * ridge regression of log(crowd) on which numbers were drawn -> one popularity weight per number.

``value_index`` of a set = estimated prize share versus an average set (1.25 = 25% bigger share).
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .config import N_NUMBERS, PICK
from .prediction.sets import rank_numbers

NUM_COLS = [f"n{i}" for i in range(1, PICK + 1)]
NEED = ["g3_share_amount", "g4_share_amount", "g4_winning_shares", "g5_winning_shares",
        "g6_winning_shares", "g7_winning_shares"]
MIN_DRAWS = 40
BIRTHDAY_PRIOR = 0.06  # used only when the file has too few draws with prize data


def _prior() -> dict:
    w = np.where(np.arange(1, N_NUMBERS + 1) <= 31, BIRTHDAY_PRIOR, -BIRTHDAY_PRIOR)
    return {"weights": (w - w.mean()).tolist(), "n_draws": 0, "source": "birthday prior", "cv_rho": None,
            "g3_least": None, "g3_most": None, "g4_least": None, "g4_most": None}


def fit_popularity(path: str | Path | None = None, raw: pd.DataFrame | None = None) -> dict:
    """Learn how popular each number is with players from the prize data in the results file."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_predict
    from scipy.stats import spearmanr

    if raw is None:
        from .store import STORE_PATH
        p = Path(path or STORE_PATH)
        if not p.exists():
            return _prior()
        raw = pd.read_csv(p)
    if not set(NEED + NUM_COLS) <= set(raw.columns):
        return _prior()
    df = raw[NEED + NUM_COLS].apply(pd.to_numeric, errors="coerce").dropna()
    df = df[(df["g4_share_amount"] > 0) & (df["g4_winning_shares"] > 0)]
    if len(df) < MIN_DRAWS:
        return _prior()
    fund = df["g4_share_amount"] * df["g4_winning_shares"]
    crowd = (df["g5_winning_shares"] + df["g6_winning_shares"] + df["g7_winning_shares"]) / fund
    ok = crowd > 0
    df, y = df[ok], np.log(crowd[ok].to_numpy())
    X = np.zeros((len(df), N_NUMBERS))
    X[np.arange(len(df))[:, None], df[NUM_COLS].to_numpy(int) - 1] = 1
    alphas = np.logspace(-2, 3, 20)
    model = RidgeCV(alphas=alphas).fit(X, y)
    cv_pred = cross_val_predict(RidgeCV(alphas=alphas), X, y, cv=KFold(5, shuffle=True, random_state=0))
    q = pd.qcut(cv_pred, 4, labels=False)
    w = model.coef_ - model.coef_.mean()
    return {
        "weights": w.tolist(), "n_draws": int(len(df)), "source": "prize data", "alpha": float(model.alpha_),
        "cv_rho": float(spearmanr(cv_pred, y).correlation),
        "g3_least": float(df["g3_share_amount"][q == 0].median()), "g3_most": float(df["g3_share_amount"][q == 3].median()),
        "g4_least": float(df["g4_share_amount"][q == 0].median()), "g4_most": float(df["g4_share_amount"][q == 3].median()),
    }


def value_index(numbers, pop: dict) -> float:
    """Estimated prize share of this set versus an average set (1.0 = average)."""
    w = np.asarray(pop["weights"])
    return float(np.exp(PICK * w.mean() - w[np.asarray(numbers, dtype=int) - 1].sum()))


def number_popularity(pop: dict) -> pd.DataFrame:
    """Per number: popularity weight and the value it adds to a set (percent vs an average number)."""
    w = np.asarray(pop["weights"])
    return pd.DataFrame({"number": range(1, N_NUMBERS + 1), "popularity": w,
                         "value_pct": (np.exp(w.mean() - w) - 1) * 100})


# ------------------------------------------------------------------ typical shape of a draw
def shape_profile(draws: pd.DataFrame) -> dict:
    """What most past draws look like: sum range (middle 80%), odd count 2-4, numbers in 3+ decades."""
    X = draws[NUM_COLS].to_numpy(int)
    sums = X.sum(1)
    lo, hi = (int(v) for v in np.percentile(sums, [10, 90]))
    prof = {"sum_lo": lo, "sum_hi": hi, "odd_lo": 2, "odd_hi": 4, "min_decades": 3}
    prof["share_of_draws"] = float(shape_ok(X, prof).mean())
    return prof


def _decades(X: np.ndarray) -> np.ndarray:
    d = X // 10
    return np.array([len(set(r)) for r in d]) if X.ndim == 2 else len(set(d))


def shape_ok(X: np.ndarray, prof: dict) -> np.ndarray:
    X = np.atleast_2d(X)
    s, odd = X.sum(1), (X % 2).sum(1)
    return ((s >= prof["sum_lo"]) & (s <= prof["sum_hi"]) & (odd >= prof["odd_lo"]) & (odd <= prof["odd_hi"])
            & (_decades(X) >= prof["min_decades"]))


# ------------------------------------------------------------------ set builder
def value_sets(probs: np.ndarray, scores: np.ndarray, pop: dict, prof: dict, n_sets: int = 2, pool: int = 18,
               max_overlap: int = 3, seed: int = 0, earlier: list | None = None) -> list[dict]:
    """Sets from the model's top ``pool`` numbers, in a typical draw shape, ordered by prize value.

    Value is rounded to 1% before ordering, so among sets of the same value the model's own
    ranking decides. A set shares at most ``max_overlap`` numbers with any set before it.
    """
    order = rank_numbers(probs, scores, seed)
    num_rank = {int(i) + 1: r + 1 for r, i in enumerate(order)}
    pool = int(np.clip(pool, PICK, N_NUMBERS))
    top = np.array([int(i) + 1 for i in order[:pool]])
    C = np.sort(top[np.array(list(combinations(range(pool), PICK)))], axis=1)
    w = np.asarray(pop["weights"])
    vi = np.exp(PICK * w.mean() - w[C - 1].sum(1))
    mp = probs[C - 1].mean(1)
    ok = shape_ok(C, prof)
    if not ok.any():
        ok[:] = True
    C, vi, mp = C[ok], vi[ok], mp[ok]
    idx = np.lexsort((-mp, -np.round(vi, 2)))
    earlier = [tuple(sorted(int(n) for n in e)) for e in (earlier or [])]
    chosen: list[int] = []
    overlap = max_overlap
    while len(chosen) < n_sets and overlap <= PICK:
        for i in idx:
            if len(chosen) >= n_sets:
                break
            c = set(C[i].tolist())
            if i in chosen or tuple(C[i].tolist()) in earlier:
                continue
            if all(len(c & set(C[j].tolist())) <= overlap for j in chosen) and \
               all(len(c & set(e)) <= overlap for e in earlier):
                chosen.append(int(i))
        overlap += 1
    out = []
    for r, i in enumerate(chosen, start=1):
        nums = C[i].tolist()
        out.append({"rank": r, "numbers": nums, "number_ranks": [num_rank[n] for n in nums],
                    "mean_probability": float(mp[i]), "value_index": float(vi[i]),
                    "value_pct": float((vi[i] - 1) * 100), "typical_shape": bool(shape_ok(np.array(nums), prof)[0]),
                    "odd": int(sum(n % 2 for n in nums)), "even": int(sum(1 - n % 2 for n in nums)),
                    "sum": int(sum(nums)), "decades": int(len({n // 10 for n in nums}))})
    return out
