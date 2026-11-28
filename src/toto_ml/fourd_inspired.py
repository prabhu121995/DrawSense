"""4D-inspired TOTO sets: numbers taken from the last three Singapore 4D draws.

A third source next to the Validated model and the Full-history model. It does not use or change them.

How the numbers are found (the method from the client's Gemini notes and the Axis & Mirror charts):

1. Take the last three 4D draws before the TOTO draw: all 23 numbers each (1st, 2nd, 3rd, 10 starter,
   10 consolation).
2. Split every 4D number into its front and back pairs: 8064 -> 80, 64.
3. Each pair counts as itself, as its axis number (0<->5, 1<->6, 2<->7, 3<->8, 4<->9: 64 -> 19) and as its
   mirror number (reversed: 64 -> 46). Only results in 1-49 are kept.
4. Every hit adds a weight: pair itself 1.0, axis or mirror 0.5; 1st/2nd/3rd prize x2; the newest 4D draw
   x1.0, the one before x0.85, the oldest x0.7.
5. Numbers are ranked by total weight (ties: smaller number first). The top 18 are the pool; sets are
   ranks 1-6, 7-12 and 13-18.

Every result also comes with a walk-forward backtest on the saved history: for each past TOTO draw the
sets are rebuilt from the three 4D draws before it and compared with what random numbers would score.
"""
from __future__ import annotations

from math import comb

import pandas as pd

from .db import FOURD_COLS, read_fourd_csv
from .metrics import axis, mirror

N_FOURD = 3
POOL = 18
SET_SIZE = 6
RECENCY = (1.0, 0.85, 0.7)  # newest first
TOP_PRIZES = {"first", "second", "third"}
TOP_WEIGHT = 2.0
VARIANT_WEIGHT = {"pair": 1.0, "axis": 0.5, "mirror": 0.5}
SOURCE = "fourd"
LABEL = "4D-inspired"


def _dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="mixed", errors="coerce")


def load_fourd(csv_path=None) -> pd.DataFrame:
    """4D draws, oldest first, with a parsed ``when`` column."""
    f = read_fourd_csv(csv_path)
    f = f.assign(when=_dates(f["draw_date"])).dropna(subset=["when"])
    return f.drop_duplicates("when", keep="last").sort_values("when").reset_index(drop=True)


def score_numbers(fourd_rows: pd.DataFrame) -> dict[int, float]:
    """Weight of every TOTO number 1-49 found in these 4D draws (rows oldest first)."""
    scores = {n: 0.0 for n in range(1, 50)}
    rows = list(fourd_rows.itertuples(index=False))[::-1]  # newest first
    for age, r in enumerate(rows):
        rec = RECENCY[age] if age < len(RECENCY) else RECENCY[-1]
        for c in FOURD_COLS:
            v = str(getattr(r, c, "") or "").strip()
            if len(v) != 4 or not v.isdigit():
                continue
            w = rec * (TOP_WEIGHT if c in TOP_PRIZES else 1.0)
            for p in (v[:2], v[2:]):
                for kind, q in (("pair", p), ("axis", axis(p)), ("mirror", mirror(p))):
                    n = int(q)
                    if 1 <= n <= 49:
                        scores[n] += w * VARIANT_WEIGHT[kind]
    return scores


def rank(scores: dict[int, float]) -> list[int]:
    return sorted(scores, key=lambda n: (-scores[n], n))


def build_sets(order: list[int], pool: int = POOL) -> list[list[int]]:
    return [sorted(order[i:i + SET_SIZE]) for i in range(0, pool, SET_SIZE)]


def window_before(fourd: pd.DataFrame, when, n: int = N_FOURD) -> pd.DataFrame:
    """The ``n`` 4D draws strictly before ``when`` (or the latest ``n`` when ``when`` is None)."""
    f = fourd if when is None else fourd[fourd["when"] < pd.Timestamp(when)]
    return f.tail(n)


def p_at_least(k: int, pool: int, drawn: int = 6, total: int = 49) -> float:
    """Chance that ``pool`` random numbers contain at least ``k`` of the ``drawn`` winning numbers."""
    return sum(comb(drawn, j) * comb(total - drawn, pool - j) for j in range(k, min(drawn, pool) + 1)) / comb(total, pool)


def backtest(draws: pd.DataFrame, fourd: pd.DataFrame) -> dict:
    """Walk-forward check over every TOTO draw that has three 4D draws before it."""
    rows = []
    d = draws.assign(when=_dates(draws["draw_date"]))
    for r in d.itertuples(index=False):
        win = window_before(fourd, r.when)
        if len(win) < N_FOURD:
            continue
        order = rank(score_numbers(win))
        won = {int(getattr(r, f"n{k}")) for k in range(1, 7)}
        sets = build_sets(order)
        rows.append({"draw_date": r.when.date().isoformat(), "winning": sorted(won),
                     **{f"set{i + 1}_hits": len(won & set(s)) for i, s in enumerate(sets)},
                     "pool_hits": len(won & set(order[:POOL]))})
    df = pd.DataFrame(rows)
    if df.empty:
        return {"n_draws": 0, "table": df}
    exp_set, exp_pool = SET_SIZE * 6 / 49, POOL * 6 / 49
    return {
        "n_draws": len(df),
        "set1_hits_avg": float(df["set1_hits"].mean()), "set_hits_expected": exp_set,
        "set1_3plus": float((df["set1_hits"] >= 3).mean()), "set_3plus_expected": p_at_least(3, SET_SIZE),
        "pool_hits_avg": float(df["pool_hits"].mean()), "pool_hits_expected": exp_pool,
        "pool_3plus": float((df["pool_hits"] >= 3).mean()), "pool_3plus_expected": p_at_least(3, POOL),
        "table": df,
    }


def predict(draws: pd.DataFrame, fourd: pd.DataFrame | None = None) -> dict:
    """4D-inspired sets for the next TOTO draw, plus the walk-forward backtest."""
    fourd = load_fourd() if fourd is None else fourd
    win = window_before(fourd, None)
    if len(win) < N_FOURD:
        return {"ok": False, "reason": f"Needs {N_FOURD} 4D draws; found {len(win)}."}
    scores = score_numbers(win)
    order = rank(scores)
    sets = [{"rank": i + 1, "source": SOURCE, "numbers": s,
             "odd": sum(n % 2 for n in s), "even": sum(1 - n % 2 for n in s), "sum": sum(s)}
            for i, s in enumerate(build_sets(order))]
    return {
        "ok": True, "label": LABEL,
        "based_on_4d": [{"draw_date": r.when.date().isoformat(), "draw_number": r.draw_number,
                         "first": r.first, "second": r.second, "third": r.third} for r in win.itertuples()],
        "sets": sets,
        "pool": [{"rank": i + 1, "number": n, "weight": round(scores[n], 2)} for i, n in enumerate(order[:POOL])],
        "backtest": backtest(draws, fourd),
    }
