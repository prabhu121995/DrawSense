"""Synthetic demo data: fair, independent, uniformly random 6/49 draws.

Only used for tests / sanity checks. On fair random data every model should
perform at chance level; a "winner" there signals a bug or leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_draws(n_draws: int = 400, seed: int = 0, start_date: str = "2020-01-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=n_draws * 4 + 7, freq="D")
    dates = dates[dates.day_of_week.isin([0, 3])][:n_draws]  # Mondays & Thursdays
    rows = []
    for i in range(n_draws):
        seven = rng.choice(np.arange(1, 50), size=7, replace=False)
        main = np.sort(seven[:6])
        rows.append({"draw_date": dates[i].strftime("%a, %d %b %Y"),
                     **{f"n{j + 1}": int(main[j]) for j in range(6)}, "additional": int(seven[6])})
    return pd.DataFrame(rows)
