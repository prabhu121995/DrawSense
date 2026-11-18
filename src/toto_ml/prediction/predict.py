"""Per-number history statistics (computed only from the draws passed in)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import N_NUMBERS
from ..data.ingest import draws_to_matrix


def number_history_stats(draws: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    M = draws_to_matrix(draws)
    T = len(draws)
    w = min(window, T)
    counts = M.sum(0)
    recent = M[T - w:].sum(0)
    dd = pd.to_datetime(draws["draw_date"]).dt.strftime("%Y-%m-%d").to_numpy()
    rows = []
    for n in range(N_NUMBERS):
        seen = np.flatnonzero(M[:, n])
        li = seen[-1] if len(seen) else -1
        rows.append({"number": n + 1, "hist_count": int(counts[n]), "hist_freq": float(counts[n] / T),
                     "recent_count": int(recent[n]), "recent_window": int(w),
                     "last_seen_date": str(dd[li]) if li >= 0 else None,
                     "draws_since_last": int(T - 1 - li) if li >= 0 else None})
    return pd.DataFrame(rows)
