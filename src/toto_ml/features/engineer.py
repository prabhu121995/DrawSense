"""Leakage-free per-(draw, number) feature engineering.

State ``t`` describes the information available *before* draw ``t`` is made:
it is computed exclusively from draws ``0 .. t-1``. At loop iteration ``t``
we first emit features from the running state (which has absorbed only draws
< t) and only afterwards absorb draw ``t`` into the running statistics.

Rows are produced for states ``min_history .. T`` inclusive. The state ``T``
row (one past the last observed draw) has no target - it is the live
"next draw" prediction row.

Prize columns are never read here (they are outcomes of the draw, not inputs).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import BASELINE_P, N_NUMBERS, PICK
from ..data.ingest import additional_to_matrix, draws_to_matrix

EXPECTED_GAP = N_NUMBERS / PICK  # ~8.17 draws between appearances under fairness


@dataclass
class FeatureSet:
    frame: pd.DataFrame          # long format: one row per (state t, number)
    feature_names: list[str]
    T: int                       # number of observed draws
    min_history: int
    config: dict

    def rows_for_states(self, states) -> pd.DataFrame:
        return self.frame[self.frame["t"].isin(states)]

    def X(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.feature_names].to_numpy(dtype=np.float64)


def _decade(n: np.ndarray) -> np.ndarray:
    # ranges 1-9, 10-19, 20-29, 30-39, 40-49
    return np.minimum(n // 10, 4)


def build_features(draws: pd.DataFrame, cfg: dict) -> FeatureSet:
    fcfg = cfg["features"]
    windows = list(fcfg["rolling_windows"])
    alphas = list(fcfg["ewm_alphas"])
    cw = int(fcfg["context_window"])
    use_add = bool(fcfg.get("include_additional_history", True))
    min_history = int(cfg["split"]["min_history"])

    draws = draws.reset_index(drop=True)
    M = draws_to_matrix(draws).astype(np.float64)
    A = additional_to_matrix(draws).astype(np.float64)
    T = len(draws)
    if T <= min_history:
        raise ValueError(f"Need more than {min_history} draws to build features (have {T}).")

    numbers = np.arange(1, N_NUMBERS + 1)
    is_odd = (numbers % 2 == 1).astype(float)
    decade = _decade(numbers)
    is_low = (numbers <= 24).astype(float)

    # Running state (absorbs draws < t)
    count = np.zeros(N_NUMBERS)
    add_count = np.zeros(N_NUMBERS)
    ewm = {a: np.full(N_NUMBERS, BASELINE_P) for a in alphas}
    last_seen = np.full(N_NUMBERS, -1, dtype=np.int64)
    gap_sum = np.zeros(N_NUMBERS)
    gap_sq = np.zeros(N_NUMBERS)
    gap_n = np.zeros(N_NUMBERS)
    gap_max = np.zeros(N_NUMBERS)
    C = np.zeros((N_NUMBERS, N_NUMBERS))  # pair co-occurrence counts
    cum = np.vstack([np.zeros((1, N_NUMBERS)), np.cumsum(M, axis=0)])  # cum[t] = counts over draws < t

    draw_sum = M @ numbers  # per draw sums (used only for draws < t)
    odd_cnt = M @ is_odd
    sorted_main = np.sort(draws[["n1", "n2", "n3", "n4", "n5", "n6"]].to_numpy(), axis=1)
    consec = (np.diff(sorted_main, axis=1) == 1).sum(axis=1).astype(float)
    rng_span = (sorted_main[:, -1] - sorted_main[:, 0]).astype(float)
    dec_onehot = np.eye(5)[decade]  # (49, 5)
    dec_cnt = M @ dec_onehot          # (T, 5) numbers per decade per draw

    blocks = []
    for t in range(T + 1):
        if t >= min_history:
            f = {}
            f["freq_all"] = count / t
            for w in windows:
                lo = max(0, t - w)
                f[f"freq_{w}"] = (cum[t] - cum[lo]) / (t - lo)
            for a in alphas:
                f[f"ewm_{a}"] = ewm[a].copy()
            gap = np.where(last_seen >= 0, t - 1 - last_seen, t).astype(float)
            f["gap"] = gap
            f["gap_ratio"] = gap / EXPECTED_GAP
            mean_gap = np.where(gap_n > 0, gap_sum / np.maximum(gap_n, 1), EXPECTED_GAP)
            var_gap = np.where(gap_n > 1, gap_sq / np.maximum(gap_n, 1) - mean_gap ** 2, EXPECTED_GAP ** 2)
            std_gap = np.sqrt(np.maximum(var_gap, 1e-6))
            f["mean_gap"] = mean_gap
            f["std_gap"] = std_gap
            f["max_gap"] = gap_max.copy()
            f["gap_z"] = (gap - mean_gap) / std_gap
            prev = M[t - 1]
            f["in_last_1"] = prev.copy()
            f["hits_last_3"] = cum[t] - cum[max(0, t - 3)]
            if use_add:
                f["additional_freq"] = add_count / t
            prev_idx = np.flatnonzero(prev)
            Cn = C / t
            f["pair_with_prev"] = Cn[:, prev_idx].sum(axis=1)
            offdiag = Cn.copy()
            np.fill_diagonal(offdiag, 0)
            f["pair_max"] = offdiag.max(axis=1)
            top5 = np.sort(offdiag, axis=1)[:, -5:]
            f["pair_top5_mean"] = top5.mean(axis=1)
            nb = np.zeros(N_NUMBERS)
            nb[1:] += prev[:-1]
            nb[:-1] += prev[1:]
            f["neighbor_in_prev"] = (nb > 0).astype(float)
            # number attributes
            f["number"] = numbers.astype(float)
            f["is_odd"] = is_odd
            f["decade"] = decade.astype(float)
            f["is_low"] = is_low
            # draw-level context from previous draws (identical for all 49 numbers of a state)
            lo = max(0, t - cw)
            k = t - lo
            f["prev_sum"] = np.full(N_NUMBERS, draw_sum[t - 1])
            f["prev_odd_count"] = np.full(N_NUMBERS, odd_cnt[t - 1])
            f["prev_consecutive"] = np.full(N_NUMBERS, consec[t - 1])
            f["prev_range"] = np.full(N_NUMBERS, rng_span[t - 1])
            f["ctx_mean_sum"] = np.full(N_NUMBERS, draw_sum[lo:t].mean())
            odd_share = odd_cnt[lo:t].sum() / (PICK * k)
            f["ctx_parity_share"] = np.where(is_odd == 1, odd_share, 1 - odd_share)
            dec_share = dec_cnt[lo:t].sum(axis=0) / (PICK * k)       # share of picks per decade
            dec_expected = dec_onehot.sum(axis=0) / N_NUMBERS         # numbers per decade / 49
            f["ctx_decade_share_ratio"] = (dec_share / dec_expected)[decade]
            block = pd.DataFrame(f)
            block.insert(0, "number_", numbers)
            block.insert(0, "t", t)
            blocks.append(block)
        if t == T:
            break
        # ---- absorb draw t (only AFTER emitting state t) ----
        row = M[t]
        idx = np.flatnonzero(row)
        for n in idx:
            if last_seen[n] >= 0:
                g = t - last_seen[n] - 1
                gap_sum[n] += g
                gap_sq[n] += g * g
                gap_n[n] += 1
                gap_max[n] = max(gap_max[n], g)
            last_seen[n] = t
        count += row
        add_count += A[t]
        for a in alphas:
            ewm[a] = (1 - a) * ewm[a] + a * row
        C[np.ix_(idx, idx)] += 1

    frame = pd.concat(blocks, ignore_index=True)
    frame = frame.rename(columns={"number_": "num"})
    # targets: did number appear in draw t? (NaN for live state t == T)
    tgt = np.full(len(frame), np.nan)
    has = frame["t"].to_numpy() < T
    tt = frame["t"].to_numpy()[has]
    nn = frame["num"].to_numpy()[has] - 1
    tgt[has] = M[tt, nn]
    frame["target"] = tgt
    dn = np.append(draws["draw_number"].to_numpy(), -1)
    frame["draw_number"] = dn[frame["t"].to_numpy()]
    feature_names = [c for c in frame.columns if c not in ("t", "num", "target", "draw_number")]
    return FeatureSet(frame=frame, feature_names=feature_names, T=T, min_history=min_history, config=fcfg)
