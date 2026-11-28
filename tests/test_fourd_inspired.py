"""4D-inspired sets: pair extraction, axis/mirror variants, no look-ahead in the backtest."""
import pandas as pd

from toto_ml import fourd_inspired as fi
from toto_ml.db import FOURD_COLS


def fourd_frame(dates, numbers):
    rows = []
    for d, nums in zip(dates, numbers):
        r = {"draw_number": "1", "draw_date": d, **{c: "" for c in FOURD_COLS}}
        r.update(dict(zip(FOURD_COLS, nums)))
        rows.append(r)
    f = pd.DataFrame(rows)
    return f.assign(when=pd.to_datetime(f["draw_date"])).sort_values("when").reset_index(drop=True)


def test_pairs_axis_and_mirror_weights():
    f = fourd_frame(["2026-11-25"], [["0000", "0000", "0000", "1427"]])  # starter_1 = 1427 -> 14, 27
    s = fi.score_numbers(f)
    assert s[14] == 1.0 and s[27] == 1.0           # the pairs themselves
    assert s[41] == 0.5                            # mirror of 14; its axis 69 is above 49
    assert {n for n, v in s.items() if v} == {14, 27, 41}  # 27 -> axis 72, mirror 72: out of range


def test_top_prize_and_recency_weights():
    f = fourd_frame(["2026-11-22", "2026-11-25"], [["1200"], ["1200"]])  # first prize in both draws
    s = fi.score_numbers(f)
    assert abs(s[12] - (2.0 * 1.0 + 2.0 * 0.85)) < 1e-9
    assert abs(s[21] - 0.5 * (2.0 + 1.7)) < 1e-9   # mirror of 12
    assert abs(s[1] - 0.0) < 1e-9 and s[5] == 0.0  # "00": axis "55" out of range, 0 not a TOTO number


def test_sets_and_backtest_use_only_earlier_4d():
    f = fourd_frame(["2026-11-01", "2026-11-04", "2026-11-07", "2026-11-11"],
                    [["0102"], ["0304"], ["0506"], ["4748"]])
    draws = pd.DataFrame([{"draw_date": "2026-11-09", "n1": 1, "n2": 2, "n3": 3, "n4": 4, "n5": 5, "n6": 6,
                           "additional": 7, "draw_number": 1}])
    b = fi.backtest(draws, f)
    assert b["n_draws"] == 1
    row = b["table"].iloc[0]
    assert row["pool_hits"] == 6                   # 47/48 from the later 4D draw are never used
    r = fi.predict(draws, f)
    assert r["ok"] and [x["draw_date"] for x in r["based_on_4d"]] == ["2026-11-04", "2026-11-07", "2026-11-11"]
    assert len(r["sets"]) == 3 and all(len(s["numbers"]) == 6 for s in r["sets"])
    assert len({n for s in r["sets"] for n in s["numbers"]}) == fi.POOL


def test_random_baseline_odds():
    assert abs(fi.p_at_least(0, 18) - 1.0) < 1e-12
    assert 0.38 < fi.p_at_least(3, 18) < 0.39
