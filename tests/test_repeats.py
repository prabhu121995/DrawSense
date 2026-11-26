from math import comb

import pandas as pd

from toto_ml.evaluation.repeats import p_any_earlier, p_share, p_share_at_least, past_matches, repeat_report


def test_exact_probabilities():
    assert abs(sum(p_share(k) for k in range(7)) - 1) < 1e-12
    assert p_share(6) == 1 / comb(49, 6)
    assert abs(p_share_at_least(3) - 260624 / comb(49, 6)) < 1e-12  # 1.864%
    assert p_any_earlier(3, 300) > 0.99  # a 3-number "repeat" of some old draw is almost certain


def test_past_matches():
    draws = pd.DataFrame({"draw_number": [1, 2, 3], "draw_date": pd.to_datetime(["2024-01-01", "2024-01-04", "2024-01-08"]),
                          "n1": [1, 1, 10], "n2": [2, 2, 11], "n3": [3, 30, 12], "n4": [4, 31, 13],
                          "n5": [5, 32, 14], "n6": [6, 33, 15]})
    m = past_matches(draws, [1, 2, 3, 4, 40, 41])
    assert list(m["draw_number"]) == [1] and m["shared"][0] == 4 and m["shared_numbers"][0] == [1, 2, 3, 4]


def test_random_history_matches_theory(synth_draws):
    rep = repeat_report(synth_draws, n_sims=50, seed=1)
    lo, hi = rep["pairs3_random_95"]
    assert lo <= rep["pairs3_expected"] <= hi
    assert rep["best_table"]["real draws"].sum() == len(synth_draws) - 1
