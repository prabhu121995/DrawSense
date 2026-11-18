"""Feature engineering & leakage-prevention tests."""
import numpy as np
import pandas as pd

from toto_ml.data.ingest import draws_to_matrix
from toto_ml.features import build_features


def test_shapes_and_targets(synth_draws, fast_cfg):
    fs = build_features(synth_draws, fast_cfg)
    T, mh = len(synth_draws), fast_cfg["split"]["min_history"]
    assert fs.frame.shape[0] == (T - mh + 1) * 49
    M = draws_to_matrix(synth_draws)
    lab = fs.frame[fs.frame["t"] < T]
    assert np.array_equal(lab["target"].to_numpy(), M[lab["t"], lab["num"] - 1])
    assert fs.frame[fs.frame["t"] == T]["target"].isna().all()  # live row has no label
    assert (lab.groupby("t")["target"].sum() == 6).all()


def test_no_target_or_label_columns_in_features(synth_draws, fast_cfg):
    fs = build_features(synth_draws, fast_cfg)
    assert not {"target", "draw_number", "t", "num"} & set(fs.feature_names)


def test_features_use_only_past_draws(synth_draws, fast_cfg):
    """Changing or deleting draws >= t must not change any feature at state t."""
    fs_full = build_features(synth_draws, fast_cfg)
    t = 100
    truncated = synth_draws.iloc[:t].copy()
    fs_trunc = build_features(truncated, fast_cfg)
    a = fs_full.frame[fs_full.frame["t"] == t][fs_full.feature_names].to_numpy()
    b = fs_trunc.frame[fs_trunc.frame["t"] == t][fs_trunc.feature_names].to_numpy()
    np.testing.assert_allclose(a, b)

    tampered = synth_draws.copy()
    rng = np.random.default_rng(0)
    for i in range(t, len(tampered)):
        nums = np.sort(rng.choice(np.arange(1, 50), 7, replace=False))
        tampered.loc[i, ["n1", "n2", "n3", "n4", "n5", "n6"]] = nums[:6]
        tampered.loc[i, "additional"] = nums[6]
    fs_tam = build_features(tampered, fast_cfg)
    early = fs_full.frame["t"] <= t
    np.testing.assert_allclose(fs_full.frame.loc[early, fs_full.feature_names].to_numpy(),
                               fs_tam.frame.loc[early, fs_tam.feature_names].to_numpy())


def test_frequency_gap_and_rolling_values(synth_draws, fast_cfg):
    fs = build_features(synth_draws, fast_cfg)
    M = draws_to_matrix(synth_draws)
    t = 77
    row = fs.frame[fs.frame["t"] == t].set_index("num")
    np.testing.assert_allclose(row["freq_all"].to_numpy(), M[:t].mean(0))
    np.testing.assert_allclose(row["freq_10"].to_numpy(), M[t - 10:t].mean(0))
    np.testing.assert_allclose(row["in_last_1"].to_numpy(), M[t - 1])
    for n in (1, 25, 49):
        seen = np.flatnonzero(M[:t, n - 1])
        expected_gap = t - 1 - seen[-1] if len(seen) else t
        assert row.loc[n, "gap"] == expected_gap
    # pair_with_prev equals co-occurrence counts with previous draw numbers / t
    C = M[:t].T.astype(int) @ M[:t].astype(int)
    prev = np.flatnonzero(M[t - 1])
    np.testing.assert_allclose(row["pair_with_prev"].to_numpy(), C[:, prev].sum(1) / t)


def test_ewm_recursion(synth_draws, fast_cfg):
    fs = build_features(synth_draws, fast_cfg)
    M = draws_to_matrix(synth_draws).astype(float)
    a = fast_cfg["features"]["ewm_alphas"][0]
    e = np.full(49, 6 / 49)
    t = 60
    for i in range(t):
        e = (1 - a) * e + a * M[i]
    np.testing.assert_allclose(fs.frame[fs.frame["t"] == t][f"ewm_{a}"].to_numpy(), e)
