import numpy as np
import pytest

from toto_ml.evaluation import EXPECTED, make_split, run_experiment
from toto_ml.evaluation import stats as st
from toto_ml.evaluation.metrics import (hypergeom_pmf_table, per_draw_ranking, probability_summary, rank_order,
                                        ranking_summary)
from toto_ml.models import registry


def _random_actual(n, seed=0):
    rng = np.random.default_rng(seed)
    A = np.zeros((n, 49), dtype=np.int8)
    for i in range(n):
        A[i, rng.choice(49, 6, replace=False)] = 1
    return A


def test_expected_constants():
    assert EXPECTED["hits@6"] == pytest.approx(0.7347, abs=1e-4)
    assert EXPECTED["hits@10"] == pytest.approx(1.2245, abs=1e-4)
    assert EXPECTED["recall@10"] == pytest.approx(0.20408, abs=1e-5)
    pmf = hypergeom_pmf_table(10)
    assert (pmf * np.arange(7)).sum() == pytest.approx(EXPECTED["hits@10"])


def test_perfect_ranking_metrics():
    A = _random_actual(20)
    order = rank_order(A.astype(float), np.arange(20))
    s = ranking_summary(per_draw_ranking(order, A))
    assert s["hits@6"] == 6 and s["hits@10"] == 6 and s["exact_six_rate"] == 1
    assert s["map@10"] == pytest.approx(1) and s["ndcg@10"] == pytest.approx(1)
    assert s["recall@10"] == 1 and s["precision@6"] == 1


def test_random_ranking_matches_theory():
    A = _random_actual(3000, seed=1)
    rng = np.random.default_rng(2)
    order = rank_order(rng.random((3000, 49)), np.arange(3000))
    s = ranking_summary(per_draw_ranking(order, A))
    assert s["hits@10"] == pytest.approx(EXPECTED["hits@10"], abs=0.05)
    assert s["hits@6"] == pytest.approx(EXPECTED["hits@6"], abs=0.04)


def test_ties_do_not_favour_low_numbers():
    order = rank_order(np.zeros((2000, 49)), np.arange(2000), seed=5)
    top = np.bincount(order[:, :10].ravel(), minlength=49)
    assert top.min() > 0.6 * top.mean() and top.max() < 1.4 * top.mean()


def test_probability_metrics_uniform_forecast():
    A = _random_actual(500)
    P = np.full(A.shape, 6 / 49)
    m, rel = probability_summary(P, A)
    assert m["brier"] == pytest.approx(EXPECTED["brier"], rel=1e-6)
    assert m["log_loss"] == pytest.approx(EXPECTED["log_loss"], rel=1e-6)
    assert m["ece"] < 1e-9 and m["roc_auc"] == 0.5
    assert m["mean_prob_sum_per_draw"] == pytest.approx(6)


def test_random_simulation_and_stats():
    A = _random_actual(100)
    null = st.simulate_random(A, 400, seed=0)
    assert null["hits10"].mean() == pytest.approx(EXPECTED["hits@10"], abs=0.03)
    assert st.empirical_p(10.0, null["hits10"]) == pytest.approx(1 / 401)
    ci = st.bootstrap_mean_ci(np.random.default_rng(0).normal(0, 1, 500), 500, 0)
    assert ci["ci_low"] < 0 < ci["ci_high"]
    adj = st.holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert adj["a"] == pytest.approx(0.03) and adj["b"] == pytest.approx(0.08) and adj["c"] == 0.5


def test_split_boundaries(fast_cfg):
    sp = make_split(180, fast_cfg)
    assert sp.test_start == 180 - 36
    assert sp.val_start == 60
    assert set(sp.val_states).isdisjoint(sp.test_states)
    with pytest.raises(ValueError):
        make_split(50, fast_cfg)


def test_backtest_never_trains_on_test_period(synth_draws, fast_cfg, monkeypatch):
    """Instrument every fit() call: final/validation models must only see states < test_start."""
    fits = []
    orig = registry.SklearnScorer.fit

    def spy(self, df):
        # keep the object itself (not id()): ids of garbage-collected scorers can be reused
        fits.append((self.name, int(df["t"].min()), int(df["t"].max()), self))
        return orig(self, df)

    monkeypatch.setattr(registry.SklearnScorer, "fit", spy)
    res = run_experiment(synth_draws, fast_cfg, ["random", "frequency", "ewm", "logistic_regression", "lightgbm"])
    sp = res.split
    prod = [o.production_scorer for o in res.outputs.values() if o.production_scorer is not o.final_scorer]
    eval_fits = [f for f in fits if not any(f[3] is p for p in prod)]
    assert eval_fits, "expected model fits"
    assert all(f[2] < sp.test_start for f in eval_fits), "a model used for evaluation saw test-period labels"
    assert all(f[1] >= sp.min_history for f in fits)
    # walk-forward: each validation block is predicted by a model trained strictly before it
    assert len([f for f in eval_fits if f[0] == "lightgbm"]) == len(range(sp.val_start, sp.test_start, 25)) + 1
    # identical test draws for every model
    n = {r["model"]: r["n_draws"] for r in res.leaderboard["test"]}
    assert len(set(n.values())) == 1 and "ensemble" in n
    # calibrated probabilities are marginal (sum ~6), not normalised to 1
    for o in res.outputs.values():
        s = o.test_probs.reshape(-1, 49).sum(1).mean()
        assert 4 < s < 8
    assert "summary" in res.significance and res.significance["n_test_draws"] == sp.T - sp.test_start


def test_synthetic_data_shows_no_significant_edge(synth_draws, fast_cfg):
    res = run_experiment(synth_draws, fast_cfg, ["random", "frequency", "ewm"])
    for name, e in res.significance["models"].items():
        assert e["p_hits10_holm"] > 0.001  # nothing can be extremely significant on fair random data
