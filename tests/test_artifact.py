"""Selection -> artifact export -> load -> next ranked sets; plus the API that serves it."""
import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from toto_ml.artifact import load_artifact, load_history, next_draw_date, predict_next, save_artifact, select_best
from toto_ml.data import generate_synthetic_draws
from toto_ml.evaluation import run_experiment
from toto_ml.prediction import ranked_sets


@pytest.fixture(scope="module")
def artifact(tmp_path_factory, synth_draws, fast_cfg):
    res = run_experiment(synth_draws, fast_cfg, ["random", "frequency", "ewm", "logistic_regression", "lightgbm"])
    best, table = select_best(res)
    out = save_artifact(res, best, synth_draws, tmp_path_factory.mktemp("art"), table)
    return out, best, res


def test_selection_uses_validation_and_excludes_random(artifact):
    out, best, res = artifact
    assert best != "random"
    val = {r["model"]: r["log_loss"] for r in res.leaderboard["validation"] if r["model"] != "random"}
    assert val[best] == min(val.values())


def test_artifact_roundtrip_and_history_order(artifact, synth_draws):
    out, best, _ = artifact
    b = load_artifact(out)
    assert b["model_name"] == best and b["metadata"]["best_model"] == best
    h = load_history(b)
    assert h["draw_date"].tolist() == synth_draws["draw_date"].tolist()  # dates & order survive the CSV roundtrip


def test_predict_next_five_ranked_sets(artifact, synth_draws):
    b = load_artifact(artifact[0])
    assert "split_scorers" in b and "split_calibrator" in b
    r = predict_next(b, n_split=0, n_full=5, strategy="model")
    assert [s["rank"] for s in r["sets"]] == [1, 2, 3, 4, 5]
    assert {s["source"] for s in r["sets"]} == {"full"}
    keys = set()
    for s in r["sets"]:
        assert len(s["numbers"]) == 6 and len(set(s["numbers"])) == 6
        assert all(1 <= n <= 49 for n in s["numbers"])
        keys.add(tuple(s["numbers"]))
    assert len(keys) == 5
    means = [s["mean_probability"] for s in r["sets"]]
    assert means == sorted(means, reverse=True)
    top6 = sorted(x["number"] for x in r["numbers"][:6])
    assert r["sets"][0]["numbers"] == top6  # set #1 = six best-ranked numbers
    assert sorted(x["number"] for x in r["numbers"]) == list(range(1, 50))
    assert 4 < r["prob_sum"] < 8  # marginal probabilities, not normalised to 100%
    assert r["based_on"]["last_draw_date"] == synth_draws["draw_date"].iloc[-1].strftime("%Y-%m-%d")


def test_refit_updates_model_with_new_draws_without_rebenchmarking(artifact, synth_draws, tmp_path):
    from toto_ml.artifact import refit
    from toto_ml.data import validate_and_clean
    b = load_artifact(artifact[0])
    more, _, _ = validate_and_clean(generate_synthetic_draws(185, seed=3))  # same 180 draws + 5 new ones
    assert more["draw_date"].iloc[:180].tolist() == synth_draws["draw_date"].tolist()
    new = refit(b, more, out_dir=tmp_path)
    assert new["model_name"] == b["model_name"]                       # same algorithm, no re-selection
    assert new["metadata"]["training_data"]["n_draws"] == 185
    assert new["metadata"]["last_update"]["new_draws"] == 5
    assert new["metadata"]["test_significance"] == b["metadata"]["test_significance"]  # evidence kept
    assert new["calibrator"] is not None
    reloaded = load_artifact(tmp_path)
    r = predict_next(reloaded)                                        # history.csv now has 185 draws
    assert r["based_on"]["n_draws"] == 185 and len(r["sets"]) == 4   # default: 2 split + 2 full sets
    assert [s["source"] for s in r["sets"]] == ["split", "split", "full", "full"]


def test_ranked_sets_diversity():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.1, 0.14, 49)
    sets = ranked_sets(p, p, n_sets=5, pool=12, max_overlap=3)
    for i, a in enumerate(sets):
        for b in sets[:i]:
            assert len(set(a["numbers"]) & set(b["numbers"])) <= 3


def test_next_draw_date_is_monday_or_thursday():
    from datetime import date
    assert next_draw_date(date(2026, 11, 16)).isoformat() == "2026-11-19"  # Mon -> Thu
    assert next_draw_date(date(2026, 11, 19)).isoformat() == "2026-11-23"  # Thu -> Mon


def test_api_serves_artifact(artifact, monkeypatch, tmp_path, synth_draws):
    from toto_ml import store
    monkeypatch.setenv("TOTO_ML_ARTIFACT", str(artifact[0]))
    results = tmp_path / "toto_results.csv"
    results.write_bytes((artifact[0] / "history.csv").read_bytes())
    monkeypatch.setattr(store, "STORE_PATH", results)
    from toto_ml.api.main import app
    with TestClient(app) as c:
        assert c.get("/api/health").json()["artifact"] is True
        assert c.get("/api/model").json()["best_model"] == artifact[1]
        r = c.post("/api/predict", data={"n_split": "2", "n_full": "3"})
        assert r.status_code == 200, r.text
        assert len(r.json()["sets"]) == 5
        nxt = (synth_draws["draw_date"].iloc[-1] + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        a = c.post("/api/draws", json={"draw_date": nxt, "numbers": [5, 9, 13, 22, 31, 44], "additional": 2})
        assert a.status_code == 200, a.text
        assert c.get("/api/draws?limit=1").json()[0]["draw_date"] == nxt
        assert c.post("/api/predict", data={"n_split": "2", "n_full": "3"}).json()["based_on"]["last_draw_date"] == nxt
        bad = c.post("/api/draws", json={"draw_date": nxt, "numbers": [5, 9, 13, 22, 31, 44], "additional": 2})
        assert bad.status_code == 400
        newer = generate_synthetic_draws(200, seed=11)
        buf = io.BytesIO()
        newer.to_csv(buf, index=False)
        r = c.post("/api/predict", data={"n_split": "2", "n_full": "3"}, files={"file": ("newer.csv", buf.getvalue(), "text/csv")})
        assert r.status_code == 200, r.text
        assert r.json()["upload_report"]["n_valid_draws"] == 200
        bad = c.post("/api/predict", data={"n_split": "2", "n_full": "3"}, files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")})
        assert bad.status_code == 422


def test_api_without_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("TOTO_ML_ARTIFACT", str(tmp_path / "missing"))
    from toto_ml.api.main import app
    with TestClient(app) as c:
        assert c.get("/api/model").status_code == 503


@pytest.fixture(scope="module")
def ensemble_artifact(tmp_path_factory, synth_draws, fast_cfg):
    """The live setup: XGBoost + Random Forest ensemble, exported as a fixed choice."""
    from toto_ml.config import deep_merge
    cfg = deep_merge(fast_cfg, {"ensemble": {"members": ["xgboost", "random_forest"]},
                                "models": {"xgboost": {"n_estimators": 20, "max_depth": 2}}})
    res = run_experiment(synth_draws, cfg, ["random", "frequency", "xgboost", "random_forest"])
    _, table = select_best(res)
    return save_artifact(res, cfg["live"]["model"], synth_draws, tmp_path_factory.mktemp("ens"), table)


def test_ensemble_gives_two_split_and_two_full_sets(ensemble_artifact):
    b = load_artifact(ensemble_artifact)
    assert b["model_name"] == "ensemble" and b["members"] == ["xgboost", "random_forest"]
    assert b["metadata"]["best_model_label"] == "Ensemble (XGBoost + Random Forest)"
    assert set(b["split_scorers"]) == set(b["scorers"]) == {"xgboost", "random_forest"}
    r = predict_next(b)
    assert [s["source"] for s in r["sets"]] == ["split", "split", "full", "full"]
    assert [s["rank"] for s in r["sets"]] == [1, 2, 3, 4]
    nums = [s["numbers"] for s in r["sets"]]
    for i, a in enumerate(nums):  # full-data sets are no near-copies of the split sets
        for c in nums[:i]:
            assert len(set(a) & set(c)) <= 3


def test_refit_changes_only_the_full_model(ensemble_artifact, synth_draws, tmp_path):
    from toto_ml.artifact import refit
    from toto_ml.data import validate_and_clean
    from toto_ml.features import build_features
    b = load_artifact(ensemble_artifact)
    more, _, _ = validate_and_clean(generate_synthetic_draws(200, seed=3))
    new = refit(b, more, out_dir=tmp_path)
    live = build_features(more, b["config"]).rows_for_states([len(more)])
    for m in ("xgboost", "random_forest"):
        assert np.allclose(new["split_scorers"][m].score(live), b["split_scorers"][m].score(live), rtol=0, atol=1e-12)
    assert any(not np.allclose(new["scorers"][m].score(live), b["scorers"][m].score(live), rtol=0, atol=1e-6)
               for m in ("xgboost", "random_forest"))


def test_ranked_sets_respect_earlier_sets():
    p = np.linspace(0.2, 0.1, 49)
    first = ranked_sets(p, p, n_sets=1)[0]["numbers"]
    again = ranked_sets(p, p, n_sets=2, earlier=[first])
    assert all(len(set(s["numbers"]) & set(first)) <= 3 for s in again)
