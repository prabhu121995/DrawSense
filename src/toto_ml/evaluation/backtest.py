"""Chronological walk-forward backtesting & final hold-out evaluation.

Timeline of feature states (state t = "just before draw t"):

    [0, min_history)                         warm-up, no rows
    [min_history, val_start)                 initial training only
    [val_start, test_start)                  DEVELOPMENT walk-forward validation
                                             (expanding window, refit every k draws)
    [test_start, T)                          FINAL TEST - untouched until the end

Development out-of-sample (OOS) predictions are used to fit calibrators and to
form the ensemble. For the final test each model is fitted ONCE on all
pre-test states, calibrators are frozen, and every test draw is scored from
features of strictly earlier draws. No test-period outcome is ever used for
training, tuning, calibration or model selection. All models are evaluated on
identical test draws.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..config import BASELINE_P, N_NUMBERS, config_hash
from ..features import FeatureSet, build_features
from ..models import BASELINES, ML_MODELS, Calibrator, make_scorer, rank_mean_ensemble
from ..models.tuning import tune_model
from . import stats as st
from .metrics import (EXPECTED, per_draw_ranking, probability_summary, rank_order,
                      ranking_summary)

log = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]


@dataclass
class SplitInfo:
    T: int
    min_history: int
    val_start: int
    test_start: int

    @property
    def val_states(self):
        return np.arange(self.val_start, self.test_start)

    @property
    def test_states(self):
        return np.arange(self.test_start, self.T)

    @property
    def dev_states(self):
        return np.arange(self.min_history, self.test_start)


def make_split(T: int, cfg: dict) -> SplitInfo:
    s = cfg["split"]
    mh = int(s["min_history"])
    n_test = max(1, int(round(T * float(s["test_fraction"]))))
    test_start = T - n_test
    val_start = mh + int(s["min_train_draws"])
    if val_start + 10 > test_start:
        raise ValueError(
            f"Not enough draws ({T}) for min_history={mh}, min_train_draws={s['min_train_draws']} "
            f"and test_fraction={s['test_fraction']}. Need at least ~{val_start + 10 + n_test}.")
    return SplitInfo(T=T, min_history=mh, val_start=val_start, test_start=test_start)


@dataclass
class ModelOutput:
    name: str
    val_scores: np.ndarray
    test_scores: np.ndarray
    val_probs: np.ndarray = None
    test_probs: np.ndarray = None
    calibrator: Calibrator = None
    final_scorer: object = None           # trained on all pre-test states (used for test)
    production_scorer: object = None      # trained on all labelled states (used for live)
    production_calibrator: Calibrator = None
    fit_seconds: float = 0.0
    params: dict = field(default_factory=dict)


@dataclass
class ExperimentResult:
    cfg: dict
    config_hash: str
    split: SplitInfo
    features: FeatureSet
    draws: pd.DataFrame
    outputs: dict[str, ModelOutput]
    leaderboard: dict            # {"validation": [...], "test": [...]}
    reliability: dict            # {model: {split: rel}}
    significance: dict
    importance: dict
    tuning: dict
    random_null: dict
    elapsed: float


def _matrix(values: np.ndarray, n_states: int) -> np.ndarray:
    return values.reshape(n_states, N_NUMBERS)


def _actual(fs: FeatureSet, states: np.ndarray) -> np.ndarray:
    df = fs.rows_for_states(states)
    return _matrix(df["target"].to_numpy(dtype=np.int8), len(states))


def run_experiment(draws: pd.DataFrame, cfg: dict, models: list[str] | None = None,
                   progress: ProgressFn | None = None) -> ExperimentResult:
    t0 = time.time()
    progress = progress or (lambda f, m: None)
    seed = int(cfg["seed"])
    models = [m for m in (models or ["random", "frequency", "ewm", *ML_MODELS]) if m != "ensemble"]
    want_ensemble = True
    cal_method = cfg["calibration"]["method"]
    n_bins = int(cfg["calibration"]["n_bins"])
    k = int(cfg["split"]["refit_every"])

    progress(0.01, "Building leakage-free features")
    fs = build_features(draws, cfg)
    split = make_split(fs.T, cfg)
    fnames = fs.feature_names
    frame = fs.frame
    dev_rows = frame[(frame["t"] >= split.min_history) & (frame["t"] < split.test_start)]
    test_rows = fs.rows_for_states(split.test_states)
    labelled_rows = frame[frame["t"] < split.T]
    val_states, test_states = split.val_states, split.test_states
    log.info("Split: T=%d, val=[%d,%d) test=[%d,%d), features=%d", split.T, split.val_start,
             split.test_start, split.test_start, split.T, len(fnames))

    # ---------- optional Optuna tuning (pre-test data only) ----------
    tuning: dict = {}
    overrides: dict[str, dict] = {}
    if cfg["tuning"].get("enabled"):
        for name in cfg["tuning"].get("models", []):
            if name in models and name in ML_MODELS:
                progress(0.03, f"Optuna tuning {name} on development data")
                tuning[name] = tune_model(name, dev_rows, fnames, cfg)
                overrides[name] = tuning[name]["best_params"]

    outputs: dict[str, ModelOutput] = {}
    n_models = len(models)
    for mi, name in enumerate(models):
        base = 0.05 + 0.80 * mi / n_models
        progress(base, f"Walk-forward validation: {name}")
        tm = time.time()
        # --- development walk-forward (expanding window) ---
        val_scores = []
        origins = list(range(split.val_start, split.test_start, k))
        for oi, origin in enumerate(origins):
            block = np.arange(origin, min(origin + k, split.test_start))
            sc = make_scorer(name, cfg, fnames, overrides.get(name))
            if sc.needs_fit:
                train = frame[(frame["t"] >= split.min_history) & (frame["t"] < origin)]
                sc.fit(train)
            val_scores.append(sc.score(fs.rows_for_states(block)))
            progress(base + 0.80 / n_models * 0.8 * (oi + 1) / len(origins),
                     f"{name}: walk-forward origin {oi + 1}/{len(origins)}")
        val_scores = np.concatenate(val_scores)
        # --- final model: fit once on all pre-test states, frozen for the test ---
        final = make_scorer(name, cfg, fnames, overrides.get(name))
        if final.needs_fit:
            final.fit(dev_rows)
        test_scores = final.score(test_rows)
        # --- production model for live predictions: all labelled states ---
        prod = final
        if final.needs_fit:
            prod = make_scorer(name, cfg, fnames, overrides.get(name)).fit(labelled_rows)
        outputs[name] = ModelOutput(name=name, val_scores=val_scores, test_scores=test_scores,
                                    final_scorer=final, production_scorer=prod,
                                    fit_seconds=time.time() - tm,
                                    params={**(cfg["models"].get(name) or {}), **overrides.get(name, {})})

    if want_ensemble:
        members = [m for m in cfg["ensemble"]["members"] if m in outputs]
        if len(members) >= 2:
            vt = fs.rows_for_states(val_states)["t"].to_numpy()
            tt = test_rows["t"].to_numpy()
            outputs["ensemble"] = ModelOutput(
                name="ensemble",
                val_scores=rank_mean_ensemble({m: outputs[m].val_scores for m in members}, vt),
                test_scores=rank_mean_ensemble({m: outputs[m].test_scores for m in members}, tt),
                params={"members": members, "method": cfg["ensemble"]["method"]})

    # ---------- calibration (fit on development OOS only) ----------
    progress(0.87, "Calibrating probabilities on development OOS predictions")
    y_val = _actual(fs, val_states).ravel()
    y_test = _actual(fs, test_states).ravel()
    for name, out in outputs.items():
        method = "constant" if name == "random" else cal_method
        out.calibrator = Calibrator(method).fit(out.val_scores, y_val)
        out.val_probs = out.calibrator.transform(out.val_scores)
        out.test_probs = out.calibrator.transform(out.test_scores)
        # Production calibrator may additionally use the (already evaluated) test OOS scores.
        out.production_calibrator = Calibrator(method).fit(
            np.concatenate([out.val_scores, out.test_scores]), np.concatenate([y_val, y_test]))

    # ---------- metrics ----------
    progress(0.9, "Computing metrics, random simulations and significance tests")
    leaderboard = {"validation": [], "test": []}
    reliability: dict = {}
    per_draw: dict = {}
    for split_name, states in (("validation", val_states), ("test", test_states)):
        actual = _actual(fs, states)
        for name, out in outputs.items():
            scores = out.val_scores if split_name == "validation" else out.test_scores
            probs = out.val_probs if split_name == "validation" else out.test_probs
            S = _matrix(scores, len(states))
            P = _matrix(probs, len(states))
            order = rank_order(S, states, seed)
            pdm = per_draw_ranking(order, actual)
            row = {"model": name, **ranking_summary(pdm)}
            psum, rel = probability_summary(P, actual, n_bins)
            row.update(psum)
            row["n_draws"] = int(len(states))
            row["fit_seconds"] = round(out.fit_seconds, 2)
            leaderboard[split_name].append(row)
            reliability.setdefault(name, {})[split_name] = rel
            per_draw.setdefault(split_name, {})[name] = {**pdm, "logloss": st.per_draw_logloss(P, actual)}

    # ---------- significance on the final test ----------
    test_actual = _actual(fs, test_states)
    n_sims = int(cfg["evaluation"]["n_random_simulations"])
    n_boot = int(cfg["evaluation"]["n_bootstrap"])
    alpha = float(cfg["evaluation"]["alpha"])
    null = st.simulate_random(test_actual, n_sims, seed + 1)
    null_summary = {m: {"mean": float(v.mean()), "q025": float(np.quantile(v, 0.025)),
                        "q975": float(np.quantile(v, 0.975))} for m, v in null.items()}
    sig: dict = {"null": null_summary, "models": {}, "n_test_draws": int(len(test_states)),
                 "alpha": alpha, "n_random_simulations": n_sims, "n_bootstrap": n_boot}
    pd_test = per_draw["test"]
    rnd = pd_test.get("random")
    const_ll = EXPECTED["log_loss"]
    raw_p = {}
    for name, d in pd_test.items():
        if name == "random":
            continue
        obs10, obs6 = d["hits10"].mean(), d["hits6"].mean()
        entry = {
            "hits@10": float(obs10), "hits@6": float(obs6),
            "p_hits10_vs_random_sim": st.empirical_p(obs10, null["hits10"]),
            "p_hits6_vs_random_sim": st.empirical_p(obs6, null["hits6"]),
            "p_ndcg10_vs_random_sim": st.empirical_p(d["ndcg10"].mean(), null["ndcg10"]),
            "percentile_vs_random_hits10": float((null["hits10"] < obs10).mean() * 100),
            "boot_hits10_minus_expected": st.bootstrap_mean_ci(d["hits10"] - EXPECTED["hits@10"], n_boot, seed + 2, alpha),
            "boot_hits6_minus_expected": st.bootstrap_mean_ci(d["hits6"] - EXPECTED["hits@6"], n_boot, seed + 3, alpha),
            "boot_logloss_minus_uniform": st.bootstrap_mean_ci(d["logloss"] - const_ll, n_boot, seed + 4, alpha),
        }
        if rnd is not None:
            entry["boot_hits10_paired_vs_random_model"] = st.bootstrap_mean_ci(
                d["hits10"] - rnd["hits10"], n_boot, seed + 5, alpha)
        raw_p[name] = entry["p_hits10_vs_random_sim"]
        sig["models"][name] = entry
    adj = st.holm(raw_p) if raw_p else {}
    for name, entry in sig["models"].items():
        entry["p_hits10_holm"] = adj.get(name)
        better_ll = entry["boot_logloss_minus_uniform"]["ci_high"] < 0
        entry["significant_hits10"] = bool(adj.get(name, 1) < alpha)
        entry["significant_logloss_improvement"] = bool(better_ll)
        if entry["significant_hits10"]:
            entry["verdict"] = ("Beats random on hits@10 after Holm correction on this test period. "
                                "Treat with caution: replicate on future draws before believing it.")
        else:
            entry["verdict"] = "No statistically significant evidence of beating random selection."
    sig["any_significant"] = any(e["significant_hits10"] for e in sig["models"].values())
    sig["summary"] = (
        "At least one model passed the Holm-corrected test on hits@10; this may still be chance or "
        "period-specific - confirm on new draws." if sig["any_significant"] else
        "No model shows statistically significant out-of-sample improvement over random selection. "
        "This is the expected result for a fair, independent lottery.")

    # ---------- feature importance (SHAP for tree models) ----------
    progress(0.95, "Computing SHAP feature importance")
    importance = compute_importance(outputs, test_rows, fnames, seed)

    for name in leaderboard:
        leaderboard[name].sort(key=lambda r: -r["hits@10"])
    progress(1.0, "Done")
    return ExperimentResult(cfg=cfg, config_hash=config_hash(cfg), split=split, features=fs, draws=draws,
                            outputs=outputs, leaderboard=leaderboard, reliability=reliability,
                            significance=sig, importance=importance, tuning=tuning,
                            random_null=null_summary, elapsed=time.time() - t0)


def compute_importance(outputs: dict, rows: pd.DataFrame, fnames: list[str], seed: int,
                       max_rows: int = 1500) -> dict:
    res = {}
    sample = rows.sample(min(max_rows, len(rows)), random_state=seed)
    X = sample[fnames].to_numpy(dtype=np.float64)
    import warnings
    for name, out in outputs.items():
        sc = out.final_scorer
        if sc is None or not getattr(sc, "needs_fit", False):
            continue
        try:
            warnings.filterwarnings("ignore", module="shap")
            if name == "logistic_regression":
                coef = sc.estimator[-1].coef_.ravel()
                vals = np.abs(coef)
                method = "|standardised coefficient|"
            elif sc.tree_model is not None:
                import shap
                expl = shap.TreeExplainer(sc.tree_model)
                sv = expl.shap_values(X, check_additivity=False) if name in ("random_forest", "extra_trees") \
                    else expl.shap_values(X)
                sv = np.asarray(sv)
                if sv.ndim == 3:  # (n, f, classes) or (classes, n, f)
                    sv = sv[..., 1] if sv.shape[-1] == 2 else sv[1]
                vals = np.abs(sv).mean(0)
                method = "mean |SHAP|"
            else:
                continue
            order = np.argsort(-vals)
            res[name] = {"method": method,
                         "features": [{"feature": fnames[i], "importance": float(vals[i])} for i in order]}
        except Exception as e:  # SHAP should never break an experiment
            log.warning("Importance for %s failed: %s", name, e)
    return res
