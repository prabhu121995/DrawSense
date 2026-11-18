"""Optuna hyper-parameter tuning restricted to PRE-TEST data.

The development states are split chronologically (first 70% train, last 30%
validate). The final test period is never touched.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from .registry import make_scorer

log = logging.getLogger(__name__)


def _space(trial, name: str) -> dict:
    if name == "lightgbm":
        return {"n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "num_leaves": trial.suggest_int("num_leaves", 4, 31),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50, log=True)}
    if name == "xgboost":
        return {"n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50, log=True)}
    if name == "catboost":
        return {"iterations": trial.suggest_int("iterations", 50, 300),
                "depth": trial.suggest_int("depth", 2, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 50, log=True)}
    if name in ("random_forest", "extra_trees"):
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 300)}
    if name == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-3, 10, log=True)}
    if name == "mlp":
        return {"alpha": trial.suggest_float("alpha", 1e-4, 1.0, log=True)}
    raise KeyError(name)


def tune_model(name: str, dev: pd.DataFrame, feature_names: list[str], cfg: dict) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    states = np.sort(dev["t"].unique())
    cut = states[int(len(states) * 0.7)]
    tr, va = dev[dev["t"] < cut], dev[dev["t"] >= cut]
    y = va["target"].to_numpy(int)

    def objective(trial):
        sc = make_scorer(name, cfg, feature_names, _space(trial, name)).fit(tr)
        p = np.clip(sc.score(va), 1e-6, 1 - 1e-6)
        return log_loss(y, p, labels=[0, 1])

    sampler = optuna.samplers.TPESampler(seed=int(cfg["seed"]))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=int(cfg["tuning"]["n_trials"]))
    log.info("Tuned %s: best logloss %.5f params %s", name, study.best_value, study.best_params)
    return {"best_params": study.best_params, "best_value": float(study.best_value),
            "n_trials": len(study.trials)}
