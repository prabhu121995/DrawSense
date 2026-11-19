"""Model registry: baselines + ML classifiers behind one scorer interface.

Every scorer maps feature rows -> a raw *ranking score* (higher = ranked
higher). Raw scores are NOT probabilities; calibrated marginal probabilities
come from :mod:`toto_ml.models.calibration`, fitted on out-of-sample data.
"""
from __future__ import annotations

import logging
import os
import warnings
import zlib

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

N_JOBS = int(os.environ.get("TOTO_ML_N_JOBS", max(1, (os.cpu_count() or 2) - 1)))

MODEL_ORDER = ["random", "frequency", "ewm", "logistic_regression", "random_forest", "extra_trees",
               "xgboost", "lightgbm", "catboost", "mlp", "ensemble"]
BASELINES = {"random", "frequency", "ewm"}
ML_MODELS = ["logistic_regression", "random_forest", "extra_trees", "xgboost", "lightgbm", "catboost", "mlp"]
LABELS = {
    "random": "Random selection baseline",
    "frequency": "Historical frequency baseline",
    "ewm": "Exponentially weighted frequency",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "mlp": "MLP Neural Network",
    "ensemble": "Ensemble ranking model",
}


class Scorer:
    name: str = "base"
    needs_fit: bool = False

    def __init__(self, params: dict | None = None, seed: int = 42, feature_names: list[str] | None = None):
        self.params = dict(params or {})
        self.seed = seed
        self.feature_names = feature_names

    def fit(self, df: pd.DataFrame) -> "Scorer":
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError


class RandomScorer(Scorer):
    """Uniform random ranking, deterministic per (seed, state)."""
    name = "random"

    def score(self, df: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(df))
        for t, idx in df.groupby("t").indices.items():
            rng = np.random.default_rng([self.seed, 7919, int(t)])
            out[idx] = rng.random(len(idx))
        return out


class FrequencyScorer(Scorer):
    name = "frequency"

    def score(self, df):
        return df["freq_all"].to_numpy(dtype=float)


class EWMScorer(Scorer):
    name = "ewm"

    def score(self, df):
        col = f"ewm_{self.params.get('alpha', 0.05)}"
        if col not in df.columns:
            raise KeyError(f"{col} not in features; add alpha to features.ewm_alphas")
        return df[col].to_numpy(dtype=float)


def _make_estimator(name: str, params: dict, seed: int):
    p = dict(params)
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, **p))
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=N_JOBS, **p)
    if name == "extra_trees":
        return ExtraTreesClassifier(random_state=seed, n_jobs=N_JOBS, **p)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=seed, n_jobs=N_JOBS, eval_metric="logloss", tree_method="hist", **p)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=seed, n_jobs=N_JOBS, verbose=-1, **p)
    if name == "catboost":
        from catboost import CatBoostClassifier
        # CatBoost results depend on the thread count, not only on the seed. A fixed thread_count from the
        # config makes every retrain reproducible on any PC (default 1, see configs/default.yaml).
        threads = p.pop("thread_count", N_JOBS)
        return CatBoostClassifier(random_seed=seed, thread_count=threads, verbose=0,
                                  allow_writing_files=False, **p)
    if name == "mlp":
        if "hidden_layer_sizes" in p:
            p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
        return make_pipeline(StandardScaler(), MLPClassifier(random_state=seed, **p))
    raise KeyError(f"Unknown model '{name}'")


class SklearnScorer(Scorer):
    needs_fit = True

    def __init__(self, name: str, params=None, seed=42, feature_names=None):
        super().__init__(params, seed, feature_names)
        self.name = name
        self.estimator = None

    def fit(self, df: pd.DataFrame) -> "SklearnScorer":
        X = df[self.feature_names].to_numpy(dtype=np.float64)
        y = df["target"].to_numpy(dtype=int)
        self.estimator = _make_estimator(self.name, self.params, self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.estimator.fit(X, y)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_names].to_numpy(dtype=np.float64)
        return self.estimator.predict_proba(X)[:, 1]

    @property
    def tree_model(self):
        """Underlying tree estimator for SHAP (None for non-tree models)."""
        if self.name in ("random_forest", "extra_trees", "xgboost", "lightgbm", "catboost"):
            return self.estimator
        return None


def make_scorer(name: str, cfg: dict, feature_names: list[str], params_override: dict | None = None) -> Scorer:
    params = dict(cfg["models"].get(name, {}) or {})
    params.update(params_override or {})
    seed = int(cfg["seed"]) + zlib.crc32(name.encode()) % 1000
    if name == "random":
        return RandomScorer(params, seed=int(cfg["seed"]), feature_names=feature_names)
    if name == "frequency":
        return FrequencyScorer(params, seed=seed, feature_names=feature_names)
    if name == "ewm":
        return EWMScorer(params, seed=seed, feature_names=feature_names)
    return SklearnScorer(name, params, seed=seed, feature_names=feature_names)


def rank_mean_ensemble(member_scores: dict[str, np.ndarray], t: np.ndarray) -> np.ndarray:
    """Average within-draw percentile rank across members (scale-free, robust)."""
    df = pd.DataFrame(member_scores)
    df["t"] = t
    ranks = df.groupby("t")[list(member_scores)].rank(pct=True, method="average")
    return ranks.mean(axis=1).to_numpy()
