from pathlib import Path

import pytest

from toto_ml.config import load_config
from toto_ml.data import generate_synthetic_draws, validate_and_clean

FAST = {
    "split": {"min_history": 20, "min_train_draws": 40, "refit_every": 25, "test_fraction": 0.2},
    "evaluation": {"n_random_simulations": 200, "n_bootstrap": 200},
    "models": {"random_forest": {"n_estimators": 20, "max_depth": 3, "min_samples_leaf": 50},
               "lightgbm": {"n_estimators": 20, "num_leaves": 4, "min_child_samples": 50}},
    "ensemble": {"members": ["frequency", "ewm", "logistic_regression", "lightgbm"]},
}
REAL_CSV = Path(__file__).resolve().parents[1] / "toto_results.csv"


@pytest.fixture(scope="session")
def fast_cfg():
    return load_config(overrides=FAST)


@pytest.fixture(scope="session")
def synth_draws():
    draws, _, _ = validate_and_clean(generate_synthetic_draws(180, seed=3))
    return draws
