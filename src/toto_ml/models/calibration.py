"""Probability calibration of raw ranking scores.

Calibrators are fitted ONLY on out-of-sample walk-forward predictions from the
development period, then applied unchanged to the final test period and to
live predictions. Outputs are *marginal* per-number probabilities; they are
not normalised to sum to 1 (they should sum to roughly 6 per draw).
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ..config import BASELINE_P

EPS = 1e-6


class Calibrator:
    def __init__(self, method: str = "sigmoid"):
        self.method = method
        self.mu = 0.0
        self.sd = 1.0
        self.model = None
        self.constant: float | None = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "Calibrator":
        scores = np.asarray(scores, float)
        y = np.asarray(y, int)
        if self.method == "constant" or np.unique(y).size < 2 or np.std(scores) < 1e-12:
            self.method_used = "constant"
            self.constant = float(np.clip(y.mean() if len(y) else BASELINE_P, EPS, 1 - EPS))
            return self
        self.method_used = self.method
        if self.method == "sigmoid":
            self.mu, self.sd = float(scores.mean()), float(scores.std() or 1.0)
            self.model = LogisticRegression(C=1.0)
            self.model.fit(((scores - self.mu) / self.sd).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS)
            self.model.fit(scores, y)
        elif self.method == "none":
            pass
        else:
            raise ValueError(f"Unknown calibration method {self.method}")
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, float)
        if getattr(self, "method_used", None) == "constant":
            return np.full(len(scores), self.constant)
        if self.method == "sigmoid":
            p = self.model.predict_proba(((scores - self.mu) / self.sd).reshape(-1, 1))[:, 1]
        elif self.method == "isotonic":
            p = self.model.predict(scores)
        else:
            p = scores
        return np.clip(p, EPS, 1 - EPS)
