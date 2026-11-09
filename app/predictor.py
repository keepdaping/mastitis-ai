"""
The model, loaded once and shared by /predict, the seed script and the live demo.

score(frame) takes a cow's recent readings + the new one (training-CSV column names),
builds features with the SAME ml/features.py used in training, and returns the risk
score, risk level and SHAP values for the newest reading.
"""
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from ml.explain import make_explainer, shap_values
from ml.features import build_features

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MASTITIS_MODEL", ROOT / "models" / "mastitis_model.joblib"))
MEDIUM_FROM = 0.25           # low < 0.25 <= medium < threshold <= high (see medium_floor)


def medium_floor(threshold: float) -> float:
    """Start of the 'medium' band: 0.25. If a retrained model's threshold were itself
    <= 0.25 the medium band would be empty, so we would use half the threshold instead."""
    return MEDIUM_FROM if threshold > MEDIUM_FROM else round(threshold / 2, 3)


def risk_level(score: float, threshold: float) -> str:
    if score >= threshold:
        return "high"
    if score >= medium_floor(threshold):
        return "medium"
    return "low"


class Predictor:
    def __init__(self, path: Path = MODEL_PATH):
        self.bundle = joblib.load(path)
        model = self.bundle["model"]
        # We score ONE row at a time: parallel trees only add thread start-up overhead.
        # (Same trees, same predictions - just faster for single rows.)
        est = model.steps[-1][1]
        if "n_jobs" in est.get_params():
            est.set_params(n_jobs=1)
        self.transform, self.explainer = make_explainer(model, self.bundle["background"])

    @property
    def version(self) -> str:
        return self.bundle["version"]

    @property
    def threshold(self) -> float:
        return self.bundle["threshold"]

    def score(self, frame: pd.DataFrame):
        """frame: cow history + new reading (last row). Returns (score, level, shap_row, feature_row)."""
        X = build_features(frame, self.bundle["reference"]).iloc[[-1]][self.bundle["features"]]
        score = float(self.bundle["model"].predict_proba(X)[0, 1])
        sv = shap_values(self.transform, self.explainer, X)[0]
        return score, risk_level(score, self.threshold), sv, X.iloc[0].to_dict()


@lru_cache(maxsize=1)
def get_predictor() -> Predictor | None:
    """The shared Predictor, or None if no model has been trained yet."""
    return Predictor() if MODEL_PATH.exists() else None
