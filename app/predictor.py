"""
The trained model, loaded once and shared by /predict, the seed script and the live demo (app/ layer).

score(frame) takes a cow's recent readings + the new one (training-CSV column names), builds
features with the SAME ml/features.py used in training, and returns the risk score, risk level
and SHAP values for the newest reading. Risk bands are defined here (limits in app/config.py).
"""

import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.config import MEDIUM_FROM
from ml.explain import make_explainer, shap_values
from ml.features import build_features

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MASTITIS_MODEL", ROOT / "models" / "mastitis_model.joblib"))


def medium_floor(threshold: float) -> float:
    """Start of the 'medium' band: MEDIUM_FROM (0.25). If a retrained model's threshold were itself
    <= MEDIUM_FROM the medium band would be empty, so we would use half the threshold instead."""
    return MEDIUM_FROM if threshold > MEDIUM_FROM else round(threshold / 2, 3)


def risk_level(score: float, threshold: float) -> str:
    """low < medium_floor <= medium < threshold <= high."""
    if score >= threshold:
        return "high"
    if score >= medium_floor(threshold):
        return "medium"
    return "low"


def risk_bands(threshold: float) -> dict[str, str]:
    """The bands in use, as text (shown by GET /model-info)."""
    floor = medium_floor(threshold)
    return {"low": f"< {floor}", "medium": f"{floor} - {threshold:.3f}", "high": f">= {threshold:.3f}"}


class Predictor:
    """The trained model bundle + its SHAP explainer."""

    def __init__(self, path: Path = MODEL_PATH):
        self.bundle = joblib.load(path)
        model = self.bundle["model"]
        # We score ONE row at a time: parallel trees only add thread start-up overhead.
        # (Same trees, same predictions - just faster for single rows.)
        estimator = model.steps[-1][1]
        if "n_jobs" in estimator.get_params():
            estimator.set_params(n_jobs=1)
        self.transform, self.explainer = make_explainer(model, self.bundle["background"])

    @property
    def version(self) -> str:
        """Model version from the saved metadata, e.g. 'rf-smote-v2'."""
        return self.bundle["version"]

    @property
    def threshold(self) -> float:
        """Decision threshold: scores at or above it are 'high'."""
        return self.bundle["threshold"]

    @property
    def features(self) -> list[str]:
        """Feature names in the order the model expects."""
        return self.bundle["features"]

    def score(self, frame: pd.DataFrame) -> tuple[float, str, np.ndarray, dict]:
        """frame: cow history + new reading (last row). Returns (score, level, shap_row, feature_row)."""
        features = build_features(frame, self.bundle["reference"]).iloc[[-1]][self.features]
        score = float(self.bundle["model"].predict_proba(features)[0, 1])
        shap_row = shap_values(self.transform, self.explainer, features)[0]
        return score, risk_level(score, self.threshold), shap_row, features.iloc[0].to_dict()


@lru_cache(maxsize=1)
def get_predictor() -> Predictor | None:
    """The shared Predictor, or None if no model has been trained yet."""
    return Predictor() if MODEL_PATH.exists() else None
