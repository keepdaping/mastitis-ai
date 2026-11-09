"""
SHAP explanations - shared by ml/evaluate.py (summary plot) and app/predictor.py (top reasons).

Our models are pipelines: [optional scaler] -> [optional SMOTE] -> model.
SMOTE only acts during training, so to explain a prediction we apply the
transform steps (e.g. the scaler) and explain the final model directly.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

Transform = Callable[[pd.DataFrame], pd.DataFrame]
POSITIVE_CLASS = 1


def _split_pipeline(pipe: Pipeline) -> tuple[Transform, object]:
    """Return (function that applies the preprocessing steps, final estimator)."""
    steps = [step for _, step in pipe.steps[:-1] if hasattr(step, "transform")]  # skips SMOTE

    def transform(features: pd.DataFrame) -> pd.DataFrame:
        """Apply the preprocessing steps (e.g. the scaler) to a feature frame."""
        transformed = features
        for step in steps:
            transformed = step.transform(transformed)
        return pd.DataFrame(np.asarray(transformed), columns=features.columns, index=features.index)

    return transform, pipe.steps[-1][1]


def make_explainer(pipe: Pipeline, background: pd.DataFrame) -> tuple[Transform, object]:
    """Build a SHAP explainer for the pipeline. `background` = sample of training features."""
    transform, estimator = _split_pipeline(pipe)
    if hasattr(estimator, "coef_"):  # logistic regression
        explainer = shap.LinearExplainer(estimator, transform(background))
    else:  # random forest / xgboost
        explainer = shap.TreeExplainer(estimator)
    return transform, explainer


def shap_values(transform: Transform, explainer: object, features: pd.DataFrame) -> np.ndarray:
    """SHAP values for the POSITIVE (mastitis) class, shape (n_rows, n_features)."""
    values = explainer.shap_values(transform(features))
    if isinstance(values, list):  # older shap: one array per class
        values = values[POSITIVE_CLASS]
    values = np.asarray(values)
    if values.ndim == 3:  # newer shap: (rows, features, classes)
        values = values[:, :, POSITIVE_CLASS]
    return values
