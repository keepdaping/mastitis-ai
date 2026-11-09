"""
SHAP explanations, shared by ml/evaluate.py (summary plot) and api/main.py (top reasons).

Our models are pipelines: [optional scaler] -> [optional SMOTE] -> model.
SMOTE only acts during training, so to explain a prediction we apply the
transform steps (e.g. the scaler) and explain the final model directly.
"""
import numpy as np
import pandas as pd
import shap


def _split_pipeline(pipe):
    """Return (function that applies the preprocessing steps, final estimator)."""
    steps = [s for _, s in pipe.steps[:-1] if hasattr(s, "transform")]   # skips SMOTE

    def transform(X):
        Z = X
        for s in steps:
            Z = s.transform(Z)
        return pd.DataFrame(np.asarray(Z), columns=X.columns, index=X.index)

    return transform, pipe.steps[-1][1]


def make_explainer(pipe, background: pd.DataFrame):
    """Build a SHAP explainer for the pipeline. `background` = sample of training features."""
    transform, est = _split_pipeline(pipe)
    if hasattr(est, "coef_"):                       # logistic regression
        explainer = shap.LinearExplainer(est, transform(background))
    else:                                           # random forest / xgboost
        explainer = shap.TreeExplainer(est)
    return transform, explainer


def shap_values(transform, explainer, X: pd.DataFrame) -> np.ndarray:
    """SHAP values for the POSITIVE (mastitis) class, shape (n_rows, n_features)."""
    sv = explainer.shap_values(transform(X))
    if isinstance(sv, list):                        # older shap: one array per class
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                                # newer shap: (rows, features, classes)
        sv = sv[:, :, 1]
    return sv
