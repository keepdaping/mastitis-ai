"""
Train and compare models for subclinical mastitis (SCM) detection - writes models/.

Steps:
  1. Load the (SIMULATED) milkings, hold out 2 whole farms as the test set.
     Splitting by farm checks the model works on farms it has never seen.
  2. Build features with ml/features.py (herd medians from TRAIN farms only), with a CLEAN
     baseline: readings that were high risk or inside an alert episode don't count towards a
     cow's "normal". Those flags need risk levels, so they come from a stage-0 model
     (see baseline_flags()), exactly as the live service stores them per reading.
  3. Keep only milkings with a CMT label; positive = cmt_score >= 2.
  4. Compare 3 models x 2 ways of handling class imbalance
     (class weights vs SMOTE oversampling) with 5-fold GroupKFold CV by farm.
  5. Pick the model with the highest recall whose precision is >= 0.5.
  6. Tune the decision threshold so recall >= 0.85 (missing a sick cow is
     worse than an extra CMT test), using out-of-fold predictions.
  7. Save the best model + threshold + feature list + version to models/.

Run:  python -m ml.train
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.config import CMT_POSITIVE_FROM
from ml.episodes import flags_for_frame
from ml.features import FEATURES, build_features, compute_reference

SEED = 42
DATA = Path("data/synthetic_milkings.csv")
MODELS = Path("models")
REPORTS = Path("reports")
N_TEST_FARMS = 2
CV_FOLDS = 5
MIN_PRECISION = 0.5  # selection constraint
TARGET_RECALL = 0.85  # threshold tuning target
CV_THRESHOLD = 0.5  # threshold for the per-fold CV metrics
SHAP_BACKGROUND_SIZE = 200  # rows saved with the model for SHAP explanations
MODEL_VERSION = 2  # bump when retraining -> model_version e.g. "rf-smote-v2" (v2: clean baseline)
SLUGS = {  # short names used in model_version (returned by the API)
    "LogReg + class_weight": "lr-cw",
    "LogReg + SMOTE": "lr-smote",
    "RandomForest + class_weight": "rf-cw",
    "RandomForest + SMOTE": "rf-smote",
    "XGBoost + scale_pos_weight": "xgb-spw",
    "XGBoost + SMOTE": "xgb-smote",
}
CV_METRICS = ["recall", "precision", "f1", "roc_auc"]


# --------------------------------------------------------------------------- data
def baseline_flags(
    df: pd.DataFrame, plain_features: pd.DataFrame, labels: pd.Series, labelled: pd.Series, is_test: pd.Series
) -> tuple[pd.Series, dict]:
    """Stage 0: which readings must be excluded from each cow's baseline.

    In the live service a reading is excluded if the model rated it high, or an alert episode
    was active (ml/episodes.py). To create the same flags for the training data we need a risk
    level for EVERY reading, so we score all readings with a simple stage-0 model, always
    OUT-OF-FARM (a farm is never scored by a model trained on its own labels):
      - training farms: 5-fold GroupKFold over farms
      - test farms:     model trained on all training farms
    The stage-0 features use the plain (unfiltered) baseline, so nothing is circular, and every
    flag uses only the current and earlier readings (no future leakage).
    """
    stage0 = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=SEED
    )
    score = pd.Series(np.nan, index=df.index)
    train_rows = df.index[~is_test]
    farms = df.loc[train_rows, "farm_id"]
    for fit_idx, score_idx in GroupKFold(n_splits=CV_FOLDS).split(train_rows, groups=farms):
        fit_farms, score_farms = set(farms.iloc[fit_idx]), set(farms.iloc[score_idx])
        fit = labelled & df["farm_id"].isin(fit_farms)
        model = clone(stage0).fit(plain_features[fit], labels[fit])
        rows = df["farm_id"].isin(score_farms)
        score[rows] = model.predict_proba(plain_features[rows])[:, 1]
    model = clone(stage0).fit(plain_features[labelled & ~is_test], labels[labelled & ~is_test])
    score[is_test] = model.predict_proba(plain_features[is_test])[:, 1]

    labelled_train = labelled & ~is_test
    threshold, _, _ = tune_threshold(labels[labelled_train], score[labelled_train])  # same rule as the final model
    flags = flags_for_frame(df, score >= threshold)
    return flags, {"stage0_threshold": round(threshold, 4), "share_excluded": round(float(flags.mean()), 4)}


def split_farms(df: pd.DataFrame) -> list[str]:
    """The held-out test farms (a whole-farm split, seeded)."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=N_TEST_FARMS, random_state=SEED)
    farms = df["farm_id"].drop_duplicates().to_numpy()
    _, test_idx = next(splitter.split(farms, groups=farms))
    return sorted(farms[test_idx])


def load_and_split(verbose: bool = False) -> dict:
    """Load data, split farms into train/test, build features. Returns labelled rows only."""
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    test_farms = split_farms(df)
    is_test = df["farm_id"].isin(test_farms)
    reference = compute_reference(df[~is_test])  # no information from test farms
    labelled = df["cmt_score"].notna()
    labels = (df["cmt_score"] >= CMT_POSITIVE_FROM).astype(int)

    # Stage 0 -> baseline exclusion flags -> final features with a CLEAN baseline
    plain_features = build_features(df, reference)
    flags, info = baseline_flags(df, plain_features, labels, labelled, is_test)
    all_features = build_features(df.assign(baseline_exclude=flags), reference)
    if verbose:
        infected = df["true_infected"] == 1
        print(
            f"Clean baseline: {info['share_excluded']:.1%} of readings excluded "
            f"(stage-0 threshold {info['stage0_threshold']:.3f}); "
            f"{flags[infected].mean():.1%} of truly infected readings and "
            f"{flags[~infected].mean():.1%} of healthy readings excluded"
        )

    train, test = labelled & ~is_test, labelled & is_test
    return {
        "X_train": all_features[train],
        "y_train": labels[train],
        "g_train": df.loc[train, "farm_id"],
        "X_test": all_features[test],
        "y_test": labels[test],
        "meta_test": df.loc[test],
        "test_farms": test_farms,
        "reference": reference,
        "baseline_info": info,
    }


# --------------------------------------------------------------------------- models
def make_candidates(pos_weight: float) -> dict[str, Pipeline]:
    """3 models x 2 imbalance strategies. SMOTE lives inside an imblearn Pipeline,
    so synthetic samples are only created from the training folds (no leakage)."""
    logreg = LogisticRegression(max_iter=2000)
    forest = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED)
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=SEED,
    )
    smote = SMOTE(random_state=SEED)
    return {
        "LogReg + class_weight": Pipeline(
            [("scale", StandardScaler()), ("model", clone(logreg).set_params(class_weight="balanced"))]
        ),
        "LogReg + SMOTE": Pipeline([("scale", StandardScaler()), ("smote", clone(smote)), ("model", clone(logreg))]),
        "RandomForest + class_weight": Pipeline([("model", clone(forest).set_params(class_weight="balanced"))]),
        "RandomForest + SMOTE": Pipeline([("smote", clone(smote)), ("model", clone(forest))]),
        "XGBoost + scale_pos_weight": Pipeline([("model", clone(xgb).set_params(scale_pos_weight=pos_weight))]),
        "XGBoost + SMOTE": Pipeline([("smote", clone(smote)), ("model", clone(xgb))]),
    }


def cross_validate(
    model: Pipeline, features: pd.DataFrame, labels: pd.Series, groups: pd.Series
) -> tuple[pd.DataFrame, np.ndarray]:
    """GroupKFold by farm. Returns per-fold metrics (at 0.5) and out-of-fold probabilities."""
    oof = np.zeros(len(labels))
    folds = []
    for train_idx, val_idx in GroupKFold(n_splits=CV_FOLDS).split(features, labels, groups):
        fitted = clone(model).fit(features.iloc[train_idx], labels.iloc[train_idx])
        proba = fitted.predict_proba(features.iloc[val_idx])[:, 1]
        oof[val_idx] = proba
        pred = (proba >= CV_THRESHOLD).astype(int)
        truth = labels.iloc[val_idx]
        folds.append(
            {
                "recall": recall_score(truth, pred),
                "precision": precision_score(truth, pred, zero_division=0),
                "f1": f1_score(truth, pred),
                "roc_auc": roc_auc_score(truth, proba),
            }
        )
    return pd.DataFrame(folds), oof


def tune_threshold(
    labels: pd.Series, proba: np.ndarray, target_recall: float = TARGET_RECALL
) -> tuple[float, float, float]:
    """Highest threshold that still reaches the target recall (= best precision at that recall).
    Returns (threshold, precision, recall) at that threshold."""
    precision, recall, thresholds = precision_recall_curve(labels, proba)
    reaching = np.where(recall[:-1] >= target_recall)[0]  # recall[i] belongs to thresholds[i]
    best = reaching[-1] if len(reaching) else 0
    return float(thresholds[best]), float(precision[best]), float(recall[best])


# --------------------------------------------------------------------------- training steps
def print_class_balance(data: dict) -> float:
    """Print train/test class balance; returns the negative/positive ratio (XGBoost weight)."""
    labels = data["y_train"]
    print(f"Class balance (CMT-labelled milkings, positive = cmt_score >= {CMT_POSITIVE_FROM}):")
    for name, split_labels in [("train", labels), ("test", data["y_test"])]:
        print(
            f"  {name:5s}: n={len(split_labels):5d}  positive={split_labels.sum():5d} "
            f"({split_labels.mean():.1%})  negative={(1 - split_labels).sum():5d}"
        )
    print(f"  test farms (held out): {data['test_farms']}")
    pos_weight = float((labels == 0).sum() / (labels == 1).sum())
    print(f"  negative/positive ratio = {pos_weight:.2f} (used as XGBoost scale_pos_weight)\n")
    return pos_weight


def compare_candidates(candidates: dict[str, Pipeline], data: dict) -> pd.DataFrame:
    """Cross-validate every candidate; one row of CV metrics + tuned threshold per model."""
    features, labels, groups = data["X_train"], data["y_train"], data["g_train"]
    rows = []
    for name, model in candidates.items():
        folds, oof = cross_validate(model, features, labels, groups)
        threshold, precision_at, recall_at = tune_threshold(labels, oof)
        row = {"model": name}
        for metric in CV_METRICS:
            row[f"cv_{metric}"] = folds[metric].mean()
            row[f"cv_{metric}_sd"] = folds[metric].std()
        row.update({"tuned_threshold": threshold, "oof_precision_at_thr": precision_at, "oof_recall_at_thr": recall_at})
        rows.append(row)
        print(
            f"  {name:28s} recall={row['cv_recall']:.3f}+-{row['cv_recall_sd']:.3f}  "
            f"precision={row['cv_precision']:.3f}  AUC={row['cv_roc_auc']:.3f}  "
            f"thr@R>={TARGET_RECALL}={threshold:.3f}"
        )
    return pd.DataFrame(rows)


def select_best(cv: pd.DataFrame) -> str:
    """Highest CV recall among models with precision >= MIN_PRECISION (tie-break: ROC-AUC)."""
    eligible = cv[cv["cv_precision"] >= MIN_PRECISION]
    if eligible.empty:
        print(f"WARNING: no model reaches precision >= {MIN_PRECISION}; selecting on recall alone")
        eligible = cv
    best = eligible.sort_values(["cv_recall", "cv_roc_auc"], ascending=False).iloc[0]
    print(
        f"\nSelected: {best['model']} (CV recall {best['cv_recall']:.3f}, "
        f"precision {best['cv_precision']:.3f}); threshold {best['tuned_threshold']:.3f}"
    )
    return best["model"]


def fit_all(candidates: dict[str, Pipeline], cv: pd.DataFrame, data: dict) -> dict:
    """Refit every candidate on all training farms (evaluate.py compares them on the test farms)."""
    fitted = {}
    for name, model in candidates.items():
        threshold = float(cv.loc[cv["model"] == name, "tuned_threshold"].iloc[0])
        fitted[name] = {"model": clone(model).fit(data["X_train"], data["y_train"]), "threshold": threshold}
    return fitted


def save_bundle(fitted: dict, best_name: str, cv: pd.DataFrame, data: dict) -> str:
    """Write models/mastitis_model.joblib + metadata.json. Returns the model version."""
    version = f"{SLUGS[best_name]}-v{MODEL_VERSION}"
    bundle = {
        "model": fitted[best_name]["model"],
        "model_name": best_name,
        "threshold": fitted[best_name]["threshold"],
        "features": FEATURES,
        "reference": data["reference"],  # herd medians for cows without history
        "baseline": {
            "rule": "clean readings only: 7 d if >= 3, else 21 d if >= 3, else herd median",
            **data["baseline_info"],
        },
        "background": data["X_train"].sample(SHAP_BACKGROUND_SIZE, random_state=SEED),  # SHAP background
        "test_farms": data["test_farms"],
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "positive_definition": f"cmt_score >= {CMT_POSITIVE_FROM} (CMT >= 3 on the 1-5 field scale)",
        "data": "SIMULATED (data/generate_synthetic.py)",
    }
    joblib.dump(bundle, MODELS / "mastitis_model.joblib")
    meta = {key: value for key, value in bundle.items() if key not in ("model", "background")}
    meta["cv_results"] = cv.round(4).to_dict(orient="records")
    (MODELS / "metadata.json").write_text(json.dumps(meta, indent=2))
    return version


def main() -> None:
    """Train, compare and save (see the module docstring)."""
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    data = load_and_split(verbose=True)
    pos_weight = print_class_balance(data)

    candidates = make_candidates(pos_weight)
    cv = compare_candidates(candidates, data)
    cv.round(4).to_csv(REPORTS / "cv_results.csv", index=False)
    best_name = select_best(cv)

    fitted = fit_all(candidates, cv, data)
    joblib.dump(fitted, MODELS / "candidates.joblib")
    version = save_bundle(fitted, best_name, cv, data)
    print(f"Saved models/mastitis_model.joblib (version {version}) and models/candidates.joblib")


if __name__ == "__main__":
    main()
