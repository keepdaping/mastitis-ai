"""
Train and compare models for subclinical mastitis (SCM) detection.

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

from ml.episodes import flags_for_frame
from ml.features import FEATURES, build_features, compute_reference

SEED = 42
DATA = Path("data/synthetic_milkings.csv")
MODELS = Path("models")
REPORTS = Path("reports")
N_TEST_FARMS = 2
CV_FOLDS = 5
MIN_PRECISION = 0.5        # selection constraint
TARGET_RECALL = 0.85       # threshold tuning target
POSITIVE_CMT = 2           # cmt_score 0-4; >= 2 means CMT >= 3 on the 1-5 field scale
MODEL_VERSION = 2          # bump when retraining -> model_version e.g. "rf-smote-v2" (v2: clean baseline)
SLUGS = {                  # short names used in model_version (returned by the API)
    "LogReg + class_weight": "lr-cw", "LogReg + SMOTE": "lr-smote",
    "RandomForest + class_weight": "rf-cw", "RandomForest + SMOTE": "rf-smote",
    "XGBoost + scale_pos_weight": "xgb-spw", "XGBoost + SMOTE": "xgb-smote",
}


def baseline_flags(df, X0, y, labelled, is_test):
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
    rf0 = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced",
                                 n_jobs=-1, random_state=SEED)
    score = pd.Series(np.nan, index=df.index)
    train_rows = df.index[~is_test]
    farms = df.loc[train_rows, "farm_id"]
    for fit_idx, score_idx in GroupKFold(n_splits=CV_FOLDS).split(train_rows, groups=farms):
        fit_farms, score_farms = set(farms.iloc[fit_idx]), set(farms.iloc[score_idx])
        fit = labelled & df["farm_id"].isin(fit_farms)
        m = clone(rf0).fit(X0[fit], y[fit])
        rows = df["farm_id"].isin(score_farms)
        score[rows] = m.predict_proba(X0[rows])[:, 1]
    m = clone(rf0).fit(X0[labelled & ~is_test], y[labelled & ~is_test])
    score[is_test] = m.predict_proba(X0[is_test])[:, 1]

    lab_train = labelled & ~is_test
    thr0, _, _ = tune_threshold(y[lab_train], score[lab_train])     # same rule as the final model
    flags = flags_for_frame(df, score >= thr0)
    return flags, {"stage0_threshold": round(thr0, 4), "share_excluded": round(float(flags.mean()), 4)}


def load_and_split(verbose: bool = False):
    """Load data, split farms into train/test, build features. Returns labelled rows only."""
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    gss = GroupShuffleSplit(n_splits=1, test_size=N_TEST_FARMS, random_state=SEED)
    farms = df["farm_id"].drop_duplicates().to_numpy()
    train_idx, test_idx = next(gss.split(farms, groups=farms))
    test_farms = sorted(farms[test_idx])

    is_test = df["farm_id"].isin(test_farms)
    reference = compute_reference(df[~is_test])        # no information from test farms
    labelled = df["cmt_score"].notna()
    y_all = (df["cmt_score"] >= POSITIVE_CMT).astype(int)

    # Stage 0 -> baseline exclusion flags -> final features with a CLEAN baseline
    X0 = build_features(df, reference)
    flags, info = baseline_flags(df, X0, y_all, labelled, is_test)
    X_all = build_features(df.assign(baseline_exclude=flags), reference)
    if verbose:
        infected = df["true_infected"] == 1
        print(f"Clean baseline: {info['share_excluded']:.1%} of readings excluded "
              f"(stage-0 threshold {info['stage0_threshold']:.3f}); "
              f"{flags[infected].mean():.1%} of truly infected readings and "
              f"{flags[~infected].mean():.1%} of healthy readings excluded")

    tr, te = labelled & ~is_test, labelled & is_test
    return {
        "X_train": X_all[tr], "y_train": y_all[tr], "g_train": df.loc[tr, "farm_id"],
        "X_test": X_all[te], "y_test": y_all[te], "meta_test": df.loc[te],
        "test_farms": test_farms, "reference": reference, "baseline_info": info,
    }


def make_candidates(pos_weight: float) -> dict:
    """3 models x 2 imbalance strategies. SMOTE lives inside an imblearn Pipeline,
    so synthetic samples are only created from the training folds (no leakage)."""
    lr = LogisticRegression(max_iter=2000)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED)
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1, random_state=SEED)
    smote = SMOTE(random_state=SEED)
    return {
        "LogReg + class_weight": Pipeline([("scale", StandardScaler()),
                                           ("model", clone(lr).set_params(class_weight="balanced"))]),
        "LogReg + SMOTE": Pipeline([("scale", StandardScaler()), ("smote", clone(smote)), ("model", clone(lr))]),
        "RandomForest + class_weight": Pipeline([("model", clone(rf).set_params(class_weight="balanced"))]),
        "RandomForest + SMOTE": Pipeline([("smote", clone(smote)), ("model", clone(rf))]),
        "XGBoost + scale_pos_weight": Pipeline([("model", clone(xgb).set_params(scale_pos_weight=pos_weight))]),
        "XGBoost + SMOTE": Pipeline([("smote", clone(smote)), ("model", clone(xgb))]),
    }


def cross_validate(model, X, y, groups):
    """GroupKFold by farm. Returns per-fold metrics (at 0.5) and out-of-fold probabilities."""
    oof = np.zeros(len(y))
    folds = []
    for tr, va in GroupKFold(n_splits=CV_FOLDS).split(X, y, groups):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        pred = (p >= 0.5).astype(int)
        folds.append({
            "recall": recall_score(y.iloc[va], pred),
            "precision": precision_score(y.iloc[va], pred, zero_division=0),
            "f1": f1_score(y.iloc[va], pred),
            "roc_auc": roc_auc_score(y.iloc[va], p),
        })
    return pd.DataFrame(folds), oof


def tune_threshold(y, proba, target_recall=TARGET_RECALL):
    """Highest threshold that still reaches the target recall (= best precision at that recall)."""
    precision, recall, thresholds = precision_recall_curve(y, proba)
    ok = np.where(recall[:-1] >= target_recall)[0]      # recall[i] belongs to thresholds[i]
    i = ok[-1] if len(ok) else 0
    return float(thresholds[i]), float(precision[i]), float(recall[i])


def main():
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    d = load_and_split(verbose=True)
    X, y, g = d["X_train"], d["y_train"], d["g_train"]

    # --- Class balance (check BEFORE choosing an imbalance strategy) ---
    print("Class balance (CMT-labelled milkings, positive = cmt_score >= 2):")
    for name, yy in [("train", y), ("test", d["y_test"])]:
        print(f"  {name:5s}: n={len(yy):5d}  positive={yy.sum():5d} ({yy.mean():.1%})  "
              f"negative={(1 - yy).sum():5d}")
    print(f"  test farms (held out): {d['test_farms']}")
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    print(f"  negative/positive ratio = {pos_weight:.2f} (used as XGBoost scale_pos_weight)\n")

    # --- Cross-validated comparison ---
    rows, oofs = [], {}
    candidates = make_candidates(pos_weight)
    for name, model in candidates.items():
        folds, oof = cross_validate(model, X, y, g)
        thr, p_at, r_at = tune_threshold(y, oof)
        oofs[name] = oof
        row = {"model": name}
        for k in ["recall", "precision", "f1", "roc_auc"]:
            row[f"cv_{k}"] = folds[k].mean()
            row[f"cv_{k}_sd"] = folds[k].std()
        row.update({"tuned_threshold": thr, "oof_precision_at_thr": p_at, "oof_recall_at_thr": r_at})
        rows.append(row)
        print(f"  {name:28s} recall={row['cv_recall']:.3f}+-{row['cv_recall_sd']:.3f}  "
              f"precision={row['cv_precision']:.3f}  AUC={row['cv_roc_auc']:.3f}  thr@R>={TARGET_RECALL}={thr:.3f}")
    cv = pd.DataFrame(rows)
    cv.round(4).to_csv(REPORTS / "cv_results.csv", index=False)

    # --- Model selection: highest recall subject to precision >= 0.5 (tie-break: ROC-AUC) ---
    eligible = cv[cv["cv_precision"] >= MIN_PRECISION]
    if eligible.empty:
        print("WARNING: no model reaches precision >= 0.5; selecting on recall alone")
        eligible = cv
    best_row = eligible.sort_values(["cv_recall", "cv_roc_auc"], ascending=False).iloc[0]
    best_name = best_row["model"]
    print(f"\nSelected: {best_name} (CV recall {best_row['cv_recall']:.3f}, "
          f"precision {best_row['cv_precision']:.3f}); threshold {best_row['tuned_threshold']:.3f}")

    # --- Refit every candidate on all training farms (evaluate.py compares them on the test farms) ---
    fitted = {}
    for name, model in candidates.items():
        thr = float(cv.loc[cv["model"] == name, "tuned_threshold"].iloc[0])
        fitted[name] = {"model": clone(model).fit(X, y), "threshold": thr}
    joblib.dump(fitted, MODELS / "candidates.joblib")

    version = f"{SLUGS[best_name]}-v{MODEL_VERSION}"
    bundle = {
        "model": fitted[best_name]["model"],
        "model_name": best_name,
        "threshold": fitted[best_name]["threshold"],
        "features": FEATURES,
        "reference": d["reference"],         # herd medians for cows without history
        "baseline": {"rule": "clean readings only: 7 d if >= 3, else 21 d if >= 3, else herd median",
                     **d["baseline_info"]},
        "background": X.sample(200, random_state=SEED),   # SHAP background sample (API explanations)
        "test_farms": d["test_farms"],
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "positive_definition": f"cmt_score >= {POSITIVE_CMT} (CMT >= 3 on the 1-5 field scale)",
        "data": "SIMULATED (data/generate_synthetic.py)",
    }
    joblib.dump(bundle, MODELS / "mastitis_model.joblib")
    meta = {k: v for k, v in bundle.items() if k not in ("model", "background")}
    meta["cv_results"] = cv.round(4).to_dict(orient="records")
    (MODELS / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved models/mastitis_model.joblib (version {version}) and models/candidates.joblib")


if __name__ == "__main__":
    main()
