"""
Evaluate all trained candidates on the 2 held-out farms and write reports/.

Outputs (reports/):
  test_comparison.csv / .md  - accuracy, precision, recall, F1, ROC-AUC, specificity
                               (each model at its own tuned threshold)
  cv_comparison.md           - cross-validation table from training
  confusion_matrix.png       - best model, held-out farms
  roc_curves.png             - all candidates, held-out farms
  shap_summary.png           - which features drive the best model
  metrics.json               - best model metrics (served by the API's /model-info)

Run:  python -m ml.evaluate   (after python -m ml.train)
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")                     # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from ml.explain import make_explainer, shap_values
from ml.train import load_and_split

MODELS = Path("models")
REPORTS = Path("reports")

# Validated categorical palette (fixed order) + recessive ink colours
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK_2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
})


def metrics_at(y, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
        "specificity": tn / (tn + fp),
        "threshold": thr,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#cde2fb", "#2a78d6", "#0d366b"]))
    labels = ["Healthy", "Mastitis"]
    ax.set_xticks([0, 1], [f"Predicted\n{l}" for l in labels])
    ax.set_yticks([0, 1], [f"Actual\n{l}" for l in labels])
    for i in range(2):
        for j in range(2):
            dark = cm[i, j] > cm.max() * 0.45
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=14,
                    color="#ffffff" if dark else INK)
    ax.set_title(title, fontsize=10, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(curves, path):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":", label="Chance")
    for i, (name, (fpr, tpr, auc)) in enumerate(curves.items()):
        # colour = model (fixed order); dashed = SMOTE, solid = class weights
        ax.plot(fpr, tpr, color=SERIES[i], lw=2, ls="--" if "SMOTE" in name else "-",
                label=f"{name} (AUC {auc:.3f})")
    ax.set_xlabel("False positive rate (healthy cows flagged)")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves - held-out farms (SIMULATED data)", loc="left")
    ax.grid(color="#e8e7e3", lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def to_markdown(df: pd.DataFrame) -> str:
    """Small markdown table writer (avoids the optional 'tabulate' dependency)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines) + "\n"


def main():
    REPORTS.mkdir(exist_ok=True)
    bundle = joblib.load(MODELS / "mastitis_model.joblib")
    candidates = joblib.load(MODELS / "candidates.joblib")
    d = load_and_split()
    X_te, y_te = d["X_test"], d["y_test"]
    assert d["test_farms"] == bundle["test_farms"], "test farms differ from training run"

    # --- Compare every candidate on the held-out farms ---
    rows, curves = [], {}
    for name, c in candidates.items():
        p = c["model"].predict_proba(X_te)[:, 1]
        m = metrics_at(y_te, p, c["threshold"])
        rows.append({"model": name, **{k: m[k] for k in
                     ["accuracy", "precision", "recall", "f1", "roc_auc", "specificity", "threshold"]},
                     "selected": "yes" if name == bundle["model_name"] else ""})
        fpr, tpr, _ = roc_curve(y_te, p)
        curves[name] = (fpr, tpr, m["roc_auc"])
    table = pd.DataFrame(rows)
    table.round(4).to_csv(REPORTS / "test_comparison.csv", index=False)
    header = (f"# Model comparison - held-out farms {', '.join(d['test_farms'])} (SIMULATED data)\n\n"
              f"Each model uses its own threshold tuned on training CV for recall >= 0.85.\n\n")
    (REPORTS / "test_comparison.md").write_text(header + to_markdown(table.round(3)))

    cv = pd.read_csv(REPORTS / "cv_results.csv")
    cv_cols = ["model", "cv_recall", "cv_precision", "cv_f1", "cv_roc_auc", "tuned_threshold"]
    (REPORTS / "cv_comparison.md").write_text(
        "# 5-fold GroupKFold CV on training farms (threshold 0.5)\n\n" + to_markdown(cv[cv_cols].round(3)))

    plot_roc(curves, REPORTS / "roc_curves.png")

    # --- Best model: confusion matrix + metrics.json ---
    best = bundle["model"]
    p_best = best.predict_proba(X_te)[:, 1]
    m_best = metrics_at(y_te, p_best, bundle["threshold"])
    cm = np.array([[m_best["tn"], m_best["fp"]], [m_best["fn"], m_best["tp"]]])
    plot_confusion(cm, f"{bundle['model_name']}\nthreshold {bundle['threshold']:.2f}, held-out farms",
                   REPORTS / "confusion_matrix.png")

    # Recall on true episodes per milking is also useful to report - per-row only here.
    metrics = {
        "model_name": bundle["model_name"], "version": bundle["version"],
        "threshold": bundle["threshold"], "test_farms": d["test_farms"],
        "n_test": int(len(y_te)), "n_test_positive": int(y_te.sum()),
        "test_metrics": {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in m_best.items()},
        "comparison": table.round(4).to_dict(orient="records"),
        "data": "SIMULATED",
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    # --- SHAP summary for the best model ---
    background = d["X_train"].sample(200, random_state=0)
    transform, explainer = make_explainer(best, background)
    sample = X_te.sample(min(500, len(X_te)), random_state=0)
    sv = shap_values(transform, explainer, sample)
    plt.figure()
    shap.summary_plot(sv, sample, show=False, max_display=12, plot_size=(7.5, 5.5))
    plt.title(f"What drives the mastitis risk score - {bundle['model_name']}", loc="left", fontsize=10)
    plt.tight_layout()
    plt.savefig(REPORTS / "shap_summary.png", dpi=150)
    plt.close("all")

    print(to_markdown(table.round(3)))
    t = m_best
    print(f"Best ({bundle['model_name']}) on held-out farms: recall={t['recall']:.3f} "
          f"precision={t['precision']:.3f} specificity={t['specificity']:.3f} ROC-AUC={t['roc_auc']:.3f}")
    print(f"  confusion: TP={t['tp']} FN={t['fn']} FP={t['fp']} TN={t['tn']}")
    print("Reports written to reports/")


if __name__ == "__main__":
    main()
