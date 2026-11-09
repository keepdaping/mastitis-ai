"""
Evaluate all trained candidates on the 2 held-out farms and write reports/ (ml/ layer).

Outputs (reports/):
  test_comparison.csv / .md  - accuracy, precision, recall, F1, ROC-AUC, specificity
                               (each model at its own tuned threshold)
  cv_comparison.md           - cross-validation table from training
  confusion_matrix.png       - best model, held-out farms
  roc_curves.png             - all candidates, held-out farms
  shap_summary.png           - which features drive the best model
  metrics.json               - best model metrics (served by GET /model-info and the /model page)

Run:  python -m ml.evaluate   (after python -m ml.train)
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ml.explain import make_explainer, shap_values
from ml.train import load_and_split

MODELS = Path("models")
REPORTS = Path("reports")
DPI = 150
SHAP_BACKGROUND_SIZE = 200
SHAP_SAMPLE_SIZE = 500
SHAP_MAX_DISPLAY = 12
SHAP_RANDOM_STATE = 0
TABLE_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc", "specificity", "threshold"]

# Validated categorical palette (fixed order) + recessive ink colours
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK_2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
CONFUSION_COLOURS = ["#cde2fb", "#2a78d6", "#0d366b"]
DARK_CELL_SHARE = 0.45  # cells above this share of the maximum get white text
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
})  # fmt: skip


def metrics_at(labels: pd.Series, proba: np.ndarray, threshold: float) -> dict:
    """Classification metrics at a decision threshold (+ ROC-AUC and the confusion counts)."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(labels, pred),
        "precision": precision_score(labels, pred, zero_division=0),
        "recall": recall_score(labels, pred),
        "f1": f1_score(labels, pred),
        "roc_auc": roc_auc_score(labels, proba),
        "specificity": tn / (tn + fp),
        "threshold": threshold,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def plot_confusion(matrix: np.ndarray, title: str, path: Path) -> None:
    """Confusion matrix in the sequential blue palette, counts written in each cell."""
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(matrix, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("blue", CONFUSION_COLOURS))
    labels = ["Healthy", "Mastitis"]
    ax.set_xticks([0, 1], [f"Predicted\n{label}" for label in labels])
    ax.set_yticks([0, 1], [f"Actual\n{label}" for label in labels])
    for row in range(2):
        for col in range(2):
            dark = matrix[row, col] > matrix.max() * DARK_CELL_SHARE
            ax.text(
                col,
                row,
                f"{matrix[row, col]:,}",
                ha="center",
                va="center",
                fontsize=14,
                color="#ffffff" if dark else INK,
            )
    ax.set_title(title, fontsize=10, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_roc(curves: dict, path: Path) -> None:
    """ROC curves of all candidates: colour = model (fixed order), dashed = SMOTE."""
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":", label="Chance")
    for index, (name, (fpr, tpr, auc)) in enumerate(curves.items()):
        ax.plot(
            fpr, tpr, color=SERIES[index], lw=2, ls="--" if "SMOTE" in name else "-", label=f"{name} (AUC {auc:.3f})"
        )
    ax.set_xlabel("False positive rate (healthy cows flagged)")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves - held-out farms (SIMULATED data)", loc="left")
    ax.grid(color="#e8e7e3", lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def to_markdown(df: pd.DataFrame) -> str:
    """Small markdown table writer (avoids the optional 'tabulate' dependency)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append(
            "| " + " | ".join(f"{value:.3f}" if isinstance(value, float) else str(value) for value in row) + " |"
        )
    return "\n".join(lines) + "\n"


def compare_on_test(candidates: dict, best_name: str, data: dict) -> tuple[pd.DataFrame, dict]:
    """Every candidate on the held-out farms at its own threshold. Returns (table, ROC curves)."""
    rows, curves = [], {}
    for name, candidate in candidates.items():
        proba = candidate["model"].predict_proba(data["X_test"])[:, 1]
        metrics = metrics_at(data["y_test"], proba, candidate["threshold"])
        rows.append(
            {
                "model": name,
                **{key: metrics[key] for key in TABLE_COLUMNS},
                "selected": "yes" if name == best_name else "",
            }
        )
        fpr, tpr, _ = roc_curve(data["y_test"], proba)
        curves[name] = (fpr, tpr, metrics["roc_auc"])
    return pd.DataFrame(rows), curves


def write_tables(table: pd.DataFrame, test_farms: list[str]) -> None:
    """test_comparison.csv/.md and cv_comparison.md."""
    table.round(4).to_csv(REPORTS / "test_comparison.csv", index=False)
    header = (
        f"# Model comparison - held-out farms {', '.join(test_farms)} (SIMULATED data)\n\n"
        f"Each model uses its own threshold tuned on training CV for recall >= 0.85.\n\n"
    )
    (REPORTS / "test_comparison.md").write_text(header + to_markdown(table.round(3)))

    cv = pd.read_csv(REPORTS / "cv_results.csv")
    cv_cols = ["model", "cv_recall", "cv_precision", "cv_f1", "cv_roc_auc", "tuned_threshold"]
    (REPORTS / "cv_comparison.md").write_text(
        "# 5-fold GroupKFold CV on training farms (threshold 0.5)\n\n" + to_markdown(cv[cv_cols].round(3))
    )


def evaluate_best(bundle: dict, table: pd.DataFrame, data: dict) -> dict:
    """Confusion matrix plot + metrics.json for the selected model. Returns its metrics."""
    proba = bundle["model"].predict_proba(data["X_test"])[:, 1]
    best = metrics_at(data["y_test"], proba, bundle["threshold"])
    matrix = np.array([[best["tn"], best["fp"]], [best["fn"], best["tp"]]])
    plot_confusion(
        matrix,
        f"{bundle['model_name']}\nthreshold {bundle['threshold']:.2f}, held-out farms",
        REPORTS / "confusion_matrix.png",
    )
    metrics = {
        "model_name": bundle["model_name"],
        "version": bundle["version"],
        "threshold": bundle["threshold"],
        "test_farms": data["test_farms"],
        "n_test": len(data["y_test"]),
        "n_test_positive": int(data["y_test"].sum()),
        "test_metrics": {
            key: (round(float(value), 4) if isinstance(value, float) else value) for key, value in best.items()
        },
        "comparison": table.round(4).to_dict(orient="records"),
        "data": "SIMULATED",
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    return best


def plot_shap_summary(bundle: dict, data: dict) -> None:
    """SHAP summary plot: which features drive the selected model."""
    background = data["X_train"].sample(SHAP_BACKGROUND_SIZE, random_state=SHAP_RANDOM_STATE)
    transform, explainer = make_explainer(bundle["model"], background)
    test = data["X_test"]
    sample = test.sample(min(SHAP_SAMPLE_SIZE, len(test)), random_state=SHAP_RANDOM_STATE)
    values = shap_values(transform, explainer, sample)
    plt.figure()
    shap.summary_plot(values, sample, show=False, max_display=SHAP_MAX_DISPLAY, plot_size=(7.5, 5.5))
    plt.title(f"What drives the mastitis risk score - {bundle['model_name']}", loc="left", fontsize=10)
    plt.tight_layout()
    plt.savefig(REPORTS / "shap_summary.png", dpi=DPI)
    plt.close("all")


def main() -> None:
    """Evaluate on the held-out farms and write every report (see the module docstring)."""
    REPORTS.mkdir(exist_ok=True)
    bundle = joblib.load(MODELS / "mastitis_model.joblib")
    candidates = joblib.load(MODELS / "candidates.joblib")
    data = load_and_split()
    assert data["test_farms"] == bundle["test_farms"], "test farms differ from training run"

    table, curves = compare_on_test(candidates, bundle["model_name"], data)
    write_tables(table, data["test_farms"])
    plot_roc(curves, REPORTS / "roc_curves.png")
    best = evaluate_best(bundle, table, data)
    plot_shap_summary(bundle, data)

    print(to_markdown(table.round(3)))
    print(
        f"Best ({bundle['model_name']}) on held-out farms: recall={best['recall']:.3f} "
        f"precision={best['precision']:.3f} specificity={best['specificity']:.3f} ROC-AUC={best['roc_auc']:.3f}"
    )
    print(f"  confusion: TP={best['tp']} FN={best['fn']} FP={best['fp']} TN={best['tn']}")
    print("Reports written to reports/")


if __name__ == "__main__":
    main()
