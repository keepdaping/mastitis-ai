# Model comparison - held-out farms F02, F09 (SIMULATED data)

Each model uses its own threshold tuned on training CV for recall >= 0.85.

| model | accuracy | precision | recall | f1 | roc_auc | specificity | threshold | selected |
|---|---|---|---|---|---|---|---|---|
| LogReg + class_weight | 0.772 | 0.481 | 0.927 | 0.633 | 0.948 | 0.730 | 0.274 |  |
| LogReg + SMOTE | 0.758 | 0.465 | 0.924 | 0.618 | 0.948 | 0.713 | 0.268 |  |
| RandomForest + class_weight | 0.855 | 0.608 | 0.902 | 0.726 | 0.940 | 0.843 | 0.205 |  |
| RandomForest + SMOTE | 0.927 | 0.793 | 0.887 | 0.837 | 0.946 | 0.937 | 0.386 | yes |
| XGBoost + scale_pos_weight | 0.800 | 0.517 | 0.890 | 0.654 | 0.932 | 0.775 | 0.138 |  |
| XGBoost + SMOTE | 0.921 | 0.775 | 0.884 | 0.826 | 0.940 | 0.931 | 0.157 |  |
