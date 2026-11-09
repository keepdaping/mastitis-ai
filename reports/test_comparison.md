# Model comparison - held-out farms F02, F09 (SIMULATED data)

Each model uses its own threshold tuned on training CV for recall >= 0.85.

| model | accuracy | precision | recall | f1 | roc_auc | specificity | threshold | selected |
|---|---|---|---|---|---|---|---|---|
| LogReg + class_weight | 0.827 | 0.556 | 0.924 | 0.694 | 0.947 | 0.801 | 0.289 |  |
| LogReg + SMOTE | 0.809 | 0.529 | 0.927 | 0.673 | 0.947 | 0.777 | 0.277 |  |
| RandomForest + class_weight | 0.843 | 0.584 | 0.902 | 0.709 | 0.936 | 0.826 | 0.187 |  |
| RandomForest + SMOTE | 0.945 | 0.865 | 0.878 | 0.871 | 0.946 | 0.963 | 0.416 | yes |
| XGBoost + scale_pos_weight | 0.756 | 0.462 | 0.902 | 0.611 | 0.927 | 0.716 | 0.117 |  |
| XGBoost + SMOTE | 0.931 | 0.811 | 0.878 | 0.843 | 0.938 | 0.945 | 0.192 |  |
