# 5-fold GroupKFold CV on training farms (threshold 0.5)

| model | cv_recall | cv_precision | cv_f1 | cv_roc_auc | tuned_threshold |
|---|---|---|---|---|---|
| LogReg + class_weight | 0.822 | 0.858 | 0.839 | 0.914 | 0.289 |
| LogReg + SMOTE | 0.821 | 0.850 | 0.835 | 0.914 | 0.277 |
| RandomForest + class_weight | 0.824 | 0.880 | 0.850 | 0.913 | 0.187 |
| RandomForest + SMOTE | 0.838 | 0.765 | 0.770 | 0.915 | 0.416 |
| XGBoost + scale_pos_weight | 0.813 | 0.902 | 0.854 | 0.910 | 0.117 |
| XGBoost + SMOTE | 0.825 | 0.767 | 0.756 | 0.911 | 0.192 |
