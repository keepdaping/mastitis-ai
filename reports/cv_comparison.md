# 5-fold GroupKFold CV on training farms (threshold 0.5)

| model | cv_recall | cv_precision | cv_f1 | cv_roc_auc | tuned_threshold |
|---|---|---|---|---|---|
| LogReg + class_weight | 0.817 | 0.835 | 0.824 | 0.910 | 0.274 |
| LogReg + SMOTE | 0.814 | 0.835 | 0.823 | 0.911 | 0.268 |
| RandomForest + class_weight | 0.820 | 0.871 | 0.844 | 0.916 | 0.205 |
| RandomForest + SMOTE | 0.830 | 0.790 | 0.799 | 0.917 | 0.386 |
| XGBoost + scale_pos_weight | 0.804 | 0.891 | 0.844 | 0.906 | 0.138 |
| XGBoost + SMOTE | 0.825 | 0.778 | 0.762 | 0.911 | 0.157 |
