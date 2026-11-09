# Clean baseline: model v1 vs v2 (SIMULATED data)

Selected model RandomForest + SMOTE, held-out test farms F02 + F09 (1,543 CMT-labelled milkings, 328 positive).
v1 = baseline from all of the cow's readings in the last 7 days. v2 = baseline from clean readings only
(not high risk, not inside an alert episode; 7 days, else 21 days, else herd median).

| Metric | v1 `rf-smote-v1` | v2 `rf-smote-v2` | Change |
|---|---|---|---|
| Threshold | 0.386 | 0.416 | +0.030 |
| Recall | 0.887 | 0.878 | -0.009 (3 more missed of 328) |
| Precision | 0.793 | 0.865 | +0.072 |
| Specificity | 0.937 | 0.963 | +0.026 |
| F1 | 0.837 | 0.871 | +0.034 |
| ROC-AUC | 0.946 | 0.946 | 0.000 |
| False alarms (FP) | 76 | 45 | -31 (-41%) |
| Missed (FN) | 37 | 40 | +3 |

**Why fewer false alarms:** with the v1 baseline an infection quickly became the cow's "normal", so `ec_vs_baseline` was nearly useless on infected milkings (median +1.4%, AUC 0.68 on its own); with the clean baseline it stays raised (median +11.6%, AUC 0.89).
The model therefore gets a second, independent signal that must agree with the quarter-conductivity difference, so a single odd reading on a healthy cow is less often enough to cross the threshold.

_One train/test split on simulated data; differences of about ±0.01 are within noise. The v1 figures come from the
evaluation run before the change (the v1 model file was replaced by v2)._
