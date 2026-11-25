# Loan Default Risk Scoring & Cost-Aware Decision Pipeline

End-to-end Python pipeline to predict 12-month loan default probabilities (`default_12m`) and make cost-aware `approve` / `reject` decisions under asymmetric business costs (`$8,000` per approved default vs. `$600` per rejected good applicant).

## How to Run & Environment
Place `applications_train.csv`, `applications_new.csv`, and `payments_history.csv` in the working directory and run:

```bash
python train_and_score.py
```

Check and record your actual runtime versions in your environment (for example by printing `pd.__version__`, `np.__version__`, and `sklearn.__version__` in your run log so they match your exact execution environment).

## Deliverables
- `train_and_score.py`: End-to-end cleaning, point-in-time payment feature engineering, temporal validation, model comparison, cost-ratio threshold application (`0.070`), full-train refit, scoring on `applications_new.csv`, and chart generation.
- `predictions.csv`: Scored decisions for all `1,800` new applications (`app_id`, `default_probability`, `decision` — `1,289` approve, `511` reject, `28.4%` reject rate, mean probability `0.0791`).
- `cost_vs_threshold.png`, `roc_curves_validation.png`, `feature_importances.png`, `new_applications_score_dist.png`: Diagnostic plots.
- `report.md`: Executive summary in English.

## Data Hygiene & Leakage Prevention
- **Deduplication:** Removed `40` exact duplicate rows in `applications_train.csv` (`6,040` -> `6,000` unique applications; `8.27%` overall default rate).
- **Post-outcome column excluded:** Dropped `collections_calls` (absent in `applications_new.csv` and reflects post-origination collections activity).
- **No calendar extrapolation features:** Excluded `app_year` and `app_month` so the model does not extrapolate fixed calendar periods into future windows.
- **Strict point-in-time payment features (`pay_date < app_date`):** Built historical and 180-day recency delinquency features (`prev_late_pay_rate`, `prev_late_pay_count`, `prev_late15_count`, `prev_late30_count`, `recent_180d_late_rate`, `prev_days_late_mean`, `prev_days_late_max`) strictly prior to each application date.
- **Missing values:** Missing values in `income` and `credit_score` are imputed using training medians inside the scikit-learn pipeline after splitting.

## Temporal Validation Summary (`2025-03-01` Cutoff)
Evaluated on `4,631` training rows (`< 2025-03-01`) and `1,369` validation rows (`>= 2025-03-01`) using the theoretical cost-ratio threshold `600 / (8,000 + 600) ≈ 0.070`:
- **LogisticRegression (`C=0.25`) [Selected]:** `ROC-AUC = 0.8795`, `PR-AUC = 0.5057`, validation cost = `$345,800` (`FN = 22`, `FP = 283`).
- **LogisticRegression (`C=1.0`):** `ROC-AUC = 0.8796`, `PR-AUC = 0.5023`, validation cost = `$352,000` (`FN = 23`, `FP = 280`).
- **GradientBoosting (`lr=0.05`):** `ROC-AUC = 0.8623`, `PR-AUC = 0.4412`, validation cost = `$363,000`.
- **HistGradientBoosting:** `ROC-AUC = 0.8570`, `PR-AUC = 0.4453`, validation cost = `$368,200`.
- **RandomForest (`depth=8`):** `ROC-AUC = 0.8586`, `PR-AUC = 0.4515`, validation cost = `$383,400`.

## Limitations & Monitoring
- `applications_train.csv` spans `2024-01-01` to `2025-06-30` and `applications_new.csv` spans `2025-07-01` to `2025-12-31`. Mean predicted default probability on the new set is `0.0791` vs. `0.0827` historical default rate; any macro shift or portfolio drift requires periodic monitoring and quarterly retraining once realized 12-month default labels mature.