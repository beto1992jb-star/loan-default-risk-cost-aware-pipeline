# Executive Summary: 12-Month Loan Default Risk & Decision Policy

## Business Problem & Solution
We built a reproducible credit risk scoring pipeline (`train_and_score.py`) to score `1,800` new loan applications (`applications_new.csv`) using `6,000` deduplicated historical applications (`applications_train.csv`, `8.27%` default rate) and point-in-time payment records (`payments_history.csv`). Because defaults carry an **`$8,000`** loss per approved default versus **`$600`** per rejected good borrower, raw accuracy is misleading and underwriting decisions should reflect asymmetric business costs.

## Validation Performance & Threshold Sensitivity
Training prior to `2025-03-01` (`4,631` applications) and validating from `2025-03-01` to `2025-06-30` (`1,369` applications), **Logistic Regression (`C=0.25`)** was selected as the primary model (`0.8795` ROC-AUC, `0.5057` PR-AUC, and `$345,800` validation cost with `FN = 22` and `FP = 283`) while avoiding calendar extrapolation (`app_year`/`app_month` excluded) and post-outcome leakage (`collections_calls` dropped).

Rather than fitting to a noisy minimum on the empirical cost curve, we apply the theoretical cost-ratio cutoff $600 / (8,000 + 600) \approx \mathbf{0.070}$ (reject if `default_probability >= 0.070`, else approve). In validation, `Approve All` costs **`$976,000`** and `Reject All` costs **`$748,200`**; at threshold `0.070` with `Logistic Regression (C=0.25)`, validation cost is **`$345,800`**, saving **`$402,400` (`53.8%`)** versus `Reject All` and **`$630,200` (`64.6%`)** versus `Approve All`.

| Threshold (`C=0.25`) | Validation Cost | Validation Reject Rate | New Cohort Reject Rate (`n=1,800`) | New Cohort Approve Rate (`n=1,800`) |
|---:|---:|---:|---:|---:|
| `0.050` | `$352,000` | `34.0%` | `34.5%` | `65.5%` |
| **`0.070` (Selected)** | **`$345,800`** | **`28.0%`** | **`28.4%` (`511`)** | **`71.6%` (`1,289`)** |
| `0.100` | `$370,400` | `21.5%` | `22.7%` | `77.3%` |

## Operational Impact, Limitations & Retraining Cadence
Applying the `0.070` threshold on the `1,800` new applications (`predictions.csv`) rejects `511` applications (`28.4%`) and approves `1,289` (`71.6%`), a noticeable underwriting shift that should be reviewed with credit operations alongside manual review for borderline cases. New applications cover a later period (`Jul–Dec 2025`, mean predicted probability `0.0791`) with unobserved `default_12m` outcomes and potential macro or cohort drift; we recommend **quarterly retraining** as 12-month default labels mature and ongoing tracking of approval rates, score distributions, and realized losses.