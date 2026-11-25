import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42
COST_FN = 8000.0  # Approve default
COST_FP = 600.0   # Reject good applicant
THEORETICAL_THRESHOLD = round(COST_FP / (COST_FN + COST_FP), 3)  # ~0.070


def build_payment_features(apps_df: pd.DataFrame, payments_df: pd.DataFrame) -> pd.DataFrame:
    m = apps_df[["app_id", "customer_id", "app_date_dt"]].merge(
        payments_df[["customer_id", "pay_date_dt", "amount", "days_late"]],
        on="customer_id",
        how="left",
    )
    prior = m[m["pay_date_dt"] < m["app_date_dt"]].copy()

    if len(prior) == 0:
        out = pd.DataFrame({"app_id": apps_df["app_id"]})
        for col in [
            "prev_pay_count", "prev_pay_total_amt", "prev_pay_mean_amt", "prev_pay_max_amt",
            "prev_days_late_mean", "prev_days_late_max", "prev_days_late_std",
            "prev_late_pay_count", "prev_late_pay_rate", "prev_late15_count", "prev_late30_count",
            "prev_days_since_last_pay", "prev_days_since_first_pay",
            "recent_180d_count", "recent_180d_late_count", "recent_180d_late_rate",
        ]:
            out[col] = 0.0
        return out

    prior["is_late"] = (prior["days_late"] > 0).astype(float)
    prior["is_late_15"] = (prior["days_late"] >= 15).astype(float)
    prior["is_late_30"] = (prior["days_late"] >= 30).astype(float)
    prior["days_before_app"] = (prior["app_date_dt"] - prior["pay_date_dt"]).dt.days.astype(float)
    prior["recent_180d"] = (prior["days_before_app"] <= 180).astype(float)
    prior["recent_late_180d"] = prior["is_late"] * prior["recent_180d"]

    agg = (
        prior.groupby("app_id")
        .agg(
            prev_pay_count=("amount", "count"),
            prev_pay_total_amt=("amount", "sum"),
            prev_pay_mean_amt=("amount", "mean"),
            prev_pay_max_amt=("amount", "max"),
            prev_days_late_mean=("days_late", "mean"),
            prev_days_late_max=("days_late", "max"),
            prev_days_late_std=("days_late", "std"),
            prev_late_pay_count=("is_late", "sum"),
            prev_late_pay_rate=("is_late", "mean"),
            prev_late15_count=("is_late_15", "sum"),
            prev_late30_count=("is_late_30", "sum"),
            prev_days_since_last_pay=("days_before_app", "min"),
            prev_days_since_first_pay=("days_before_app", "max"),
            recent_180d_count=("recent_180d", "sum"),
            recent_180d_late_count=("recent_late_180d", "sum"),
        )
        .reset_index()
    )

    out = apps_df[["app_id"]].merge(agg, on="app_id", how="left")
    for col in [
        "prev_pay_count", "prev_pay_total_amt", "prev_late_pay_count", "prev_late_pay_rate",
        "prev_late15_count", "prev_late30_count", "recent_180d_count", "recent_180d_late_count",
    ]:
        out[col] = out[col].fillna(0.0)
    out["recent_180d_late_rate"] = np.where(
        out["recent_180d_count"] > 0,
        out["recent_180d_late_count"] / out["recent_180d_count"],
        0.0,
    )
    return out


def add_domain_features(df: pd.DataFrame, pay_feats: pd.DataFrame) -> pd.DataFrame:
    d = df.merge(pay_feats, on="app_id", how="left")
    safe_income = d["income"].replace(0, np.nan)
    safe_term = d["term_months"].replace(0, np.nan)

    d["debt_to_income"] = d["existing_debt"] / safe_income
    d["loan_to_income"] = d["loan_amount"] / safe_income
    d["est_monthly_installment"] = d["loan_amount"] / safe_term
    d["payment_to_income"] = d["est_monthly_installment"] / safe_income
    d["total_debt_after_loan"] = d["existing_debt"] + d["loan_amount"]
    d["total_debt_to_income"] = d["total_debt_after_loan"] / safe_income
    d["credit_x_dti"] = d["credit_score"] * d["debt_to_income"]
    d["credit_x_late_rate"] = d["credit_score"] * d["prev_late_pay_rate"]
    d["has_prev_payments"] = (d["prev_pay_count"] > 0).astype(int)
    return d


def calc_cost(y_true: np.ndarray, y_pred_reject: np.ndarray) -> tuple[float, int, int]:
    fn = int(np.sum((y_true == 1) & (y_pred_reject == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred_reject == 1)))
    return fn * COST_FN + fp * COST_FP, fn, fp


def main() -> None:
    np.random.seed(RANDOM_SEED)

    train = pd.read_csv("applications_train.csv").drop_duplicates().copy()
    new = pd.read_csv("applications_new.csv").copy()
    pay = pd.read_csv("payments_history.csv").copy()

    train["app_date_dt"] = pd.to_datetime(train["app_date"])
    new["app_date_dt"] = pd.to_datetime(new["app_date"])
    pay["pay_date_dt"] = pd.to_datetime(pay["pay_date"])

    train_feat = add_domain_features(train, build_payment_features(train, pay))
    new_feat = add_domain_features(new, build_payment_features(new, pay))

    val_cutoff = pd.Timestamp("2025-03-01")
    tr_sub = train_feat[train_feat["app_date_dt"] < val_cutoff].copy()
    va_sub = train_feat[train_feat["app_date_dt"] >= val_cutoff].copy()

    drop_cols = ["app_id", "customer_id", "app_date", "app_date_dt", "default_12m", "collections_calls"]
    feature_cols = [c for c in tr_sub.columns if c not in drop_cols]
    cat_cols = ["purpose", "region"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    pre_linear = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    pre_tree = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    candidates = {
        "LogisticRegression (C=1.0)": Pipeline([
            ("pre", pre_linear),
            ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_SEED)),
        ]),
        "LogisticRegression (C=0.25)": Pipeline([
            ("pre", pre_linear),
            ("clf", LogisticRegression(C=0.25, max_iter=2000, random_state=RANDOM_SEED)),
        ]),
        "RandomForest (depth=8)": Pipeline([
            ("pre", pre_tree),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20, random_state=RANDOM_SEED, n_jobs=-1)),
        ]),
        "GradientBoosting (lr=0.05)": Pipeline([
            ("pre", pre_tree),
            ("clf", GradientBoostingClassifier(learning_rate=0.05, max_depth=3, n_estimators=200, subsample=0.8, random_state=RANDOM_SEED)),
        ]),
        "HistGradientBoosting": Pipeline([
            ("pre", pre_tree),
            ("clf", HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=1.0, random_state=RANDOM_SEED)),
        ]),
    }

    X_tr, y_tr = tr_sub[feature_cols], tr_sub["default_12m"].values
    X_va, y_va = va_sub[feature_cols], va_sub["default_12m"].values

    best_model_name = None
    best_model_pipe = None
    best_val_cost = np.inf
    probs_va = {}

    for name, pipe in candidates.items():
        pipe.fit(X_tr, y_tr)
        p_va = pipe.predict_proba(X_va)[:, 1]
        probs_va[name] = p_va

        auc = roc_auc_score(y_va, p_va)
        prauc = average_precision_score(y_va, p_va)
        c_theo, fn_t, fp_t = calc_cost(y_va, (p_va >= THEORETICAL_THRESHOLD).astype(int))
        print(
            f"{name:30s} | ROC-AUC: {auc:.4f} | PR-AUC: {prauc:.4f} | "
            f"Cost@{THEORETICAL_THRESHOLD:.3f}: ${c_theo:,.0f} (FN={fn_t}, FP={fp_t})"
        )
        if c_theo < best_val_cost:
            best_val_cost = c_theo
            best_model_name = name
            best_model_pipe = pipe

    chosen_thresh = THEORETICAL_THRESHOLD
    print(f"\nSelected model: {best_model_name} at cost-ratio threshold {chosen_thresh:.3f}")

    best_model_pipe.fit(train_feat[feature_cols], train_feat["default_12m"].values)
    new_probs = best_model_pipe.predict_proba(new_feat[feature_cols])[:, 1]
    new_decisions = np.where(new_probs >= chosen_thresh, "reject", "approve")

    preds = pd.DataFrame({
        "app_id": new_feat["app_id"],
        "default_probability": np.round(new_probs, 6),
        "decision": new_decisions,
    })
    preds.to_csv("predictions.csv", index=False)

    approve_all_cost = calc_cost(y_va, np.zeros_like(y_va))[0]
    p_va_best = probs_va[best_model_name]
    ts = np.linspace(0.01, 0.50, 491)
    costs_curve = [calc_cost(y_va, (p_va_best >= t).astype(int))[0] / 1000.0 for t in ts]

    plt.figure(figsize=(8, 5))
    plt.plot(ts, costs_curve, label="Validation Cost ($k)", linewidth=2)
    plt.axhline(approve_all_cost / 1000.0, color="red", linestyle="--", label=f"Approve All (${approve_all_cost/1000:.0f}k)")
    plt.axvline(chosen_thresh, color="green", linestyle=":", label=f"Cost-Ratio Cutoff ({chosen_thresh:.3f})")
    plt.title("Validation Expected Cost by Probability Threshold")
    plt.xlabel("Default Probability Cutoff (Reject if >= threshold)")
    plt.ylabel("Total Expected Cost ($ Thousands)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("cost_vs_threshold.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, p_arr in probs_va.items():
        fpr, tpr, _ = roc_curve(y_va, p_arr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_va, p_arr):.3f})", linewidth=1.8)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Baseline")
    plt.title("ROC Curves on Temporal Validation Set (Mar–Jun 2025)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=8.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curves_validation.png", dpi=150)
    plt.close()

    ohe_names = list(best_model_pipe.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(cat_cols))
    all_feat_names = num_cols + ohe_names
    coefs = best_model_pipe.named_steps["clf"].coef_[0]
    coef_df = pd.DataFrame({"feature": all_feat_names, "coef": coefs, "abs_coef": np.abs(coefs)}).sort_values("abs_coef", ascending=False).head(12)

    plt.figure(figsize=(8, 5.5))
    colors = ["#c0392b" if c > 0 else "#2980b9" for c in coef_df["coef"][::-1]]
    plt.barh(coef_df["feature"][::-1], coef_df["coef"][::-1], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Top Standardized Coefficients (Logistic Regression)")
    plt.xlabel("Standardized Log-Odds Coefficient (+ increases default risk)")
    plt.tight_layout()
    plt.savefig("feature_importances.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(new_probs, bins=40, color="#34495e", alpha=0.85, edgecolor="white")
    plt.axvline(chosen_thresh, color="#e74c3c", linestyle="--", linewidth=2, label=f"Cutoff = {chosen_thresh:.3f}")
    plt.title("Predicted Default Probability Distribution (New Applications, n=1,800)")
    plt.xlabel("Predicted Default Probability (12m)")
    plt.ylabel("Number of Applications")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("new_applications_score_dist.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()