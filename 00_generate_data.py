"""
Generates the CSVs that the numbered problem files read from. Run this
once (`uv run 00_generate_data.py`) before running any other file.

This file is infrastructure, not part of the interview walkthroughs --
in a real interview you'd be handed a CSV, not asked to generate one.
"""

import os

import numpy as np
import pandas as pd


def make_housing_data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    sqft = rng.normal(1800, 600, n).clip(min=300)
    bedrooms = rng.integers(1, 6, n)
    age = rng.integers(0, 80, n)
    neighborhood = rng.choice(["A", "B", "C", "D"], n, p=[0.4, 0.3, 0.2, 0.1])
    premium = {"A": 0, "B": 20_000, "C": 50_000, "D": -15_000}
    price = (
        50_000 + sqft * 120 + bedrooms * 8_000 - age * 300
        + np.array([premium[x] for x in neighborhood])
        + rng.normal(0, 25_000, n)
    )
    df = pd.DataFrame({"sqft": sqft, "bedrooms": bedrooms, "age": age,
                        "neighborhood": neighborhood, "price": price})
    missing_idx = rng.choice(n, size=int(0.03 * n), replace=False)
    df.loc[missing_idx, "sqft"] = np.nan
    outlier_idx = rng.choice(n, size=5, replace=False)
    df.loc[outlier_idx, "price"] *= 6
    return df


def make_churn_data(n=5000, seed=1):
    rng = np.random.default_rng(seed)
    tenure_months = rng.integers(1, 60, n)
    monthly_usage = rng.normal(50, 20, n).clip(min=0)
    support_tickets = rng.poisson(1.0, n)
    plan = rng.choice(["basic", "pro", "enterprise"], n, p=[0.5, 0.35, 0.15])
    logit = (-2.2 - 0.03 * tenure_months - 0.02 * monthly_usage
             + 0.5 * support_tickets + np.where(plan == "basic", 0.4, 0.0))
    prob = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, prob)
    df = pd.DataFrame({"tenure_months": tenure_months, "monthly_usage": monthly_usage,
                        "support_tickets": support_tickets, "plan": plan, "churn": churn})
    missing_idx = rng.choice(n, size=int(0.02 * n), replace=False)
    df.loc[missing_idx, "monthly_usage"] = np.nan
    return df


def make_experiment_data(n=2000, n_stores=40, seed=2):
    """Randomized experiment: treatment is a discount email, outcome is
    purchase_amount. Errors are heteroskedastic (noisier for high
    baseline spenders) and correlated within store, so HC and cluster
    robust SEs give a different (correct) answer than classical OLS SEs."""
    rng = np.random.default_rng(seed)
    store_id = rng.integers(0, n_stores, n)
    store_shock = rng.normal(0, 5, n_stores)[store_id]
    treatment = rng.binomial(1, 0.5, n)
    baseline_spend = rng.normal(100, 30, n).clip(min=0)
    true_ate = 15.0
    noise = rng.normal(0, 10 + 0.3 * baseline_spend, n)
    purchase_amount = 50 + true_ate * treatment + 0.5 * baseline_spend + store_shock + noise
    df = pd.DataFrame({"customer_id": np.arange(n), "store_id": store_id,
                        "treatment": treatment, "baseline_spend": baseline_spend,
                        "purchase_amount": purchase_amount})

    # a few missing baseline_spend values and duplicate rows, like a real
    # experiment export from a CRM would have
    missing_idx = rng.choice(n, size=int(0.01 * n), replace=False)
    df.loc[missing_idx, "baseline_spend"] = np.nan
    dupes = df.sample(10, random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)

    return df


def make_panel_data(n_units=400, seed=3):
    """2-period panel for diff-in-differences. Both groups share the same
    underlying trend (parallel trends holds by construction); only the
    treated group gets a bump, and only in the post period."""
    rng = np.random.default_rng(seed)
    unit_id = np.arange(n_units)
    treated_group = rng.binomial(1, 0.5, n_units)
    unit_fe = rng.normal(0, 5, n_units)
    true_effect = 8.0
    rows = []
    for period in [0, 1]:
        trend = 3.0 * period
        outcome = (50 + unit_fe + trend
                   + true_effect * treated_group * period
                   + rng.normal(0, 4, n_units))
        for u, tg, o in zip(unit_id, treated_group, outcome):
            rows.append({"unit_id": u, "period": period, "treated_group": tg, "outcome": o})
    return pd.DataFrame(rows)


def make_observational_data(n=3000, seed=4):
    """Confounded observational data for the IV / DML / heterogeneous
    effects extensions. engagement_score drives both treatment
    (probability of opening the email) and the outcome directly, so a
    naive OLS/regression of outcome on treatment is biased. `instrument`
    is a random encouragement (e.g. a push-notification nudge) that
    shifts open probability but has no direct effect on the outcome --
    satisfies the exclusion restriction by construction. The treatment
    effect itself is heterogeneous (rises with engagement_score), so the
    heterogeneous-effects extension has real signal to recover."""
    rng = np.random.default_rng(seed)
    engagement_score = rng.normal(50, 15, n).clip(0, 100)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    instrument = rng.binomial(1, 0.5, n)
    open_prob = np.clip(0.15 + 0.5 * instrument + 0.006 * engagement_score, 0.01, 0.99)
    treatment = rng.binomial(1, open_prob)
    true_effect = 1.0 + 0.08 * engagement_score  # higher-engagement customers respond more
    purchase_amount = (20 + true_effect * treatment + 0.8 * engagement_score
                        + 3 * x1 - 2 * x2 + rng.normal(0, 8, n))
    df = pd.DataFrame({"customer_id": np.arange(n), "engagement_score": engagement_score,
                        "instrument": instrument, "treatment": treatment,
                        "x1": x1, "x2": x2, "purchase_amount": purchase_amount})

    # a few missing covariate values, same rationale as above
    missing_idx = rng.choice(n, size=int(0.01 * n), replace=False)
    df.loc[missing_idx, "engagement_score"] = np.nan

    return df


def make_transactions_data(n=3000, seed=5):
    """Messy transaction log: duplicate rows, missing amounts,
    inconsistent category casing/whitespace, a few negative amounts
    (refunds/errors), and a few unparseable dates."""
    rng = np.random.default_rng(seed)
    customer_id = rng.integers(1, 300, n)
    amount = rng.gamma(2, 40, n)
    categories = ["Groceries", "groceries", "  Electronics", "electronics", "Dining", "dining ", "Travel"]
    category = rng.choice(categories, n)
    start = pd.Timestamp("2024-01-01")
    dates = start + pd.to_timedelta(rng.integers(0, 400, n), unit="D")
    date_strs = dates.strftime("%Y-%m-%d").to_numpy().astype(object)

    df = pd.DataFrame({"transaction_id": np.arange(n), "customer_id": customer_id,
                        "amount": amount, "category": category, "transaction_date": date_strs})

    dupes = df.sample(50, random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)

    miss_idx = rng.choice(df.index, size=60, replace=False)
    df.loc[miss_idx, "amount"] = np.nan

    neg_idx = rng.choice(df.index, size=20, replace=False)
    df.loc[neg_idx, "amount"] = -df.loc[neg_idx, "amount"]

    bad_idx = rng.choice(df.index, size=10, replace=False)
    df.loc[bad_idx, "transaction_date"] = "not_a_date"

    return df


def make_customers_data(n=300, seed=6):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"customer_id": np.arange(1, n + 1),
                          "region": rng.choice(["West", "East", "Midwest", "South"], n)})


def make_ols_data(n=500, seed=7):
    """Small dataset for the numpy-from-scratch OLS problem. Noise
    variance grows with |x1| so heteroskedastic-robust SEs visibly
    differ from the classical (homoskedastic) ones."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    noise = rng.normal(0, 1 + 0.8 * np.abs(x1), n)
    y = 3 + 2 * x1 - 1.5 * x2 + 0.5 * x3 + noise
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})


def make_cv_data(n=400, seed=8):
    """x1/x2 are correlated so the ridge penalty in the CV problem
    actually matters (plain OLS would be unstable)."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = x1 * 0.8 + rng.normal(0, 0.3, n)
    x3 = rng.normal(0, 1, n)
    y = 4 + 2 * x1 + 2 * x2 - x3 + rng.normal(0, 3, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    make_housing_data().to_csv("data/housing.csv", index=False)
    make_churn_data().to_csv("data/churn.csv", index=False)
    make_experiment_data().to_csv("data/experiment.csv", index=False)
    make_panel_data().to_csv("data/panel.csv", index=False)
    make_observational_data().to_csv("data/observational.csv", index=False)
    make_transactions_data().to_csv("data/transactions.csv", index=False)
    make_customers_data().to_csv("data/customers.csv", index=False)
    make_ols_data().to_csv("data/ols_data.csv", index=False)
    make_cv_data().to_csv("data/cv_data.csv", index=False)
    print("wrote data/*.csv")
