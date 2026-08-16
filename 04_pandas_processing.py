# Prompt: "Here's a transaction log. Clean it up and produce a
# per-customer monthly spend summary."
# data/transactions.csv: transaction_id, customer_id, amount, category,
# transaction_date (string) -- has dupes, missing values, messy strings.
# data/customers.csv: customer_id, region.

import numpy as np
import pandas as pd

tx = pd.read_csv("data/transactions.csv")
cust = pd.read_csv("data/customers.csv")

print(tx.shape)
print(tx.dtypes)
print(tx.head())

# duplicate rows
print(tx.duplicated().sum())
tx = tx.drop_duplicates()

# missing values
print(tx.isna().sum())
tx = tx.dropna(subset=["amount", "transaction_date"])  # can't use a row without these

# inconsistent category casing/whitespace
tx["category"] = tx["category"].str.strip().str.lower()

# parse dates, drop rows that fail to parse instead of silently coercing
tx["transaction_date"] = pd.to_datetime(tx["transaction_date"], errors="coerce")
print(tx["transaction_date"].isna().sum(), "unparseable dates")
tx = tx.dropna(subset=["transaction_date"])

# negative amounts look like data errors for this business -- check before dropping
print((tx["amount"] <= 0).sum())
tx = tx[tx["amount"] > 0]

# bring in customer attributes
df = tx.merge(cust, on="customer_id", how="left")
print(df["region"].isna().sum(), "transactions with no matching customer")

df["month"] = df["transaction_date"].dt.to_period("M")

# groupby + agg -- multiple stats in one pass
summary = df.groupby(["customer_id", "month"]).agg(
    total_spend=("amount", "sum"),
    n_transactions=("amount", "count"),
    avg_txn=("amount", "mean"),
).reset_index()
print(summary.head())

# wide view: customer x month total spend
pivot = summary.pivot_table(index="customer_id", columns="month", values="total_spend", fill_value=0)
print(pivot.head())

# category mix per customer -- vectorized (groupby + div), not apply
category_totals = df.groupby(["customer_id", "category"])["amount"].sum().unstack(fill_value=0)
category_share = category_totals.div(category_totals.sum(axis=1), axis=0)
print(category_share.head())

# rolling 3-month spend per customer -- needs sorted index within each group
monthly = df.groupby(["customer_id", "month"])["amount"].sum().reset_index()
monthly = monthly.sort_values(["customer_id", "month"])
monthly["rolling_3mo"] = monthly.groupby("customer_id")["amount"].transform(
    lambda s: s.rolling(3, min_periods=1).sum()
)
print(monthly.head(10))
