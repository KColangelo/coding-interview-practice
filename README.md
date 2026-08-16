# DS interview prep

Canonical walkthroughs for common data scientist coding-interview
questions. Each numbered file is meant to be readable top to bottom in
about 30 minutes -- the length a real interview slot would give you.

## Setup

```
uv sync
uv run 00_generate_data.py   # writes data/*.csv, run once
```

Then run any problem file, e.g. `uv run 01_regression.py`.

## Files

- **00_generate_data.py** -- generates all the synthetic CSVs the other
  files read from. Not part of the interview walkthrough itself (in a
  real interview you'd be handed a file, not asked to make one).
- **01_regression.py** -- predict house price from sqft/bedrooms/age/
  neighborhood. EDA, missing values, outlier handling (winsorize),
  preprocessing pipeline, Ridge tuned via `GridSearchCV`, RMSE/MAE/R2.
- **02_classification.py** -- predict churn from an imbalanced dataset
  (~4% positive). Same skeleton as regression, but stratified CV,
  `class_weight="balanced"`, PR-AUC/ROC-AUC instead of accuracy, and
  threshold tuning.
- **03_causal_inference.py** -- estimate a treatment effect from a
  randomized experiment: missing-value/duplicate checks, balance check
  (t-test + boxplot), outcome histogram by arm, OLS, heteroskedasticity-
  robust (HC1) and cluster-robust SEs. Extensions cover diff-in-diff on
  panel data (with a parallel-trends plot), instrumental variables
  (manual 2SLS, with an instrument-relevance plot), double/debiased ML
  and heterogeneous treatment effects with `econml` (plus from-scratch
  reference versions of DML and a T-learner). Saves its plots to
  `plots/`.
- **04_pandas_processing.py** -- clean a messy transaction log
  (duplicates, missing values, inconsistent strings, unparseable
  dates), merge in customer attributes, and build a per-customer
  monthly summary with `groupby`/`agg`/`pivot_table`/rolling windows.
- **05_numpy_ols.py** -- implement OLS from scratch via the normal
  equations, with classical standard errors, then ridge and
  heteroskedasticity-robust (White/HC0/HC1) standard errors.
- **06_cross_validation.py** -- implement k-fold CV from scratch
  (no `cross_val_score`), then nested CV for hyperparameter tuning.

## How to use these

Read the prompt comment at the top of a file, then try writing the
solution yourself before reading the rest. Each file also lists, in
comments, the follow-up questions/extensions an interviewer is likely
to ask next.
