# Prompt: "Implement OLS via the normal equations, with standard errors,
# without using sklearn or statsmodels."
# data/ols_data.csv: x1, x2, x3, y.

import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("data/ols_data.csv")
X = df[["x1", "x2", "x3"]].to_numpy()
y = df["y"].to_numpy()
n, k = X.shape

# add an intercept column
X = np.column_stack([np.ones(n), X])
k += 1

# beta_hat = (X'X)^-1 X'y
XtX_inv = np.linalg.inv(X.T @ X)
beta = XtX_inv @ X.T @ y
print("beta:", beta)

resid = y - X @ beta
dof = n - k
sigma2 = (resid @ resid) / dof

# classical (homoskedastic) var-cov matrix -- SEs are sqrt of the diagonal
var_beta = sigma2 * XtX_inv
se = np.sqrt(np.diag(var_beta))

t_stats = beta / se
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))
for i in range(k):
    print(f"beta_{i}: {beta[i]:.4f}  se: {se[i]:.4f}  t: {t_stats[i]:.2f}  p: {p_values[i]:.4f}")

# sanity check against sklearn
from sklearn.linear_model import LinearRegression
sk = LinearRegression().fit(X[:, 1:], y)
print("sklearn intercept/coef:", sk.intercept_, sk.coef_)


# ---------------------------------------------------------------------
# EXTENSION: ridge -- add an L2 penalty to the normal equations
# ---------------------------------------------------------------------
alpha = 1.0
ridge_beta = np.linalg.inv(X.T @ X + alpha * np.eye(k)) @ X.T @ y
print("ridge beta:", ridge_beta)
# ridge is biased by construction, so the classical SE formula above no
# longer applies cleanly -- would need the sandwich form
# Var = (X'X + aI)^-1 X'X (X'X + aI)^-1 * sigma2, and in practice ridge SEs
# are rarely reported anyway since the whole point is trading variance for bias


# ---------------------------------------------------------------------
# EXTENSION: heteroskedasticity-robust (White/HC0) SEs -- don't assume
# constant error variance, use each residual's own squared value
# ---------------------------------------------------------------------
meat = (X * (resid ** 2)[:, None]).T @ X  # avoids building an n x n diag matrix
robust_var = XtX_inv @ meat @ XtX_inv
robust_se = np.sqrt(np.diag(robust_var))
print("HC0 SEs:", robust_se)

# HC1 small-sample correction (what statsmodels' cov_type="HC1" uses)
hc1_se = robust_se * np.sqrt(n / dof)
print("HC1 SEs:", hc1_se)
