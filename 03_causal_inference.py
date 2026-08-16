# Prompt: "We randomized whether customers got a discount email
# (treatment). What's the effect on purchase_amount?"
# data/experiment.csv: customer_id, store_id, treatment, baseline_spend,
# purchase_amount.

import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

# Agg backend since we're saving figures to disk rather than showing them
# inline (as you would in a notebook) -- keeps this runnable as a plain script
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

df = pd.read_csv("data/experiment.csv")

print(df.shape)
print(df.dtypes)
print(df.isna().sum())
print(df.duplicated().sum())
df = df.drop_duplicates().dropna(subset=["baseline_spend"])

# treatment split should be close to 50/50 if randomization worked as intended
print(df["treatment"].value_counts(normalize=True))
print(df.groupby("treatment")["purchase_amount"].describe())

# outcome distribution by arm -- quick visual gut check before any modeling
df["purchase_amount"].hist(by=df["treatment"], bins=30)
plt.suptitle("purchase_amount by arm (0=control, 1=treatment)")
plt.savefig("plots/experiment_outcome_hist.png")
plt.close()

# balance check -- covariate means/distributions should look similar across
# arms if randomization actually worked
t, p = stats.ttest_ind(
    df.loc[df.treatment == 1, "baseline_spend"],
    df.loc[df.treatment == 0, "baseline_spend"],
)
print("baseline_spend balance p-value:", p)

df.boxplot(column="baseline_spend", by="treatment")
plt.suptitle("")  # drop pandas' default "Boxplot grouped by..." suptitle
plt.title("baseline_spend balance by arm")
plt.savefig("plots/experiment_balance_boxplot.png")
plt.close()

# naive difference in means -- unbiased here since treatment is randomized
diff = df.loc[df.treatment == 1, "purchase_amount"].mean() - df.loc[df.treatment == 0, "purchase_amount"].mean()
print("naive ATE:", diff)

# OLS controlling for baseline_spend -- not needed for unbiasedness (already
# randomized) but soaks up residual variance and tightens the SE
ols = smf.ols("purchase_amount ~ treatment + baseline_spend", data=df).fit()
print(ols.summary())

# heteroskedasticity-robust SEs (HC1) in case error variance isn't constant
# across customers (it isn't here -- noise scales with baseline_spend)
robust = smf.ols("purchase_amount ~ treatment + baseline_spend", data=df).fit(cov_type="HC1")
print(robust.summary())

# cluster-robust SEs by store -- outcomes are correlated within store
# (shared shocks), which violates the iid assumption and understates
# classical SEs if ignored
cluster = smf.ols("purchase_amount ~ treatment + baseline_spend", data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["store_id"]}
)
print(cluster.summary())
print("ATE (cluster-robust):", cluster.params["treatment"], "+/-", cluster.bse["treatment"])


# ---------------------------------------------------------------------
# EXTENSION: what if we couldn't randomize? panel data / diff-in-diff.
# data/panel.csv: unit_id, period (0=pre, 1=post), treated_group, outcome.
# ---------------------------------------------------------------------
panel = pd.read_csv("data/panel.csv")
print(panel.isna().sum())

# the classic DiD visual: mean outcome by period for each group. with only
# 2 periods we can't fully test parallel pre-trends, but this is the plot
# you'd make with more pre-periods to sanity-check the assumption --
# converging/diverging lines before treatment would be a red flag
trend = panel.groupby(["period", "treated_group"])["outcome"].mean().unstack()
print(trend)

trend.plot(marker="o", xticks=[0, 1], xlabel="period (0=pre, 1=post)", ylabel="mean outcome")
plt.title("parallel trends check")
plt.savefig("plots/panel_parallel_trends.png")
plt.close()

did = smf.ols("outcome ~ treated_group * period", data=panel).fit(
    cov_type="cluster", cov_kwds={"groups": panel["unit_id"]}
)
print(did.summary())
# the treated_group:period interaction coefficient is the DiD estimate --
# effect relative to the counterfactual trend, assuming parallel trends


# ---------------------------------------------------------------------
# EXTENSION: treatment is confounded (not randomized), but we have an
# instrument. data/observational.csv: customer_id, engagement_score
# (confounder), instrument (random encouragement), treatment (email
# opened), x1, x2, purchase_amount.
# ---------------------------------------------------------------------
obs = pd.read_csv("data/observational.csv")

print(obs.isna().sum())
obs = obs.dropna(subset=["engagement_score"])

# instrument relevance -- does it actually move treatment take-up? if this
# were flat, the instrument would be too weak to identify anything
print(obs.groupby("instrument")["treatment"].mean())

obs.groupby("instrument")["treatment"].mean().plot(kind="bar", ylabel="P(treatment=1)")
plt.title("instrument relevance: treatment rate by instrument value")
plt.savefig("plots/iv_relevance.png")
plt.close()

# confounding check -- engagement_score should differ by treatment status
# here (unlike the randomized experiment above), which is exactly why the
# naive OLS below is biased
obs.boxplot(column="engagement_score", by="treatment")
plt.suptitle("")
plt.title("engagement_score by treatment (confounding)")
plt.savefig("plots/iv_confounding_boxplot.png")
plt.close()

# naive OLS is biased -- engagement_score drives both treatment and outcome
naive = smf.ols("purchase_amount ~ treatment", data=obs).fit()
print("naive (biased) effect:", naive.params["treatment"])

# manual 2SLS: first stage predicts treatment from the instrument, second
# stage regresses outcome on the *predicted* treatment
first_stage = smf.ols("treatment ~ instrument", data=obs).fit()
obs["treatment_hat"] = first_stage.predict(obs)
second_stage = smf.ols("purchase_amount ~ treatment_hat", data=obs).fit()
print("2SLS effect:", second_stage.params["treatment_hat"])
# note: SEs from this manual two-step regression are wrong (don't account
# for first-stage estimation uncertainty) -- would use linearmodels.IV2SLS
# for correct SEs in practice, this is just to show the mechanics


# ---------------------------------------------------------------------
# EXTENSION: double/debiased ML -- control for confounders nonlinearly /
# with many covariates instead of assuming a linear functional form.
# use econml in practice -- it handles cross-fitting internally and gives
# proper inference (CIs), so there's no reason to hand-roll this at work.
# ---------------------------------------------------------------------
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from econml.dml import LinearDML

rf_grid = {"n_estimators": [100, 300], "max_depth": [3, 5, None]}


def tuned_rf():
    # GridSearchCV is itself a valid sklearn estimator (has fit/predict), so
    # econml can use it directly as a nuisance model and it'll get re-tuned
    # on every cross-fit fold internally
    return GridSearchCV(RandomForestRegressor(random_state=0), rf_grid, cv=3, scoring="neg_mean_squared_error")


covs = obs[["engagement_score", "x1", "x2"]]

# model_y/model_t partial out covs from outcome/treatment; treatment is
# 0/1 here but we're treating it as continuous (discrete_treatment=False,
# the default) -- fine for a simple residual-on-residual ATE. cv=5 does
# the cross-fitting so nuisance models never see the fold they predict on.
est = LinearDML(model_y=tuned_rf(), model_t=tuned_rf(), cv=5, random_state=0)
est.fit(obs["purchase_amount"], obs["treatment"], X=None, W=covs)
print("DML effect:", est.const_marginal_effect())
print("DML 95% CI:", est.const_marginal_effect_interval())


# ---------------------------------------------------------------------
# EXTENSION: heterogeneous effects -- who benefits most? use econml's
# CausalForestDML for per-customer CATE estimates with valid CIs.
# ---------------------------------------------------------------------
from econml.dml import CausalForestDML

cf = CausalForestDML(model_y=tuned_rf(), model_t=tuned_rf(), cv=5, random_state=0)
cf.fit(obs["purchase_amount"], obs["treatment"], X=covs, W=None)

obs["cate"] = cf.effect(covs)
cate_by_segment = obs.groupby(pd.qcut(obs.engagement_score, 4))["cate"].mean()
print(cate_by_segment)

cate_by_segment.plot(kind="bar", ylabel="estimated CATE")
plt.title("treatment effect by engagement_score quartile")
plt.savefig("plots/cate_by_segment.png")
plt.close()

lb, ub = cf.effect_interval(covs)
print("mean CI width:", (ub - lb).mean())
# segments with the biggest predicted effect are the best targeting candidates


# ---------------------------------------------------------------------
# REFERENCE: if asked to implement DML by hand (no econml available) --
# this is what LinearDML is doing under the hood.
# ---------------------------------------------------------------------
from sklearn.model_selection import KFold


def fit_tuned_rf(X, y):
    search = GridSearchCV(RandomForestRegressor(random_state=0), rf_grid, cv=3, scoring="neg_mean_squared_error")
    search.fit(X, y)
    return search.best_estimator_


kf = KFold(5, shuffle=True, random_state=0)
y_resid = np.zeros(len(obs))
t_resid = np.zeros(len(obs))

for train_idx, test_idx in kf.split(obs):
    # fit nuisance models on the training fold only, predict on the held-out
    # fold -- this out-of-fold residualization is what "cross-fitting" means
    m_y = fit_tuned_rf(covs.iloc[train_idx], obs["purchase_amount"].iloc[train_idx])
    m_t = fit_tuned_rf(covs.iloc[train_idx], obs["treatment"].iloc[train_idx])
    y_resid[test_idx] = obs["purchase_amount"].iloc[test_idx] - m_y.predict(covs.iloc[test_idx])
    t_resid[test_idx] = obs["treatment"].iloc[test_idx] - m_t.predict(covs.iloc[test_idx])

# regress outcome residual on treatment residual to get the debiased effect
manual_dml = smf.ols("y_resid ~ t_resid", data=pd.DataFrame({"y_resid": y_resid, "t_resid": t_resid})).fit()
print("manual DML effect:", manual_dml.params["t_resid"])

# ...and the from-scratch equivalent of the T-learner for heterogeneity:
treated = obs[obs.treatment == 1]
control = obs[obs.treatment == 0]
m1 = fit_tuned_rf(treated[["engagement_score", "x1", "x2"]], treated["purchase_amount"])
m0 = fit_tuned_rf(control[["engagement_score", "x1", "x2"]], control["purchase_amount"])
obs["cate_tlearner"] = m1.predict(covs) - m0.predict(covs)
print(obs.groupby(pd.qcut(obs.engagement_score, 4))["cate_tlearner"].mean())
# no valid CIs here and it can be unstable with small treated/control groups
# -- CausalForestDML above is the better default whenever the library's available
