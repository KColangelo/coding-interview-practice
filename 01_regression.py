# Prompt: "Here's home sale data -- sqft, bedrooms, age, neighborhood.
# Build a model to predict sale price. Walk me through your process."
# data/housing.csv has these columns plus price.

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("data/housing.csv")

# quick look at shape/dtypes/head before anything else
print(df.shape)
print(df.dtypes)
print(df.head())

# any missing values?
print(df.isna().sum())

# summary stats to sanity-check ranges and spot anything weird
print(df.describe(include="all"))

# check for target outliers with IQR fences
q1, q3 = df["price"].quantile([0.25, 0.75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print("n outliers:", ((df["price"] < lo) | (df["price"] > hi)).sum())

# cap rather than drop -- keeps sample size, limits leverage on the linear model
df["price"] = df["price"].clip(lo, hi)

y = df["price"]
X = df.drop(columns="price")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

num_cols = ["sqft", "bedrooms", "age"]
cat_cols = ["neighborhood"]

# impute (median, robust to skew) + scale numeric since Ridge's penalty is scale-sensitive
# one-hot the categorical, drop first level to avoid the dummy trap
pre = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
])

pipe = Pipeline([("pre", pre), ("model", Ridge())])

# tune alpha with CV instead of eyeballing it -- imputer/scaler refit per fold, no leakage
grid = GridSearchCV(
    pipe,
    {"model__alpha": [0.01, 0.1, 1, 10, 100]},
    cv=KFold(5, shuffle=True, random_state=0),
    scoring="neg_root_mean_squared_error",
)
grid.fit(X_train, y_train)
print("best alpha:", grid.best_params_)
print("best CV RMSE:", -grid.best_score_)

best = grid.best_estimator_
preds = best.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("test RMSE:", rmse)
print("test MAE:", mean_absolute_error(y_test, preds))
print("test R2:", r2_score(y_test, preds))

# residuals should be ~0 mean with no obvious pattern -- quick check, would
# plot resid vs fitted if I had more time to look for nonlinearity
resid = y_test - preds
print("resid mean:", resid.mean(), "resid std:", resid.std())

# if this weren't good enough: try a tree ensemble (RandomForest/GBM) to see
# if nonlinearities/interactions beat the linear model, and check error by
# neighborhood for any systematic bias in one segment
