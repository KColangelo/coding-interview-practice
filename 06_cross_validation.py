# Prompt: "Implement k-fold cross-validation from scratch (don't use
# cross_val_score) and use it to evaluate a model."
# data/cv_data.csv: x1, x2, x3, y.

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

df = pd.read_csv("data/cv_data.csv")
X = df[["x1", "x2", "x3"]].to_numpy()
y = df["y"].to_numpy()


def kfold_indices(n, k, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, test_idx


def cross_validate(model_fn, X, y, k=5):
    scores = []
    for train_idx, test_idx in kfold_indices(len(y), k):
        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        scores.append(np.sqrt(mean_squared_error(y[test_idx], preds)))
    return np.array(scores)


scores = cross_validate(lambda: Ridge(alpha=1.0), X, y, k=5)
print("fold RMSEs:", scores)
print("mean RMSE:", scores.mean(), "+/-", scores.std())

# sanity check against sklearn's cross_val_score
from sklearn.model_selection import KFold, cross_val_score
sk_scores = -cross_val_score(
    Ridge(alpha=1.0), X, y, cv=KFold(5, shuffle=True, random_state=0), scoring="neg_root_mean_squared_error"
)
print("sklearn fold RMSEs:", sk_scores)


# ---------------------------------------------------------------------
# EXTENSION: nested CV -- outer loop evaluates, inner loop tunes alpha, so
# the alpha choice doesn't leak into the outer performance estimate
# ---------------------------------------------------------------------
def cross_validate_with_tuning(alphas, X, y, k=5):
    outer_scores = []
    for train_idx, test_idx in kfold_indices(len(y), k):
        X_train, y_train = X[train_idx], y[train_idx]

        best_alpha, best_score = None, np.inf
        for a in alphas:
            inner_scores = cross_validate(lambda a=a: Ridge(alpha=a), X_train, y_train, k=4)
            if inner_scores.mean() < best_score:
                best_alpha, best_score = a, inner_scores.mean()

        model = Ridge(alpha=best_alpha).fit(X_train, y_train)
        preds = model.predict(X[test_idx])
        outer_scores.append(np.sqrt(mean_squared_error(y[test_idx], preds)))
    return np.array(outer_scores)


nested_scores = cross_validate_with_tuning([0.01, 0.1, 1, 10, 100], X, y)
print("nested CV RMSEs:", nested_scores)

# other variants worth mentioning if asked: StratifiedKFold for
# classification (keeps class ratio per fold), GroupKFold when
# observations are clustered (e.g. multiple rows per customer -- don't
# want the same customer split across train/test), TimeSeriesSplit /
# walk-forward validation for time-ordered data (never train on the future)
