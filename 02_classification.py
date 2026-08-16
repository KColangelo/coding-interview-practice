# Prompt: "Predict whether a customer churns in the next 30 days, from
# tenure, usage, support tickets, and plan. Heads up: only ~4% churn."
# data/churn.csv has these columns plus churn (0/1).

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("data/churn.csv")

print(df.shape)
# base rate check first -- ~4% positive means accuracy is a useless metric
# here, predicting "no churn" for everyone gets ~96% accuracy
print(df["churn"].value_counts(normalize=True))

print(df.isna().sum())
print(df.describe(include="all"))

y = df["churn"]
X = df.drop(columns="churn")

# stratify so the rare positive class keeps the same ratio in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

num_cols = ["tenure_months", "monthly_usage", "support_tickets"]
cat_cols = ["plan"]

pre = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
])

# class_weight="balanced" upweights the rare class in the loss -- simpler
# than resampling (SMOTE) and usually fine for a linear model
pipe = Pipeline([("pre", pre), ("model", LogisticRegression(class_weight="balanced", max_iter=1000))])

# StratifiedKFold keeps ~4% positive rate per fold; score on average
# precision (PR-AUC) since it's far more informative than ROC-AUC or
# accuracy when positives are rare
cv = StratifiedKFold(5, shuffle=True, random_state=0)
grid = GridSearchCV(pipe, {"model__C": [0.01, 0.1, 1, 10]}, cv=cv, scoring="average_precision")
grid.fit(X_train, y_train)
print("best C:", grid.best_params_)
print("best CV avg precision:", grid.best_score_)

best = grid.best_estimator_
proba = best.predict_proba(X_test)[:, 1]

print("test ROC-AUC:", roc_auc_score(y_test, proba))
print("test PR-AUC:", average_precision_score(y_test, proba))

preds = (proba >= 0.5).astype(int)
print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds))

# 0.5 is an arbitrary default threshold -- sweep and pick by F1 (or by a
# business-driven recall/precision target) instead
thresholds = np.linspace(0.1, 0.9, 17)
f1s = [f1_score(y_test, (proba >= t).astype(int)) for t in thresholds]
best_t = thresholds[int(np.argmax(f1s))]
print("best threshold:", best_t, "F1:", max(f1s))

# other options if pushed further on imbalance: SMOTE/undersampling
# (imblearn), tree ensembles with class_weight/scale_pos_weight, or
# precision@k if the actual deliverable is a ranked target list
