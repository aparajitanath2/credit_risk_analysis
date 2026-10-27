"""Shared settings and helper functions for the credit risk project."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import RocCurveDisplay, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "SouthGermanCredit.asc"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
OUTPUTS = ROOT / "outputs"
MODELS = OUTPUTS / "models"
for folder in (PROCESSED, FIGURES, OUTPUTS, MODELS):
    folder.mkdir(parents=True, exist_ok=True)

SEED = 42
CONTINUOUS = ["duration", "amount", "age"]

# German column names in the raw file -> English names (see data/codetable.txt)
ENGLISH_NAMES = {
    "laufkont": "status", "laufzeit": "duration", "moral": "credit_history", "verw": "purpose",
    "hoehe": "amount", "sparkont": "savings", "beszeit": "employment_duration",
    "rate": "installment_rate", "famges": "personal_status_sex", "buerge": "other_debtors",
    "wohnzeit": "present_residence", "verm": "property", "alter": "age",
    "weitkred": "other_installment_plans", "wohn": "housing", "bishkred": "number_credits",
    "beruf": "job", "pers": "people_liable", "telef": "telephone", "gastarb": "foreign_worker",
    "kredit": "credit_risk",
}

# Numeric codes -> readable labels, copied from the UCI code table (data/codetable.txt)
LABELS = {
    "status": {1: "no checking account", 2: "< 0 DM", 3: "0 to 200 DM", 4: ">= 200 DM or salary"},
    "credit_history": {0: "delay in paying off in the past", 1: "critical account or other credits elsewhere",
                       2: "no credits taken or all paid back duly", 3: "existing credits paid back duly",
                       4: "all credits at this bank paid back duly"},
    "purpose": {0: "others", 1: "car (new)", 2: "car (used)", 3: "furniture/equipment", 4: "radio/television",
                5: "domestic appliances", 6: "repairs", 7: "education", 8: "vacation", 9: "retraining",
                10: "business"},
    "savings": {1: "unknown/no savings", 2: "< 100 DM", 3: "100 to 500 DM", 4: "500 to 1000 DM", 5: ">= 1000 DM"},
    "employment_duration": {1: "unemployed", 2: "< 1 yr", 3: "1 to 4 yrs", 4: "4 to 7 yrs", 5: ">= 7 yrs"},
    "installment_rate": {1: ">= 35", 2: "25 to 35", 3: "20 to 25", 4: "< 20"},
    "personal_status_sex": {1: "male: divorced/separated", 2: "female: non-single or male: single",
                            3: "male: married/widowed", 4: "female: single"},
    "other_debtors": {1: "none", 2: "co-applicant", 3: "guarantor"},
    "present_residence": {1: "< 1 yr", 2: "1 to 4 yrs", 3: "4 to 7 yrs", 4: ">= 7 yrs"},
    "property": {1: "unknown/no property", 2: "car or other", 3: "savings agreement or life insurance",
                 4: "real estate"},
    "other_installment_plans": {1: "bank", 2: "stores", 3: "none"},
    "housing": {1: "for free", 2: "rent", 3: "own"},
    "number_credits": {1: "1 credit", 2: "2-3 credits", 3: "4-5 credits", 4: ">= 6 credits"},
    "job": {1: "unemployed/unskilled non-resident", 2: "unskilled resident", 3: "skilled employee/official",
            4: "manager/self-employed/highly qualified"},
    "people_liable": {1: "3 or more", 2: "0 to 2"},
    "telephone": {1: "no", 2: "yes"},
    "foreign_worker": {1: "yes", 2: "no"},
}


def load_raw():
    """Read the raw file and rename the German columns to English.

    returns:
        DataFrame with 21 columns; credit_risk is 1 = good, 0 = bad (source coding)
    """
    df = pd.read_csv(DATA_RAW, sep=r"\s+").rename(columns=ENGLISH_NAMES)
    assert list(df.columns) == list(ENGLISH_NAMES.values()), "Unexpected columns"
    assert set(df["credit_risk"].unique()) == {0, 1}, "Unexpected target labels"
    return df


def make_preprocess(categorical):
    """Preprocessing for the models: numeric = median impute + standardise,
    categorical = most-frequent impute + one-hot. Used inside a Pipeline so it
    is fitted on training data only (also inside each cross-validation fold).

    args:
        categorical: list of categorical column names
    """
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), CONTINUOUS),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ], verbose_feature_names_out=False)


def load_split():
    """Load the train/test split written by notebook B.

    returns:
        X_train, X_test, y_train, y_test (y: 1 = bad credit)
    """
    X_train = pd.read_csv(PROCESSED / "X_train.csv", index_col=0)
    X_test = pd.read_csv(PROCESSED / "X_test.csv", index_col=0)
    y_train = pd.read_csv(PROCESSED / "y_train.csv", index_col=0).iloc[:, 0]
    y_test = pd.read_csv(PROCESSED / "y_test.csv", index_col=0).iloc[:, 0]
    cats = [c for c in X_train.columns if c not in CONTINUOUS]
    X_train[cats] = X_train[cats].astype(str)
    X_test[cats] = X_test[cats].astype(str)
    return X_train, X_test, y_train, y_test


def categorical_columns(X):
    """Names of the categorical predictors (everything except the numeric ones)."""
    return [c for c in X.columns if c not in CONTINUOUS]


def bootstrap_auc_ci(y_true, p, n=1000, seed=SEED):
    """95% bootstrap interval for ROC-AUC (resample the test cases with replacement)."""
    rng = np.random.default_rng(seed)
    y_arr, p_arr = np.asarray(y_true), np.asarray(p)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_arr), len(y_arr))
        if len(np.unique(y_arr[idx])) == 2:
            aucs.append(roc_auc_score(y_arr[idx], p_arr[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def evaluate(y_true, p, name):
    """One row of performance metrics from predicted probabilities (not hard labels).

    returns:
        dict with ROC-AUC, 95% interval, Gini, Brier score, mean predicted PD, observed bad rate
    """
    auc = roc_auc_score(y_true, p)
    lo, hi = bootstrap_auc_ci(y_true, p)
    return {"model": name, "roc_auc": auc, "auc_95ci_low": lo, "auc_95ci_high": hi,
            "gini": 2 * auc - 1, "brier_score": brier_score_loss(y_true, p),
            "mean_predicted_pd": float(np.mean(p)), "observed_bad_rate": float(np.mean(y_true))}


def calibration_table(y_true, p, bins=5):
    """Mean predicted PD vs observed bad rate in equal-sized score bins."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(p)})
    # rank first so that bins stay equal-sized even when many cases share the same score (e.g. a shallow tree)
    df["score_bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=[f"Bin {i}" for i in range(1, bins + 1)])
    return df.groupby("score_bin", observed=True).agg(
        cases=("y", "size"), mean_predicted_pd=("p", "mean"), observed_bad_rate=("y", "mean"))


def psi(baseline, comparison, bins=10, eps=1e-6):
    """Population Stability Index of two score distributions, with bins fixed from the baseline."""
    edges = np.quantile(baseline, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.histogram(baseline, edges)[0] / len(baseline)
    c = np.histogram(comparison, edges)[0] / len(comparison)
    b, c = np.clip(b, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - b) * np.log(c / b)))


def plot_roc(y_true, preds, ax=None):
    """ROC curves for several models. preds = {model name: predicted probabilities}."""
    ax = ax or plt.gca()
    for name, p in preds.items():
        RocCurveDisplay.from_predictions(y_true, p, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title("ROC curves, held-out sample")


def plot_calibration(y_true, preds, ax=None, bins=5):
    """Calibration curves (equal-sized score bins). preds = {model name: predicted probabilities}."""
    ax = ax or plt.gca()
    for name, p in preds.items():
        CalibrationDisplay.from_predictions(y_true, p, name=name, n_bins=bins, strategy="quantile", ax=ax)
    ax.set_title(f"Calibration, held-out sample ({bins} equal-sized score bins)")
