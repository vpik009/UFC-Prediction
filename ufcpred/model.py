"""Train and serve the fight-outcome probability model.

We compare a regularized logistic regression against a gradient-boosted
tree on a time-ordered holdout (train on older fights, evaluate on the
most recent slice) and keep whichever has lower log loss, then refit that
choice on the full dataset for deployment. Logistic regression on
engineered Elo/bio/output-rate diffs is the standard approach for
sports win-probability models (well-calibrated, low variance on a
dataset of this size); gradient boosting is given a fair shot in case it
captures non-linear interactions logistic regression can't.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "model"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


@dataclass
class EvalResult:
    name: str
    accuracy: float
    log_loss: float
    n_test: int


def _time_split(df: pd.DataFrame, test_frac: float = 0.15):
    df = df.sort_values("event_date")
    cutoff = df["event_date"].quantile(1 - test_frac)
    train = df[df["event_date"] <= cutoff]
    test = df[df["event_date"] > cutoff]
    return train, test


def _fit_logreg(X: np.ndarray, y: np.ndarray) -> Pipeline:
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=0.5, max_iter=2000)),
        ]
    )
    pipe.fit(X, y)
    return pipe


def _fit_gbm(X: np.ndarray, y: np.ndarray) -> HistGradientBoostingClassifier:
    clf = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=0
    )
    clf.fit(X, y)
    return clf


def train_and_select(df: pd.DataFrame, verbose: bool = True) -> tuple[object, str, list[EvalResult]]:
    train_df, test_df = _time_split(df)
    Xtr, ytr = train_df[FEATURE_NAMES].values, train_df["label"].values
    Xte, yte = test_df[FEATURE_NAMES].values, test_df["label"].values

    candidates = {
        "logistic_regression": _fit_logreg(Xtr, ytr),
        "gradient_boosting": _fit_gbm(Xtr, ytr),
    }

    results = []
    for name, model in candidates.items():
        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        res = EvalResult(
            name=name,
            accuracy=accuracy_score(yte, pred),
            log_loss=log_loss(yte, proba, labels=[0, 1]),
            n_test=len(yte),
        )
        results.append(res)
        if verbose:
            print(f"  {name}: accuracy={res.accuracy:.3f} log_loss={res.log_loss:.3f} (n={res.n_test})")

    best = min(results, key=lambda r: r.log_loss)
    if verbose:
        print(f"  -> selected {best.name}")

    # refit the chosen model family on the FULL dataset for deployment
    X_all, y_all = df[FEATURE_NAMES].values, df["label"].values
    final_model = _fit_logreg(X_all, y_all) if best.name == "logistic_regression" else _fit_gbm(X_all, y_all)
    return final_model, best.name, results


def save_model(model, model_name: str, results: list[EvalResult]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "model_name": model_name, "feature_names": FEATURE_NAMES}, MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps(
            {
                "selected_model": model_name,
                "results": [r.__dict__ for r in results],
            },
            indent=2,
        )
    )


def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["model_name"], bundle["feature_names"]


def predict_proba(model, feature_row: dict) -> float:
    x = np.array([[feature_row[f] for f in FEATURE_NAMES]])
    return float(model.predict_proba(x)[0, 1])
