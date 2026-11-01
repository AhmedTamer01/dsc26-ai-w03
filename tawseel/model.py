"""Failure-risk baseline for Tawseel COD parcels.

You write this file. Keep every name and signature exactly as it is: the grader imports them.
Allowed dependencies: numpy, pandas, scikit-learn (and the standard library).
"""
from __future__ import annotations

import pandas as pd
import math
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier

TARGET = "failed"

# The raw columns of make_parcels() that the model may use at decision time (when the parcel
# is created). Nothing known only after the delivery attempt, not the label, not the row id.
# TODO: this is every column of the table. Keep only the ones known at decision time.
FEATURE_COLUMNS: list[str] = [
    "customer_id", "merchant_id", "area", "created_at", "promised_slot",
    "cod_amount", "n_items", "address_text",
]



def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df (same index and row order) with two new columns.

    Needs customer_id, created_at and failed.
    - customer_prior_parcels (int): the same customer's parcels with created_at strictly
      earlier than this row.
    - customer_prior_fail_rate (float): share of those that failed; NaN when there are none.
    Parcels with the same timestamp are not prior to each other. The result must not depend
    on the input row order. Never modify the input.
    """
    result = df.copy()

    sorted_df = result.sort_values(
        ["customer_id", "created_at"]
    )

    history = sorted_df.groupby(
        ["customer_id", "created_at"]
    ).agg(
        parcels_at_time=("failed", "size"),
        fails_at_time=("failed", "sum")
    ).reset_index()

    grouped = history.groupby("customer_id")

    history["cumulative_parcels"] = grouped["parcels_at_time"].cumsum()
    history["cumulative_fails"] = grouped["fails_at_time"].cumsum()

    history["customer_prior_parcels"] = (
        history.groupby("customer_id")["cumulative_parcels"].shift(1)
    )

    history["customer_prior_fails"] = (
        history.groupby("customer_id")["cumulative_fails"].shift(1)
    )

    history["customer_prior_fail_rate"] = (
        history["customer_prior_fails"] / history["customer_prior_parcels"]
    )

    history["customer_prior_parcels"] = (
        history["customer_prior_parcels"].fillna(0).astype(int)
    )

    history_lookup = history.set_index(
        ["customer_id", "created_at"]
    )[
        [
            "customer_prior_parcels",
            "customer_prior_fail_rate"
        ]
    ]

    result = result.join(
        history_lookup,
        on=["customer_id", "created_at"]
    )

    return result


def time_split(df: pd.DataFrame, time_col: str = "created_at",
               test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test).

    ValueError unless 0 < test_fraction < 1. Sort by time; n_test = ceil(len(df) * test_fraction);
    the boundary is the timestamp of the first of the last n_test rows. Rows at or after the
    boundary are test, rows before it are train.
    """
    check_empty = df.empty
    if check_empty:
        raise ValueError("Input DataFrame is empty")
    if not (0 < test_fraction < 1):
        raise ValueError("test_fraction must be between 0 and 1")
    test_size = math.ceil(len(df) * test_fraction)
    boundary_time = df[time_col].sort_values().iloc[-test_size]
    train_df = df[df[time_col] < boundary_time]
    test_df = df[df[time_col] >= boundary_time]
    return train_df, test_df


def address_to_text(X):


    return X["address_text"].fillna("").astype(str)

def extract_date_features(X):
        
        
        ts = pd.to_datetime(X["created_at"])
        return pd.DataFrame({"month": ts.dt.month, "weekday": ts.dt.weekday,
                         "hour": ts.dt.hour}, index=X.index)


def build_pipeline(seed: int = 0) -> Pipeline:
    """Return an UNFITTED Pipeline.

    Input: a DataFrame with at least FEATURE_COLUMNS plus the two history columns; ignore any
    other column. The final step must support predict_proba. Must not fail on unseen
    categories, missing numbers, missing categories or empty address strings. Same seed and
    same data must give identical predictions.
    """
    num_features = ["cod_amount", "n_items", "customer_prior_parcels", "customer_prior_fail_rate"]
    cat_features = ["customer_id", "merchant_id", "area", "promised_slot"]
    text_features = "address_text"
    date_features = ["created_at"]

    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True) , StandardScaler()), num_features),
        ("cat",make_pipeline(SimpleImputer(strategy="constant", fill_value="missing") , OneHotEncoder(handle_unknown="ignore", min_frequency=5)), cat_features),
        ("text", make_pipeline(FunctionTransformer(address_to_text),
                       TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3)),
         ["address_text"]),
        ("date", make_pipeline(FunctionTransformer(extract_date_features),
                       SimpleImputer(strategy="median"), StandardScaler()),
         ["created_at"]),
    ])
    pipeline = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(random_state=seed, max_iter=1000))
    ])
    return pipeline


def evaluate(model, test_df: pd.DataFrame) -> dict:
    """{"roc_auc", "pr_auc", "brier", "base_rate", "n"} from model.predict_proba(test_df)[:, 1]
    against test_df["failed"] (roc_auc_score, average_precision_score, brier_score_loss)."""
    y_true = test_df[TARGET]
    y_prob = model.predict_proba(test_df)[:, 1]

    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "base_rate": float(y_true.mean()),
        "n": int(len(test_df)),
    }


def compare_to_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int = 0) -> dict:
    """Fit build_pipeline(seed) and DummyClassifier(strategy="prior") on train_df only, evaluate
    both on test_df, return {"model": {...}, "dummy": {...}, "lift_pr_auc": model - dummy PR-AUC}."""

    y_train = train_df["failed"]

    model = build_pipeline(seed)
    dummy = DummyClassifier(strategy="prior")

    model.fit(train_df, y_train)
    dummy.fit(train_df, y_train)

    model_results = evaluate(model, test_df)
    dummy_results = evaluate(dummy, test_df)

    lift_pr_auc = model_results["pr_auc"] - dummy_results["pr_auc"]

    return {
        "model": model_results,
        "dummy": dummy_results,
        "lift_pr_auc": lift_pr_auc,
    }
