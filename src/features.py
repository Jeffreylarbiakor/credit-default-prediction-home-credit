"""Feature engineering helpers for the application table.

Stubs to be implemented in 02_feature_engineering.ipynb. Keeping the function
signatures here so notebooks import from a stable location.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def clean_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Clean known data anomalies in the application table.

    Home Credit uses 365243 (~1000 years) as a sentinel for 'not applicable'
    in DAYS_* columns. Convert those to NaN so imputation handles them
    consistently.

    Returns a copy of the DataFrame with the anomalies replaced.
    """
    df = df.copy()
    days_employed_anomaly = df["DAYS_EMPLOYED"] == 365243
    df.loc[days_employed_anomaly, "DAYS_EMPLOYED"] = np.nan
    # Add a binary flag - the anomaly itself may be predictive
    df["DAYS_EMPLOYED_ANOMALY"] = days_employed_anomaly.astype(int)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived ratio features.

    These are well-documented Home Credit features:
    - CREDIT_INCOME_RATIO: credit amount / annual income (DTI-style)
    - ANNUITY_INCOME_RATIO: annuity / annual income (payment burden)
    - CREDIT_TERM: annuity / credit (implied loan term in years)
    - DAYS_EMPLOYED_PERCENT: days employed / age in days
    """
    df = df.copy()
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    df["DAYS_EMPLOYED_PERCENT"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    return df


def encode_categoricals(
    df: pd.DataFrame, fit_columns: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode categorical columns.

    Returns the encoded DataFrame and the list of categorical column names
    that were encoded, so the test set can be aligned to the same columns.
    """
    if fit_columns is None:
        fit_columns = df.select_dtypes(include=["object"]).columns.tolist()
    encoded = pd.get_dummies(df, columns=fit_columns, dummy_na=True)
    return encoded, fit_columns


def align_columns(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align test columns to train columns.

    After one-hot encoding, train and test may have different columns
    (a category appearing only in one set). This drops mismatched columns
    so the model sees the same feature space at train and inference time.
    """
    target = train["TARGET"] if "TARGET" in train.columns else None
    train_features = train.drop(columns=["TARGET"]) if target is not None else train
    common = train_features.columns.intersection(test.columns)
    aligned_train = train_features[common]
    aligned_test = test[common]
    if target is not None:
        aligned_train = aligned_train.copy()
        aligned_train["TARGET"] = target
    return aligned_train, aligned_test
