"""Data loading helpers for the Home Credit application tables.

Kept deliberately thin - the goal is to centralise the path convention and
dtype handling so notebooks don't repeat themselves.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Repo-relative paths. Resolves correctly whether called from notebooks/ or repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def load_application_train(path: Path | None = None) -> pd.DataFrame:
    """Load application_train.csv.

    Parameters
    ----------
    path : Path, optional
        Override the default path. Useful for testing on a sampled copy.

    Returns
    -------
    pd.DataFrame
        307,511 rows x 122 columns. Includes the TARGET column.
    """
    if path is None:
        path = RAW_DIR / "application_train.csv"
    return pd.read_csv(path)


def load_application_test(path: Path | None = None) -> pd.DataFrame:
    """Load application_test.csv (no TARGET column - this is the Kaggle holdout)."""
    if path is None:
        path = RAW_DIR / "application_test.csv"
    return pd.read_csv(path)


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted summary of missing values per column.

    Returns columns: missing_count, missing_pct, dtype. Sorted by missing_pct desc.
    Only columns with at least one missing value are returned.
    """
    missing = df.isna().sum()
    missing = missing[missing > 0]
    summary = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": (missing / len(df) * 100).round(2),
        "dtype": df[missing.index].dtypes.astype(str),
    })
    return summary.sort_values("missing_pct", ascending=False)
