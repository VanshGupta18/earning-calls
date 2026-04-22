"""
Leakage control utilities for the Multimodal Earnings Call Intelligence System.

Functions:
    time_based_split          — strict temporal train/test split
    validate_no_future_leakage — assert no feature row post-dates its label
    check_overlapping_windows  — detect overlap between train and test windows
"""

from __future__ import annotations

import logging
from typing import Union

import polars as pl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Time-based split
# ---------------------------------------------------------------------------


def time_based_split(
    df: pl.DataFrame,
    date_column: str,
    train_end_date: str,
    test_start_date: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Split a DataFrame into non-overlapping train and test sets based on date.

    Rows with date <= train_end_date go to train.
    Rows with date >= test_start_date go to test.
    Rows in between (the gap) are dropped — this is intentional to avoid
    any temporal bleed around the boundary.

    Args:
        df: Polars DataFrame containing at least one date column.
        date_column: Name of the column holding the date (str 'YYYY-MM-DD' or Date dtype).
        train_end_date: Last date (inclusive) for the training set, 'YYYY-MM-DD'.
        test_start_date: First date (inclusive) for the test set, 'YYYY-MM-DD'.

    Returns:
        (train_df, test_df) tuple of Polars DataFrames.

    Raises:
        ValueError: If train_end_date >= test_start_date (no gap).
        KeyError: If date_column is not in df.
    """
    if date_column not in df.columns:
        raise KeyError(f"Column '{date_column}' not found in DataFrame. "
                       f"Available: {df.columns}")

    if train_end_date >= test_start_date:
        raise ValueError(
            f"train_end_date ({train_end_date}) must be strictly before "
            f"test_start_date ({test_start_date}) to ensure no overlap."
        )

    # Normalise the date column to string for comparison
    col = pl.col(date_column)
    if df[date_column].dtype in (pl.Date, pl.Datetime):
        col = col.dt.strftime("%Y-%m-%d")
        date_col = df.with_columns(col.alias("__date_str__"))["__date_str__"]
    else:
        date_col = df[date_column].cast(pl.Utf8)

    df = df.with_columns(date_col.alias("__date_str__"))
    train = df.filter(pl.col("__date_str__") <= train_end_date).drop("__date_str__")
    test = df.filter(pl.col("__date_str__") >= test_start_date).drop("__date_str__")

    gap_rows = df.height - train.height - test.height
    logger.info(
        "Time split: train=%d rows (≤ %s), test=%d rows (≥ %s), gap_dropped=%d",
        train.height, train_end_date, test.height, test_start_date, gap_rows,
    )
    return train, test


# ---------------------------------------------------------------------------
# Future leakage detection
# ---------------------------------------------------------------------------


def validate_no_future_leakage(
    features_df: pl.DataFrame,
    labels_df: pl.DataFrame,
    date_column: str,
    join_key: str = "call_id",
) -> None:
    """
    Assert that no feature row has a date >= the corresponding label's date.

    In an earnings call context, this means verifying that features derived
    from a call dated T are not contaminated with information from after T.

    Args:
        features_df: DataFrame with at least [join_key, date_column].
        labels_df: DataFrame with at least [join_key, date_column].
            The label date is the date when the target (e.g., next-day return)
            is computed — this must be strictly after the feature date.
        date_column: Name of the date column in both DataFrames.
        join_key: Column used to join features and labels (default 'call_id').

    Raises:
        ValueError: If any feature date >= its corresponding label date.
        KeyError: If required columns are missing.
    """
    for name, df in [("features_df", features_df), ("labels_df", labels_df)]:
        for col in [join_key, date_column]:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' missing from {name}.")

    joined = features_df.select([join_key, pl.col(date_column).alias("feature_date")]).join(
        labels_df.select([join_key, pl.col(date_column).alias("label_date")]),
        on=join_key,
        how="inner",
    )

    # Cast to string for consistent comparison
    joined = joined.with_columns([
        pl.col("feature_date").cast(pl.Utf8),
        pl.col("label_date").cast(pl.Utf8),
    ])

    leaky = joined.filter(pl.col("feature_date") >= pl.col("label_date"))

    if leaky.height > 0:
        raise ValueError(
            f"Future leakage detected: {leaky.height} rows where feature_date >= label_date.\n"
            f"First offender:\n{leaky.head(1)}"
        )

    logger.info(
        "Leakage check passed: %d joined rows, no future leakage detected.",
        joined.height,
    )


# ---------------------------------------------------------------------------
# Window overlap check
# ---------------------------------------------------------------------------


def check_overlapping_windows(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    date_column: str,
    window_days: int = 5,
) -> bool:
    """
    Return True if any train and test dates fall within `window_days` of each other.

    Used to detect cases where a rolling window feature (e.g., 5-day return)
    computed from a test-set call might have used data that was also in training.

    Args:
        train_df: Training DataFrame with a date column.
        test_df: Test DataFrame with a date column.
        date_column: Name of the date column.
        window_days: If any test date is within `window_days` of any train date,
                     the function returns True (overlap detected).

    Returns:
        True if overlap detected, False otherwise.
    """
    for name, df in [("train_df", train_df), ("test_df", test_df)]:
        if date_column not in df.columns:
            raise KeyError(f"Column '{date_column}' missing from {name}.")

    def to_date_series(df: pl.DataFrame) -> pl.Series:
        col = df[date_column]
        if col.dtype == pl.Utf8:
            return col.str.strptime(pl.Date, "%Y-%m-%d")
        return col.cast(pl.Date)

    train_dates = to_date_series(train_df)
    test_dates = to_date_series(test_df)

    train_max = train_dates.max()
    test_min = test_dates.min()

    if train_max is None or test_min is None:
        logger.warning("Empty train or test set — cannot check overlap.")
        return False

    import datetime
    gap = (test_min - train_max).days  # type: ignore[operator]

    if gap < window_days:
        logger.warning(
            "Window overlap detected: gap between train_max (%s) and test_min (%s) "
            "is %d days, which is less than window_days=%d.",
            train_max, test_min, gap, window_days,
        )
        return True

    logger.info(
        "No window overlap: gap=%d days (train_max=%s, test_min=%s, window=%d days).",
        gap, train_max, test_min, window_days,
    )
    return False
