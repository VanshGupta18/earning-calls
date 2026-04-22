"""
Tests for src/evaluation/leakage.py

Run with: pytest tests/test_leakage.py -v
"""

import pytest
import polars as pl
from datetime import date

from src.evaluation.leakage import (
    time_based_split,
    validate_no_future_leakage,
    check_overlapping_windows,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_df() -> pl.DataFrame:
    """Simple DataFrame with call_id, date, and a value column."""
    return pl.DataFrame({
        "call_id": ["A_2023Q1", "B_2023Q2", "C_2023Q3", "D_2023Q4", "E_2024Q1"],
        "call_date": ["2023-02-01", "2023-05-01", "2023-08-01", "2023-11-01", "2024-02-01"],
        "realized_vol_1d": [0.01, 0.02, 0.015, 0.025, 0.018],
    })


# ---------------------------------------------------------------------------
# time_based_split
# ---------------------------------------------------------------------------


class TestTimeBasedSplit:
    def test_basic_split(self, sample_df):
        train, test = time_based_split(
            sample_df, "call_date", "2023-08-01", "2023-11-01"
        )
        assert set(train["call_id"].to_list()) == {"A_2023Q1", "B_2023Q2", "C_2023Q3"}
        assert set(test["call_id"].to_list()) == {"D_2023Q4", "E_2024Q1"}

    def test_gap_is_dropped(self, sample_df):
        # gap between 2023-06-01 and 2023-09-01 — B goes to train, C to neither
        train, test = time_based_split(
            sample_df, "call_date", "2023-06-01", "2023-09-01"
        )
        assert "C_2023Q3" not in train["call_id"].to_list()
        assert "C_2023Q3" not in test["call_id"].to_list()

    def test_missing_column_raises(self, sample_df):
        with pytest.raises(KeyError, match="nonexistent"):
            time_based_split(sample_df, "nonexistent", "2023-06-01", "2023-09-01")

    def test_reversed_dates_raise(self, sample_df):
        with pytest.raises(ValueError, match="strictly before"):
            time_based_split(sample_df, "call_date", "2023-09-01", "2023-06-01")

    def test_equal_dates_raise(self, sample_df):
        with pytest.raises(ValueError, match="strictly before"):
            time_based_split(sample_df, "call_date", "2023-06-01", "2023-06-01")

    def test_all_in_train(self, sample_df):
        train, test = time_based_split(
            sample_df, "call_date", "2025-01-01", "2025-06-01"
        )
        assert train.height == sample_df.height
        assert test.height == 0

    def test_all_in_test(self, sample_df):
        train, test = time_based_split(
            sample_df, "call_date", "2020-01-01", "2021-01-01"
        )
        assert train.height == 0
        assert test.height == sample_df.height


# ---------------------------------------------------------------------------
# validate_no_future_leakage
# ---------------------------------------------------------------------------


class TestValidateNoFutureLeakage:
    def test_no_leakage_passes(self):
        features = pl.DataFrame({
            "call_id": ["A", "B"],
            "call_date": ["2023-01-15", "2023-04-15"],
        })
        labels = pl.DataFrame({
            "call_id": ["A", "B"],
            "call_date": ["2023-01-16", "2023-04-16"],
        })
        # Should not raise
        validate_no_future_leakage(features, labels, "call_date")

    def test_leakage_raises(self):
        features = pl.DataFrame({
            "call_id": ["A"],
            "call_date": ["2023-01-17"],  # feature date AFTER label date
        })
        labels = pl.DataFrame({
            "call_id": ["A"],
            "call_date": ["2023-01-16"],
        })
        with pytest.raises(ValueError, match="Future leakage detected"):
            validate_no_future_leakage(features, labels, "call_date")

    def test_same_date_raises(self):
        df = pl.DataFrame({
            "call_id": ["A"],
            "call_date": ["2023-01-16"],
        })
        with pytest.raises(ValueError, match="Future leakage detected"):
            validate_no_future_leakage(df, df, "call_date")

    def test_missing_column_raises(self):
        features = pl.DataFrame({"call_id": ["A"], "call_date": ["2023-01-15"]})
        labels = pl.DataFrame({"call_id": ["A"], "wrong_col": ["2023-01-16"]})
        with pytest.raises(KeyError):
            validate_no_future_leakage(features, labels, "call_date")


# ---------------------------------------------------------------------------
# check_overlapping_windows
# ---------------------------------------------------------------------------


class TestCheckOverlappingWindows:
    def test_no_overlap(self):
        train = pl.DataFrame({"call_date": ["2023-01-01", "2023-06-01"]})
        test = pl.DataFrame({"call_date": ["2023-12-01", "2023-12-15"]})
        assert check_overlapping_windows(train, test, "call_date", window_days=5) is False

    def test_overlap_detected(self):
        train = pl.DataFrame({"call_date": ["2023-01-01", "2023-06-28"]})
        test = pl.DataFrame({"call_date": ["2023-06-30", "2023-07-15"]})
        # gap = 2 days < window_days=5 → overlap
        assert check_overlapping_windows(train, test, "call_date", window_days=5) is True

    def test_exactly_at_boundary(self):
        train = pl.DataFrame({"call_date": ["2023-01-01", "2023-06-25"]})
        test = pl.DataFrame({"call_date": ["2023-06-30"]})
        # gap = 5 days == window_days=5 → no overlap (strict less-than)
        result = check_overlapping_windows(train, test, "call_date", window_days=5)
        assert result is False

    def test_missing_column_raises(self):
        train = pl.DataFrame({"call_date": ["2023-01-01"]})
        test = pl.DataFrame({"wrong": ["2023-12-01"]})
        with pytest.raises(KeyError):
            check_overlapping_windows(train, test, "call_date")
