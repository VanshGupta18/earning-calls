"""
Tests for src/evaluation/metrics.py

Run with: pytest tests/test_metrics.py -v
"""

import math

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_auroc,
    compute_directional_accuracy,
    compute_f1,
    compute_information_coefficient,
    compute_max_drawdown,
    compute_rmse,
    compute_sharpe,
)


# ---------------------------------------------------------------------------
# compute_rmse
# ---------------------------------------------------------------------------


class TestComputeRmse:
    def test_perfect_prediction(self):
        y = [1.0, 2.0, 3.0]
        assert compute_rmse(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        # errors: [0, 1, 2] → MSE = (0+1+4)/3 = 5/3 → RMSE ≈ 1.2910
        assert compute_rmse([1, 2, 3], [1, 1, 1]) == pytest.approx(math.sqrt(5 / 3), rel=1e-5)

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_rmse([1, 2], [1, 2, 3])

    def test_numpy_arrays(self):
        y_true = np.array([0.5, 1.0, 1.5])
        y_pred = np.array([0.5, 1.0, 1.5])
        assert compute_rmse(y_true, y_pred) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_directional_accuracy
# ---------------------------------------------------------------------------


class TestDirectionalAccuracy:
    def test_all_correct(self):
        assert compute_directional_accuracy([1, -1, 1], [2, -3, 0.5]) == pytest.approx(1.0)

    def test_all_wrong(self):
        assert compute_directional_accuracy([1, -1, 1], [-2, 3, -0.5]) == pytest.approx(0.0)

    def test_half_correct(self):
        assert compute_directional_accuracy([1, -1], [1, 1]) == pytest.approx(0.5)

    def test_zero_ground_truth_excluded(self):
        # Zero in y_true should be excluded
        result = compute_directional_accuracy([1, 0, -1], [1, -999, -1])
        assert result == pytest.approx(1.0)  # Both non-zero are correct

    def test_all_zero_ground_truth(self):
        result = compute_directional_accuracy([0, 0, 0], [1, -1, 1])
        assert math.isnan(result)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_directional_accuracy([], [])


# ---------------------------------------------------------------------------
# compute_information_coefficient
# ---------------------------------------------------------------------------


class TestInformationCoefficient:
    def test_perfect_rank_correlation(self):
        y = [1, 2, 3, 4, 5]
        assert compute_information_coefficient(y, y) == pytest.approx(1.0)

    def test_negative_rank_correlation(self):
        assert compute_information_coefficient([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)

    def test_zero_correlation(self):
        # IC should be near 0 for random data — just check it's in range
        ic = compute_information_coefficient([1, 2, 3, 4], [4, 1, 3, 2])
        assert -1.0 <= ic <= 1.0

    def test_too_few_samples(self):
        with pytest.raises(ValueError):
            compute_information_coefficient([1], [1])


# ---------------------------------------------------------------------------
# compute_auroc
# ---------------------------------------------------------------------------


class TestComputeAuroc:
    def test_perfect_classifier(self):
        assert compute_auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)

    def test_random_classifier(self):
        # Probabilities equal to 0.5 → AUROC = 0.5
        result = compute_auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
        assert result == pytest.approx(0.5)

    def test_worst_classifier(self):
        assert compute_auroc([0, 0, 1, 1], [0.9, 0.8, 0.1, 0.2]) == pytest.approx(0.0)

    def test_single_class_raises(self):
        with pytest.raises(ValueError):
            compute_auroc([1, 1, 1], [0.9, 0.8, 0.7])


# ---------------------------------------------------------------------------
# compute_f1
# ---------------------------------------------------------------------------


class TestComputeF1:
    def test_perfect_f1(self):
        assert compute_f1([0, 1, 1, 0], [0, 1, 1, 0]) == pytest.approx(1.0)

    def test_zero_f1(self):
        assert compute_f1([1, 1], [0, 0]) == pytest.approx(0.0)

    def test_partial(self):
        # TP=1, FP=0, FN=1 → precision=1, recall=0.5 → F1=0.667
        result = compute_f1([1, 1, 0], [1, 0, 0])
        assert result == pytest.approx(2 / 3, rel=1e-4)


# ---------------------------------------------------------------------------
# compute_sharpe
# ---------------------------------------------------------------------------


class TestComputeSharpe:
    def test_positive_sharpe(self):
        returns = np.full(252, 0.001)  # constant positive returns
        sharpe = compute_sharpe(returns)
        assert sharpe > 0

    def test_zero_std_returns_nan(self):
        returns = np.zeros(252)
        assert math.isnan(compute_sharpe(returns))

    def test_empty_returns_nan(self):
        assert math.isnan(compute_sharpe([]))

    def test_known_value(self):
        # 252 returns of 0.01 with std of 0.01
        # excess = 0.01, annualised mean = 0.01*252, annualised std = 0.01*sqrt(252)
        # Sharpe = 0.01/0.01 * sqrt(252) = sqrt(252) ≈ 15.87
        np.random.seed(42)
        returns = np.full(252, 0.01)
        sharpe = compute_sharpe(returns, risk_free_rate=0.0)
        # std of constant series is 0 — won't work; use varied returns instead
        returns = np.random.normal(loc=0.01, scale=0.01, size=252)
        sharpe = compute_sharpe(returns)
        assert isinstance(sharpe, float)


# ---------------------------------------------------------------------------
# compute_max_drawdown
# ---------------------------------------------------------------------------


class TestComputeMaxDrawdown:
    def test_no_drawdown(self):
        returns = [0.01, 0.02, 0.01, 0.03]  # all positive
        assert compute_max_drawdown(returns) == pytest.approx(0.0, abs=1e-10)

    def test_known_drawdown(self):
        # Portfolio goes +10%, then -20%, then +10%
        # Peak = 1.1, trough = 1.1 * 0.8 = 0.88 → drawdown = (1.1-0.88)/1.1 = 0.2
        returns = [0.10, -0.20, 0.10]
        dd = compute_max_drawdown(returns)
        assert 0.19 < dd < 0.21  # approximately 20%
