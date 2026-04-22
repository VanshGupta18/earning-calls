"""
Evaluation metrics for the Multimodal Earnings Call Intelligence System.

All functions follow a consistent signature:
    compute_*(y_true, y_pred, ...) -> float

Targets:
    - Volatility:       compute_rmse
    - Returns:          compute_directional_accuracy
    - Risk ranking:     compute_information_coefficient
    - Classification:   compute_auroc, compute_f1
    - Trading:          compute_sharpe
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def compute_rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Root Mean Squared Error.

    Primary metric for volatility prediction (realized_vol_1d, realized_vol_5d).

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        RMSE as a float. Lower is better.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}"
        )
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------------------------------------------------------
# Directional accuracy
# ---------------------------------------------------------------------------


def compute_directional_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Fraction of predictions with the correct sign.

    Primary metric for return direction prediction.

    Args:
        y_true: Ground-truth return values (can be positive or negative).
        y_pred: Predicted return values or direction scores.

    Returns:
        Directional accuracy in [0, 1]. 0.5 = random, 1.0 = perfect.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        raise ValueError("Empty arrays provided.")
    correct = np.sign(y_true) == np.sign(y_pred)
    # Exclude zero ground-truth (no direction) from the count
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(correct[mask].mean())


# ---------------------------------------------------------------------------
# Rank correlation
# ---------------------------------------------------------------------------


def compute_information_coefficient(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Information Coefficient (IC) — Spearman rank correlation between
    predicted and actual values.

    Primary metric for risk ranking and factor evaluation.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted scores or values.

    Returns:
        IC in [-1, 1]. 0 = no rank correlation, 1 = perfect rank correlation.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 3:
        raise ValueError("Need at least 3 samples to compute rank correlation.")
    ic, _ = spearmanr(y_true, y_pred)
    return float(ic)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def compute_auroc(y_true: ArrayLike, y_proba: ArrayLike) -> float:
    """
    Area Under the ROC Curve (AUROC).

    Used for binary classification (e.g., return_up / return_down).

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_proba: Predicted probability for the positive class.

    Returns:
        AUROC in [0, 1]. 0.5 = random, 1.0 = perfect.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    if len(np.unique(y_true)) < 2:
        raise ValueError("AUROC requires both positive and negative examples.")
    return float(roc_auc_score(y_true, y_proba))


def compute_f1(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str = "binary",
) -> float:
    """
    F1 score for classification.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels (hard, not probabilities).
        average: Averaging strategy — 'binary', 'macro', 'weighted'.

    Returns:
        F1 score in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


# ---------------------------------------------------------------------------
# Trading / portfolio
# ---------------------------------------------------------------------------


def compute_sharpe(
    returns: ArrayLike,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Annualized Sharpe ratio for a series of periodic returns.

    Args:
        returns: Array of period returns (e.g., daily log or simple returns).
        risk_free_rate: Annualized risk-free rate (default 0.0).
        periods_per_year: Number of periods per year (252 for daily,
                          52 for weekly, 12 for monthly).

    Returns:
        Annualized Sharpe ratio. Higher is better.
        Returns NaN if standard deviation is zero.
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0:
        return float("nan")
    excess = returns - (risk_free_rate / periods_per_year)
    std = np.std(excess, ddof=1)
    if std == 0:
        return float("nan")
    return float((np.mean(excess) / std) * np.sqrt(periods_per_year))


def compute_max_drawdown(returns: ArrayLike) -> float:
    """
    Maximum drawdown of a cumulative return series.

    Args:
        returns: Array of period simple returns.

    Returns:
        Maximum drawdown as a positive float (e.g., 0.15 = 15% drawdown).
    """
    returns = np.asarray(returns, dtype=float)
    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / peak
    return float(np.max(drawdown))
