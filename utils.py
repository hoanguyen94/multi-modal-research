"""Shared evaluation and price-conversion utilities for forecasting notebooks."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)


ArrayLike = Any


def _float_array(values: ArrayLike) -> np.ndarray:
    """Convert array-like input to a NumPy float64 array without flattening it."""
    return np.asarray(values, dtype=np.float64)


def prices_to_log_returns(origin_price: ArrayLike, target_price: ArrayLike) -> np.ndarray:
    """Calculate log(target/origin), requiring finite, strictly positive prices."""
    origin = _float_array(origin_price)
    target = _float_array(target_price)
    origin, target = np.broadcast_arrays(origin, target)
    if np.any(~np.isfinite(origin)) or np.any(~np.isfinite(target)):
        raise ValueError("Prices must be finite.")
    if np.any(origin <= 0) or np.any(target <= 0):
        raise ValueError("Prices must be strictly positive.")
    return np.log(target / origin)


def returns_to_direction(log_returns: ArrayLike, *, tie_is_up: bool = False) -> np.ndarray:
    """Convert log returns to binary directions; zero is down unless tie_is_up=True."""
    values = _float_array(log_returns)
    return (values >= 0 if tie_is_up else values > 0).astype(np.int8)


def prices_to_direction(
    origin_price: ArrayLike, target_price: ArrayLike, *, tie_is_up: bool = False
) -> np.ndarray:
    """Convert origin and target prices directly to binary directional labels."""
    return returns_to_direction(
        prices_to_log_returns(origin_price, target_price), tie_is_up=tie_is_up
    )


def predicted_direction(probability_up: ArrayLike, threshold: ArrayLike = 0.5) -> np.ndarray:
    """Classify upward probabilities using a scalar or row-specific threshold."""
    probability = _float_array(probability_up)
    cutoff = _float_array(threshold)
    probability, cutoff = np.broadcast_arrays(probability, cutoff)
    if np.any(~np.isfinite(probability)) or np.any((probability < 0) | (probability > 1)):
        raise ValueError("probability_up must contain finite values in [0, 1].")
    if np.any(~np.isfinite(cutoff)) or np.any((cutoff <= 0) | (cutoff >= 1)):
        raise ValueError("threshold must contain finite values strictly between 0 and 1.")
    return (probability >= cutoff).astype(np.int8)


def probability_to_direction_score(
    probability_up: ArrayLike, threshold: ArrayLike = 0.5
) -> np.ndarray:
    """Map probabilities to [-1, 1], with the decision threshold mapped to zero."""
    probability = _float_array(probability_up)
    cutoff = _float_array(threshold)
    probability, cutoff = np.broadcast_arrays(probability, cutoff)
    # Reuse validation and preserve the original broadcast shape.
    predicted_direction(probability, cutoff)
    return np.where(
        probability >= cutoff,
        (probability - cutoff) / (1.0 - cutoff),
        (probability - cutoff) / cutoff,
    )


def direction_to_log_return(direction: ArrayLike, return_scale: ArrayLike) -> np.ndarray:
    """Assign -scale to down predictions and +scale to up predictions."""
    labels = np.asarray(direction)
    scale = _float_array(return_scale)
    labels, scale = np.broadcast_arrays(labels, scale)
    if np.any(~np.isin(labels, (0, 1))):
        raise ValueError("direction must contain only 0 (down) and 1 (up).")
    if np.any(~np.isfinite(scale)) or np.any(scale < 0):
        raise ValueError("return_scale must be finite and non-negative.")
    return (2.0 * labels.astype(np.float64) - 1.0) * scale


def probability_to_log_return(
    probability_up: ArrayLike,
    return_scale: ArrayLike,
    threshold: ArrayLike = 0.5,
) -> np.ndarray:
    """Convert upward probabilities to signed log returns with bounded magnitude."""
    scale = _float_array(return_scale)
    if np.any(~np.isfinite(scale)) or np.any(scale < 0):
        raise ValueError("return_scale must be finite and non-negative.")
    return probability_to_direction_score(probability_up, threshold) * scale


def log_returns_to_prices(origin_price: ArrayLike, log_return: ArrayLike) -> np.ndarray:
    """Convert log returns to positive target prices: target = origin * exp(return)."""
    origin = _float_array(origin_price)
    returns = _float_array(log_return)
    origin, returns = np.broadcast_arrays(origin, returns)
    if np.any(~np.isfinite(origin)) or np.any(origin <= 0):
        raise ValueError("origin_price must be finite and strictly positive.")
    if np.any(~np.isfinite(returns)):
        raise ValueError("log_return must be finite.")
    return origin * np.exp(returns)


def probability_to_price(
    origin_price: ArrayLike,
    probability_up: ArrayLike,
    return_scale: ArrayLike,
    threshold: ArrayLike = 0.5,
) -> np.ndarray:
    """Convert a directional probability to price via a threshold-centered log return."""
    return log_returns_to_prices(
        origin_price,
        probability_to_log_return(probability_up, return_scale, threshold),
    )


def directional_classification_metrics(
    y_true: ArrayLike,
    probability_up: ArrayLike,
    threshold: ArrayLike = 0.5,
) -> dict[str, float]:
    """Return hit rate, balanced accuracy, ROC AUC, and Brier score."""
    truth = np.asarray(y_true, dtype=np.int8).reshape(-1)
    probability = _float_array(probability_up).reshape(-1)
    if truth.size != probability.size:
        raise ValueError("y_true and probability_up must have equal length.")
    if np.any(~np.isin(truth, (0, 1))):
        raise ValueError("y_true must contain only 0 and 1.")
    prediction = predicted_direction(probability, threshold).reshape(-1)
    clipped = np.clip(probability, 0.0, 1.0)
    auc = roc_auc_score(truth, clipped) if np.unique(truth).size == 2 else float("nan")
    hit_rate = float(accuracy_score(truth, prediction))
    return {
        "hit_rate": hit_rate,
        "accuracy": hit_rate,
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "roc_auc": float(auc),
        "brier_score": float(brier_score_loss(truth, clipped)),
    }
