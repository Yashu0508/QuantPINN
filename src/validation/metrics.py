"""Metrics used for chronological option-price validation."""

from __future__ import annotations

import numpy as np


def regression_metrics(actual, predicted) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("actual and predicted must have the same non-empty shape")
    error = predicted - actual
    relative = np.abs(error) / np.maximum(np.abs(actual), 1e-12)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_abs_error": float(np.max(np.abs(error))),
        "mean_absolute_relative_error": float(np.mean(relative)),
        "n": float(actual.size),
    }
