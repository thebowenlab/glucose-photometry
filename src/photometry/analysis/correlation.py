from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from scipy.stats import rankdata

from photometry.analysis._numeric import time_derivative
from photometry.analysis._session import (
    SessionLike,
    get_aligned_session_arrays,
    infer_dt_minutes,
)
from photometry.signal.fp_series import lowpass_minutes


@dataclass(slots=True)
class CrossCorrelationConfig:
    """Configuration for lagged correlation within one session.

    Positive lag means ``x`` occurs later than ``y``: the calculation compares
    ``x(t + lag)`` with ``y(t)``.
    """

    x_key: str = "fp"
    y_key: str = "bg"
    y_transform: str = "raw"  # 'raw' or 'derivative'
    max_lag_min: float = 20.0
    method: str = "spearman"  # 'spearman' or 'pearson'
    min_overlap: int = 10
    x_lowpass_period_min: float | None = 4.0
    x_lowpass_order: int = 3
    y_derivative_smooth_minutes: float = 1.0


def _rank_finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    ranked = np.full_like(values, np.nan, dtype=float)
    finite = np.isfinite(values)
    if np.any(finite):
        ranked[finite] = rankdata(values[finite], method="average")
    return ranked


def _pearson_pair(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return np.nan
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    x_scale = float(np.sqrt(np.mean(x_centered**2)))
    y_scale = float(np.sqrt(np.mean(y_centered**2)))
    if x_scale <= 0 or y_scale <= 0:
        return np.nan
    return float(np.mean((x_centered / x_scale) * (y_centered / y_scale)))


def _lagged_correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_lag_samples: int,
    method: str,
    min_overlap: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have matching shapes")

    method_normalized = method.strip().lower()
    if method_normalized == "spearman":
        x_work = _rank_finite(x)
        y_work = _rank_finite(y)
    elif method_normalized == "pearson":
        x_work = x
        y_work = y
    else:
        raise ValueError("method must be 'spearman' or 'pearson'")

    sample_lags = np.arange(-max_lag_samples, max_lag_samples + 1, dtype=int)
    correlations = np.full(sample_lags.size, np.nan, dtype=float)
    overlaps = np.zeros(sample_lags.size, dtype=int)
    n = x_work.size

    for index, lag in enumerate(sample_lags):
        if lag >= 0:
            x_segment = x_work[lag:]
            y_segment = y_work[: n - lag]
        else:
            x_segment = x_work[: n + lag]
            y_segment = y_work[-lag:]

        valid = np.isfinite(x_segment) & np.isfinite(y_segment)
        overlaps[index] = int(np.count_nonzero(valid))
        if overlaps[index] >= int(min_overlap):
            correlations[index] = _pearson_pair(
                x_segment[valid],
                y_segment[valid],
            )

    return correlations, overlaps


def compute_cross_correlation(
    session: SessionLike,
    *,
    config: CrossCorrelationConfig | None = None,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute a lagged correlation curve from one canonical session.

    ``x`` and ``y`` may override the configured session fields. This allows the
    same function to analyze model residuals or predictions while still using the
    session's canonical time axis.
    """
    cfg = config or CrossCorrelationConfig()
    t = get_aligned_session_arrays(session, ("t",), min_length=3)[0]

    if x is None:
        _, x_values = get_aligned_session_arrays(
            session,
            ("t", cfg.x_key),
            min_length=3,
        )
    else:
        x_values = np.asarray(x, dtype=float)
        if x_values.shape != t.shape:
            raise ValueError("x override must have the same shape as session['t']")

    if y is None:
        _, y_values = get_aligned_session_arrays(
            session,
            ("t", cfg.y_key),
            min_length=3,
        )
    else:
        y_values = np.asarray(y, dtype=float)
        if y_values.shape != t.shape:
            raise ValueError("y override must have the same shape as session['t']")

    if cfg.x_lowpass_period_min is not None:
        x_processed = lowpass_minutes(
            t,
            x_values,
            cutoff_period_min=cfg.x_lowpass_period_min,
            order=cfg.x_lowpass_order,
        )
    else:
        x_processed = x_values.copy()

    transform = cfg.y_transform.strip().lower()
    if transform == "raw":
        y_processed = y_values.copy()
    elif transform == "derivative":
        y_processed = time_derivative(
            t,
            y_values,
            smooth_minutes=cfg.y_derivative_smooth_minutes,
        )
    else:
        raise ValueError("y_transform must be 'raw' or 'derivative'")

    dt = infer_dt_minutes(t)
    max_lag_samples = max(0, int(round(float(cfg.max_lag_min) / dt)))
    correlations, overlaps = _lagged_correlation(
        x_processed,
        y_processed,
        max_lag_samples=max_lag_samples,
        method=cfg.method,
        min_overlap=cfg.min_overlap,
    )
    lags_min = np.arange(-max_lag_samples, max_lag_samples + 1) * dt

    finite = np.isfinite(correlations)
    if np.any(finite):
        candidates = np.where(finite, np.abs(correlations), np.nan)
        best_index = int(np.nanargmax(candidates))
        best_lag_min = float(lags_min[best_index])
        best_correlation = float(correlations[best_index])
    else:
        best_index = -1
        best_lag_min = np.nan
        best_correlation = np.nan

    return {
        "lags_min": lags_min,
        "correlation": correlations,
        "overlap_count": overlaps,
        "best_index": best_index,
        "best_lag_min": best_lag_min,
        "best_correlation": best_correlation,
        "dt_min": dt,
        "lag_convention": "positive lag means x occurs later than y",
        "config": asdict(cfg),
        "series": {
            "t": t.copy(),
            "x": x_values.copy(),
            "y": y_values.copy(),
            "x_processed": x_processed,
            "y_processed": y_processed,
        },
    }


def compute_session_cross_correlations(
    session: SessionLike,
    *,
    config: CrossCorrelationConfig | None = None,
    x: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute FP-to-glucose and FP-to-glucose-derivative curves."""
    cfg = config or CrossCorrelationConfig()
    raw = compute_cross_correlation(
        session,
        config=replace(cfg, y_transform="raw"),
        x=x,
    )
    derivative = compute_cross_correlation(
        session,
        config=replace(cfg, y_transform="derivative"),
        x=x,
    )
    return {"glucose": raw, "glucose_derivative": derivative}
