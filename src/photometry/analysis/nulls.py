from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from photometry.analysis._numeric import time_derivative
from photometry.analysis._session import (
    SessionLike,
    get_aligned_session_arrays,
    infer_dt_minutes,
)
from photometry.analysis.correlation import (
    CrossCorrelationConfig,
    _lagged_correlation,
    compute_cross_correlation,
)


@dataclass(slots=True)
class CircularShiftConfig:
    """Configuration for within-session circular-shift null generation."""

    signal_key: str = "bg"
    n_null: int = 200
    min_shift_min: float = 20.0
    max_shift_min: float | None = None
    seed: int | None = 0
    replace: bool = True


def _candidate_shifts(
    n_samples: int,
    dt_min: float,
    min_shift_min: float,
    max_shift_min: float | None,
) -> np.ndarray:
    if n_samples < 3:
        raise ValueError("at least three samples are required for circular shifts")

    min_samples = max(1, int(np.ceil(float(min_shift_min) / dt_min)))
    maximum_possible = n_samples // 2
    if max_shift_min is None:
        max_samples = maximum_possible
    else:
        max_samples = min(
            maximum_possible,
            int(np.floor(float(max_shift_min) / dt_min)),
        )

    if max_samples < min_samples:
        raise ValueError(
            "the requested minimum circular shift is too large for this session"
        )

    raw = np.arange(1, n_samples, dtype=int)
    circular_distance = np.minimum(raw, n_samples - raw)
    return raw[
        (circular_distance >= min_samples)
        & (circular_distance <= max_samples)
    ]


def generate_circular_shift_nulls(
    session: SessionLike,
    *,
    config: CircularShiftConfig | None = None,
    signal: np.ndarray | None = None,
) -> dict[str, Any]:
    """Create circularly shifted versions of one signal from one session.

    Returned null signals have shape ``[n_null, n_samples]``. Shift metadata is
    reported as signed shifts in ``[-n/2, n/2]`` for easier interpretation, while
    NumPy's non-negative roll indices are retained as ``roll_samples``.
    """
    cfg = config or CircularShiftConfig()
    t = get_aligned_session_arrays(session, ("t",), min_length=3)[0]

    if signal is None:
        _, values = get_aligned_session_arrays(
            session,
            ("t", cfg.signal_key),
            min_length=3,
        )
    else:
        values = np.asarray(signal, dtype=float)
        if values.shape != t.shape:
            raise ValueError("signal override must have the same shape as session['t']")

    if cfg.n_null < 1:
        raise ValueError("n_null must be at least 1")

    dt = infer_dt_minutes(t)
    candidates = _candidate_shifts(
        values.size,
        dt,
        cfg.min_shift_min,
        cfg.max_shift_min,
    )
    if not cfg.replace and cfg.n_null > candidates.size:
        raise ValueError(
            "n_null exceeds the number of distinct valid shifts; use replace=True"
        )

    rng = np.random.default_rng(cfg.seed)
    roll_samples = rng.choice(
        candidates,
        size=int(cfg.n_null),
        replace=bool(cfg.replace),
    ).astype(int)
    null_signals = np.vstack([np.roll(values, int(k)) for k in roll_samples])

    signed_samples = roll_samples.copy()
    signed_samples[signed_samples > values.size // 2] -= values.size
    effective_samples = np.minimum(roll_samples, values.size - roll_samples)

    return {
        "null_signals": null_signals,
        "roll_samples": roll_samples,
        "signed_shift_samples": signed_samples,
        "signed_shift_min": signed_samples.astype(float) * dt,
        "effective_shift_min": effective_samples.astype(float) * dt,
        "dt_min": dt,
        "source_signal": values.copy(),
        "config": asdict(cfg),
    }


def compute_circular_shift_correlation_null(
    session: SessionLike,
    *,
    correlation_config: CrossCorrelationConfig | None = None,
    null_config: CircularShiftConfig | None = None,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate a lagged-correlation curve against within-session shift nulls."""
    corr_cfg = correlation_config or CrossCorrelationConfig()
    null_cfg = null_config or CircularShiftConfig(signal_key=corr_cfg.y_key)

    real = compute_cross_correlation(
        session,
        config=corr_cfg,
        x=x,
        y=y,
    )
    shift_result = generate_circular_shift_nulls(
        session,
        config=null_cfg,
        signal=y,
    )

    x_processed = real["series"]["x_processed"]
    dt = float(real["dt_min"])
    max_lag_samples = (real["lags_min"].size - 1) // 2
    null_curves = np.full(
        (null_cfg.n_null, real["lags_min"].size),
        np.nan,
        dtype=float,
    )

    transform = corr_cfg.y_transform.strip().lower()
    for index, shifted_y in enumerate(shift_result["null_signals"]):
        if transform == "raw":
            y_processed = shifted_y
        elif transform == "derivative":
            y_processed = time_derivative(
                real["series"]["t"],
                shifted_y,
                smooth_minutes=corr_cfg.y_derivative_smooth_minutes,
            )
        else:
            raise ValueError("y_transform must be 'raw' or 'derivative'")

        curve, _ = _lagged_correlation(
            x_processed,
            y_processed,
            max_lag_samples=max_lag_samples,
            method=corr_cfg.method,
            min_overlap=corr_cfg.min_overlap,
        )
        null_curves[index] = curve

    observed = real["correlation"]
    abs_null = np.abs(null_curves)
    abs_observed = np.abs(observed)
    pointwise_p = (
        1.0 + np.sum(abs_null >= abs_observed[None, :], axis=0)
    ) / (null_curves.shape[0] + 1.0)

    max_abs_null = np.nanmax(abs_null, axis=1)
    familywise_p = (
        1.0 + np.sum(max_abs_null[:, None] >= abs_observed[None, :], axis=0)
    ) / (null_curves.shape[0] + 1.0)
    invalid_observed = ~np.isfinite(observed)
    pointwise_p[invalid_observed] = np.nan
    familywise_p[invalid_observed] = np.nan

    lower = np.nanpercentile(null_curves, 2.5, axis=0)
    upper = np.nanpercentile(null_curves, 97.5, axis=0)

    return {
        "observed": real,
        "null_correlation": null_curves,
        "pointwise_p_two_sided": pointwise_p,
        "familywise_p_max_abs": familywise_p,
        "null_lower_2_5": lower,
        "null_upper_97_5": upper,
        "shift_metadata": {
            key: value
            for key, value in shift_result.items()
            if key != "null_signals"
        },
        "dt_min": dt,
    }
