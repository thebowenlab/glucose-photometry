from __future__ import annotations

import numpy as np

from photometry.analysis._session import finite_interp, infer_dt_minutes


def rolling_mean_causal(values: np.ndarray, window_samples: int) -> np.ndarray:
    """Causal moving average with a progressively shorter leading window."""
    x = np.asarray(values, dtype=float)
    window = max(int(window_samples), 1)
    if window == 1 or x.size == 0:
        return x.copy()

    finite = np.isfinite(x)
    filled = np.where(finite, x, 0.0)
    csum = np.cumsum(filled)
    counts = np.cumsum(finite.astype(float))

    out = np.empty_like(x, dtype=float)
    for i in range(x.size):
        start = max(0, i - window + 1)
        total = csum[i] - (csum[start - 1] if start > 0 else 0.0)
        count = counts[i] - (counts[start - 1] if start > 0 else 0.0)
        out[i] = total / count if count > 0 else np.nan
    return out


def time_derivative(
    t: np.ndarray,
    y: np.ndarray,
    *,
    smooth_minutes: float = 0.0,
    preserve_nans: bool = False,
) -> np.ndarray:
    """Differentiate a one-dimensional signal on a time axis measured in minutes.

    When ``preserve_nans`` is false, missing values are linearly interpolated before
    differentiation. When true, derivatives are computed independently within each
    contiguous finite run and remain NaN elsewhere.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.shape != y.shape:
        raise ValueError("t and y must have matching shapes")
    if t.size < 3:
        raise ValueError("at least three samples are required to compute a derivative")

    if preserve_nans:
        derivative = np.full_like(y, np.nan, dtype=float)
        finite = np.isfinite(t) & np.isfinite(y)
        idx = np.flatnonzero(finite)
        if idx.size >= 3:
            split_at = np.flatnonzero(np.diff(idx) > 1)
            starts = np.r_[0, split_at + 1]
            stops = np.r_[split_at + 1, idx.size]
            for start, stop in zip(starts, stops):
                run = idx[start:stop]
                if run.size >= 3:
                    derivative[run] = np.gradient(y[run], t[run])
    else:
        y_filled = finite_interp(t, y)
        derivative = np.gradient(y_filled, t).astype(float)

    if smooth_minutes > 0:
        dt = infer_dt_minutes(t)
        window = max(1, int(round(float(smooth_minutes) / dt)))
        derivative = rolling_mean_causal(derivative, window)
        if preserve_nans:
            derivative[~(np.isfinite(t) & np.isfinite(y))] = np.nan

    return derivative


def shifted_template(
    template_t: np.ndarray,
    template_y: np.ndarray,
    target_t: np.ndarray,
    lag_min: float,
    *,
    fill: str = "hold",
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate ``template(t - lag)`` on ``target_t``.

    Positive lag means the evaluated template occurs later than the unshifted
    template. The returned mask marks samples whose source times lie within the
    original template domain.
    """
    template_t = np.asarray(template_t, dtype=float)
    template_y = np.asarray(template_y, dtype=float)
    target_t = np.asarray(target_t, dtype=float)
    if template_t.ndim != 1 or template_y.ndim != 1 or target_t.ndim != 1:
        raise ValueError("template_t, template_y, and target_t must be one-dimensional")
    if template_t.size != template_y.size:
        raise ValueError("template_t and template_y must have matching lengths")
    if template_t.size < 2:
        raise ValueError("template must contain at least two samples")

    source_t = target_t - float(lag_min)
    valid = (source_t >= template_t[0]) & (source_t <= template_t[-1])

    if fill == "hold":
        left = float(template_y[0])
        right = float(template_y[-1])
    elif fill == "zero":
        left = right = 0.0
    elif fill == "nan":
        left = right = np.nan
    else:
        raise ValueError("fill must be 'hold', 'zero', or 'nan'")

    shifted = np.interp(
        source_t,
        template_t,
        template_y,
        left=left,
        right=right,
    ).astype(float)
    return shifted, valid
