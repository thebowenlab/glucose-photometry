from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


SessionLike = Mapping[str, Any]


def get_session_array(
    session: SessionLike,
    key: str,
    *,
    min_length: int = 1,
) -> np.ndarray:
    """Return a one-dimensional floating-point array from a canonical session."""
    if not isinstance(session, Mapping):
        raise TypeError("session must be a mapping containing canonical session fields")
    if key not in session:
        raise KeyError(f"session is missing required field {key!r}")

    value = np.asarray(session[key], dtype=float)
    if value.ndim != 1:
        raise ValueError(f"session field {key!r} must be one-dimensional")
    if value.size < min_length:
        raise ValueError(
            f"session field {key!r} must contain at least {min_length} samples"
        )
    return value


def get_aligned_session_arrays(
    session: SessionLike,
    keys: Sequence[str],
    *,
    min_length: int = 3,
    require_strict_time: bool = True,
) -> tuple[np.ndarray, ...]:
    """Validate and return equally sized arrays from one canonical session.

    The first requested field must be ``"t"`` when ``require_strict_time`` is true.
    Canonical sessions loaded by :func:`photometry.load_sessions` already satisfy
    these conditions; the checks make direct calls with user-created dictionaries
    fail early and clearly.
    """
    if not keys:
        raise ValueError("at least one session field must be requested")

    arrays = tuple(get_session_array(session, key, min_length=min_length) for key in keys)
    lengths = {arr.size for arr in arrays}
    if len(lengths) != 1:
        sizes = ", ".join(f"{key}={arr.size}" for key, arr in zip(keys, arrays))
        raise ValueError(f"session arrays must have matching lengths; got {sizes}")

    if require_strict_time:
        if keys[0] != "t":
            raise ValueError("the first requested field must be 't' when validating time")
        t = arrays[0]
        if not np.all(np.isfinite(t)):
            raise ValueError("session time field 't' must contain only finite values")
        if np.any(np.diff(t) <= 0):
            raise ValueError("session time field 't' must be strictly increasing")

    return arrays


def finite_interp(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Linearly fill non-finite values of ``y`` on the supplied time axis."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.shape != y.shape:
        raise ValueError("t and y must have matching shapes")

    finite = np.isfinite(t) & np.isfinite(y)
    if finite.all():
        return y.copy()
    if np.count_nonzero(finite) < 2:
        raise ValueError("at least two finite samples are required for interpolation")
    return np.interp(t, t[finite], y[finite]).astype(float)


def infer_dt_minutes(t: np.ndarray) -> float:
    """Infer a robust positive sample interval in minutes."""
    t = np.asarray(t, dtype=float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("cannot infer a positive time step from t")
    return float(np.median(dt))


def window_mask(
    t: np.ndarray,
    window: tuple[float | None, float | None],
    *,
    finite: np.ndarray | None = None,
) -> np.ndarray:
    """Return a mask for an inclusive absolute-time window."""
    lo, hi = window
    lo_value = -np.inf if lo is None else float(lo)
    hi_value = np.inf if hi is None else float(hi)
    mask = (t >= min(lo_value, hi_value)) & (t <= max(lo_value, hi_value))
    if finite is not None:
        mask &= np.asarray(finite, dtype=bool)
    return mask


def baseline_value(
    t: np.ndarray,
    y: np.ndarray,
    window: tuple[float, float],
    *,
    statistic: str = "median",
) -> float:
    """Compute a finite baseline statistic within an inclusive window."""
    mask = window_mask(t, window, finite=np.isfinite(y))
    if not np.any(mask):
        raise ValueError(f"no finite samples fall inside baseline window {window}")

    values = np.asarray(y, dtype=float)[mask]
    if statistic == "median":
        return float(np.nanmedian(values))
    if statistic == "mean":
        return float(np.nanmean(values))
    raise ValueError("statistic must be 'median' or 'mean'")


def as_template_matrix(templates: Any) -> np.ndarray:
    """Normalize one template or a bank of templates to shape ``[K, T]``."""
    matrix = np.asarray(templates, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2 or matrix.shape[1] < 3:
        raise ValueError("templates must have shape [T] or [K, T] with T >= 3")
    if matrix.shape[0] < 1:
        raise ValueError("at least one template is required")
    return matrix
