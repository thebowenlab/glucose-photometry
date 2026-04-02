# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:56:30 2026

@author: Adams
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def coerce_time_series(s: pd.Series) -> tuple[pd.Series, str]:
    """
    Returns:
      (series, kind)
    where kind is one of: 'numeric', 'datetime', 'index'

    More tolerant than strict all-or-none coercion:
    - if most values are numeric, keep numeric and fill gaps by interpolation
    - if most values are datetime, keep datetime
    - otherwise fall back to sample index
    """
    if s.empty:
        return pd.Series([], dtype=float), "index"

    # Try numeric first
    num = pd.to_numeric(s, errors="coerce")
    frac_num = float(num.notna().mean()) if len(num) else 0.0
    if frac_num > 0.5:
        if num.notna().sum() >= 2:
            num = num.interpolate(method="linear", limit_direction="both")
        elif num.notna().sum() == 1:
            # If only one valid point exists, just forward/back fill
            num = num.ffill().bfill()
        else:
            return pd.Series(np.arange(len(s), dtype=float)), "index"
        return num.astype(float), "numeric"

    # Try datetime
    dt = pd.to_datetime(s, errors="coerce")
    frac_dt = float(dt.notna().mean()) if len(dt) else 0.0
    if frac_dt > 0.5:
        return dt, "datetime"

    return pd.Series(np.arange(len(s), dtype=float)), "index"


def infer_dt(arr: np.ndarray) -> float | None:
    arr = np.asarray(arr, dtype=float)
    if arr.size < 2:
        return None
    dt = np.diff(arr)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return None
    return float(np.median(dt))


def to_seconds(
    series: pd.Series,
    kind: str,
    *,
    numeric_units: str = "minutes",
    relative: bool = True,
) -> tuple[np.ndarray, float | None]:
    """
    Convert time to seconds.
    numeric_units: 'seconds' or 'minutes'
    """
    if kind == "datetime":
        arr = (pd.to_datetime(series).astype("int64") / 1e9).to_numpy(dtype=float)
        if relative and arr.size > 0:
            arr = arr - arr[0]
        return arr, infer_dt(arr)

    if kind == "numeric":
        arr = series.to_numpy(dtype=float)
        if numeric_units == "minutes":
            arr = arr * 60.0
        elif numeric_units != "seconds":
            raise ValueError("numeric_units must be 'minutes' or 'seconds'")
        if relative and arr.size > 0:
            arr = arr - arr[0]
        return arr, infer_dt(arr)

    arr = np.arange(len(series), dtype=float)
    return arr, infer_dt(arr)


def odd_window_for_seconds(win_sec: float, dt_sec: float | None, *, min_samples: int = 3) -> int:
    if dt_sec is None or dt_sec <= 0:
        n = max(min_samples, 1)
    else:
        n = max(int(round(win_sec / dt_sec)), min_samples)
    return n if n % 2 == 1 else n + 1