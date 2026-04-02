# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:57:06 2026

@author: Adams
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def trim_signal_and_time(
    time_series: pd.Series,
    value_series: pd.Series,
) -> tuple[pd.Series, pd.Series, dict[str, int]]:
    """
    Align time and values to shared length, then trim NaN edges based on the value array.
    """
    n = min(len(time_series), len(value_series))
    time_series = time_series.iloc[:n].reset_index(drop=True)
    value_series = value_series.iloc[:n].reset_index(drop=True)

    if n == 0:
        return time_series.iloc[0:0], value_series.iloc[0:0], {"head": 0, "tail": 0}

    valid = value_series.notna().to_numpy()
    if not valid.any():
        return time_series.iloc[0:0], value_series.iloc[0:0], {"head": n, "tail": 0}

    first = int(np.argmax(valid))
    last = int(n - 1 - np.argmax(valid[::-1]))

    return (
        time_series.iloc[first:last + 1].reset_index(drop=True),
        value_series.iloc[first:last + 1].reset_index(drop=True),
        {"head": first, "tail": n - 1 - last},
    )


def mad_stats(y: np.ndarray, c: float = 1.4826) -> tuple[float, float]:
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    if not finite.any():
        return np.nan, 0.0
    med = np.nanmedian(y[finite])
    mad = np.nanmedian(np.abs(y[finite] - med))
    return float(med), float(c * mad)


def remove_mad_outliers(
    y: np.ndarray,
    *,
    k: float = 6.0,
    fill: str = "interp",  # 'interp' | 'nan' | 'ffill' | 'bfill'
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    y = np.asarray(y, dtype=float)
    med, mad_scaled = mad_stats(y)

    stats = {
        "count": 0,
        "pct": 0.0,
        "median": med,
        "mad_scaled": mad_scaled,
        "k": float(k),
        "fill": fill,
    }

    if not np.isfinite(med) or mad_scaled <= 0:
        return y, stats

    mask = np.abs(y - med) > (k * mad_scaled)
    n_out = int(mask.sum())
    stats["count"] = n_out
    stats["pct"] = (100.0 * n_out / y.size) if y.size else 0.0

    if n_out == 0:
        return y, stats

    s = pd.Series(y.copy())
    s.loc[mask] = np.nan

    if fill == "nan":
        out = s.to_numpy(dtype=float)
    elif fill == "interp":
        out = s.interpolate(method="linear", limit_direction="both").to_numpy(dtype=float)
    elif fill == "ffill":
        out = s.ffill().bfill().to_numpy(dtype=float)
    elif fill == "bfill":
        out = s.bfill().ffill().to_numpy(dtype=float)
    else:
        raise ValueError("fill must be one of {'interp', 'nan', 'ffill', 'bfill'}")

    return out, stats