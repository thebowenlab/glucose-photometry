# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:57:43 2026

@author: Adams
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def rolling_median(y: np.ndarray, window: int) -> np.ndarray:
    window = max(int(window), 1)
    return pd.Series(y).rolling(window, center=True, min_periods=1).median().to_numpy(dtype=float)


def rolling_mean_causal(y: np.ndarray, window: int) -> np.ndarray:
    window = max(int(window), 1)
    return pd.Series(y).rolling(window, center=False, min_periods=1).mean().to_numpy(dtype=float)


def savgol_safe(y: np.ndarray, window: int, polyorder: int = 3) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y

    mask = ~np.isfinite(y)
    s = pd.Series(y).interpolate(method="linear", limit_direction="both")

    window = max(int(window), polyorder + 2)
    if window % 2 == 0:
        window += 1

    try:
        out = savgol_filter(s.to_numpy(dtype=float), window_length=window, polyorder=polyorder, mode="interp")
    except ValueError:
        out = pd.Series(s).rolling(window, center=True, min_periods=1).mean().to_numpy(dtype=float)

    out[mask] = np.nan
    return out