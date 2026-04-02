# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 18:34:29 2026

@author: Adams
"""

from __future__ import annotations

import numpy as np
from matplotlib.ticker import MaxNLocator


def normalize_fp(
    fp: np.ndarray,
    *,
    baseline_mask: np.ndarray | None = None,
    mode: str = "zscore",
) -> np.ndarray:
    fp = np.asarray(fp, dtype=float)

    if fp.size == 0 or not np.any(np.isfinite(fp)):
        return fp

    mode = mode.lower().strip()

    if mode == "zscore":
        if baseline_mask is not None and np.any(baseline_mask):
            base_vals = fp[baseline_mask]
        else:
            base_vals = fp[np.isfinite(fp)]

        m = np.nanmean(base_vals)
        s = np.nanstd(base_vals)
        s = s if np.isfinite(s) and s > 0 else 1.0
        return (fp - m) / s

    if mode == "minmax":
        lo = np.nanmin(fp)
        hi = np.nanmax(fp)
        return (fp - lo) / (hi - lo + 1e-12)

    if mode == "raw" or mode == "none":
        return fp

    raise ValueError("mode must be one of {'zscore', 'minmax', 'raw', 'none'}")


def sem(a: np.ndarray, axis: int = 0) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    n = np.sum(np.isfinite(a), axis=axis)
    sd = np.nanstd(a, axis=axis, ddof=1)
    out = sd / np.sqrt(np.maximum(n, 1))
    out[~np.isfinite(out)] = np.nan
    return out


def interp_to_grid(
    t: np.ndarray,
    y: np.ndarray,
    t_grid: np.ndarray,
) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)

    finite = np.isfinite(t) & np.isfinite(y)
    if np.sum(finite) < 2:
        return np.full_like(t_grid, np.nan, dtype=float)

    return np.interp(t_grid, t[finite], y[finite]).astype(float)

def apply_y_ticks(ax):
    """Limit to 4–5 'nice' ticks."""
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3, steps=[1, 2, 2.5, 5, 10]))