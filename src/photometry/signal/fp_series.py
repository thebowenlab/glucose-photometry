# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 19:54:07 2026

@author: Adams
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt


def safe_dt_from_t(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    diffs = np.diff(t)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 1.0 / 60.0
    med = float(np.median(diffs))
    return med if np.isfinite(med) and med > 0 else 1.0 / 60.0


def interp_nans(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)

    if not np.any(~np.isfinite(y)):
        return y

    finite = np.isfinite(y) & np.isfinite(t)
    if np.sum(finite) < 2:
        return np.nan_to_num(y, nan=0.0)

    return np.interp(t, t[finite], y[finite]).astype(float)


def lowpass_minutes(
    t: np.ndarray,
    x: np.ndarray,
    cutoff_period_min: float,
    order: int = 3,
) -> np.ndarray:
    """
    Zero-phase IIR low-pass with cutoff specified as period in minutes.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    if t.size < 5 or not np.any(np.isfinite(x)):
        return x.astype(float)

    dt = safe_dt_from_t(t)
    fs_cpm = 1.0 / max(dt, 1e-9)  # samples per minute
    cutoff_cpm = 1.0 / max(float(cutoff_period_min), 1e-9)
    nyq = fs_cpm / 2.0
    wn = cutoff_cpm / max(nyq, 1e-9)

    if wn >= 1.0:
        return x.astype(float)

    b, a = butter(int(order), wn, btype="low", analog=False)
    x_fill = interp_nans(x, t)

    try:
        y = filtfilt(b, a, x_fill, method="pad")
        return y.astype(float)
    except Exception:
        return x_fill.astype(float)


@dataclass
class FPConfig:
    lp_cutoff_period_min: float = 4.0
    lp_order: int = 3
    baseline_window: tuple[float, float] = (-25.0, 0.0)


class FpSeries:
    def __init__(self, t: np.ndarray, fp: np.ndarray):
        self.t = np.asarray(t, dtype=float)
        self.fp = np.asarray(fp, dtype=float)

        n = min(self.t.size, self.fp.size)
        self.t = self.t[:n]
        self.fp = self.fp[:n]

        if n < 8:
            raise ValueError("series too short")

        self.dt = safe_dt_from_t(self.t)
        self._lp_sig: tuple[float, int] | None = None
        self.fp_lp: np.ndarray | None = None

    @classmethod
    def from_arrays(cls, t, fp):
        return cls(np.asarray(t, float), np.asarray(fp, float))

    def compute_lowpass(self, cutoff_period_min: float, order: int) -> np.ndarray:
        sig = (float(cutoff_period_min), int(order))
        if self._lp_sig != sig or self.fp_lp is None:
            self.fp_lp = lowpass_minutes(self.t, self.fp, cutoff_period_min, order).astype(float)
            self._lp_sig = sig
        return self.fp_lp

    def baseline_mask(self, window: tuple[float, float]) -> np.ndarray:
        a, b = float(window[0]), float(window[1])
        return (self.t >= min(a, b)) & (self.t <= max(a, b))

    def zscore_in_window(self, x: np.ndarray, window: tuple[float, float]) -> tuple[np.ndarray, float, float]:
        x = np.asarray(x, dtype=float)
        m = self.baseline_mask(window)
        mu = float(np.nanmean(x[m]))
        sd = float(np.nanstd(x[m]) + 1e-12)
        z = (x - mu) / sd
        return z.astype(float), mu, sd


def lowpass_fp_trace(
    t: np.ndarray,
    fp: np.ndarray,
    *,
    cutoff_period_min: float = 4.0,
    order: int = 3,
) -> np.ndarray:
    series = FpSeries.from_arrays(t, fp)
    return series.compute_lowpass(cutoff_period_min, order)