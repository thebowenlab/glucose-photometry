# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:59:14 2026

@author: Adams
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from photometry.preprocess.time import coerce_time_series, to_seconds, odd_window_for_seconds
from photometry.preprocess.cleaning import trim_signal_and_time, remove_mad_outliers
from photometry.preprocess.smoothing import rolling_median, rolling_mean_causal


@dataclass
class PrepareBundleConfig:
    time_numeric_units: str = "minutes"   # for numeric time columns
    fp_outlier_k: float = 2.0
    fp_outlier_fill: str = "interp"
    median_window_sec: float = 7.0
    fp_smooth_window_sec: float = 60.0
    cgm_smooth_window_sec: float = 60.0


def _as_numeric_series(values: list[Any]) -> pd.Series:
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _as_time_series(values: list[Any]) -> pd.Series:
    return pd.Series(values)


def _to_list(values: Any) -> list[Any]:
    """Convert arrays/series to a list without ambiguous truth-value checks."""
    if values is None:
        return []
    if isinstance(values, list):
        return values.copy()
    if isinstance(values, tuple):
        return list(values)
    if isinstance(values, pd.Series):
        return values.tolist()
    if isinstance(values, np.ndarray):
        return np.ravel(values).tolist()
    try:
        return list(values)
    except TypeError:
        return [values]


def _prepare_single_signal(
    *,
    raw_time: list[Any],
    raw_values: list[Any],
    numeric_units: str,
    median_window_sec: float,
    smooth_window_sec: float,
    outlier_k: float | None = None,
    outlier_fill: str = "interp",
) -> dict[str, Any]:
    """
    Generic helper for one time/value pair.
    """
    y = _as_numeric_series(raw_values)
    t = _as_time_series(raw_time)

    if len(y) == 0:
        return {
            "t_raw": np.array([], dtype=float),
            "t_seconds": np.array([], dtype=float),
            "t_minutes": np.array([], dtype=float),
            "y_raw": np.array([], dtype=float),
            "y_clean": np.array([], dtype=float),
            "y_proc": np.array([], dtype=float),
            "dt_seconds": None,
            "time_kind": "index",
            "trim_edges": {"head": 0, "tail": 0},
            "outliers": None,
        }

    # If no explicit time was provided, fall back to sample index.
    if len(t) == 0:
        t = pd.Series(np.arange(len(y), dtype=float))

    t_trim, y_trim, trim_info = trim_signal_and_time(t, y)

    t_coerced, time_kind = coerce_time_series(t_trim)
    t_seconds, dt_seconds = to_seconds(
        t_coerced,
        time_kind,
        numeric_units=numeric_units,
        relative=False,
    )

    if dt_seconds is None or dt_seconds <= 0:
        dt_seconds = 1.0

    y_raw = y_trim.to_numpy(dtype=float)

    if outlier_k is not None:
        y_clean, outlier_stats = remove_mad_outliers(
            y_raw,
            k=outlier_k,
            fill=outlier_fill,
        )
    else:
        y_clean = y_raw.copy()
        outlier_stats = None

    win_med = odd_window_for_seconds(median_window_sec, dt_seconds, min_samples=3)
    win_smooth = max(1, int(round(smooth_window_sec / dt_seconds)))

    y_med = rolling_median(y_clean, win_med)
    y_proc = rolling_mean_causal(y_med, win_smooth)

    return {
        "t_raw": t_coerced.to_numpy(),
        "t_seconds": t_seconds,
        "t_minutes": t_seconds / 60.0,
        "y_raw": y_raw,
        "y_clean": y_clean,
        "y_proc": y_proc,
        "dt_seconds": float(dt_seconds),
        "time_kind": time_kind,
        "trim_edges": trim_info,
        "outliers": outlier_stats,
        "windows": {
            "median_samples": int(win_med),
            "smooth_samples": int(win_smooth),
        },
    }


def prepare_bundle(
    bundle: dict[str, Any],
    config: PrepareBundleConfig | None = None,
) -> dict[str, Any]:
    """
    Prepare a bundle for downstream metrics.

    Returns a dict with:
      - fp: processed photometry time series
      - cgm: processed glucose time series
      - compatibility aliases:
          time_minutes, time_seconds, fp_proc, cgm_proc
    """
    cfg = config or PrepareBundleConfig()

    timestamps = bundle.get("timestamps", {}) or {}
    fp_data = bundle.get("fp_data", {}) or {}
    cgm_data = bundle.get("cgm_data", {}) or {}

    raw_t_fp = _to_list(timestamps.get("fp", []))
    raw_t_cgm = _to_list(timestamps.get("cgm", []))
    raw_fp = _to_list(fp_data.get("orig", []))
    raw_cgm = _to_list(cgm_data.get("orig", []))

    fp = _prepare_single_signal(
        raw_time=raw_t_fp,
        raw_values=raw_fp,
        numeric_units=cfg.time_numeric_units,
        median_window_sec=cfg.median_window_sec,
        smooth_window_sec=cfg.fp_smooth_window_sec,
        outlier_k=cfg.fp_outlier_k,
        outlier_fill=cfg.fp_outlier_fill,
    )

    cgm = _prepare_single_signal(
        raw_time=raw_t_cgm if raw_t_cgm else raw_t_fp,
        raw_values=raw_cgm,
        numeric_units=cfg.time_numeric_units,
        median_window_sec=cfg.median_window_sec,
        smooth_window_sec=cfg.cgm_smooth_window_sec,
        outlier_k=None,
    )

    out = {
        "bundle": bundle,
        "raw": {
            "timestamps": {
                "fp": raw_t_fp,
                "cgm": raw_t_cgm,
            },
            "fp_data": {"orig": raw_fp},
            "cgm_data": {"orig": raw_cgm},
        },
        "fp": fp,
        "cgm": cgm,
        "meta": {
            "subject": bundle.get("name"),
            "date": bundle.get("date"),
            "route": bundle.get("route"),
            "dose": bundle.get("dose"),
        },
        # Compatibility aliases for early scripts.
        "time_seconds": fp["t_seconds"],
        "time_minutes": fp["t_minutes"],
        "fp_proc": fp["y_proc"],
        "cgm_proc": cgm["y_proc"],
    }
    return out