# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:01:30 2026

@author: Adams
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _baseline_from_window(
    t: np.ndarray,
    y: np.ndarray,
    baseline_window: tuple[float, float] | None,
    *,
    t0: float = 0.0,
) -> float:
    if baseline_window is None:
        mask = (t < t0) & np.isfinite(y)
    else:
        a, b = baseline_window
        mask = (t >= min(a, b)) & (t <= max(a, b)) & np.isfinite(y)

    if not np.any(mask):
        return 0.0
    return float(np.nanmedian(y[mask]))


def _resolve_window(
    window: tuple[float | None, float | None],
    *,
    event_time: float | None = None,
) -> tuple[float, float]:
    lo, hi = window

    lo_abs = -np.inf if lo is None else float(lo)
    hi_abs = np.inf if hi is None else float(hi)

    if event_time is not None:
        if np.isfinite(lo_abs):
            lo_abs += float(event_time)
        if np.isfinite(hi_abs):
            hi_abs += float(event_time)

    return lo_abs, hi_abs


def _window_mask(
    t: np.ndarray,
    y: np.ndarray,
    window: tuple[float | None, float | None],
    *,
    event_time: float | None = None,
) -> np.ndarray:
    lo_abs, hi_abs = _resolve_window(window, event_time=event_time)
    return (t >= min(lo_abs, hi_abs)) & (t <= max(lo_abs, hi_abs)) & np.isfinite(y)


def _baseline_median_relative(
    t: np.ndarray,
    y: np.ndarray,
    baseline_window: tuple[float, float] | None,
    *,
    event_time: float | None = None,
) -> float:
    if baseline_window is None:
        t_ref = 0.0 if event_time is None else float(event_time)
        mask = (t < t_ref) & np.isfinite(y)
    else:
        mask = _window_mask(t, y, baseline_window, event_time=event_time)

    if not np.any(mask):
        return 0.0
    return float(np.nanmedian(y[mask]))


def _trapz_auc(
    t: np.ndarray,
    y: np.ndarray,
    *,
    direction: str | None = None,
    baseline_window: tuple[float, float] | None = (-25.0, 0.0),
    auc_window: tuple[float | None, float | None] = (0.0, None),
    event_time: float | None = None,
) -> float:
    """
    Baseline-subtracted trapezoidal AUC.

    direction:
        None        -> signed AUC of (y - baseline)
        "negative"  -> magnitude of area below baseline
        "positive"  -> area above baseline
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    n = min(t.size, y.size)
    t = t[:n]
    y = y[:n]

    if n < 2:
        return np.nan

    baseline = _baseline_median_relative(
        t,
        y,
        baseline_window,
        event_time=event_time,
    )
    y_eval = y - baseline

    mask = _window_mask(t, y_eval, auc_window, event_time=event_time)
    if np.sum(mask) < 2:
        return np.nan

    t_sel = t[mask]
    y_sel = y_eval[mask]

    if direction is None:
        return float(np.trapz(y_sel, t_sel))

    if direction == "negative":
        return float(-np.trapz(np.minimum(y_sel, 0.0), t_sel))

    if direction == "positive":
        return float(np.trapz(np.maximum(y_sel, 0.0), t_sel))

    raise ValueError("direction must be None, 'negative', or 'positive'")


def time_to_peak_inhibition(
    t: np.ndarray,
    y: np.ndarray,
    *,
    t0: float = 0.0,
    search_window: tuple[float | None, float | None] = (0.0, None),
    baseline_window: tuple[float, float] | None = (-5.0, 0.0),
    subtract_baseline: bool = True,
) -> dict[str, float | int | np.ndarray]:
    """
    Find the time of the minimum signal value after t0.

    Parameters
    ----------
    t : np.ndarray
        Time vector, typically in minutes.
    y : np.ndarray
        Processed photometry trace.
    t0 : float
        Event time.
    search_window : tuple
        Relative window around t0, e.g. (0, 30) means search in [t0, t0+30].
    baseline_window : tuple or None
        Window used to compute baseline.
    subtract_baseline : bool
        If True, analyze y - baseline.

    Returns
    -------
    dict with:
      - idx_peak_inhibition
      - t_peak_inhibition
      - y_peak_inhibition
      - baseline
      - y_eval
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    n = min(t.size, y.size)
    t = t[:n]
    y = y[:n]

    if n == 0:
        return {
            "idx_peak_inhibition": -1,
            "t_peak_inhibition": np.nan,
            "y_peak_inhibition": np.nan,
            "baseline": np.nan,
            "y_eval": np.array([], dtype=float),
        }

    baseline = _baseline_from_window(t, y, baseline_window, t0=t0)
    y_eval = y - baseline if subtract_baseline else y.copy()

    rel_lo, rel_hi = search_window
    abs_lo = t0 if rel_lo is None else (t0 + float(rel_lo))
    abs_hi = np.inf if rel_hi is None else (t0 + float(rel_hi))

    mask = (t >= abs_lo) & (t <= abs_hi) & np.isfinite(y_eval)
    if not np.any(mask):
        return {
            "idx_peak_inhibition": -1,
            "t_peak_inhibition": np.nan,
            "y_peak_inhibition": np.nan,
            "baseline": float(baseline),
            "y_eval": y_eval,
        }

    y_search = y_eval.copy()
    y_search[~mask] = np.nan
    idx = int(np.nanargmin(y_search))

    return {
        "idx_peak_inhibition": idx,
        "t_peak_inhibition": float(t[idx]),
        "y_peak_inhibition": float(y_eval[idx]),
        "baseline": float(baseline),
        "y_eval": y_eval,
    }


def compute_fp_peak_delta_z(
    t: np.ndarray,
    fp: np.ndarray,
    *,
    baseline_window: tuple[float, float] = (-25.0, 0.0),
    response_window: tuple[float, float] = (0.0, 10.0),
    event_time: float | None = None,
    use_abs_peak: bool = True,
    eps: float = 1e-12,
) -> dict[str, float]:
    """
    Compute peak delta-z from a baseline-defined z-score.

    This is a direct port of the original logic.
    """
    t = np.asarray(t, dtype=float)
    fp = np.asarray(fp, dtype=float)

    n = min(t.size, fp.size)
    t = t[:n]
    fp = fp[:n]

    bl = _window_mask(t, fp, baseline_window, event_time=event_time)
    rs = _window_mask(t, fp, response_window, event_time=event_time)

    if bl.sum() < 3 or rs.sum() < 1:
        return {
            "fp_peak_dz": np.nan,
            "fp_peak_z": np.nan,
            "fp_peak_time": np.nan,
            "fp_peak_sign": np.nan,
            "fp_baseline_mu0": np.nan,
            "fp_baseline_sd0": np.nan,
            "fp_baseline_median_z": np.nan,
        }

    mu0 = float(np.nanmean(fp[bl]))
    sd0 = float(np.nanstd(fp[bl], ddof=1))
    if not np.isfinite(sd0) or sd0 <= 0:
        sd0 = eps

    z = (fp - mu0) / sd0
    z_bl = z[bl]
    z_rs = z[rs]
    t_rs = t[rs]

    if use_abs_peak:
        idx = int(np.nanargmax(np.abs(z_rs)))
        peak_z = float(z_rs[idx])
        peak_val = float(abs(peak_z))
    else:
        idx = int(np.nanargmax(z_rs))
        peak_z = float(z_rs[idx])
        peak_val = peak_z

    peak_t = float(t_rs[idx])
    med_bl = float(np.nanmedian(z_bl))

    return {
        "fp_peak_dz": float(peak_val - med_bl),
        "fp_peak_z": peak_z,
        "fp_peak_time": peak_t,
        "fp_peak_sign": float(np.sign(peak_z)) if np.isfinite(peak_z) else np.nan,
        "fp_baseline_mu0": mu0,
        "fp_baseline_sd0": sd0,
        "fp_baseline_median_z": med_bl,
    }


def auc_blocks(
    t: np.ndarray,
    fp: np.ndarray,
    *,
    baseline_window: tuple[float, float] | None = (-25.0, 0.0),
    auc_window_early: tuple[float | None, float | None] = (0.0, None),
    auc_window_late: tuple[float | None, float | None] = (None, None),
    event_time: float | None = None,
    early_anchor: float | None = None,
    anchor_early_to_time: bool = False,
) -> dict[str, Any]:
    """
    Compute AUC summary blocks.

    By default, this preserves the original behavior more closely:
    - if early_anchor is passed, the REPORTED early window start is updated
    - but the actual early AUC integration window is unchanged

    If you want the early AUC to truly start at early_anchor, set:
        anchor_early_to_time=True
    """
    t = np.asarray(t, dtype=float)
    fp = np.asarray(fp, dtype=float)

    n = min(t.size, fp.size)
    t = t[:n]
    fp = fp[:n]

    if n < 2 or not np.any(np.isfinite(t)):
        return {
            "fp_neg_auc_early": np.nan,
            "fp_neg_auc_early_win": auc_window_early,
            "fp_neg_auc_late": np.nan,
            "fp_neg_auc_late_win": auc_window_late,
            "fp_auc": np.nan,
            "fp_auc_win": (0.0, np.nan),
        }

    early_auc_window = auc_window_early
    reported_early_win = auc_window_early

    if early_anchor is not None and np.isfinite(early_anchor):
        reported_early_win = (float(early_anchor), auc_window_early[1])
        if anchor_early_to_time:
            early_auc_window = (float(early_anchor), auc_window_early[1])

    t_max = float(np.nanmax(t))
    total_start = 0.0 if event_time is None else float(event_time)

    out = {
        "fp_neg_auc_early": _trapz_auc(
            t,
            fp,
            direction="negative",
            baseline_window=baseline_window,
            auc_window=early_auc_window,
            event_time=event_time,
        ),
        "fp_neg_auc_early_win": reported_early_win,
        "fp_neg_auc_late": _trapz_auc(
            t,
            fp,
            direction="negative",
            baseline_window=baseline_window,
            auc_window=auc_window_late,
            event_time=event_time,
        ),
        "fp_neg_auc_late_win": auc_window_late,
        "fp_auc": _trapz_auc(
            t,
            fp,
            direction=None,
            baseline_window=baseline_window,
            auc_window=(total_start, t_max),
            event_time=None,
        ),
        "fp_auc_win": (total_start, t_max),
    }
    return out


def auc_blocks_from_cfg(
    t: np.ndarray,
    fp: np.ndarray,
    cfg,
    *,
    early_anchor: float | None = None,
    anchor_early_to_time: bool = False,
) -> dict[str, Any]:
    """
    Convenience wrapper if you want to drive this from a config object.
    """
    return auc_blocks(
        t,
        fp,
        baseline_window=getattr(cfg, "baseline_window", (-25.0, 0.0)),
        auc_window_early=getattr(cfg, "auc_window_early", (0.0, None)),
        auc_window_late=getattr(cfg, "auc_window_late", (None, None)),
        event_time=getattr(cfg, "event_time", None),
        early_anchor=early_anchor,
        anchor_early_to_time=anchor_early_to_time,
    )