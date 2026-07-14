from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from photometry.analysis._numeric import rolling_mean_causal, time_derivative
from photometry.analysis._session import (
    SessionLike,
    baseline_value,
    get_aligned_session_arrays,
    infer_dt_minutes,
    window_mask,
)


@dataclass(slots=True)
class GlucoseAnalysisConfig:
    """Configuration for single-session glucose kinetics.

    Time values and all windows are expressed in minutes. The defaults mirror the
    core OGTT settings in ``compute_ogtt_v5.py``, while ``iauc`` is implemented as
    a true incremental AUC above the estimated baseline.
    """

    event_time: float = 0.0
    baseline_window: tuple[float, float] = (-25.0, 0.0)
    onset_baseline_window: tuple[float, float] = (-5.0, 0.0)
    peak_window: tuple[float | None, float | None] = (0.0, 120.0)
    iauc_window: tuple[float | None, float | None] = (0.0, 120.0)
    derivative_smooth_minutes: float = 1.0
    onset_slope_smooth_minutes: float = 2.0
    onset_slope_z: float = 3.0
    onset_level_threshold: float | None = None
    onset_min_run_minutes: float = 1.0
    onset_fallback_time: float | None = 0.0
    baseline_statistic: str = "median"


def _robust_scale(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median))) * 1.4826
    if np.isfinite(mad) and mad > 0:
        return mad
    std = float(np.std(finite))
    return std if np.isfinite(std) and std > 0 else 0.0


def compute_glucose_derivative(
    session: SessionLike,
    *,
    smooth_minutes: float = 1.0,
) -> dict[str, Any]:
    """Compute the aligned glucose derivative for one canonical session."""
    t, bg = get_aligned_session_arrays(session, ("t", "bg"), min_length=3)
    derivative = time_derivative(t, bg, smooth_minutes=smooth_minutes)
    return {
        "t": t.copy(),
        "bg_derivative": derivative,
        "dt_min": infer_dt_minutes(t),
        "smooth_minutes": float(smooth_minutes),
    }


def detect_glucose_onset(
    session: SessionLike,
    *,
    config: GlucoseAnalysisConfig | None = None,
    derivative: np.ndarray | None = None,
) -> dict[str, Any]:
    """Detect the first sustained glucose rise after the configured event time.

    Detection follows the slope-or-level logic of the original script:

    * estimate baseline slope and level variability before the event;
    * flag samples exceeding either threshold;
    * require a sustained run for ``onset_min_run_minutes``.
    """
    cfg = config or GlucoseAnalysisConfig()
    t, bg = get_aligned_session_arrays(session, ("t", "bg"), min_length=3)
    dt = infer_dt_minutes(t)

    baseline = baseline_value(
        t,
        bg,
        cfg.onset_baseline_window,
        statistic=cfg.baseline_statistic,
    )
    bg_increment = bg - baseline

    if derivative is None:
        derivative = time_derivative(
            t,
            bg,
            smooth_minutes=cfg.onset_slope_smooth_minutes,
        )
    else:
        derivative = np.asarray(derivative, dtype=float)
        if derivative.shape != t.shape:
            raise ValueError("derivative must have the same shape as session['t']")

    baseline_mask = window_mask(
        t,
        cfg.onset_baseline_window,
        finite=np.isfinite(derivative),
    )
    baseline_derivative = derivative[baseline_mask]
    if baseline_derivative.size:
        slope_center = float(np.nanmedian(baseline_derivative))
        slope_scale = _robust_scale(baseline_derivative)
        slope_threshold = slope_center + cfg.onset_slope_z * slope_scale
    else:
        finite_derivative = derivative[np.isfinite(derivative)]
        slope_threshold = (
            float(np.nanpercentile(finite_derivative, 80.0))
            if finite_derivative.size
            else np.nan
        )

    if cfg.onset_level_threshold is None:
        level_values = bg_increment[
            window_mask(
                t,
                cfg.onset_baseline_window,
                finite=np.isfinite(bg_increment),
            )
        ]
        level_center = float(np.nanmedian(level_values)) if level_values.size else 0.0
        level_scale = _robust_scale(level_values)
        level_threshold = level_center + 3.0 * level_scale
    else:
        level_threshold = float(cfg.onset_level_threshold)

    after_event = t >= float(cfg.event_time)
    condition = after_event & np.isfinite(bg_increment) & (
        (derivative > slope_threshold) | (bg_increment > level_threshold)
    )

    run_samples = max(1, int(round(cfg.onset_min_run_minutes / dt)))
    onset_index = -1
    if run_samples == 1:
        hits = np.flatnonzero(condition)
        if hits.size:
            onset_index = int(hits[0])
    elif condition.size >= run_samples:
        run_counts = np.convolve(
            condition.astype(np.int32),
            np.ones(run_samples, dtype=np.int32),
            mode="valid",
        )
        starts = np.flatnonzero(run_counts >= run_samples)
        if starts.size:
            onset_index = int(starts[0])

    found = onset_index >= 0
    if found:
        onset_time = float(t[onset_index])
    elif cfg.onset_fallback_time is None:
        onset_time = np.nan
    else:
        onset_time = float(cfg.onset_fallback_time)

    return {
        "onset_index": onset_index,
        "onset_time": onset_time,
        "found_onset": found,
        "slope_threshold": float(slope_threshold),
        "level_threshold": float(level_threshold),
        "run_samples": int(run_samples),
        "baseline": float(baseline),
        "condition": condition,
    }


def compute_incremental_auc(
    session: SessionLike,
    *,
    baseline_window: tuple[float, float] = (-25.0, 0.0),
    auc_window: tuple[float | None, float | None] = (0.0, 120.0),
    positive_only: bool = True,
    baseline_statistic: str = "median",
) -> dict[str, Any]:
    """Compute baseline-subtracted incremental glucose AUC for one session."""
    t, bg = get_aligned_session_arrays(session, ("t", "bg"), min_length=2)
    baseline = baseline_value(
        t,
        bg,
        baseline_window,
        statistic=baseline_statistic,
    )
    increment = bg - baseline
    mask = window_mask(t, auc_window, finite=np.isfinite(increment))
    if np.count_nonzero(mask) < 2:
        iauc = np.nan
    else:
        values = np.maximum(increment[mask], 0.0) if positive_only else increment[mask]
        iauc = float(np.trapezoid(values, t[mask]))

    return {
        "iauc": iauc,
        "baseline": float(baseline),
        "auc_window": tuple(auc_window),
        "positive_only": bool(positive_only),
        "bg_increment": increment,
        "auc_mask": mask,
    }


def compute_glucose_metrics(
    session: SessionLike,
    *,
    config: GlucoseAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Run the core glucose-processing and kinetic analysis on one session.

    The returned dictionary contains scalar metrics plus the processed arrays needed
    by an external integration or visualization script.
    """
    cfg = config or GlucoseAnalysisConfig()
    t, bg = get_aligned_session_arrays(session, ("t", "bg"), min_length=3)
    dt = infer_dt_minutes(t)
    baseline = baseline_value(
        t,
        bg,
        cfg.baseline_window,
        statistic=cfg.baseline_statistic,
    )
    increment = bg - baseline
    derivative = time_derivative(
        t,
        bg,
        smooth_minutes=cfg.derivative_smooth_minutes,
    )

    onset = detect_glucose_onset(session, config=cfg)

    peak_mask = window_mask(t, cfg.peak_window, finite=np.isfinite(increment))
    if np.any(peak_mask):
        peak_candidates = np.where(peak_mask, increment, np.nan)
        peak_index = int(np.nanargmax(peak_candidates))
        peak_time = float(t[peak_index])
        peak_increment = float(increment[peak_index])
        peak_raw = float(bg[peak_index])
    else:
        peak_index = -1
        peak_time = peak_increment = peak_raw = np.nan

    derivative_peak_mask = window_mask(
        t,
        cfg.peak_window,
        finite=np.isfinite(derivative),
    )
    if np.any(derivative_peak_mask):
        derivative_candidates = np.where(derivative_peak_mask, derivative, np.nan)
        derivative_peak_index = int(np.nanargmax(derivative_candidates))
        derivative_peak_time = float(t[derivative_peak_index])
        derivative_peak_value = float(derivative[derivative_peak_index])
    else:
        derivative_peak_index = -1
        derivative_peak_time = derivative_peak_value = np.nan

    auc = compute_incremental_auc(
        session,
        baseline_window=cfg.baseline_window,
        auc_window=cfg.iauc_window,
        positive_only=True,
        baseline_statistic=cfg.baseline_statistic,
    )

    return {
        "baseline": float(baseline),
        "dt_min": float(dt),
        "onset_index": int(onset["onset_index"]),
        "onset_time": float(onset["onset_time"]),
        "found_onset": bool(onset["found_onset"]),
        "peak_index": int(peak_index),
        "peak_time": float(peak_time),
        "peak_increment": float(peak_increment),
        "peak_raw": float(peak_raw),
        "derivative_peak_index": int(derivative_peak_index),
        "derivative_peak_time": float(derivative_peak_time),
        "derivative_peak_value": float(derivative_peak_value),
        "iauc": float(auc["iauc"]),
        "config": asdict(cfg),
        "thresholds": {
            "onset_slope": onset["slope_threshold"],
            "onset_level": onset["level_threshold"],
            "onset_run_samples": onset["run_samples"],
        },
        "series": {
            "t": t.copy(),
            "bg_raw": bg.copy(),
            "bg_increment": increment,
            "bg_derivative": derivative,
            "onset_condition": onset["condition"],
            "iauc_mask": auc["auc_mask"],
        },
    }
