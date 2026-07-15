from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from photometry.analysis.glucose import GlucoseAnalysisConfig, compute_glucose_metrics
from photometry.plotting.utils import apply_y_ticks, normalize_fp


def _session_title(session: dict[str, Any]) -> str:
    parts = [
        str(session.get("subject", "Session")),
        str(session.get("date", "")),
        str(session.get("route", "")),
    ]
    dose = session.get("dose_num")
    if dose is not None:
        parts.append(f"{dose:g} g/kg")
    return " • ".join(part for part in parts if part and part != "None")


def _draw_marker(ax, x: float, label: str, *, linestyle: str = "--") -> None:
    if not np.isfinite(x):
        return
    ax.axvline(x, color="k", linestyle=linestyle, linewidth=1.0, alpha=0.75)
    y0, y1 = ax.get_ylim()
    ax.text(
        x,
        y1 - 0.03 * (y1 - y0),
        label,
        rotation=90,
        ha="right",
        va="top",
        fontsize=8,
        clip_on=False,
    )


def plot_session_summary(
    session: dict[str, Any],
    *,
    glucose_result: dict[str, Any] | None = None,
    glucose_config: GlucoseAnalysisConfig | None = None,
    fp_normalization: str = "zscore",
    fp_baseline_window: tuple[float, float] = (-25.0, 0.0),
    xlim: tuple[float, float] | None = (-10.0, 40.0),
    title: str | None = None,
    figsize: tuple[float, float] = (10.0, 7.0),
    axes: tuple[Any, Any] | None = None,
) -> dict[str, Any]:
    """Plot one session in the style of the original stacked OGTT figure.

    The top panel shows baseline-subtracted glucose and its derivative. The bottom
    panel shows normalized photometry. Event, onset, derivative-peak, and glucose-
    peak times are annotated using outputs from :func:`compute_glucose_metrics`.
    """
    result = glucose_result or compute_glucose_metrics(
        session,
        config=glucose_config,
    )
    t = np.asarray(result["series"]["t"], dtype=float)
    bg_increment = np.asarray(result["series"]["bg_increment"], dtype=float)
    bg_derivative = np.asarray(result["series"]["bg_derivative"], dtype=float)
    fp = np.asarray(session["fp"], dtype=float)
    if fp.shape != t.shape:
        raise ValueError("session['fp'] must have the same shape as session['t']")

    if axes is None:
        fig, (ax_glucose, ax_fp) = plt.subplots(
            2,
            1,
            figsize=figsize,
            sharex=True,
            constrained_layout=True,
        )
    else:
        ax_glucose, ax_fp = axes
        fig = ax_glucose.figure

    ax_derivative = ax_glucose.twinx()
    line_glucose, = ax_glucose.plot(t, bg_increment, linewidth=2.0, label="Δ glucose")
    line_derivative, = ax_derivative.plot(
        t,
        bg_derivative,
        linewidth=1.4,
        alpha=0.8,
        label="d(Δ glucose)/dt",
    )

    auc_mask = np.asarray(result["series"]["iauc_mask"], dtype=bool)
    positive = np.maximum(bg_increment, 0.0)
    ax_glucose.fill_between(
        t,
        0.0,
        positive,
        where=auc_mask,
        color=line_glucose.get_color(),
        alpha=0.12,
        linewidth=0,
        label="Positive iAUC",
    )
    ax_glucose.axhline(0.0, color="0.4", linewidth=0.8)
    ax_glucose.set_ylabel("Δ glucose (mg/dL)")
    ax_derivative.set_ylabel("d glucose/dt (mg/dL/min)")
    apply_y_ticks(ax_glucose)
    apply_y_ticks(ax_derivative)

    handles = [line_glucose, line_derivative]
    labels = [handle.get_label() for handle in handles]
    ax_glucose.legend(handles, labels, frameon=False, loc="upper right")

    baseline_mask = (t >= fp_baseline_window[0]) & (t <= fp_baseline_window[1])
    fp_plot = normalize_fp(
        fp,
        baseline_mask=baseline_mask,
        mode=fp_normalization,
    )
    ax_fp.plot(t, fp_plot, linewidth=1.6, label=f"FP ({fp_normalization})")
    ax_fp.axhline(0.0, color="0.4", linestyle="--", linewidth=0.8)
    ax_fp.set_ylabel("FP (z-score)" if fp_normalization == "zscore" else "FP")
    ax_fp.set_xlabel("Time from glucose administration (min)")
    ax_fp.legend(frameon=False, loc="best")
    apply_y_ticks(ax_fp)

    markers = (
        (0.0, "Time 0", "-"),
        (float(result["onset_time"]), "Glucose onset", "--"),
        (float(result["derivative_peak_time"]), "Max rise", ":"),
        (float(result["peak_time"]), "Glucose peak", "--"),
    )
    for marker, label, linestyle in markers:
        _draw_marker(ax_glucose, marker, label, linestyle=linestyle)
        ax_fp.axvline(marker, color="k", linestyle=linestyle, linewidth=0.9, alpha=0.55)

    if xlim is not None:
        ax_fp.set_xlim(xlim)
    ax_glucose.set_title(title or _session_title(session))

    for axis in (ax_glucose, ax_derivative, ax_fp):
        axis.grid(False)
    for axis in (ax_glucose, ax_fp):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    return {
        "fig": fig,
        "axes": {
            "glucose": ax_glucose,
            "derivative": ax_derivative,
            "fp": ax_fp,
        },
        "glucose_result": result,
        "fp_normalized": fp_plot,
    }
