# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 18:34:56 2026

@author: Adams
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from photometry.plotting.utils import normalize_fp, sem, interp_to_grid


def build_matrix(
    traces: list[np.ndarray],
    times: list[np.ndarray],
    *,
    t_grid: np.ndarray | None = None,
    normalization: str = "none",
    baseline_window: tuple[float, float] = (-5.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a list of traces + time vectors into an (N, T) matrix on a common grid.
    """
    if len(traces) != len(times):
        raise ValueError("traces and times must have the same length")
    if len(traces) == 0:
        raise ValueError("No traces provided")

    if t_grid is None:
        t_grid = np.asarray(times[0], dtype=float)
    else:
        t_grid = np.asarray(t_grid, dtype=float)

    rows = []
    for t, y in zip(times, traces):
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)

        base_mask = (t >= baseline_window[0]) & (t <= baseline_window[1])
        y_norm = normalize_fp(y, baseline_mask=base_mask, mode=normalization)
        y_grid = interp_to_grid(t, y_norm, t_grid)
        rows.append(y_grid)

    mat = np.vstack(rows)
    return mat, t_grid


def compute_group_average(
    mat: np.ndarray,
) -> dict[str, np.ndarray]:
    mat = np.asarray(mat, dtype=float)
    return {
        "mean": np.nanmean(mat, axis=0),
        "sem": sem(mat, axis=0),
        "n": np.sum(np.any(np.isfinite(mat), axis=1)),
    }


def plot_group_average(
    traces: list[np.ndarray],
    times: list[np.ndarray],
    *,
    t_grid: np.ndarray | None = None,
    normalization: str = "none",
    baseline_window: tuple[float, float] = (-5.0, 0.0),
    group_color: str | None = None,          # accepts hex like "#1f77b4"
    label: str | None = None,
    alpha_sem: float = 0.25,
    linewidth: float = 2.0,
    plot_window: tuple[float, float] | None = None,  # x-axis limits
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    mat, t_grid = build_matrix(
        traces,
        times,
        t_grid=t_grid,
        normalization=normalization,
        baseline_window=baseline_window,
    )
    summary = compute_group_average(mat)

    mean = summary["mean"]
    se = summary["sem"]

    # Draw the line first, then reuse its actual color for the SEM fill
    line, = ax.plot(
        t_grid,
        mean,
        linewidth=linewidth,
        color=group_color,
        label=label,
    )
    line_color = line.get_color()

    ax.fill_between(
        t_grid,
        mean - se,
        mean + se,
        alpha=alpha_sem,
        color=line_color,
        linewidth=0,
    )

    if t_grid.size > 1 and t_grid[0] <= 0 <= t_grid[-1]:
        ax.axvline(0, color="k", linewidth=1.5, alpha=0.8)

    if plot_window is not None:
        ax.set_xlim(plot_window)

    ax.set_xlabel("Time (min)")
    ax.set_ylabel(normalization if normalization != "none" else "Signal")

    if label is not None:
        ax.legend()

    return {
        "fig": fig,
        "ax": ax,
        "t": t_grid,
        "mat": mat,
        "mean": mean,
        "sem": se,
        "n": summary["n"],
        "color": line_color,
    }