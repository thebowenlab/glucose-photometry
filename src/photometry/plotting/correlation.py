from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from photometry.plotting.utils import apply_y_ticks, interp_to_grid, sem


def _equal(left: Any, right: Any) -> bool:
    try:
        return bool(np.isclose(float(left), float(right), equal_nan=False))
    except (TypeError, ValueError):
        return left == right


def _label(value: Any) -> str:
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return f"{float(value):g}"
    return str(value)


def plot_cross_correlation_summary(
    sessions: Sequence[Mapping[str, Any]],
    correlation_results: Sequence[Mapping[str, Any]],
    *,
    target: str = "glucose_derivative",
    group_key: str = "dose_num",
    group_order: Sequence[Any] | None = None,
    group_colors: Mapping[Any, Any] | Sequence[Any] | None = None,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    vmin: float = -0.6,
    vmax: float = 0.6,
    cmap: str = "RdBu_r",
    show_best_lag: bool = True,
    figsize: tuple[float, float] = (10.0, 5.5),
) -> dict[str, Any]:
    """Plot session-level lag curves as a heatmap and group mean ± SEM lines."""
    if len(sessions) != len(correlation_results):
        raise ValueError("sessions and correlation_results must have the same length")
    if not sessions:
        raise ValueError("sessions must contain at least one session")

    first = correlation_results[0][target]
    lags = np.asarray(first["lags_min"], dtype=float)
    curves = []
    best_lags = []
    for result in correlation_results:
        item = result[target]
        source_lags = np.asarray(item["lags_min"], dtype=float)
        source_curve = np.asarray(item["correlation"], dtype=float)
        if source_lags.shape == lags.shape and np.allclose(source_lags, lags):
            curve = source_curve
        else:
            curve = interp_to_grid(source_lags, source_curve, lags)
        curves.append(curve)
        best_lags.append(float(item["best_lag_min"]))
    matrix = np.vstack(curves)
    best_lags_array = np.asarray(best_lags, dtype=float)

    observed = []
    for session in sessions:
        value = session.get(group_key)
        if not any(_equal(value, item) for item in observed):
            observed.append(value)
    groups = list(group_order) if group_order is not None else observed
    group_rows = [
        np.asarray([i for i, session in enumerate(sessions) if _equal(session.get(group_key), group)], dtype=int)
        for group in groups
    ]
    retained = [(g, rows) for g, rows in zip(groups, group_rows) if rows.size]
    groups = [item[0] for item in retained]
    group_rows = [item[1] for item in retained]
    ordered = np.concatenate(group_rows)

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if isinstance(group_colors, Mapping):
        colors = [group_colors.get(group, default_colors[i % len(default_colors)]) for i, group in enumerate(groups)]
    elif group_colors is not None:
        colors = [group_colors[i % len(group_colors)] for i in range(len(groups))]
    else:
        colors = [default_colors[i % len(default_colors)] for i in range(len(groups))]

    fig, (ax_heat, ax_line) = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": (1.35, 1.0)},
        constrained_layout=True,
    )
    ordered_matrix = matrix[ordered]
    im = ax_heat.imshow(
        ordered_matrix,
        aspect="auto",
        interpolation="nearest",
        extent=(lags[0], lags[-1], ordered_matrix.shape[0] - 0.5, -0.5),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax_heat.axvline(0.0, color="k", linewidth=0.9)
    if show_best_lag:
        ax_heat.scatter(
            best_lags_array[ordered],
            np.arange(ordered.size),
            marker="x",
            s=22,
            linewidths=0.9,
            color="k",
            label="max |r|",
        )

    centers = []
    labels = []
    cursor = 0
    for group, rows in zip(groups, group_rows):
        start = cursor
        cursor += rows.size
        centers.append((start + cursor - 1) / 2.0)
        labels.append(f"{_label(group)} (n={rows.size})")
        if cursor < len(sessions):
            ax_heat.axhline(cursor - 0.5, color="k", linewidth=0.8, alpha=0.6)
    ax_heat.set_yticks(centers)
    ax_heat.set_yticklabels(labels)
    ax_heat.set_ylabel(group_key)
    ax_heat.set_xlabel("Lag (min)")
    if xlim is not None:
        ax_heat.set_xlim(xlim)
    ax_heat.set_title(title or f"FP × {target.replace('_', ' ')}")
    fig.colorbar(im, ax=ax_heat, label="Correlation (r)", fraction=0.046, pad=0.02)

    group_summaries = {}
    for group, rows, color in zip(groups, group_rows, colors):
        group_matrix = matrix[rows]
        mean = np.nanmean(group_matrix, axis=0)
        error = sem(group_matrix, axis=0)
        group_summaries[group] = {"mean": mean, "sem": error, "n": rows.size}
        ax_line.plot(lags, mean, color=color, linewidth=1.8, label=f"{_label(group)} (n={rows.size})")
        ax_line.fill_between(lags, mean - error, mean + error, color=color, alpha=0.2, linewidth=0)
    ax_line.axvline(0.0, color="k", linewidth=0.9)
    ax_line.axhline(0.0, color="0.5", linewidth=0.8)
    if xlim is not None:
        ax_line.set_xlim(xlim)
    ax_line.set_xlabel("Lag (min)")
    ax_line.set_ylabel("Correlation (r)")
    ax_line.legend(frameon=False, fontsize=8)
    ax_line.grid(False)
    apply_y_ticks(ax_line)

    return {
        "fig": fig,
        "axes": {"heatmap": ax_heat, "mean": ax_line},
        "lags_min": lags,
        "matrix": matrix,
        "ordered_indices": ordered,
        "best_lag_min": best_lags_array,
        "groups": groups,
        "group_summaries": group_summaries,
    }
