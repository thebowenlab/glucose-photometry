from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from photometry.analysis.glucose import compute_glucose_metrics
from photometry.plotting.group_average import build_matrix, compute_group_average
from photometry.plotting.utils import apply_y_ticks


def _group_equal(left: Any, right: Any) -> bool:
    try:
        return bool(np.isclose(float(left), float(right), equal_nan=False))
    except (TypeError, ValueError):
        return left == right


def _group_indices(
    sessions: Sequence[Mapping[str, Any]],
    *,
    group_key: str,
    group_order: Sequence[Any] | None,
) -> tuple[list[Any], list[np.ndarray]]:
    observed: list[Any] = []
    for session in sessions:
        value = session.get(group_key)
        if not any(_group_equal(value, item) for item in observed):
            observed.append(value)

    groups = list(group_order) if group_order is not None else observed
    indices = [
        np.asarray(
            [i for i, session in enumerate(sessions) if _group_equal(session.get(group_key), group)],
            dtype=int,
        )
        for group in groups
    ]
    retained = [(group, index) for group, index in zip(groups, indices) if index.size]
    return [item[0] for item in retained], [item[1] for item in retained]


def _format_group(value: Any) -> str:
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return f"{float(value):g}"
    return str(value)


def prepare_session_matrix(
    sessions: Sequence[Mapping[str, Any]],
    *,
    value: str,
    normalization: str = "none",
    glucose_results: Sequence[Mapping[str, Any]] | None = None,
    baseline_window: tuple[float, float] = (-25.0, 0.0),
    t_grid: np.ndarray | None = None,
    xlim: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Build a common-grid matrix for a canonical session field or glucose output.

    Supported values are ``fp``, ``bg``, ``bg_increment``, and ``bg_derivative``.
    Normalization is delegated to :func:`photometry.plotting.utils.normalize_fp`.
    """
    if not sessions:
        raise ValueError("sessions must contain at least one session")

    if glucose_results is None and value in {"bg_increment", "bg_derivative"}:
        glucose_results = [compute_glucose_metrics(dict(session)) for session in sessions]
    if glucose_results is not None and len(glucose_results) != len(sessions):
        raise ValueError("glucose_results and sessions must have the same length")

    times: list[np.ndarray] = []
    traces: list[np.ndarray] = []
    for index, session in enumerate(sessions):
        t = np.asarray(session["t"], dtype=float)
        if value in {"fp", "bg"}:
            trace = np.asarray(session[value], dtype=float)
        elif value in {"bg_increment", "bg_derivative"}:
            assert glucose_results is not None
            trace = np.asarray(glucose_results[index]["series"][value], dtype=float)
        else:
            raise ValueError(
                "value must be one of {'fp', 'bg', 'bg_increment', 'bg_derivative'}"
            )
        if t.shape != trace.shape:
            raise ValueError(f"session {index} has mismatched time and {value} shapes")
        times.append(t)
        traces.append(trace)

    if t_grid is None:
        t_grid = np.asarray(times[0], dtype=float)
        if xlim is not None:
            t_grid = t_grid[(t_grid >= xlim[0]) & (t_grid <= xlim[1])]
    if t_grid.size < 2:
        raise ValueError("the requested time grid contains fewer than two samples")

    matrix, t_grid = build_matrix(
        traces,
        times,
        t_grid=t_grid,
        normalization=normalization,
        baseline_window=baseline_window,
    )
    return {
        "matrix": matrix,
        "t": t_grid,
        "value": value,
        "normalization": normalization,
    }


def plot_multi_signal_summary(
    sessions: Sequence[Mapping[str, Any]],
    panels: Sequence[Mapping[str, Any]],
    *,
    glucose_results: Sequence[Mapping[str, Any]] | None = None,
    group_key: str = "dose_num",
    group_order: Sequence[Any] | None = None,
    group_label: str | None = None,
    xlim: tuple[float, float] = (-10.0, 40.0),
    baseline_window: tuple[float, float] = (-25.0, 0.0),
    group_colors: Mapping[Any, Any] | Sequence[Any] | None = None,
    figsize: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Plot legacy-style cohort heatmaps with group mean ± SEM traces below.

    ``panels`` is a sequence of dictionaries. Each dictionary requires ``value``
    and may define ``normalization``, ``title``, ``ylabel``, ``cmap``, ``vmin``,
    and ``vmax``.
    """
    if not sessions:
        raise ValueError("sessions must contain at least one session")
    if not panels:
        raise ValueError("panels must contain at least one panel specification")

    groups, group_rows = _group_indices(
        sessions,
        group_key=group_key,
        group_order=group_order,
    )
    ordered_indices = np.concatenate(group_rows)
    if ordered_indices.size != len(sessions):
        raise RuntimeError("grouping did not assign every session")

    if figsize is None:
        figsize = (4.8 * len(panels), 7.5)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    outer = fig.add_gridspec(2, len(panels), height_ratios=(2.2, 1.0))

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if isinstance(group_colors, Mapping):
        colors = [group_colors.get(group, default_colors[i % len(default_colors)]) for i, group in enumerate(groups)]
    elif group_colors is not None:
        colors = [group_colors[i % len(group_colors)] for i in range(len(groups))]
    else:
        colors = [default_colors[i % len(default_colors)] for i in range(len(groups))]

    heat_axes = []
    line_axes = []
    prepared_panels = []

    for column, panel in enumerate(panels):
        heat_ax = fig.add_subplot(outer[0, column])
        line_ax = fig.add_subplot(outer[1, column])
        heat_axes.append(heat_ax)
        line_axes.append(line_ax)

        prepared = prepare_session_matrix(
            sessions,
            value=str(panel["value"]),
            normalization=str(panel.get("normalization", "none")),
            glucose_results=glucose_results,
            baseline_window=baseline_window,
            xlim=xlim,
        )
        prepared_panels.append(prepared)
        matrix = prepared["matrix"]
        t = prepared["t"]
        matrix_ordered = matrix[ordered_indices]

        im = heat_ax.imshow(
            matrix_ordered,
            aspect="auto",
            interpolation="nearest",
            extent=(t[0], t[-1], matrix_ordered.shape[0] - 0.5, -0.5),
            cmap=panel.get("cmap", "RdBu_r"),
            vmin=panel.get("vmin"),
            vmax=panel.get("vmax"),
        )
        heat_ax.axvline(0.0, color="k", linewidth=0.9, alpha=0.8)
        heat_ax.set_xlim(xlim)
        heat_ax.set_title(str(panel.get("title", panel["value"])))
        heat_ax.set_xlabel("Time (min)")
        fig.colorbar(im, ax=heat_ax, fraction=0.045, pad=0.02)

        centers = []
        labels = []
        cursor = 0
        for group, rows in zip(groups, group_rows):
            start = cursor
            cursor += rows.size
            centers.append((start + cursor - 1) / 2.0)
            labels.append(f"{_format_group(group)} (n={rows.size})")
            if cursor < len(sessions):
                heat_ax.axhline(cursor - 0.5, color="k", linewidth=0.8, alpha=0.6)
        heat_ax.set_yticks(centers)
        heat_ax.set_yticklabels(labels)
        heat_ax.set_ylabel(group_label or group_key)

        for group, rows, color in zip(groups, group_rows, colors):
            summary = compute_group_average(matrix[rows])
            label = f"{_format_group(group)} (n={summary['n']})"
            line_ax.plot(t, summary["mean"], linewidth=1.8, color=color, label=label)
            line_ax.fill_between(
                t,
                summary["mean"] - summary["sem"],
                summary["mean"] + summary["sem"],
                color=color,
                alpha=0.2,
                linewidth=0,
            )
        line_ax.axvline(0.0, color="k", linewidth=0.9, alpha=0.8)
        line_ax.set_xlim(xlim)
        line_ax.set_xlabel("Time (min)")
        line_ax.set_ylabel(str(panel.get("ylabel", panel["value"])))
        line_ax.grid(False)
        apply_y_ticks(line_ax)
        if column == len(panels) - 1:
            line_ax.legend(frameon=False, fontsize=8)

    return {
        "fig": fig,
        "axes": {"heatmaps": heat_axes, "means": line_axes},
        "groups": groups,
        "group_indices": group_rows,
        "ordered_indices": ordered_indices,
        "panels": prepared_panels,
    }
