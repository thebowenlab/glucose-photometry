from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from photometry.plotting.utils import apply_y_ticks


def _equal(left: Any, right: Any) -> bool:
    try:
        return bool(np.isclose(float(left), float(right), equal_nan=False))
    except (TypeError, ValueError):
        return left == right


def _label(value: Any) -> str:
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return f"{float(value):g}"
    return str(value)


def plot_metrics_by_group(
    records: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, str],
    *,
    group_key: str = "dose_num",
    group_order: Sequence[Any] | None = None,
    group_colors: Mapping[Any, Any] | Sequence[Any] | None = None,
    ncols: int = 2,
    figsize_per_panel: tuple[float, float] = (3.8, 3.4),
    jitter: float = 0.08,
) -> dict[str, Any]:
    """Plot metric distributions by a categorical or numeric session group."""
    if not records:
        raise ValueError("records must contain at least one record")
    if not metrics:
        raise ValueError("metrics must contain at least one label-to-key mapping")

    observed = []
    for record in records:
        value = record.get(group_key)
        if not any(_equal(value, item) for item in observed):
            observed.append(value)
    groups = list(group_order) if group_order is not None else observed
    groups = [group for group in groups if any(_equal(record.get(group_key), group) for record in records)]

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if isinstance(group_colors, Mapping):
        colors = [group_colors.get(group, default_colors[i % len(default_colors)]) for i, group in enumerate(groups)]
    elif group_colors is not None:
        colors = [group_colors[i % len(group_colors)] for i in range(len(groups))]
    else:
        colors = [default_colors[i % len(default_colors)] for i in range(len(groups))]

    n_metrics = len(metrics)
    ncols = max(1, min(int(ncols), n_metrics))
    nrows = int(np.ceil(n_metrics / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    rng = np.random.default_rng(0)
    summaries = {}

    for axis, (display_label, key) in zip(axes.flat, metrics.items()):
        values_by_group = []
        active_positions = []
        active_colors = []
        for position, (group, color) in enumerate(zip(groups, colors), start=1):
            values = np.asarray(
                [record.get(key, np.nan) for record in records if _equal(record.get(group_key), group)],
                dtype=float,
            )
            values = values[np.isfinite(values)]
            values_by_group.append(values)
            if values.size:
                active_positions.append(position)
                active_colors.append(color)
                x = position + rng.uniform(-jitter, jitter, size=values.size)
                axis.scatter(x, values, s=28, color=color, alpha=0.85, zorder=3)

        active_values = [values_by_group[pos - 1] for pos in active_positions]
        if active_values:
            box = axis.boxplot(
                active_values,
                positions=active_positions,
                widths=0.5,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "k", "linewidth": 1.2},
            )
            for patch, color in zip(box["boxes"], active_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.22)
                patch.set_edgecolor(color)

        axis.set_xticks(np.arange(1, len(groups) + 1))
        axis.set_xticklabels([_label(group) for group in groups])
        axis.set_xlabel(group_key)
        axis.set_ylabel(display_label)
        axis.set_title(display_label)
        axis.grid(False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        apply_y_ticks(axis)
        summaries[key] = {
            group: values for group, values in zip(groups, values_by_group)
        }

    for axis in axes.flat[n_metrics:]:
        axis.set_visible(False)

    return {
        "fig": fig,
        "axes": list(axes.flat[:n_metrics]),
        "groups": groups,
        "summaries": summaries,
    }
