import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from photometry.plotting.group_average import build_matrix, compute_group_average
from photometry.plotting.utils import apply_y_ticks

def plot_dataset_comparison(
    *,
    traces_a: list[np.ndarray],
    times_a: list[np.ndarray],
    traces_b: list[np.ndarray],
    times_b: list[np.ndarray],
    label_a: str = "Dataset A",
    label_b: str = "Dataset B",
    t_grid: np.ndarray | None = None,
    normalization: str = "none",
    baseline_window: tuple[float, float] = (-5.0, 0.0),
    group_colors: tuple[str | None, str | None] = (None, None),
    alpha_sem: float = 0.22,
    linewidth: float = 2.0,
    title: str | None = None,
    remove_spines: tuple[str, ...] | None = ("top", "right"),
    plot_window: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (7.5, 4.5),
    ax=None,
):
    if t_grid is None:
        t_grid = np.asarray(times_a[0], dtype=float)

    mat_a, t_grid = build_matrix(
        traces_a,
        times_a,
        t_grid=t_grid,
        normalization=normalization,
        baseline_window=baseline_window,
    )
    mat_b, _ = build_matrix(
        traces_b,
        times_b,
        t_grid=t_grid,
        normalization=normalization,
        baseline_window=baseline_window,
    )

    s_a = compute_group_average(mat_a)
    s_b = compute_group_average(mat_b)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        made_fig = True
    else:
        fig = ax.figure
        made_fig = False

    line_a, = ax.plot(
        t_grid,
        s_a["mean"],
        color=group_colors[0],
        linewidth=linewidth,
        label=f"{label_a} (n={s_a['n']})",
    )
    color_a = line_a.get_color()
    ax.fill_between(
        t_grid,
        s_a["mean"] - s_a["sem"],
        s_a["mean"] + s_a["sem"],
        color=color_a,
        alpha=alpha_sem,
        linewidth=0,
    )

    line_b, = ax.plot(
        t_grid,
        s_b["mean"],
        color=group_colors[1],
        linewidth=linewidth,
        label=f"{label_b} (n={s_b['n']})",
    )
    color_b = line_b.get_color()
    ax.fill_between(
        t_grid,
        s_b["mean"] - s_b["sem"],
        s_b["mean"] + s_b["sem"],
        color=color_b,
        alpha=alpha_sem,
        linewidth=0,
    )

    if t_grid.size > 1 and t_grid[0] <= 0 <= t_grid[-1]:
        ax.axvline(0, color="k", linewidth=1.5, alpha=0.8)

    if plot_window is not None:
        ax.set_xlim(plot_window)
    
    apply_y_ticks(ax)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(normalization if normalization != "none" else "Signal")
    
    for s in remove_spines or ():
        if s in ax.spines:
            ax.spines[s].set_visible(False)

    if title is not None:
        ax.set_title(title)

    ax.legend()

    if made_fig:
        fig.tight_layout()

    return {
        "fig": fig,
        "ax": ax,
        "t": t_grid,
        "a": {"mat": mat_a, **s_a, "color": color_a},
        "b": {"mat": mat_b, **s_b, "color": color_b},
    }


def plot_two_group_bar(
    values_a,
    values_b,
    *,
    label_a: str = "Group A",
    label_b: str = "Group B",
    group_colors: tuple[str | None, str | None] = (None, None),
    ylabel: str = "Value",
    title: str | None = None,
    figsize: tuple[float, float] = (4.8, 4.5),
    jitter: float = 0.08,
    point_size: float = 42.0,
    bar_alpha: float = 0.35,
    bar_width: float = 0.6,
    show_points: bool = True,
    show_mean_bar: bool = True,
    show_pvalue: bool = True,
    remove_spines: tuple[str, ...] | None = ("top", "right"),
    ax=None,
):
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if a.size == 0 or b.size == 0:
        raise ValueError("Both groups must contain at least one finite value.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        made_fig = True
    else:
        fig = ax.figure
        made_fig = False

    x = np.array([0.0, 1.0], dtype=float)

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))

    if show_mean_bar:
        bars = ax.bar(
            x,
            [mean_a, mean_b],
            width=bar_width,
            color=[group_colors[0], group_colors[1]],
            alpha=bar_alpha,
            edgecolor=[group_colors[0], group_colors[1]],
            linewidth=1.5,
            zorder=1,
        )
        color_a = bars[0].get_facecolor()
        color_b = bars[1].get_facecolor()
    else:
        sc_a = ax.scatter([], [], color=group_colors[0])
        sc_b = ax.scatter([], [], color=group_colors[1])
        color_a = sc_a.get_facecolor()[0]
        color_b = sc_b.get_facecolor()[0]
        sc_a.remove()
        sc_b.remove()

    rng = np.random.RandomState(0)

    if show_points:
        xa = x[0] + rng.uniform(-jitter, jitter, size=a.size)
        xb = x[1] + rng.uniform(-jitter, jitter, size=b.size)

        ax.scatter(xa, a, s=point_size, color=color_a, zorder=3)
        ax.scatter(xb, b, s=point_size, color=color_b, zorder=3)

    stat, p = mannwhitneyu(a, b, alternative="two-sided")

    if show_pvalue:
        y_max = max(np.max(a), np.max(b), mean_a, mean_b)
        y_min = min(np.min(a), np.min(b), mean_a, mean_b)
        y_range = y_max - y_min if y_max > y_min else 1.0
        pad = 0.08 * y_range

        y_bracket = y_max + pad
        y_text = y_bracket + 0.45 * pad

        ax.plot(
            [x[0], x[0], x[1], x[1]],
            [y_bracket, y_bracket + 0.3 * pad, y_bracket + 0.3 * pad, y_bracket],
            color="k",
            linewidth=1.2,
            zorder=4,
        )
        ax.text(
            np.mean(x),
            y_text,
            f"Mann–Whitney p={p:.3g}",
            ha="center",
            va="bottom",
            zorder=5,
        )

        cur_y0, cur_y1 = ax.get_ylim()
        y_top_needed = y_text + 0.6 * pad
        if y_top_needed > cur_y1:
            ax.set_ylim(cur_y0, y_top_needed)

    ax.set_xticks(x)
    ax.set_xticklabels([label_a, label_b])
    ax.set_ylabel(ylabel)
    apply_y_ticks(ax)
    
    for s in remove_spines or ():
        if s in ax.spines:
            ax.spines[s].set_visible(False)

    if title is not None:
        ax.set_title(title)

    ax.set_xlim(-0.5, 1.5)

    if made_fig:
        fig.tight_layout()

    return {
        "fig": fig,
        "ax": ax,
        "means": (mean_a, mean_b),
        "ns": (a.size, b.size),
        "statistic": float(stat),
        "pvalue": float(p),
        "colors": (color_a, color_b),
    }