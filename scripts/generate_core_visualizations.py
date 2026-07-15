from __future__ import annotations

"""Generate core glucose-photometry visualizations from the released dataset.

Edit DATASET_DIR and OUTPUT_DIR in the USER SETTINGS section before running this
file in PyCharm. The script intentionally performs analysis/aggregation outside
of the package core; reusable plotting and single-session computations remain in
``photometry``.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from photometry import (
    CrossCorrelationConfig,
    compute_glucose_metrics,
    compute_session_cross_correlations,
    load_sessions,
    plot_cross_correlation_summary,
    plot_metrics_by_group,
    plot_multi_signal_summary,
    plot_session_summary,
)


# ---------------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------------

# Folder containing agrp-photometry-bg.pkl.
DATASET_DIR = Path("/Users/annabowen/Downloads/31891579")
DATASET_FILENAME = "agrp-photometry-bg.pkl"

# Figures and the session-metrics CSV will be written here.
OUTPUT_DIR = Path("/Users/annabowen/Downloads/glucose-photometry/core-visualizations")

# Shared plotting/analysis selections.
PLOT_WINDOW_MIN = (-10.0, 40.0)
OG_DOSES = (0.5, 1.0, 2.0, 2.5)
ROUTE_ORDER = ("IV", "OG")
FIGURE_DPI = 250
SHOW_FIGURES = True

# Example session. Leave subject as None to use the first matching OG 2 g/kg session.
EXAMPLE_SUBJECT: str | None = None
EXAMPLE_ROUTE = "OG"
EXAMPLE_DOSE = 2.0

# Lagged-correlation settings.
CORRELATION_CONFIG = CrossCorrelationConfig(
    max_lag_min=20.0,
    method="spearman",
    min_overlap=10,
    x_lowpass_period_min=4.0,
    x_lowpass_order=3,
    y_derivative_smooth_minutes=1.0,
)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------


def values_match(left: Any, right: Any) -> bool:
    try:
        return bool(np.isclose(float(left), float(right), equal_nan=False))
    except (TypeError, ValueError):
        return left == right


def select_sessions(
    sessions: list[dict[str, Any]],
    results: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
    *,
    route: str | None = None,
    doses: tuple[float, ...] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    keep = []
    for index, session in enumerate(sessions):
        if route is not None and str(session.get("route")).upper() != route.upper():
            continue
        if doses is not None and not any(values_match(session.get("dose_num"), dose) for dose in doses):
            continue
        keep.append(index)
    return (
        [sessions[index] for index in keep],
        [results[index] for index in keep],
        [correlations[index] for index in keep],
    )


def save_figure(fig: plt.Figure, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    if not SHOW_FIGURES:
        plt.close(fig)
    print(f"Saved: {path}")
    return path


def build_metric_records(
    sessions: list[dict[str, Any]],
    glucose_results: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for session, glucose, coupling in zip(sessions, glucose_results, correlations):
        records.append(
            {
                "subject": session.get("subject"),
                "date": session.get("date"),
                "route": session.get("route"),
                "dose_num": session.get("dose_num"),
                "baseline": glucose["baseline"],
                "onset_time": glucose["onset_time"],
                "derivative_peak_time": glucose["derivative_peak_time"],
                "peak_time": glucose["peak_time"],
                "peak_increment": glucose["peak_increment"],
                "iauc": glucose["iauc"],
                "fp_bg_best_r": coupling["glucose"]["best_correlation"],
                "fp_bg_best_lag_min": coupling["glucose"]["best_lag_min"],
                "fp_dbg_best_r": coupling["glucose_derivative"]["best_correlation"],
                "fp_dbg_best_lag_min": coupling["glucose_derivative"]["best_lag_min"],
            }
        )
    return records


def find_example_index(sessions: list[dict[str, Any]]) -> int:
    for index, session in enumerate(sessions):
        if str(session.get("route")).upper() != EXAMPLE_ROUTE.upper():
            continue
        if not values_match(session.get("dose_num"), EXAMPLE_DOSE):
            continue
        if EXAMPLE_SUBJECT is not None and str(session.get("subject")) != str(EXAMPLE_SUBJECT):
            continue
        return index
    raise RuntimeError(
        "No session matched EXAMPLE_SUBJECT/EXAMPLE_ROUTE/EXAMPLE_DOSE. "
        "Edit those settings near the top of the script."
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main() -> None:
    dataset_file = DATASET_DIR.expanduser().resolve() / DATASET_FILENAME
    if not dataset_file.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_file}\n"
            "Edit DATASET_DIR near the top of this script."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading: {dataset_file}")
    sessions = load_sessions(dataset_file)
    print(f"Loaded {len(sessions)} sessions")

    print("Computing glucose kinetics and lagged correlations...")
    glucose_results = [compute_glucose_metrics(session) for session in sessions]
    correlations = [
        compute_session_cross_correlations(session, config=CORRELATION_CONFIG)
        for session in sessions
    ]
    records = build_metric_records(sessions, glucose_results, correlations)
    pd.DataFrame(records).to_csv(OUTPUT_DIR / "session_core_metrics.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'session_core_metrics.csv'}")

    # 1. Annotated single-session stacked trace.
    example_index = find_example_index(sessions)
    out = plot_session_summary(
        sessions[example_index],
        glucose_result=glucose_results[example_index],
        xlim=PLOT_WINDOW_MIN,
    )
    save_figure(out["fig"], "01_example_session_summary.png")

    # 2. OG dose heatmaps with group mean ± SEM lines.
    og_sessions, og_glucose, og_correlations = select_sessions(
        sessions,
        glucose_results,
        correlations,
        route="OG",
        doses=OG_DOSES,
    )
    out = plot_multi_signal_summary(
        og_sessions,
        panels=(
            {
                "value": "fp",
                "normalization": "zscore",
                "title": "Photometry",
                "ylabel": "FP (z-score)",
                "cmap": "RdBu_r",
                "vmin": -3.0,
                "vmax": 3.0,
            },
            {
                "value": "bg",
                "normalization": "percent",
                "title": "Glucose",
                "ylabel": "Glucose (% baseline)",
                "cmap": "RdBu_r",
                "vmin": 75.0,
                "vmax": 250.0,
            },
            {
                "value": "bg_derivative",
                "normalization": "none",
                "title": "Glucose derivative",
                "ylabel": "d glucose/dt",
                "cmap": "RdBu_r",
                "vmin": -15.0,
                "vmax": 15.0,
            },
        ),
        glucose_results=og_glucose,
        group_key="dose_num",
        group_order=OG_DOSES,
        group_label="Dose (g/kg)",
        xlim=PLOT_WINDOW_MIN,
        figsize=(15.0, 7.5),
    )
    save_figure(out["fig"], "02_og_dose_heatmaps_and_means.png")

    # 3. IV versus OG comparison at 2 g/kg.
    route_sessions, route_glucose, _ = select_sessions(
        sessions,
        glucose_results,
        correlations,
        doses=(2.0,),
    )
    out = plot_multi_signal_summary(
        route_sessions,
        panels=(
            {
                "value": "fp",
                "normalization": "zscore",
                "title": "Photometry: 2 g/kg",
                "ylabel": "FP (z-score)",
                "cmap": "RdBu_r",
                "vmin": -3.0,
                "vmax": 3.0,
            },
            {
                "value": "bg",
                "normalization": "percent",
                "title": "Glucose: 2 g/kg",
                "ylabel": "Glucose (% baseline)",
                "cmap": "RdBu_r",
                "vmin": 75.0,
                "vmax": 350.0,
            },
        ),
        glucose_results=route_glucose,
        group_key="route",
        group_order=ROUTE_ORDER,
        group_label="Route",
        xlim=PLOT_WINDOW_MIN,
        figsize=(10.0, 7.5),
    )
    save_figure(out["fig"], "03_route_comparison_2gkg.png")

    # 4. Core glucose kinetics by oral dose.
    og_records = [record for record in records if str(record["route"]).upper() == "OG" and any(values_match(record["dose_num"], dose) for dose in OG_DOSES)]
    out = plot_metrics_by_group(
        og_records,
        {
            "Glucose onset (min)": "onset_time",
            "Maximum rise time (min)": "derivative_peak_time",
            "Glucose peak time (min)": "peak_time",
            "Peak Δ glucose (mg/dL)": "peak_increment",
            "Incremental AUC": "iauc",
        },
        group_key="dose_num",
        group_order=OG_DOSES,
        ncols=3,
    )
    save_figure(out["fig"], "04_glucose_metrics_by_og_dose.png")

    # 5–6. FP cross-correlation with glucose and its derivative across OG doses.
    out = plot_cross_correlation_summary(
        og_sessions,
        og_correlations,
        target="glucose",
        group_key="dose_num",
        group_order=OG_DOSES,
        title="FP × glucose cross-correlation",
        xlim=(-20.0, 20.0),
        vmin=-0.6,
        vmax=0.6,
    )
    save_figure(out["fig"], "05_xcorr_fp_glucose.png")

    out = plot_cross_correlation_summary(
        og_sessions,
        og_correlations,
        target="glucose_derivative",
        group_key="dose_num",
        group_order=OG_DOSES,
        title="FP × glucose-derivative cross-correlation",
        xlim=(-20.0, 20.0),
        vmin=-0.6,
        vmax=0.6,
    )
    save_figure(out["fig"], "06_xcorr_fp_glucose_derivative.png")

    print("Visualization generation completed successfully.")
    if SHOW_FIGURES:
        plt.show()


if __name__ == "__main__":
    main()
