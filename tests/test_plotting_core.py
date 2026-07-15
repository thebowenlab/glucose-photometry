from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photometry import (
    CrossCorrelationConfig,
    compute_glucose_metrics,
    compute_session_cross_correlations,
    plot_cross_correlation_summary,
    plot_metrics_by_group,
    plot_multi_signal_summary,
    plot_session_summary,
    prepare_session_matrix,
)
from photometry.plotting.utils import normalize_fp


def _session(subject: str, dose: float, phase: float = 0.0):
    t = np.arange(-25.0, 60.0 + 0.1, 0.1)
    response = np.where(
        t >= 0,
        (25.0 + 10.0 * dose) * (1.0 - np.exp(-t / 3.0)) * np.exp(-t / 28.0),
        0.0,
    )
    fp = -np.exp(-((t - 3.0 - phase) / 4.0) ** 2) + 0.05 * np.sin(t / 2.0)
    return {
        "subject": subject,
        "date": "2025-01-01",
        "route": "OG",
        "dose_num": dose,
        "t": t,
        "bg": 100.0 + response,
        "fp": fp,
    }


def test_percent_baseline_normalization():
    values = np.asarray([100.0, 100.0, 150.0])
    normalized = normalize_fp(
        values,
        baseline_mask=np.asarray([True, True, False]),
        mode="percent",
    )
    np.testing.assert_allclose(normalized, [100.0, 100.0, 150.0])


def test_session_and_cohort_plotting_smoke():
    sessions = [_session("M1", 1.0), _session("M2", 2.0, 0.3)]
    glucose = [compute_glucose_metrics(session) for session in sessions]

    single = plot_session_summary(sessions[0], glucose_result=glucose[0])
    assert single["fp_normalized"].shape == sessions[0]["t"].shape

    prepared = prepare_session_matrix(
        sessions,
        value="bg",
        normalization="percent",
        xlim=(-10.0, 40.0),
    )
    assert prepared["matrix"].shape[0] == 2

    cohort = plot_multi_signal_summary(
        sessions,
        panels=(
            {"value": "fp", "normalization": "zscore"},
            {"value": "bg", "normalization": "percent"},
            {"value": "bg_derivative", "normalization": "none"},
        ),
        glucose_results=glucose,
        group_order=(1.0, 2.0),
    )
    assert len(cohort["axes"]["heatmaps"]) == 3
    plt.close(single["fig"])
    plt.close(cohort["fig"])


def test_correlation_and_metric_plotting_smoke():
    sessions = [_session("M1", 1.0), _session("M2", 2.0, 0.3)]
    config = CrossCorrelationConfig(
        max_lag_min=5.0,
        min_overlap=20,
        x_lowpass_period_min=None,
    )
    correlations = [
        compute_session_cross_correlations(session, config=config)
        for session in sessions
    ]
    corr = plot_cross_correlation_summary(
        sessions,
        correlations,
        group_order=(1.0, 2.0),
    )
    assert corr["matrix"].shape[0] == 2

    records = [
        {"dose_num": 1.0, "iauc": 20.0, "peak_time": 8.0},
        {"dose_num": 2.0, "iauc": 35.0, "peak_time": 10.0},
    ]
    metrics = plot_metrics_by_group(
        records,
        {"iAUC": "iauc", "Peak time": "peak_time"},
        group_order=(1.0, 2.0),
    )
    assert len(metrics["axes"]) == 2
    plt.close(corr["fig"])
    plt.close(metrics["fig"])
