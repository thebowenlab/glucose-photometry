from __future__ import annotations

import numpy as np

from photometry import (
    CircularShiftConfig,
    CrossCorrelationConfig,
    EarlyModelConfig,
    GlucoseAnalysisConfig,
    compute_circular_shift_correlation_null,
    compute_cross_correlation,
    compute_glucose_metrics,
    fit_early_model,
    generate_circular_shift_nulls,
    residualize_early_model,
)
from photometry.analysis._numeric import shifted_template


def test_glucose_metrics_return_incremental_kinetics():
    t = np.arange(-10.0, 30.0 + 0.1, 0.1)
    response = np.where(
        t >= 0,
        45.0 * (1.0 - np.exp(-t / 3.0)) * np.exp(-t / 22.0),
        0.0,
    )
    session = {
        "t": t,
        "bg": 100.0 + response,
        "fp": np.zeros_like(t),
    }
    result = compute_glucose_metrics(
        session,
        config=GlucoseAnalysisConfig(
            baseline_window=(-10.0, 0.0),
            onset_baseline_window=(-5.0, 0.0),
            peak_window=(0.0, 30.0),
            iauc_window=(0.0, 30.0),
            onset_min_run_minutes=0.5,
        ),
    )

    assert result["found_onset"]
    assert 0.0 <= result["onset_time"] <= 1.0
    assert result["peak_time"] > result["onset_time"]
    assert result["peak_increment"] > 20.0
    assert result["iauc"] > 0.0
    assert result["series"]["bg_derivative"].shape == t.shape
    assert abs(result["baseline"] - 100.0) < 0.1


def test_cross_correlation_recovers_known_positive_lag():
    rng = np.random.default_rng(3)
    t = np.arange(-20.0, 20.0, 0.1)
    y = np.convolve(rng.normal(size=t.size), np.ones(9) / 9.0, mode="same")
    lag_samples = 6
    x = np.full_like(y, np.nan)
    x[lag_samples:] = y[:-lag_samples]
    session = {"t": t, "fp": x, "bg": y}

    result = compute_cross_correlation(
        session,
        config=CrossCorrelationConfig(
            max_lag_min=2.0,
            method="pearson",
            min_overlap=50,
            x_lowpass_period_min=None,
        ),
    )

    assert abs(result["best_lag_min"] - lag_samples * 0.1) <= 0.11
    assert result["best_correlation"] > 0.99


def test_circular_shift_nulls_are_reproducible_and_respect_guard():
    t = np.arange(0.0, 100.0, 0.5)
    signal = np.arange(t.size, dtype=float)
    session = {"t": t, "bg": signal, "fp": signal[::-1]}
    config = CircularShiftConfig(
        n_null=20,
        min_shift_min=10.0,
        max_shift_min=30.0,
        seed=11,
    )

    first = generate_circular_shift_nulls(session, config=config)
    second = generate_circular_shift_nulls(session, config=config)

    np.testing.assert_array_equal(first["null_signals"], second["null_signals"])
    assert first["null_signals"].shape == (20, t.size)
    assert np.all(first["effective_shift_min"] >= 10.0)
    assert np.all(first["effective_shift_min"] <= 30.0)


def test_early_model_fit_and_residualization_recover_synthetic_component():
    t = np.arange(-25.0, 10.0 + 0.05, 0.05)
    base = (
        0.7 * np.sin((t + 8.0) / 3.2)
        + 0.35 * np.cos((t - 2.0) / 1.7)
        + 0.02 * t
    )
    true_lag = 0.5
    shifted, valid = shifted_template(t, base, t, true_lag, fill="hold")
    target = 20.0 + 4.0 * (0.3 + 1.4 * shifted)
    session = {"t": t, "fp": target, "bg": np.full_like(t, 100.0)}
    templates = np.vstack([base, base, base])
    config = EarlyModelConfig(
        baseline_window=(-25.0, 0.0),
        fit_window=(-25.0, 1.0),
        lowpass_period_min=None,
        lag_min=-1.0,
        lag_max=1.0,
        lag_step=0.25,
        minimum_template_scale=None,
        min_fit_samples=100,
    )

    fit = fit_early_model(session, t, templates, config=config)
    residualization = residualize_early_model(
        session,
        fit,
        t,
        base,
        config=config,
    )

    assert abs(fit["lag_min"] - true_lag) <= config.lag_step
    assert fit["cv_r2"] > 0.99
    mask = residualization["valid_mask"] & valid
    residual_rmse = float(np.sqrt(np.mean(residualization["residual"][mask] ** 2)))
    signal_sd = float(np.std(residualization["signal_filtered"][mask]))
    assert residual_rmse < signal_sd * 0.02


def test_circular_shift_correlation_null_returns_inference_arrays():
    rng = np.random.default_rng(7)
    t = np.arange(-25.0, 75.0, 0.25)
    bg = np.convolve(rng.normal(size=t.size), np.ones(7) / 7.0, mode="same")
    fp = np.roll(bg, 4)
    session = {"t": t, "bg": bg, "fp": fp}

    result = compute_circular_shift_correlation_null(
        session,
        correlation_config=CrossCorrelationConfig(
            max_lag_min=3.0,
            x_lowpass_period_min=None,
            method="spearman",
        ),
        null_config=CircularShiftConfig(
            n_null=12,
            min_shift_min=10.0,
            seed=5,
        ),
    )

    n_lags = result["observed"]["lags_min"].size
    assert result["null_correlation"].shape == (12, n_lags)
    assert result["pointwise_p_two_sided"].shape == (n_lags,)
    assert result["familywise_p_max_abs"].shape == (n_lags,)
