from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from photometry.analysis._numeric import shifted_template, time_derivative
from photometry.analysis._session import (
    SessionLike,
    as_template_matrix,
    get_aligned_session_arrays,
    window_mask,
)
from photometry.signal.fp_series import lowpass_minutes


@dataclass(slots=True)
class EarlyModelConfig:
    """Configuration for fitting the early photometry template model.

    Candidate templates must be on a shared ``template_t`` axis and should be
    baseline-z-scored. The external driver is responsible for choosing which
    sessions form the candidate bank and which template is later subtracted.
    """

    signal_key: str = "fp"
    baseline_window: tuple[float, float] = (-25.0, 0.0)
    fit_window: tuple[float, float] = (-25.0, 1.0)
    lowpass_period_min: float | None = 4.0
    lowpass_order: int = 3
    lag_min: float = -1.0
    lag_max: float = 1.0
    lag_step: float = 0.25
    include_template_derivative: bool = False
    ridge_alpha: float = 1e-4
    minimum_template_scale: float | None = 0.10
    min_fit_samples: int = 20
    refit_on_mean_after_cv: bool = True

    def lag_grid(self) -> np.ndarray:
        if self.lag_step <= 0:
            raise ValueError("lag_step must be positive")
        if self.lag_max < self.lag_min:
            raise ValueError("lag_max must be greater than or equal to lag_min")
        return np.arange(
            self.lag_min,
            self.lag_max + self.lag_step * 0.5,
            self.lag_step,
            dtype=float,
        )


def prepare_early_model_session(
    session: SessionLike,
    *,
    config: EarlyModelConfig | None = None,
) -> dict[str, Any]:
    """Select, filter, and baseline-normalize one session for early-model fitting."""
    cfg = config or EarlyModelConfig()
    t, signal = get_aligned_session_arrays(
        session,
        ("t", cfg.signal_key),
        min_length=3,
    )

    if cfg.lowpass_period_min is None:
        filtered = signal.copy()
    else:
        filtered = lowpass_minutes(
            t,
            signal,
            cutoff_period_min=cfg.lowpass_period_min,
            order=cfg.lowpass_order,
        )

    baseline_mask = window_mask(t, cfg.baseline_window, finite=np.isfinite(filtered))
    if np.count_nonzero(baseline_mask) < 3:
        raise ValueError(
            f"fewer than three finite samples fall in baseline window {cfg.baseline_window}"
        )

    baseline_mean = float(np.nanmean(filtered[baseline_mask]))
    baseline_sd = float(np.nanstd(filtered[baseline_mask]))
    if not np.isfinite(baseline_sd) or baseline_sd <= 0:
        raise ValueError("baseline standard deviation is zero or non-finite")

    signal_z = (filtered - baseline_mean) / baseline_sd
    fit_mask = window_mask(t, cfg.fit_window, finite=np.isfinite(signal_z))
    if np.count_nonzero(fit_mask) < cfg.min_fit_samples:
        raise ValueError(
            f"only {np.count_nonzero(fit_mask)} samples fall in fit window "
            f"{cfg.fit_window}; at least {cfg.min_fit_samples} are required"
        )

    return {
        "t": t.copy(),
        "signal_raw": signal.copy(),
        "signal_filtered": filtered,
        "signal_z": signal_z,
        "baseline_mask": baseline_mask,
        "fit_mask": fit_mask,
        "baseline_mean": baseline_mean,
        "baseline_sd": baseline_sd,
        "config": asdict(cfg),
    }


def _build_design(
    t: np.ndarray,
    template: np.ndarray,
    valid: np.ndarray,
    *,
    include_derivative: bool,
) -> np.ndarray:
    columns = [np.ones_like(template), template]
    if include_derivative:
        derivative = time_derivative(
            t,
            np.where(valid, template, np.nan),
            preserve_nans=True,
        )
        columns.append(derivative)
    return np.column_stack(columns)


def _fit_linear_model(
    design: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    *,
    ridge_alpha: float,
    minimum_template_scale: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(valid, dtype=bool) & np.isfinite(target)
    mask &= np.all(np.isfinite(design), axis=1)
    if np.count_nonzero(mask) < design.shape[1] + 3:
        return np.full(design.shape[1], np.nan), mask

    x = design[mask]
    y = target[mask]
    try:
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        penalty = np.eye(x.shape[1], dtype=float) * float(ridge_alpha)
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(x.T @ x + penalty, x.T @ y)

    if minimum_template_scale is not None and beta.size >= 2:
        beta[1] = max(float(beta[1]), float(minimum_template_scale))
    return beta.astype(float), mask


def _r2_score(observed: np.ndarray, predicted: np.ndarray) -> float:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if np.count_nonzero(valid) < 5:
        return np.nan
    y = observed[valid]
    yhat = predicted[valid]
    mse = float(np.mean((y - yhat) ** 2))
    variance = float(np.mean((y - np.mean(y)) ** 2))
    if variance <= 0 or not np.isfinite(variance):
        return np.nan
    return 1.0 - mse / variance


def select_early_model_data(
    session: SessionLike,
    template_t: np.ndarray,
    templates: np.ndarray,
    *,
    config: EarlyModelConfig | None = None,
) -> dict[str, Any]:
    """Prepare one target session and interpolate a candidate template bank.

    This function performs only data selection/alignment. It is useful when an
    external script needs to inspect the exact samples used by the model.
    """
    cfg = config or EarlyModelConfig()
    prepared = prepare_early_model_session(session, config=cfg)
    template_t = np.asarray(template_t, dtype=float)
    matrix = as_template_matrix(templates)
    if template_t.ndim != 1 or template_t.size != matrix.shape[1]:
        raise ValueError("template_t length must match the template time dimension")
    if np.any(np.diff(template_t) <= 0):
        raise ValueError("template_t must be strictly increasing")

    fit_t = prepared["t"][prepared["fit_mask"]]
    fit_target_z = prepared["signal_z"][prepared["fit_mask"]]
    templates_on_fit_t = np.vstack(
        [np.interp(fit_t, template_t, row) for row in matrix]
    )

    return {
        **prepared,
        "fit_t": fit_t,
        "fit_target_z": fit_target_z,
        "template_t": template_t.copy(),
        "templates": matrix.copy(),
        "templates_on_fit_t": templates_on_fit_t,
    }


def fit_early_model(
    session: SessionLike,
    template_t: np.ndarray,
    templates: np.ndarray,
    *,
    config: EarlyModelConfig | None = None,
) -> dict[str, Any]:
    """Fit lag, intercept, template scale, and optional derivative scale.

    With multiple templates, lag selection uses the original leave-one-template-out
    scoring idea. After selecting lag, the default behavior refits coefficients on
    the mean template, which is more stable than retaining one best-fold estimate.
    Set ``refit_on_mean_after_cv=False`` for behavior closer to the original script.
    """
    cfg = config or EarlyModelConfig()
    selected = select_early_model_data(
        session,
        template_t,
        templates,
        config=cfg,
    )
    fit_t = selected["fit_t"]
    target_z = selected["fit_target_z"]
    bank = selected["templates_on_fit_t"]
    lag_grid = cfg.lag_grid()

    lag_scores = np.full(lag_grid.size, np.nan, dtype=float)
    fold_scores = np.full((lag_grid.size, bank.shape[0]), np.nan, dtype=float)
    fold_betas: list[list[np.ndarray]] = []

    for lag_index, lag in enumerate(lag_grid):
        betas_for_lag: list[np.ndarray] = []
        for held_out in range(bank.shape[0]):
            if bank.shape[0] == 1:
                train_template = bank[0]
                validation_template = bank[0]
            else:
                train_template = np.nanmean(
                    np.delete(bank, held_out, axis=0),
                    axis=0,
                )
                validation_template = bank[held_out]

            shifted_train, valid_train = shifted_template(
                fit_t,
                train_template,
                fit_t,
                lag,
                fill="hold",
            )
            train_design = _build_design(
                fit_t,
                shifted_train,
                valid_train,
                include_derivative=cfg.include_template_derivative,
            )
            beta, train_mask = _fit_linear_model(
                train_design,
                target_z,
                valid_train,
                ridge_alpha=cfg.ridge_alpha,
                minimum_template_scale=cfg.minimum_template_scale,
            )
            betas_for_lag.append(beta)
            if not np.all(np.isfinite(beta)):
                continue

            shifted_validation, valid_validation = shifted_template(
                fit_t,
                validation_template,
                fit_t,
                lag,
                fill="hold",
            )
            validation_design = _build_design(
                fit_t,
                shifted_validation,
                valid_validation,
                include_derivative=cfg.include_template_derivative,
            )
            prediction = validation_design @ beta
            prediction[~valid_validation] = np.nan
            fold_scores[lag_index, held_out] = _r2_score(target_z, prediction)

        fold_betas.append(betas_for_lag)
        finite_scores = fold_scores[lag_index, np.isfinite(fold_scores[lag_index])]
        if finite_scores.size:
            lag_scores[lag_index] = float(np.median(finite_scores))

    if not np.any(np.isfinite(lag_scores)):
        raise RuntimeError("no lag produced a valid early-model fit")

    best_lag_index = int(np.nanargmax(lag_scores))
    best_lag = float(lag_grid[best_lag_index])
    cv_score = float(lag_scores[best_lag_index])

    if cfg.refit_on_mean_after_cv:
        final_template = np.nanmean(bank, axis=0)
        shifted_final, valid_final = shifted_template(
            fit_t,
            final_template,
            fit_t,
            best_lag,
            fill="hold",
        )
        final_design = _build_design(
            fit_t,
            shifted_final,
            valid_final,
            include_derivative=cfg.include_template_derivative,
        )
        beta, final_mask = _fit_linear_model(
            final_design,
            target_z,
            valid_final,
            ridge_alpha=cfg.ridge_alpha,
            minimum_template_scale=cfg.minimum_template_scale,
        )
    else:
        row = fold_scores[best_lag_index]
        best_fold = int(np.nanargmax(row))
        beta = np.asarray(fold_betas[best_lag_index][best_fold], dtype=float)
        final_mask = np.isfinite(target_z)

    if not np.all(np.isfinite(beta)):
        raise RuntimeError("final early-model coefficients are non-finite")

    return {
        "intercept": float(beta[0]),
        "template_scale": float(beta[1]),
        "template_derivative_scale": float(beta[2]) if beta.size >= 3 else 0.0,
        "lag_min": best_lag,
        "cv_r2": cv_score,
        "lag_grid_min": lag_grid,
        "lag_scores": lag_scores,
        "fold_scores": fold_scores,
        "baseline_mean": float(selected["baseline_mean"]),
        "baseline_sd": float(selected["baseline_sd"]),
        "fit_sample_count": int(np.count_nonzero(final_mask)),
        "config": asdict(cfg),
        "selection": {
            "fit_t": fit_t,
            "fit_target_z": target_z,
            "fit_mask": selected["fit_mask"],
        },
    }


def residualize_early_model(
    session: SessionLike,
    fit: dict[str, Any],
    template_t: np.ndarray,
    residual_templates: np.ndarray,
    *,
    config: EarlyModelConfig | None = None,
    subtract_window: tuple[float | None, float | None] | None = None,
) -> dict[str, Any]:
    """Predict and subtract an early component from one session.

    ``residual_templates`` may contain one template or a bank. A bank is averaged
    before applying the fitted coefficients, matching the original use of a mean
    water/low-dose template for subtraction.
    """
    if config is None:
        fit_config = fit.get("config", {})
        config = EarlyModelConfig(**fit_config) if fit_config else EarlyModelConfig()
    cfg = config
    prepared = prepare_early_model_session(session, config=cfg)

    template_t = np.asarray(template_t, dtype=float)
    matrix = as_template_matrix(residual_templates)
    if template_t.ndim != 1 or template_t.size != matrix.shape[1]:
        raise ValueError("template_t length must match residual template length")

    mean_template = np.nanmean(matrix, axis=0)
    target_t = prepared["t"]
    template_on_target = np.interp(target_t, template_t, mean_template)
    shifted, valid = shifted_template(
        target_t,
        template_on_target,
        target_t,
        float(fit["lag_min"]),
        fill="hold",
    )

    prediction_z = float(fit["intercept"]) + float(fit["template_scale"]) * shifted
    derivative_scale = float(fit.get("template_derivative_scale", 0.0))
    if derivative_scale != 0.0:
        template_derivative = time_derivative(
            target_t,
            np.where(valid, shifted, np.nan),
            preserve_nans=True,
        )
        prediction_z = prediction_z + derivative_scale * template_derivative

    if subtract_window is not None:
        valid &= window_mask(target_t, subtract_window)

    prediction_signal = (
        float(fit["baseline_mean"])
        + float(fit["baseline_sd"]) * prediction_z
    )
    prediction_signal = np.where(valid, prediction_signal, 0.0)
    prediction_z_masked = np.where(valid, prediction_z, np.nan)
    residual = prepared["signal_filtered"] - prediction_signal

    return {
        "t": target_t.copy(),
        "signal_filtered": prepared["signal_filtered"],
        "template_mean_z": mean_template,
        "template_shifted_z": shifted,
        "prediction_z": prediction_z_masked,
        "prediction_signal": prediction_signal,
        "residual": residual,
        "valid_mask": valid,
        "fit": fit,
        "subtract_window": subtract_window,
    }


def fit_and_residualize_early_model(
    session: SessionLike,
    template_t: np.ndarray,
    fit_templates: np.ndarray,
    residual_templates: np.ndarray,
    *,
    config: EarlyModelConfig | None = None,
    subtract_window: tuple[float | None, float | None] | None = None,
) -> dict[str, Any]:
    """Convenience wrapper combining fit and residualization for one session."""
    fit = fit_early_model(
        session,
        template_t,
        fit_templates,
        config=config,
    )
    residualization = residualize_early_model(
        session,
        fit,
        template_t,
        residual_templates,
        config=config,
        subtract_window=subtract_window,
    )
    return {"fit": fit, "residualization": residualization}
