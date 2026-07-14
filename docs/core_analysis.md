# Core single-session analysis API

The production package keeps reusable numerical analysis separate from cohort
selection, plotting, and file export. All functions in `photometry.analysis`
accept one canonical session as their primary input. A canonical session is the
dictionary returned by `load_sessions(...)` and contains aligned one-dimensional
arrays:

```python
session["t"]   # minutes
session["fp"]  # processed photometry
session["bg"]  # processed blood glucose, mg/dL
```

The external driver may loop over sessions, combine returned dictionaries into a
DataFrame, save arrays, and create figures without duplicating numerical methods.

## Package organization

```text
photometry.analysis.glucose      glucose derivative and kinetic metrics
photometry.analysis.early_model  early-template selection, fitting, prediction,
                                 and residualization
photometry.analysis.correlation  lagged FP/glucose correlation
photometry.analysis.nulls        circular-shift null generation and inference
```

## 1. Glucose processing and kinetics

```python
from photometry import GlucoseAnalysisConfig, compute_glucose_metrics

result = compute_glucose_metrics(
    session,
    config=GlucoseAnalysisConfig(
        baseline_window=(-25, 0),
        onset_baseline_window=(-5, 0),
        peak_window=(0, 120),
        iauc_window=(0, 120),
        derivative_smooth_minutes=1.0,
    ),
)

print(result["onset_time"])
print(result["peak_time"], result["peak_increment"])
print(result["iauc"])

t = result["series"]["t"]
dbg_dt = result["series"]["bg_derivative"]
```

This returns the baseline, derivative, onset, peak glucose response, derivative
peak, and positive incremental AUC. `iauc` is calculated from glucose minus the
baseline and therefore has units of `(mg/dL) * min`.

Lower-level calls are also available:

```python
from photometry import (
    compute_glucose_derivative,
    compute_incremental_auc,
    detect_glucose_onset,
)
```

## 2. Early-template model

The early model has two distinct kinds of templates, following the original
script:

1. **fit templates** select lag and model coefficients;
2. **residualization templates** define the component that is predicted and
   subtracted from the target session.

The package does not choose cohorts internally. Cohort selection is an analysis
decision and belongs in the external driver. This avoids hidden global state and
makes leakage controls explicit.

### Prepare each candidate session

```python
from photometry import EarlyModelConfig, prepare_early_model_session

model_cfg = EarlyModelConfig(
    baseline_window=(-25, 0),
    fit_window=(-25, 1),
    lowpass_period_min=4.0,
    lag_min=-1.0,
    lag_max=1.0,
    lag_step=0.25,
    include_template_derivative=False,
)

prepared = prepare_early_model_session(candidate_session, config=model_cfg)
prepared_z = prepared["signal_z"]
```

Each candidate is low-pass filtered and z-scored using its own baseline. Session
time grids can differ by one or more samples, so the external driver should
interpolate candidate traces onto a common grid before stacking them.

```python
import numpy as np

common_t = target_session["t"]
fit_templates = np.vstack([
    np.interp(common_t, item["t"], item["signal_z"])
    for item in prepared_fit_sessions
])

residual_templates = np.vstack([
    np.interp(common_t, item["t"], item["signal_z"])
    for item in prepared_residual_sessions
])
```

For reproduction of the original analysis, the fit bank was based on OG sessions
at 1.0, 2.0, and 2.5 g/kg, while the subtraction template used water/low-dose OG
sessions. In the released dataset, water sessions may have `dose_num is None`;
the driver should handle that explicitly.

For unbiased analysis, exclude the target session from its own fit bank. When the
same animal contributes multiple sessions, consider leaving the entire subject
out of the fit bank.

### Inspect selected model data

```python
from photometry import select_early_model_data

selection = select_early_model_data(
    target_session,
    common_t,
    fit_templates,
    config=model_cfg,
)

print(selection["fit_t"].shape)
print(selection["templates_on_fit_t"].shape)
```

### Fit one target session

```python
from photometry import fit_early_model

fit = fit_early_model(
    target_session,
    common_t,
    fit_templates,
    config=model_cfg,
)

print(fit["lag_min"])
print(fit["cv_r2"])
print(fit["intercept"], fit["template_scale"])
```

With multiple candidate templates, lag selection uses leave-one-template-out
scores summarized by median R². By default, coefficients are then refit on the
mean fit template at the selected lag. Set
`refit_on_mean_after_cv=False` for behavior closer to the legacy script's
best-fold coefficient selection.

### Predict and residualize one target session

```python
from photometry import residualize_early_model

model_output = residualize_early_model(
    target_session,
    fit,
    common_t,
    residual_templates,
    config=model_cfg,
)

prediction = model_output["prediction_signal"]
residual = model_output["residual"]
valid = model_output["valid_mask"]
```

The prediction is converted back into the target session's FP units before
subtraction. Outside the valid shifted-template interval, prediction is zero and
the residual equals the filtered target signal.

A combined convenience function is available:

```python
from photometry import fit_and_residualize_early_model
```

## 3. Cross-correlation

```python
from photometry import CrossCorrelationConfig, compute_session_cross_correlations

coupling = compute_session_cross_correlations(
    session,
    config=CrossCorrelationConfig(
        max_lag_min=20,
        method="spearman",
        x_lowpass_period_min=4.0,
        y_derivative_smooth_minutes=1.0,
    ),
)

fp_bg = coupling["glucose"]
fp_dbg = coupling["glucose_derivative"]

print(fp_bg["best_lag_min"], fp_bg["best_correlation"])
```

The default reproduces the legacy rank-based correlation approach. The lag
convention is explicit: **positive lag means FP (`x`) occurs later than glucose
(`y`)**.

To analyze a model residual instead of the session's stored FP trace:

```python
from photometry import compute_cross_correlation

residual_coupling = compute_cross_correlation(
    session,
    config=CrossCorrelationConfig(y_transform="raw"),
    x=model_output["residual"],
)
```

## 4. Circular-shift nulls

Create null signals without running a statistic:

```python
from photometry import CircularShiftConfig, generate_circular_shift_nulls

shift_null = generate_circular_shift_nulls(
    session,
    config=CircularShiftConfig(
        signal_key="bg",
        n_null=200,
        min_shift_min=20,
        seed=0,
    ),
)

null_bg = shift_null["null_signals"]  # [n_null, n_time]
shift_minutes = shift_null["signed_shift_min"]
```

The minimum-shift rule is based on circular distance, so a wraparound shift cannot
silently become a small effective shift.

Run cross-correlation and circular-shift inference together:

```python
from photometry import compute_circular_shift_correlation_null

null_result = compute_circular_shift_correlation_null(
    session,
    correlation_config=CrossCorrelationConfig(
        y_transform="raw",
        max_lag_min=20,
    ),
    null_config=CircularShiftConfig(
        n_null=500,
        min_shift_min=20,
        seed=0,
    ),
    x=model_output["residual"],  # omit to use session["fp"]
)

observed_r = null_result["observed"]["correlation"]
null_r = null_result["null_correlation"]
pointwise_p = null_result["pointwise_p_two_sided"]
familywise_p = null_result["familywise_p_max_abs"]
```

The familywise p-value uses the maximum absolute correlation across lags in each
null draw.

## Suggested external-driver workflow

For each target session:

1. call `compute_glucose_metrics(session)`;
2. construct fit and residual template banks from explicitly selected sessions;
3. call `fit_early_model(...)` and `residualize_early_model(...)`;
4. call `compute_session_cross_correlations(...)` for raw FP, prediction, and
   residual as needed;
5. call `compute_circular_shift_correlation_null(...)` for inferential curves;
6. store scalar outputs in a row-oriented table and arrays in session-level result
   dictionaries;
7. pass those stored outputs to separate plotting functions.

This structure keeps numerical analysis testable and reusable while allowing the
paper-specific driver to control cohort definitions, multiple-comparison rules,
visual design, and file naming.
