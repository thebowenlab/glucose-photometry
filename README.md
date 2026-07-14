# Glat et al. 2026 — glucose photometry utilities

Code accompanying **_AgRP Neuron Activity Predicts and Tracks the Glycemic Response to Oral Glucose_**.

## Installation

From the repository root:

```bash
python -m pip install -e .
```

The package requires Python 3.10 or newer.

## Load the released Figshare pickle

The released `agrp-photometry-bg.pkl` file is a versioned dataset dictionary, not a bare list. Its top-level fields are:

- `format_version`
- `signal_units`
- `sessions`

Each session contains aligned, already processed arrays named `t`, `fp`, and `bg`, along with subject/session metadata and retained raw/processing information. The loader therefore **does not preprocess these v2 signals again** when `preprocess="auto"` is used.

```python
from photometry import load_session_dataset

dataset = load_session_dataset("/path/to/agrp-photometry-bg.pkl")
print(dataset["format_version"])
print(dataset["signal_units"])
print(len(dataset["sessions"]))

session = dataset["sessions"][0]
print(session["subject"], session["date"], session["route"], session["dose_num"])
print(session["t"].shape, session["fp"].shape, session["bg"].shape)
```

For the supplied Figshare file, this returns 26 sessions. Time is in minutes and blood glucose is in mg/dL, as recorded in `dataset["signal_units"]`.

To retrieve only the normalized session list:

```python
from photometry import load_sessions

sessions = load_sessions("/path/to/agrp-photometry-bg.pkl")
```

Filter by administration route and dose:

```python
from photometry import extract_processed_fp_sessions, load_processed_bundles

raw_sessions = load_processed_bundles("/path/to/agrp-photometry-bg.pkl")
og_2gkg = extract_processed_fp_sessions(
    raw_sessions,
    route_filter="OG",
    dose_filter=2.0,
)
```

The compatibility loader now unwraps the top-level `sessions` field, so the existing `load_processed_bundles(...)` followed by `extract_processed_fp_sessions(...)` pattern works with both the released v2 dataset and legacy bundle-list pickles.

## Legacy files and preprocessing

Older files may contain one bundle or a list of bundles with this structure:

```text
timestamps: {fp, cgm}
fp_data: {orig and/or processed traces}
cgm_data: {orig and/or processed traces}
```

Use:

```python
sessions = load_sessions("legacy-bundles.pkl", preprocess="auto")
```

Preprocessing modes are:

- `auto`: keep an existing processed trace; preprocess a legacy `orig` trace only when needed. Modern v2 sessions are used as supplied.
- `always`: force preprocessing for legacy bundles.
- `never`: use the best available legacy trace without running the preprocessing pipeline.

Legacy CGM/BG traces are interpolated onto the photometry time axis so every returned session has matching `t`, `fp`, and `bg` shapes.

## Pickle safety

Python pickle files can execute code while loading. By default, these loaders use a restricted unpickler that permits the plain containers and NumPy reconstruction required by the released file. Use `trusted=True` only for a trusted legacy pickle that genuinely requires custom Python classes:

```python
sessions = load_sessions("trusted-local-legacy.pkl", trusted=True)
```

## Run tests

```bash
python -m pip install pytest
pytest -q
```

## Core single-session analysis

The reusable analysis port from the original `compute_ogtt_v5.py` workflow is in
`photometry.analysis`. Numerical functions accept one canonical session at a time
and return dictionaries containing scalar metrics and processed arrays. Cohort
selection, result aggregation, and plotting remain the responsibility of an
external driver script.

```python
from photometry import (
    compute_glucose_metrics,
    compute_session_cross_correlations,
    generate_circular_shift_nulls,
)

session = load_sessions("/path/to/agrp-photometry-bg.pkl")[0]

glucose = compute_glucose_metrics(session)
coupling = compute_session_cross_correlations(session)
shift_null = generate_circular_shift_nulls(session)
```

The core API includes:

- glucose derivative, sustained-rise onset, peak detection, and incremental AUC;
- early-template data preparation, lag/model fitting, prediction, and residualization;
- FP-to-glucose and FP-to-glucose-derivative cross-correlation;
- reproducible circular-shift null signals and correlation-null inference.

See [`docs/core_analysis.md`](docs/core_analysis.md) for complete examples,
including construction of externally selected fit and residual template banks.
