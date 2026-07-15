from pathlib import Path
import os

from photometry import (
    CircularShiftConfig,
    compute_glucose_metrics,
    compute_session_cross_correlations,
    generate_circular_shift_nulls,
    load_sessions,
)

data_dir = Path(r'/Users/annabowen/Downloads/31891579')
if not data_dir:
    raise RuntimeError("GLUCOSE_PHOTOMETRY_DATA_DIR is not set.")

data_file = Path(data_dir) / "agrp-photometry-bg.pkl"
if not data_file.is_file():
    raise FileNotFoundError(data_file)

sessions = load_sessions(data_file)
print(f"Loaded {len(sessions)} sessions")

for index, session in enumerate(sessions):
    glucose = compute_glucose_metrics(session)
    coupling = compute_session_cross_correlations(session)

    print(
        f"{index + 1:02d} "
        f"subject={session.get('subject')} "
        f"route={session.get('route')} "
        f"dose={session.get('dose_num')} "
        f"peak={glucose['peak_time']:.3g} min "
        f"iAUC={glucose['iauc']:.3g} "
        f"FP-BG r={coupling['glucose']['best_correlation']:.3g}"
    )

null_result = generate_circular_shift_nulls(
    sessions[0],
    config=CircularShiftConfig(
        signal_key="bg",
        n_null=20,
        min_shift_min=20,
        seed=0,
    ),
)

print("Circular null shape:", null_result["null_signals"].shape)
print("Dataset smoke test passed.")