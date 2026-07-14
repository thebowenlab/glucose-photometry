# -*- coding: utf-8 -*-
"""Load and normalize photometry/glucose session bundles.

The public Figshare dataset uses a versioned container of the form::

    {
        "format_version": "2.0",
        "signal_units": {"t": "minutes", "bg": "mg/dL", ...},
        "sessions": [
            {"subject": ..., "t": ndarray, "fp": ndarray, "bg": ndarray, ...},
            ...
        ],
    }

Older analysis files used either a list of bundle dictionaries or a single
bundle with ``timestamps``, ``fp_data`` and ``cgm_data`` fields.  This module
accepts all three layouts and exposes a common session representation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping, Sequence
import pickle
import re

import numpy as np

PreprocessMode = Literal["auto", "always", "never"]


class _RestrictedNumpyUnpickler(pickle.Unpickler):
    """Restricted loader for plain containers and NumPy arrays.

    Pickle is not intrinsically safe.  The released dataset only needs NumPy
    array reconstruction, so the default loader rejects arbitrary globals.
    ``trusted=True`` can be used for older local files that contain custom
    classes and were created by a trusted source.
    """

    _ALLOWED_GLOBALS = {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.numeric", "_frombuffer"),
        ("numpy._core.numeric", "_frombuffer"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
        ("datetime", "date"),
        ("datetime", "datetime"),
    }

    def find_class(self, module: str, name: str):
        if (module, name) not in self._ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(
                f"Refusing to load unsupported pickle global {module}.{name}. "
                "Only load files from a trusted source with trusted=True."
            )

        # Map NumPy 1.x/2.x module spellings to objects in the installed NumPy.
        # This is important because NumPy 2 pickles may reference ``numpy._core``
        # while a NumPy 1.x environment exposes the equivalent under
        # ``numpy.core``.
        if (module, name) == ("numpy", "dtype"):
            return np.dtype
        if (module, name) == ("numpy", "ndarray"):
            return np.ndarray
        if name == "_frombuffer":
            numeric = getattr(getattr(np, "_core", None), "numeric", None)
            if numeric is None:
                numeric = np.core.numeric
            return numeric._frombuffer
        if name in {"_reconstruct", "scalar"}:
            multiarray = getattr(getattr(np, "_core", None), "multiarray", None)
            if multiarray is None:
                multiarray = np.core.multiarray
            return getattr(multiarray, name)
        return super().find_class(module, name)


def _load_pickle(path: Path, *, trusted: bool) -> Any:
    with path.open("rb") as f:
        if trusted:
            return pickle.load(f)
        return _RestrictedNumpyUnpickler(f).load()


def _get(bundle: Any, key: str, default: Any = None) -> Any:
    return bundle.get(key, default) if isinstance(bundle, Mapping) else getattr(bundle, key, default)


def parse_dose_to_number(dose: Any) -> float | None:
    """Convert common dose strings to a numeric value in g/kg.

    Water/vehicle/NA labels are treated as 0.  Missing values remain ``None``
    so they can be distinguished from an explicit zero-dose control.
    """
    if dose is None:
        return None
    if isinstance(dose, (int, float, np.number)):
        value = float(dose)
        return value if np.isfinite(value) else None

    s = str(dose).strip().lower()
    if s in {"", "na", "n/a", "none", "nan"}:
        return None
    if "water" in s or "vehicle" in s:
        return 0.0

    match = re.search(r"[-+]?\d*\.?\d+", s)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None


def normalize_route(route: Any) -> str:
    s = "" if route is None else str(route).strip().lower()
    if "iv" in s or "intraven" in s:
        return "IV"
    if "og" in s or "oral" in s or "gavage" in s:
        return "OG"
    return s.upper() if s else "UNKNOWN"


def _as_1d_float(values: Any, *, name: str, allow_empty: bool = True) -> np.ndarray:
    if values is None:
        arr = np.array([], dtype=float)
    else:
        try:
            arr = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} could not be converted to a numeric array") from exc

    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim != 1:
        arr = np.ravel(arr)

    if not allow_empty and arr.size == 0:
        raise ValueError(f"{name} is empty")
    return arr.astype(float, copy=False)


def _first_numeric_series_with_key(
    data: Any,
    preferred_keys: tuple[str, ...],
) -> tuple[np.ndarray | None, str | None]:
    if data is None:
        return None, None

    if isinstance(data, Mapping):
        keys = list(preferred_keys) + [str(k) for k in data.keys() if str(k) not in preferred_keys]
        for key in keys:
            if key not in data:
                continue
            try:
                arr = _as_1d_float(data[key], name=key)
            except ValueError:
                continue
            if arr.size > 0:
                return arr, key
        return None, None

    try:
        arr = _as_1d_float(data, name="signal")
    except ValueError:
        return None, None
    return (arr, None) if arr.size > 0 else (None, None)


def first_numeric_series(
    data: Any,
    preferred_keys: tuple[str, ...] = ("proc_med5_sg10", "proc_med5_causal30", "orig", "fp"),
) -> np.ndarray | None:
    arr, _ = _first_numeric_series_with_key(data, preferred_keys)
    return arr


def align_time_and_trace(
    t: np.ndarray,
    y: np.ndarray,
    *,
    interp_y_nans: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    t = _as_1d_float(t, name="time")
    y = _as_1d_float(y, name="trace")

    n = min(t.size, y.size)
    t = t[:n]
    y = y[:n]
    if n == 0:
        return t, y

    finite_t = np.isfinite(t)
    if np.sum(finite_t) >= 2:
        first = int(np.argmax(finite_t))
        last = int(n - 1 - np.argmax(finite_t[::-1]))
        t = t[first : last + 1]
        y = y[first : last + 1]

    if interp_y_nans:
        finite = np.isfinite(t) & np.isfinite(y)
        if np.sum(finite) >= 2 and np.any(~np.isfinite(y)):
            y = np.interp(t, t[finite], y[finite]).astype(float)

    return t, y


def _time_to_minutes(values: Any, *, numeric_units: str = "minutes") -> np.ndarray:
    """Convert a legacy time vector to relative analysis units (minutes)."""
    if values is None:
        return np.array([], dtype=float)

    # Reuse the tolerant time coercion used by the preprocessing pipeline.
    import pandas as pd
    from photometry.preprocess.time import coerce_time_series, to_seconds

    series = pd.Series(values)
    coerced, kind = coerce_time_series(series)
    seconds, _ = to_seconds(
        coerced,
        kind,
        numeric_units=numeric_units,
        relative=False,
    )
    return np.asarray(seconds, dtype=float) / 60.0


def _interpolate_to_time(
    source_t: np.ndarray,
    source_y: np.ndarray,
    target_t: np.ndarray,
    *,
    interp_internal_nans: bool,
) -> np.ndarray:
    source_t, source_y = align_time_and_trace(
        source_t,
        source_y,
        interp_y_nans=interp_internal_nans,
    )
    target_t = _as_1d_float(target_t, name="target time")

    finite = np.isfinite(source_t) & np.isfinite(source_y)
    if np.sum(finite) < 2:
        return np.full(target_t.shape, np.nan, dtype=float)

    x = source_t[finite]
    y = source_y[finite]
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]
    x, unique_idx = np.unique(x, return_index=True)
    y = y[unique_idx]
    if x.size < 2:
        return np.full(target_t.shape, np.nan, dtype=float)

    out = np.interp(target_t, x, y).astype(float)
    out[(target_t < x[0]) | (target_t > x[-1]) | ~np.isfinite(target_t)] = np.nan
    return out


def _looks_like_modern_session(session: Any) -> bool:
    return isinstance(session, Mapping) and "t" in session and "fp" in session


def _looks_like_legacy_bundle(session: Any) -> bool:
    return (
        isinstance(session, Mapping)
        and ("timestamps" in session or "fp_data" in session or "cgm_data" in session)
    ) or (
        not isinstance(session, Mapping)
        and any(hasattr(session, key) for key in ("timestamps", "fp_data", "cgm_data"))
    )


def _unwrap_payload(payload: Any) -> tuple[dict[str, Any], list[Any], str]:
    if isinstance(payload, Mapping) and "sessions" in payload:
        raw_sessions = payload["sessions"]
        if not isinstance(raw_sessions, (list, tuple)):
            raise TypeError("Dataset field 'sessions' must be a list or tuple")
        metadata = {k: v for k, v in payload.items() if k != "sessions"}
        return metadata, list(raw_sessions), "dataset"

    if isinstance(payload, (list, tuple)):
        return {"format_version": "legacy-list"}, list(payload), "legacy-list"

    if _looks_like_modern_session(payload) or _looks_like_legacy_bundle(payload):
        return {"format_version": "single-session"}, [payload], "single-session"

    raise TypeError(
        "Unsupported pickle layout. Expected a versioned {'sessions': [...]} dataset, "
        "a list/tuple of bundles, or one session bundle."
    )


def _canonicalize_modern_session(
    session: Mapping[str, Any],
    *,
    interp_fp_nans: bool,
    validate: bool,
) -> dict[str, Any]:
    t = _as_1d_float(session.get("t"), name="session['t']", allow_empty=False)
    fp = _as_1d_float(session.get("fp"), name="session['fp']", allow_empty=False)
    bg = _as_1d_float(session.get("bg"), name="session['bg']")

    if t.size != fp.size:
        if validate:
            raise ValueError(f"Modern session has mismatched t/fp lengths: {t.size} vs {fp.size}")
        n = min(t.size, fp.size)
        t, fp = t[:n], fp[:n]

    if bg.size not in {0, t.size}:
        if validate:
            raise ValueError(f"Modern session has mismatched t/bg lengths: {t.size} vs {bg.size}")
        n = min(t.size, bg.size)
        bg_fixed = np.full(t.shape, np.nan, dtype=float)
        bg_fixed[:n] = bg[:n]
        bg = bg_fixed
    elif bg.size == 0:
        bg = np.full(t.shape, np.nan, dtype=float)

    if validate:
        if np.sum(np.isfinite(t)) < 2:
            raise ValueError("Modern session has fewer than two finite time points")
        if not np.all(np.isfinite(t)):
            raise ValueError("Modern session time contains non-finite values")
        if np.any(np.diff(t) <= 0):
            raise ValueError("Modern session time must be strictly increasing")

    # Keep all modern signals synchronized if non-finite time values occur at
    # an edge.  The released v2 file has fully finite time arrays, but doing
    # this jointly avoids offsetting BG relative to FP in less-clean files.
    finite_t = np.isfinite(t)
    if np.sum(finite_t) >= 2:
        first = int(np.argmax(finite_t))
        last = int(t.size - 1 - np.argmax(finite_t[::-1]))
        t = t[first : last + 1]
        fp = fp[first : last + 1]
        bg = bg[first : last + 1]

    if interp_fp_nans and np.any(~np.isfinite(fp)):
        finite = np.isfinite(t) & np.isfinite(fp)
        if np.sum(finite) >= 2:
            fp = np.interp(t, t[finite], fp[finite]).astype(float)

    dose_raw = session.get("dose_num", session.get("dose"))
    out = dict(session)
    out.update(
        {
            "subject": session.get("subject", session.get("name")),
            "date": session.get("date"),
            "route": normalize_route(session.get("route")),
            "dose_num": parse_dose_to_number(dose_raw),
            "t": t,
            "fp": fp,
            "bg": bg,
            "source_schema": "session-v2",
        }
    )
    return out


def _canonicalize_legacy_bundle(
    bundle: Any,
    *,
    preprocess: PreprocessMode,
    config: Any,
    interp_fp_nans: bool,
) -> dict[str, Any]:
    timestamps = _get(bundle, "timestamps", {}) or {}
    fp_data = _get(bundle, "fp_data", {}) or {}
    bg_data = _get(bundle, "cgm_data", _get(bundle, "bg_data", {})) or {}

    fp_selected, fp_key = _first_numeric_series_with_key(
        fp_data,
        ("proc_med5_sg10", "proc_med5_causal30", "fp_proc", "orig", "fp"),
    )
    bg_selected, bg_key = _first_numeric_series_with_key(
        bg_data,
        ("proc_med5_causal30", "proc_med5_sg10", "cgm_proc", "bg_proc", "orig", "bg", "cgm"),
    )

    fp_is_processed = fp_key not in {None, "orig", "fp"}
    bg_is_processed = bg_key not in {None, "orig", "bg", "cgm"}
    need_prepared = preprocess == "always" or (
        preprocess == "auto" and (not fp_is_processed or (bg_selected is not None and not bg_is_processed))
    )

    prepared = None
    if need_prepared:
        from photometry.pipelines.preprocess_bundle import PrepareBundleConfig, prepare_bundle

        cfg = config if config is not None else PrepareBundleConfig()
        # prepare_bundle is defined for mapping-style legacy bundles. Attribute
        # objects are normalized first to keep old trusted pickles usable.
        if not isinstance(bundle, Mapping):
            bundle_for_prepare = {
                "name": _get(bundle, "name", _get(bundle, "subject")),
                "date": _get(bundle, "date"),
                "route": _get(bundle, "route"),
                "dose": _get(bundle, "dose", _get(bundle, "dose_num")),
                "timestamps": timestamps,
                "fp_data": fp_data,
                "cgm_data": bg_data,
            }
        else:
            bundle_for_prepare = dict(bundle)
        prepared = prepare_bundle(bundle_for_prepare, cfg)

    numeric_units = getattr(config, "time_numeric_units", "minutes") if config is not None else "minutes"

    use_prepared_fp = prepared is not None and (preprocess == "always" or not fp_is_processed)
    if use_prepared_fp:
        t_fp = _as_1d_float(prepared["fp"]["t_minutes"], name="prepared FP time")
        fp = _as_1d_float(prepared["fp"]["y_proc"], name="prepared FP")
        fp_source = "preprocessed:orig"
    else:
        if fp_selected is None:
            raise ValueError("Legacy bundle contains no usable photometry series")
        t_fp = _time_to_minutes(
            timestamps.get("fp", []) if isinstance(timestamps, Mapping) else [],
            numeric_units=numeric_units,
        )
        if t_fp.size == 0:
            t_fp = np.arange(fp_selected.size, dtype=float)
        fp = fp_selected
        fp_source = fp_key

    t, fp = align_time_and_trace(t_fp, fp, interp_y_nans=interp_fp_nans)
    if t.size == 0:
        raise ValueError("Legacy bundle contains no aligned photometry samples")

    use_prepared_bg = prepared is not None and (preprocess == "always" or not bg_is_processed)
    if bg_selected is None and not use_prepared_bg:
        bg = np.full(t.shape, np.nan, dtype=float)
        bg_source = None
    else:
        if use_prepared_bg:
            t_bg = _as_1d_float(prepared["cgm"]["t_minutes"], name="prepared BG time")
            bg_values = _as_1d_float(prepared["cgm"]["y_proc"], name="prepared BG")
            bg_source = "preprocessed:orig"
        else:
            t_bg = _time_to_minutes(
                timestamps.get("cgm", timestamps.get("bg", [])) if isinstance(timestamps, Mapping) else [],
                numeric_units=numeric_units,
            )
            bg_values = bg_selected if bg_selected is not None else np.array([], dtype=float)
            if t_bg.size == 0:
                t_bg = t_fp[: bg_values.size]
            bg_source = bg_key
        bg = _interpolate_to_time(
            t_bg,
            bg_values,
            t,
            interp_internal_nans=False,
        )

    dose_raw = _get(bundle, "dose_num", _get(bundle, "dose"))
    return {
        "subject": _get(bundle, "subject", _get(bundle, "name")),
        "date": _get(bundle, "date"),
        "route": normalize_route(_get(bundle, "route")),
        "dose_num": parse_dose_to_number(dose_raw),
        "t": t,
        "fp": fp,
        "bg": bg,
        "raw": bundle,
        "processing": {
            "loader_preprocess_mode": preprocess,
            "fp_source_key": fp_source,
            "bg_source_key": bg_source,
            "bg_interpolated_to_fp_time": True,
        },
        "source_schema": "legacy-bundle",
    }


def canonicalize_session(
    session: Any,
    *,
    preprocess: PreprocessMode = "auto",
    config: Any = None,
    interp_fp_nans: bool = True,
    validate: bool = True,
) -> dict[str, Any]:
    """Convert one modern or legacy session into the common session schema."""
    if preprocess not in {"auto", "always", "never"}:
        raise ValueError("preprocess must be one of {'auto', 'always', 'never'}")

    if _looks_like_modern_session(session):
        # v2 Figshare sessions already contain processed, aligned t/fp/bg arrays.
        # In auto mode they are intentionally not processed a second time.
        return _canonicalize_modern_session(
            session,
            interp_fp_nans=interp_fp_nans,
            validate=validate,
        )
    if _looks_like_legacy_bundle(session):
        return _canonicalize_legacy_bundle(
            session,
            preprocess=preprocess,
            config=config,
            interp_fp_nans=interp_fp_nans,
        )
    raise TypeError("Object is neither a modern session nor a legacy bundle")


def load_session_dataset(
    path: str | Path,
    *,
    preprocess: PreprocessMode = "auto",
    config: Any = None,
    trusted: bool = False,
    interp_fp_nans: bool = True,
    validate: bool = True,
) -> dict[str, Any]:
    """Load a pickle and return dataset metadata plus canonical sessions.

    Parameters
    ----------
    path:
        Dataset pickle, legacy bundle-list pickle, or a one-session pickle.
    preprocess:
        ``"auto"`` (default) keeps modern v2 signals as supplied and only
        preprocesses legacy ``orig`` traces when no processed trace is present.
        ``"always"`` forces preprocessing of legacy inputs; ``"never"`` uses
        the best available legacy trace without applying the pipeline.
    trusted:
        By default, reject arbitrary pickle classes and permit only the NumPy
        reconstruction needed by the released dataset. Set ``True`` only for a
        trusted local legacy pickle containing custom Python classes.
    """
    path = Path(path)
    payload = _load_pickle(path, trusted=trusted)
    metadata, raw_sessions, source_layout = _unwrap_payload(payload)

    sessions: list[dict[str, Any]] = []
    for index, raw_session in enumerate(raw_sessions):
        try:
            session = canonicalize_session(
                raw_session,
                preprocess=preprocess,
                config=config,
                interp_fp_nans=interp_fp_nans,
                validate=validate,
            )
        except Exception as exc:
            raise type(exc)(f"Failed to load session index {index}: {exc}") from exc
        session.setdefault("session_index", index)
        sessions.append(session)

    result = dict(metadata)
    result["sessions"] = sessions
    result["source_layout"] = source_layout
    result["source_path"] = str(path)
    result.setdefault("signal_units", {"t": "minutes", "bg": "mg/dL", "dose_num": "g/kg"})
    return result


def load_sessions(
    path: str | Path,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Load only the canonical session list from a pickle."""
    return load_session_dataset(path, **kwargs)["sessions"]


def load_processed_bundles(
    path: str | Path,
    *,
    trusted: bool = False,
) -> list[Any]:
    """Compatibility loader returning the raw session/bundle list.

    Unlike the original implementation, this unwraps the v2 dataset's
    top-level ``sessions`` field, so existing code that calls
    ``extract_processed_fp_sessions(load_processed_bundles(path))`` works.
    Use :func:`load_session_dataset` for metadata and canonicalized arrays.
    """
    path = Path(path)
    payload = _load_pickle(path, trusted=trusted)
    _, sessions, _ = _unwrap_payload(payload)
    return sessions


def extract_processed_fp_sessions(
    bundles: Sequence[Any] | Mapping[str, Any],
    *,
    route_filter: str | None = None,
    dose_filter: float | None = None,
    interp_y_nans: bool = True,
    preprocess: PreprocessMode = "auto",
    config: Any = None,
    validate: bool = True,
) -> list[dict[str, Any]]:
    """Extract canonical processed sessions from modern or legacy bundles.

    Returned dictionaries contain ``subject``, ``date``, ``route``,
    ``dose_num``, ``t`` (minutes), ``fp``, and ``bg`` (mg/dL when available),
    plus source ``raw`` and ``processing`` metadata.
    """
    if isinstance(bundles, Mapping) and "sessions" in bundles:
        items = bundles["sessions"]
    elif _looks_like_modern_session(bundles) or _looks_like_legacy_bundle(bundles):
        items = [bundles]
    else:
        items = bundles

    route_filter_norm = normalize_route(route_filter) if route_filter is not None else None
    out: list[dict[str, Any]] = []

    for index, bundle in enumerate(items):
        session = canonicalize_session(
            bundle,
            preprocess=preprocess,
            config=config,
            interp_fp_nans=interp_y_nans,
            validate=validate,
        )
        if route_filter_norm is not None and session["route"] != route_filter_norm:
            continue
        if dose_filter is not None:
            dose_num = session.get("dose_num")
            if dose_num is None or not np.isclose(float(dose_num), float(dose_filter)):
                continue
        session.setdefault("session_index", index)
        out.append(session)

    return out
