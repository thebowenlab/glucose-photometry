# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 19:17:53 2026

@author: Adams
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import pickle

import numpy as np


def _get(b, key, default=None):
    return b.get(key, default) if isinstance(b, dict) else getattr(b, key, default)


def parse_dose_to_number(dose: Any) -> float:
    if dose is None:
        return 0.0
    if isinstance(dose, (int, float)):
        return float(dose)

    s = str(dose).strip().lower()
    if "water" in s or s in {"na", "n/a", "none", ""}:
        return 0.0

    import re
    m = re.search(r"[-+]?\d*\.?\d+", s)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return 0.0
    return 0.0


def normalize_route(route: Any) -> str:
    s = "" if route is None else str(route).strip().lower()
    if "iv" in s:
        return "IV"
    if "og" in s or "oral" in s:
        return "OG"
    return s.upper() if s else "UNKNOWN"


def first_numeric_series(
    d: Any,
    preferred_keys: tuple[str, ...] = ("proc_med5_sg10", "proc_med5_causal30", "orig", "fp"),
) -> np.ndarray | None:
    if d is None:
        return None

    if isinstance(d, dict):
        for k in preferred_keys:
            if k in d:
                try:
                    arr = np.asarray(d[k], dtype=float)
                    if arr.ndim == 1 and arr.size > 0:
                        return arr
                except Exception:
                    pass

        for v in d.values():
            try:
                arr = np.asarray(v, dtype=float)
                if arr.ndim == 1 and arr.size > 0:
                    return arr
            except Exception:
                continue
        return None

    try:
        arr = np.asarray(d, dtype=float)
        if arr.ndim == 1 and arr.size > 0:
            return arr
    except Exception:
        pass

    return None


def align_time_and_trace(
    t: np.ndarray,
    y: np.ndarray,
    *,
    interp_y_nans: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    n = min(t.size, y.size)
    t = t[:n]
    y = y[:n]

    if n == 0:
        return t, y

    # Fix non-finite times by simple clipping to finite region if possible
    finite_t = np.isfinite(t)
    if np.sum(finite_t) >= 2:
        first = int(np.argmax(finite_t))
        last = int(n - 1 - np.argmax(finite_t[::-1]))
        t = t[first:last + 1]
        y = y[first:last + 1]

    if interp_y_nans:
        finite = np.isfinite(t) & np.isfinite(y)
        if np.sum(finite) >= 2 and np.any(~np.isfinite(y)):
            y = np.interp(t, t[finite], y[finite]).astype(float)

    return t, y


def load_processed_bundles(path: str | Path) -> list[Any]:
    path = Path(path)
    with open(path, "rb") as f:
        bundles = pickle.load(f)
    return bundles


def extract_processed_fp_sessions(
    bundles: Sequence[Any],
    *,
    route_filter: str | None = None,
    dose_filter: float | None = None,
    interp_y_nans: bool = True,
) -> list[dict[str, Any]]:
    """
    Extract already-processed FP sessions from legacy processed bundles.

    Returns a list of dicts with:
      - subject
      - date
      - route
      - dose_raw
      - dose_num
      - t
      - fp
    """
    out: list[dict[str, Any]] = []

    route_filter_norm = normalize_route(route_filter) if route_filter is not None else None

    for b in bundles:
        route = _get(b, "route", None)
        route_norm = normalize_route(route)

        if route_filter_norm is not None and route_norm != route_filter_norm:
            continue

        dose_raw = _get(b, "dose", None)
        dose_num = parse_dose_to_number(dose_raw)
        if dose_filter is not None and not np.isclose(dose_num, float(dose_filter)):
            continue

        tdict = _get(b, "timestamps", {})
        if not isinstance(tdict, dict) or "fp" not in tdict:
            continue
        t = np.asarray(tdict["fp"], dtype=float)

        fp_data = _get(b, "fp_data", {})
        fp = first_numeric_series(fp_data)
        if fp is None or not np.any(np.isfinite(fp)):
            continue

        t, fp = align_time_and_trace(t, fp, interp_y_nans=interp_y_nans)

        out.append({
            "subject": _get(b, "name", None),
            "date": _get(b, "date", None),
            "route": route_norm,
            "dose_raw": dose_raw,
            "dose_num": dose_num,
            "t": t,
            "fp": fp,
        })

    return out