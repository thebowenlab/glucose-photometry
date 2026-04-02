# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:53:29 2026

@author: Adams
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_COLUMN_MAP: dict[str, list[str]] = {
    "fp_time": ["fp_time", "time", "timestamp", "timestamps", "t", "minutes", "time_min"],
    "fp": ["fp", "photometry", "signal", "465", "465nm", "dff", "zscore"],
    "cgm_time": ["cgm_time", "glucose_time", "bg_time"],
    "cgm": ["cgm", "glucose", "bg", "blood_glucose"],
    "subject": ["subject", "mouse", "animal", "name", "id"],
    "date": ["date", "session_date"],
    "route": ["route", "condition"],
    "dose": ["dose", "infusion", "treatment"],
}


def _normalize_column_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {_normalize_column_name(c): c for c in df.columns}
    for cand in candidates:
        key = _normalize_column_name(cand)
        if key in normalized:
            return normalized[key]
    return None


def _first_nonnull_from_column(df: pd.DataFrame, col: str | None) -> Any:
    if col is None or col not in df.columns:
        return None
    s = df[col].dropna()
    if s.empty:
        return None
    return s.iloc[0]


def _series_to_list(df: pd.DataFrame, col: str | None) -> list[Any]:
    if col is None or col not in df.columns:
        return []
    return df[col].tolist()


def load_excel(
    path: str | Path,
    *,
    sheet_name: str | int = 0,
    column_map: dict[str, list[str]] | None = None,
    subject: str | None = None,
    date: str | None = None,
    route: str | None = "intraduodenal_glucose",
    dose: str | float | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Load a single Excel sheet and convert it into a bundle-like dict that matches
    the older pipeline conventions.

    Expected output bundle keys:
      - name
      - date
      - route
      - dose
      - timestamps: {"fp": [...], "cgm": [...]}
      - fp_data: {"orig": [...]}
      - cgm_data: {"orig": [...]}
      - metadata
    """
    path = Path(path)
    df = pd.read_excel(path, sheet_name=sheet_name)

    cmap = DEFAULT_COLUMN_MAP.copy()
    if column_map:
        for key, value in column_map.items():
            cmap[key] = value

    fp_time_col = _find_column(df, cmap["fp_time"])
    fp_col = _find_column(df, cmap["fp"])
    cgm_time_col = _find_column(df, cmap["cgm_time"])
    cgm_col = _find_column(df, cmap["cgm"])

    subject_col = _find_column(df, cmap["subject"])
    date_col = _find_column(df, cmap["date"])
    route_col = _find_column(df, cmap["route"])
    dose_col = _find_column(df, cmap["dose"])

    if fp_col is None:
        raise ValueError(
            f"Could not find an FP column. Available columns: {list(df.columns)}"
        )

    subject_value = subject if subject is not None else _first_nonnull_from_column(df, subject_col)
    date_value = date if date is not None else _first_nonnull_from_column(df, date_col)
    route_value = route if route is not None else _first_nonnull_from_column(df, route_col)
    dose_value = dose if dose is not None else _first_nonnull_from_column(df, dose_col)

    fp_time = _series_to_list(df, fp_time_col)
    fp_values = _series_to_list(df, fp_col)

    cgm_time = _series_to_list(df, cgm_time_col)
    cgm_values = _series_to_list(df, cgm_col)

    # If CGM exists but no separate time column, reuse the FP time axis.
    if cgm_values and not cgm_time:
        cgm_time = fp_time.copy()

    bundle = {
        "name": subject_value,
        "date": date_value,
        "route": route_value,
        "dose": dose_value,
        "timestamps": {
            "fp": fp_time,
            "cgm": cgm_time,
        },
        "fp_data": {
            "orig": fp_values,
        },
        "cgm_data": {
            "orig": cgm_values,
        },
        "metadata": {
            "source_file": str(path),
            "sheet_name": sheet_name,
            "resolved_columns": {
                "fp_time": fp_time_col,
                "fp": fp_col,
                "cgm_time": cgm_time_col,
                "cgm": cgm_col,
                "subject": subject_col,
                "date": date_col,
                "route": route_col,
                "dose": dose_col,
            },
            **(metadata or {}),
        },
    }
    return bundle