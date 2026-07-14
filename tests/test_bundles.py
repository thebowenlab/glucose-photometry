from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from photometry.io.bundles import (
    extract_processed_fp_sessions,
    load_processed_bundles,
    load_session_dataset,
    load_sessions,
)


def _modern_session(subject: str = "M1", route: str = "OG", dose: float = 2.0):
    t = np.linspace(-25.0, 90.0, 101)
    return {
        "subject": subject,
        "date": "2025-01-02",
        "route": route,
        "dose_num": dose,
        "t": t,
        "fp": np.sin(t / 10.0),
        "bg": 100.0 + t,
        "raw": {"t": t.copy()},
        "processing": {"source": "test"},
    }


def test_load_v2_dataset_and_preserve_metadata(tmp_path: Path):
    path = tmp_path / "dataset.pkl"
    payload = {
        "format_version": "2.0",
        "signal_units": {"t": "minutes", "bg": "mg/dL", "dose_num": "g/kg"},
        "sessions": [_modern_session()],
    }
    path.write_bytes(pickle.dumps(payload, protocol=5))

    dataset = load_session_dataset(path)
    assert dataset["format_version"] == "2.0"
    assert dataset["source_layout"] == "dataset"
    assert len(dataset["sessions"]) == 1
    session = dataset["sessions"][0]
    assert session["source_schema"] == "session-v2"
    np.testing.assert_allclose(session["t"], payload["sessions"][0]["t"])
    np.testing.assert_allclose(session["fp"], payload["sessions"][0]["fp"])
    np.testing.assert_allclose(session["bg"], payload["sessions"][0]["bg"])


def test_compatibility_loader_unwraps_sessions(tmp_path: Path):
    path = tmp_path / "dataset.pkl"
    path.write_bytes(pickle.dumps({"format_version": "2.0", "sessions": [_modern_session()]}, protocol=5))

    raw_sessions = load_processed_bundles(path)
    assert isinstance(raw_sessions, list)
    assert len(raw_sessions) == 1
    assert raw_sessions[0]["subject"] == "M1"


def test_single_modern_session_file(tmp_path: Path):
    path = tmp_path / "one-session.pkl"
    path.write_bytes(pickle.dumps(_modern_session(), protocol=5))

    sessions = load_sessions(path)
    assert len(sessions) == 1
    assert sessions[0]["source_schema"] == "session-v2"


def test_route_and_dose_filters_accept_modern_sessions():
    sessions = [
        _modern_session("M1", "oral gavage", 2.0),
        _modern_session("M2", "IV", 2.0),
        _modern_session("M3", "OG", 1.0),
    ]
    selected = extract_processed_fp_sessions(sessions, route_filter="OG", dose_filter=2.0)
    assert [s["subject"] for s in selected] == ["M1"]
    assert selected[0]["route"] == "OG"


def test_legacy_ndarray_bundle_auto_preprocesses_and_aligns_bg(tmp_path: Path):
    t = np.linspace(-5.0, 10.0, 901)
    fp = np.sin(t) + 0.05 * np.cos(10 * t)
    bg_t = np.linspace(-5.0, 10.0, 301)
    bg = 100.0 + 10.0 / (1.0 + np.exp(-bg_t))
    bundle = {
        "name": "legacy-mouse",
        "date": "2024-01-01",
        "route": "oral",
        "dose": "2 g/kg",
        "timestamps": {"fp": t, "cgm": bg_t},
        "fp_data": {"orig": fp},
        "cgm_data": {"orig": bg},
    }
    path = tmp_path / "legacy.pkl"
    path.write_bytes(pickle.dumps([bundle], protocol=5))

    session = load_sessions(path, preprocess="auto")[0]
    assert session["source_schema"] == "legacy-bundle"
    assert session["route"] == "OG"
    assert session["dose_num"] == 2.0
    assert session["t"].shape == session["fp"].shape == session["bg"].shape
    assert session["processing"]["fp_source_key"] == "preprocessed:orig"
    assert np.isfinite(session["fp"]).all()
    assert np.isfinite(session["bg"]).sum() > 100


def test_modern_length_mismatch_is_reported(tmp_path: Path):
    bad = _modern_session()
    bad["fp"] = bad["fp"][:-1]
    path = tmp_path / "bad.pkl"
    path.write_bytes(pickle.dumps({"sessions": [bad]}, protocol=5))

    with pytest.raises(ValueError, match="session index 0.*mismatched t/fp lengths"):
        load_session_dataset(path)
