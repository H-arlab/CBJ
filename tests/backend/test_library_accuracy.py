"""End-to-end accuracy test for every Library card the user can click.

User asked for two things:
  1. "라이브러리 플랏들이 정확하게 그려지는 지 확인해" — every graph
     card that appears in the Library must produce a real, non-mock,
     non-error figure when clicked against a real H-Walker CSV.
  2. "이거 어떻게 데이터 정리하고 분석할 지 확인해" — the upload →
     analyze → compute → render pipeline must work end-to-end on a
     plausible synthetic CSV.

This file builds a synthetic Robot CSV that matches the H-Walker
firmware schema (GCP sawtooth per side, Event rising edges at heel
strike, paired Des/Act force / vel / pos / curr, IMU pitch ramp,
Sync), runs it through every API the Library exposes, and asserts
realistic outputs.

The test does NOT spin up a real HTTP server — it imports the
FastAPI app and exercises the routers directly via pydantic
models. That keeps the test fast and avoids the python-multipart
dependency.
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
import pytest


# ============================================================
# Synthetic H-Walker firmware CSV
# ============================================================

def _synthetic_robot_csv(
    fs: float = 111.0,
    duration_s: float = 6.0,
    stride_s: float = 1.05,         # slow walking, ~57 spm per side
    stance_frac: float = 0.62,
    peak_force_l_n: float = 45.0,
    peak_force_r_n: float = 47.0,
) -> pd.DataFrame:
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    df = pd.DataFrame({
        "Time_ms": t * 1000.0,
        "Freq_Hz": np.full(n, fs),
        "Sync": (t > 0.5).astype(float),  # falling edge at t=0.5
    })

    # Per-side GCP sawtooth + heel-strike Event + paired controller channels.
    for side, peak_force, phase_offset in (
        ("L", peak_force_l_n, 0.0),
        ("R", peak_force_r_n, stride_s / 2),  # R offset by half-stride
    ):
        gcp = np.zeros(n)
        event = np.zeros(n)
        force_des = np.zeros(n)
        force_act = np.zeros(n)
        pitch = np.zeros(n)

        # Walk through every stride bin
        for i_stride in range(int(duration_s / stride_s)):
            start_t = i_stride * stride_s + phase_offset
            stance_end_t = start_t + stride_s * stance_frac
            stance_mask = (t >= start_t) & (t < stance_end_t)
            stride_mask = (t >= start_t) & (t < start_t + stride_s)

            # Heel strike event — single-sample pulse at start
            hs_idx = np.searchsorted(t, start_t)
            if 0 <= hs_idx < n:
                event[hs_idx:hs_idx + 2] = 1.0

            # GCP ramp 0→1 over stance only
            if stance_mask.any():
                stance_ts = t[stance_mask] - start_t
                gcp[stance_mask] = stance_ts / (stride_s * stance_frac)

            # Force = sinusoidal hump during stance, 0 during swing
            if stance_mask.any():
                local_t = t[stance_mask] - start_t
                hump = np.sin(np.pi * local_t / (stride_s * stance_frac)) * peak_force
                force_act[stance_mask] = hump
                force_des[stance_mask] = hump * 1.05  # 5% over-target

            # Pitch oscillation per stride (sine over the whole gait cycle)
            if stride_mask.any():
                local = t[stride_mask] - start_t
                pitch[stride_mask] = 12.0 * np.sin(2 * np.pi * local / stride_s)

        df[f"{side}_GCP"] = gcp
        df[f"{side}_Event"] = event
        df[f"{side}_Phase"] = (gcp > 0.01).astype(float)  # 0=swing, 1=stance
        df[f"{side}_DesForce_N"] = force_des
        df[f"{side}_ActForce_N"] = force_act
        df[f"{side}_ErrForce_N"] = force_act - force_des
        df[f"{side}_DesVel_mps"]  = np.gradient(pitch) * 0.01
        df[f"{side}_ActVel_mps"]  = df[f"{side}_DesVel_mps"] * 0.95
        df[f"{side}_ErrVel_mps"]  = df[f"{side}_ActVel_mps"] - df[f"{side}_DesVel_mps"]
        df[f"{side}_DesPos_deg"] = pitch
        df[f"{side}_ActPos_deg"] = pitch * 0.97
        df[f"{side}_ErrPos_deg"] = df[f"{side}_ActPos_deg"] - df[f"{side}_DesPos_deg"]
        df[f"{side}_DesCurr_A"]  = np.gradient(pitch) * 0.05
        df[f"{side}_ActCurr_A"]  = df[f"{side}_DesCurr_A"] * 1.02
        df[f"{side}_ErrCurr_A"]  = df[f"{side}_ActCurr_A"] - df[f"{side}_DesCurr_A"]
        df[f"{side}_Pitch"] = pitch
        df[f"{side}_Roll"]  = pitch * 0.3
        df[f"{side}_Yaw"]   = np.zeros(n)
        df[f"{side}_Ax"] = np.gradient(pitch) * 0.1
        df[f"{side}_Ay"] = np.zeros(n)
        df[f"{side}_Az"] = np.full(n, 9.8)
        df[f"{side}_Gx"] = np.gradient(pitch)
        df[f"{side}_Gy"] = np.zeros(n)
        df[f"{side}_Gz"] = np.zeros(n)

    return df


# ============================================================
# Fixtures: stash the synthetic CSV into the dataset registry
# ============================================================

@pytest.fixture
def robot_dataset(tmp_path, monkeypatch):
    """Write the synthetic CSV to disk and register it in
    backend.routers.datasets without going through the multipart
    upload (we don't depend on python-multipart in the test env)."""
    df = _synthetic_robot_csv()
    path = tmp_path / "Robot_synthetic.csv"
    df.to_csv(path, index=False)

    from backend.services import dataset_registry as ds_mod
    ds_id = "ds_synth_robot"
    ds_mod._REGISTRY[ds_id] = {
        "id": ds_id,
        "name": path.name,
        "tag": "mixed",
        "kind": "mixed",
        "source_kind": "robot",
        "source_confidence": 1.0,
        "source_cues": ["filename starts with 'robot'"],
        "rows": len(df),
        "_path": str(path),
    }
    yield ds_id, df, path
    ds_mod._REGISTRY.pop(ds_id, None)


# ============================================================
# 1. Library graph cards — every template must render
# ============================================================

# These are exactly the keys exposed in frontend/src/data/graphTemplates.ts
# (must match `Object.keys(GRAPH_TPLS)` so a new template added on the
# frontend without a renderer trips this test).
LIBRARY_GRAPH_TEMPLATES = [
    "force", "force_avg", "force_lr_subplot", "asymmetry",
    "force_tracking_L", "force_tracking_R",
    "trials", "imu_avg", "cyclogram", "rom_bar",
    "stride_time_trend", "stance_swing_bar", "symmetry_radar",
]


@pytest.mark.parametrize("template", LIBRARY_GRAPH_TEMPLATES)
def test_every_library_graph_renders_with_real_data(robot_dataset, template):
    """Click-through accuracy contract: every Library card the user
    can click must reach a real-data renderer (no 422 'mock removed'
    fall-through, no ValueError from publication_engine)."""
    ds_id, _, _ = robot_dataset
    from backend.routers.graphs import REAL_DATA_TEMPLATES

    # Library is allowed to advertise extras only if the backend
    # explicitly knows how to render them.
    assert template in REAL_DATA_TEMPLATES, (
        f"Template '{template}' is in the Library but not in "
        f"REAL_DATA_TEMPLATES — clicking it would 422. Either add a "
        f"real-data renderer or drop it from graphTemplates.ts."
    )

    from backend.routers.graphs import RenderRequest, _render_real_data
    req = RenderRequest(
        template=template,
        preset="ieee",
        variant="col2",
        format="svg",
        dataset_id=ds_id,
    )
    result = _render_real_data(req)
    assert result is not None, (
        f"_render_real_data returned None for '{template}' even though "
        f"it's listed in REAL_DATA_TEMPLATES."
    )
    data, mime = result
    assert isinstance(data, (bytes, bytearray))
    assert len(data) > 200, (
        f"'{template}' produced only {len(data)} bytes — render likely "
        f"failed silently."
    )
    assert mime in {"image/svg+xml", "application/pdf",
                    "image/png", "image/tiff", "application/postscript"}


def test_library_advertises_no_template_without_renderer():
    """Hard contract: the set of templates exposed by the Library
    (graphTemplates.ts) must be a subset of REAL_DATA_TEMPLATES.

    Read the TS file directly so this fires whenever the frontend
    adds a card that isn't wired up yet."""
    import re
    from pathlib import Path
    from backend.routers.graphs import REAL_DATA_TEMPLATES

    ts = Path(__file__).resolve().parents[2] / \
        "frontend/src/data/graphTemplates.ts"
    src = ts.read_text(encoding="utf-8")
    # Match "  identifier: {" inside the GRAPH_TPLS object.
    keys = re.findall(r"^\s*([a-z_][a-z_0-9]*)\s*:\s*\{", src, re.MULTILINE)
    # Drop any matches before GRAPH_TPLS opens (interface fields, etc.)
    start = src.find("GRAPH_TPLS")
    keys_in_tpls = re.findall(r"^\s*([a-z_][a-z_0-9]*)\s*:\s*\{",
                              src[start:], re.MULTILINE)

    missing = [k for k in keys_in_tpls if k not in REAL_DATA_TEMPLATES]
    assert not missing, (
        f"graphTemplates.ts advertises {missing} but the backend has "
        f"no real-data renderer for them. Either implement the "
        f"renderer or remove the card."
    )
    _ = keys  # silence unused


# ============================================================
# 2. Library compute metrics — every metric must compute
# ============================================================

LIBRARY_COMPUTE_METRICS = [
    "per_stride", "impulse", "loading_rate", "rom",
    "cadence", "target_dev",
    "stride_length", "stance_time", "swing_time",
    "fatigue_index", "symmetry_summary",
    # Phase 2J motor / control-channel tracking
    "velocity_tracking", "position_tracking", "current_tracking",
    "feedforward",
]


@pytest.mark.parametrize("metric", LIBRARY_COMPUTE_METRICS)
def test_every_library_compute_metric_returns_data(robot_dataset, metric):
    """Each metric must return cols + rows that the frontend table
    can render. No empty rows for a well-formed H-Walker trial."""
    ds_id, df, _ = robot_dataset
    from backend.routers.analyze import analyze_cached
    from backend.services import compute_engine

    res, _payload = analyze_cached(ds_id)
    assert res is not None, (
        "analyze_cached returned None on a synthetic Robot CSV — "
        "indicates the analyzer's H-Walker detection regressed."
    )

    out = compute_engine.compute(metric, df, res)
    assert isinstance(out, dict)
    assert isinstance(out.get("label"), str) and out["label"]
    assert isinstance(out.get("cols"), list) and out["cols"]
    assert isinstance(out.get("rows"), list)

    # Whole-trial scalar metrics must always have at least 1 row.
    must_be_scalar = {
        "cadence", "stride_length",
        "velocity_tracking", "position_tracking",
        "current_tracking",
    }
    if metric in must_be_scalar:
        assert len(out["rows"]) >= 1, (
            f"'{metric}' returned 0 rows — synthetic data has gait "
            f"events, so this is a regression."
        )

    # Per-stride tables must have multiple strides.
    if metric == "per_stride":
        assert len(out["rows"]) >= 3, (
            f"per_stride returned only {len(out['rows'])} rows on "
            f"6 s of synthetic gait — analyzer didn't detect strides."
        )


def test_compute_cadence_lands_in_physiological_range(robot_dataset):
    """Stride period 1.05 s → cadence ≈ 120 / 1.05 ≈ 114 spm. Must
    not regress to 60 / T (≈ 57 spm — the *2 dropped) or 120 / T_step
    (≈ 228 spm — Event mistakenly used as primary)."""
    ds_id, df, _ = robot_dataset
    from backend.routers.analyze import analyze_cached
    from backend.services import compute_engine

    res, _ = analyze_cached(ds_id)
    out = compute_engine.compute("cadence", df, res)
    # Combined column is the third column ("from L HS / from R HS / Combined")
    combined_str = out["rows"][0][2]
    combined = float(combined_str.replace("—", "0"))
    assert 95 < combined < 135, (
        f"cadence {combined:.1f} spm is implausible for stride_s=1.05 "
        f"(expected ≈ 114). Either the * 2 stride→step conversion or "
        f"the GCP-primary HS detection regressed."
    )


def test_compute_stance_pct_above_50(robot_dataset):
    """Synthetic stance fraction is 0.62; the metric must return a
    value > 50 % (otherwise the inverted swing/stance bug is back)."""
    ds_id, df, _ = robot_dataset
    from backend.routers.analyze import analyze_cached
    from backend.services import compute_engine

    res, _ = analyze_cached(ds_id)
    out = compute_engine.compute("stance_time", df, res)
    # stance_time returns absolute seconds, not %; we instead check
    # via the analysis result's stance_pct fields directly.
    pct_l = res.left_stride.stance_pct_mean
    pct_r = res.right_stride.stance_pct_mean
    assert pct_l > 55, (
        f"L stance % is {pct_l:.1f} — should be ≈ 62 for "
        f"synthetic data. Inverted swing/stance regression?"
    )
    assert pct_r > 55, (
        f"R stance % is {pct_r:.1f} — should be ≈ 62."
    )
    _ = out  # not directly asserted


# ============================================================
# 3. Multi-source pipeline (Robot + Motion + Loadcell)
# ============================================================

def test_multi_source_pipeline_aligns_and_resamples(tmp_path, monkeypatch):
    """End-to-end: three CSVs in one experiment session land on a
    single common time grid after rising-edge sync alignment."""
    from backend.services import dataset_registry as ds_mod
    from backend.routers.sync import align, AlignRequest

    def _sync_pulses(n, fs, windows):
        out = np.zeros(n, dtype=float)
        t = np.arange(n) / fs
        for r, f in windows:
            out[(t >= r) & (t < f)] = 1.0
        return out

    # Robot @ 111 Hz, sync rising at t=0.5, falling at t=4.0.
    robot = _synthetic_robot_csv(fs=111.0, duration_s=6.0)
    n_r = len(robot)
    robot = robot.assign(Sync=_sync_pulses(n_r, 111.0, [(0.5, 4.0)]))
    robot_path = tmp_path / "Robot_pilot01.csv"
    robot.to_csv(robot_path, index=False)

    # Motion @ 1 kHz, sync rising at t=0.7, falling at t=4.2 (clock
    # offset vs robot — anchored to the same physical event though).
    n_m = 6000
    t_m = np.arange(n_m) / 1000.0
    motion = pd.DataFrame({
        "Time": t_m,
        "Sync": _sync_pulses(n_m, 1000.0, [(0.7, 4.2)]),
        "FP1_Fz": np.sin(2 * np.pi * t_m) * 200 + 600,
        "EMG_VL": np.cos(2 * np.pi * 4 * t_m) * 0.3,
    })
    motion_path = tmp_path / "Motion_pilot01.csv"
    motion.to_csv(motion_path, index=False)

    # Loadcell calibration (no sync — alignment routine should warn
    # but continue with the synced sources).
    loadcell = pd.DataFrame({
        "time": np.linspace(0, 5, 50),
        "applied_N": np.linspace(0, 50, 50),
        "robot_N": np.linspace(0, 51, 50),
    })
    loadcell_path = tmp_path / "Loadcell_calib_pilot01.csv"
    loadcell.to_csv(loadcell_path, index=False)

    ds_mod._REGISTRY.update({
        "ds_r": {"id": "ds_r", "name": robot_path.name,
                 "_path": str(robot_path), "source_kind": "robot"},
        "ds_m": {"id": "ds_m", "name": motion_path.name,
                 "_path": str(motion_path), "source_kind": "motion"},
        "ds_l": {"id": "ds_l", "name": loadcell_path.name,
                 "_path": str(loadcell_path), "source_kind": "loadcell"},
    })

    try:
        # Robot + Motion: align on first sync window.
        resp = align(AlignRequest(
            dataset_ids=["ds_r", "ds_m"],
            window_idx=0,
            target_fs_hz=500.0,
            columns_per_source={
                "ds_r": ["L_ActForce_N", "Sync"],
                "ds_m": ["FP1_Fz", "Sync", "EMG_VL"],
            },
        ))
        assert resp.target_fs_hz == 500.0
        assert resp.window_idx == 0
        assert resp.n_grid_samples > 100
        assert len(resp.t_aligned_s) == resp.n_grid_samples
        assert "ds_r" in resp.series and "ds_m" in resp.series
        assert "L_ActForce_N" in resp.series["ds_r"]
        assert "FP1_Fz" in resp.series["ds_m"]
        assert "EMG_VL" in resp.series["ds_m"]
        # Each source's window 0 was found and has positive duration
        assert resp.sync_windows["ds_r"] is not None
        assert resp.sync_windows["ds_m"] is not None
        assert resp.sync_windows["ds_r"].duration_s > 1.0
        assert resp.sync_windows["ds_m"].duration_s > 1.0

        # Adding the loadcell (no sync) must succeed for the synced
        # pair, with a warning naming the loadcell.
        resp2 = align(AlignRequest(
            dataset_ids=["ds_r", "ds_m", "ds_l"],
            window_idx=0,
            target_fs_hz=500.0,
        ))
        assert any("ds_l" in w for w in resp2.warnings)
        assert resp2.sync_windows["ds_l"] is None
    finally:
        for ds_id in ("ds_r", "ds_m", "ds_l"):
            ds_mod._REGISTRY.pop(ds_id, None)


def test_source_kind_classifies_three_kinds_from_filenames():
    """The Library can be source-aware (separate Robot / Motion /
    Loadcell sections) only if the upload classifier is reliable.
    This re-verifies the contract on filenames the user said they'd
    use."""
    from backend.services.source_kind import detect_source_kind

    cases = {
        "Robot CBJ_4.csv":      "robot",
        "robot_high_30.csv":    "robot",
        "Loadcell CBJ_5.csv":   "loadcell",
        "loadcell_low.csv":     "loadcell",
        "Motion CBJ.csv":       "motion",
        "motion_pilot01.csv":   "motion",
    }
    for filename, expected in cases.items():
        d = detect_source_kind(filename, ["Time_ms"])
        assert d.kind == expected, f"{filename}: expected {expected}, got {d.kind}"
        assert d.confidence >= 0.6


# Silence unused import warnings for fixtures the file uses.
_ = (io, Any)
