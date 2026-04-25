"""Per-sync inspector — MATLAB-style zoom/pan over raw CSV columns.

The user's definition of `sync` (CLAUDE.md, declarative rules):
    "디지털/아날로그 sync 신호의 한 사이클 — falling edge 후 rising
     edge 부터 다시 falling edge 까지가 1 sync."

For the H-Walker firmware CSV the canonical source is the `Sync` column
(see tools/graph_analyzer/data_manager.py CANONICAL_COLUMNS). It is a
boolean-ish square wave: 0 / 1 transitions denote sync cycles.

Two endpoints:

    GET /api/inspector/{ds_id}/syncs
        Detect every sync cycle and return its [t_start, t_end] in
        seconds. Empty list when the dataset has no Sync column.

    POST /api/inspector/{ds_id}/window
        Body: { columns: [str], t_start: float, t_end: float,
                max_points: int = 4000 }
        Returns the requested raw columns inside the time window,
        downsampled to at most max_points (LTTB-ish stride). This is
        what the frontend re-fetches on every zoom / pan gesture so
        we never ship the entire trial in one payload.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# datasets router pulls in python-multipart at import time (the upload
# endpoint uses Form data). Inspector is decoupled from that — we only
# need the path-lookup helper. Lazy import inside _read_df keeps the
# test suite runnable without multipart installed.

router = APIRouter(prefix="/api/inspector", tags=["inspector"])


# ============================================================
# Sync cycle detection
# ============================================================

def _read_df(ds_id: str) -> pd.DataFrame:
    from backend.services.dataset_registry import get_path
    path = get_path(ds_id)
    if not path:
        raise HTTPException(status_code=404, detail=f"dataset '{ds_id}' not found")
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"CSV unreadable: {exc}") from exc


def _time_axis(df: pd.DataFrame) -> np.ndarray:
    """Return a seconds-axis. Prefer existing time columns; otherwise
    fall back to a best-effort sample-rate estimate."""
    for col in ("Time_s", "Timestamp", "Time"):
        if col in df.columns:
            t = df[col].to_numpy(dtype=float)
            if np.isfinite(t).sum() >= 2:
                return t
    for col in ("Time_ms", "time_ms"):
        if col in df.columns:
            t = df[col].to_numpy(dtype=float) / 1000.0
            if np.isfinite(t).sum() >= 2:
                return t
    # Last resort: assume 111 Hz (Teensy default in DataManager).
    fs = 111.0
    return np.arange(len(df)) / fs


def _detect_sync_cycles(sync: np.ndarray, t: np.ndarray) -> list[tuple[float, float]]:
    """Find every full sync cycle = [falling → rising → next falling].

    Sync is boolean-ish; we threshold at the midpoint. Each falling
    edge starts a cycle, the next falling edge ends it. The very last
    falling edge has no end, so its cycle is dropped.
    """
    if sync.size == 0 or not np.isfinite(sync).any():
        return []
    finite = sync[np.isfinite(sync)]
    if finite.size == 0:
        return []
    lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if hi - lo < 1e-9:
        return []  # constant signal — no cycles
    threshold = (lo + hi) / 2.0
    high = sync > threshold

    # Falling edges: high[i-1] && !high[i]
    # We use np.diff on int(high) to find transitions.
    h = high.astype(np.int8)
    d = np.diff(h)
    falling = np.where(d == -1)[0] + 1  # index right after the transition
    if falling.size < 2:
        return []
    cycles: list[tuple[float, float]] = []
    for i in range(falling.size - 1):
        s, e = falling[i], falling[i + 1]
        # Only return cycles that contain a rising edge in between
        # (otherwise it's just a noise blip, not a real cycle).
        between = h[s:e]
        if np.any(between == 1):
            cycles.append((float(t[s]), float(t[e])))
    return cycles


# ============================================================
# Endpoints
# ============================================================

class SyncBoundary(BaseModel):
    index: int           # 0-based sync number
    t_start: float
    t_end: float
    duration: float


class SyncsResponse(BaseModel):
    column: Optional[str]
    n_samples: int
    sample_rate_hz: Optional[float]
    cycles: list[SyncBoundary]


@router.get("/{ds_id}/syncs", response_model=SyncsResponse)
def list_syncs(ds_id: str) -> SyncsResponse:
    df = _read_df(ds_id)
    t = _time_axis(df)
    fs = float(1.0 / np.median(np.diff(t))) if len(t) > 1 else None

    if "Sync" not in df.columns:
        return SyncsResponse(
            column=None, n_samples=len(df), sample_rate_hz=fs, cycles=[],
        )

    sync = df["Sync"].to_numpy(dtype=float)
    cycles = _detect_sync_cycles(sync, t)
    return SyncsResponse(
        column="Sync",
        n_samples=len(df),
        sample_rate_hz=fs,
        cycles=[
            SyncBoundary(index=i, t_start=s, t_end=e, duration=e - s)
            for i, (s, e) in enumerate(cycles)
        ],
    )


class WindowRequest(BaseModel):
    columns: list[str]
    t_start: float
    t_end: float
    max_points: int = 4000


class WindowSeries(BaseModel):
    name: str
    y: list[float]


class WindowResponse(BaseModel):
    t: list[float]
    series: list[WindowSeries]
    n_total: int        # samples inside [t_start, t_end] before downsample
    n_returned: int     # after downsample
    columns_missing: list[str]


def _downsample_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    # Even-stride decimation. Good enough for visual inspection; if a
    # spike falls between samples the user can tighten the window via
    # zoom and re-fetch — same UX trick MATLAB plot uses.
    return np.linspace(0, n - 1, max_points).astype(int)


@router.post("/{ds_id}/window", response_model=WindowResponse)
def fetch_window(ds_id: str, req: WindowRequest) -> WindowResponse:
    if not req.columns:
        raise HTTPException(status_code=400, detail="columns is empty")
    if req.t_end <= req.t_start:
        raise HTTPException(status_code=400, detail="t_end must be > t_start")
    if req.max_points < 50 or req.max_points > 50000:
        raise HTTPException(status_code=400, detail="max_points out of range [50, 50000]")

    df = _read_df(ds_id)
    t_full = _time_axis(df)
    mask = (t_full >= req.t_start) & (t_full <= req.t_end)
    n_total = int(mask.sum())
    if n_total == 0:
        return WindowResponse(t=[], series=[], n_total=0, n_returned=0,
                              columns_missing=[])

    idx_in_window = np.where(mask)[0]
    keep = idx_in_window[_downsample_indices(n_total, req.max_points)]

    t_out = t_full[keep].tolist()
    series: list[WindowSeries] = []
    missing: list[str] = []
    for col in req.columns:
        if col not in df.columns:
            missing.append(col)
            continue
        y = df[col].to_numpy(dtype=float)[keep]
        # NaN → null on the wire. JSON doesn't allow NaN, so we
        # forward-fill the last finite value (visualization only).
        if not np.all(np.isfinite(y)):
            last = 0.0
            cleaned = np.empty_like(y)
            for i, v in enumerate(y):
                if np.isfinite(v):
                    last = v
                cleaned[i] = last
            y = cleaned
        series.append(WindowSeries(name=col, y=y.tolist()))

    return WindowResponse(
        t=t_out, series=series,
        n_total=n_total, n_returned=len(keep),
        columns_missing=missing,
    )
