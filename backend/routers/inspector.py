"""Per-sync inspector — MATLAB-style zoom/pan over raw CSV columns.

User-confirmed sync definition (2026-04-25):
    Rising edge = sync STARTS  (operator pressed button → trial begins)
    Falling edge = sync ENDS   (operator released → trial ends)
    Window = [rising, falling] half-open interval

A recording can contain multiple windows; each window is one trial.
This router exposes the windows for visualization + a generic
zoom-window data-fetch.

Endpoints:

    GET /api/inspector/{ds_id}/syncs
        Every [rising, falling] window in this dataset's Sync column.

    POST /api/inspector/{ds_id}/window
        { columns, t_start, t_end, max_points } → downsampled traces
        of the requested raw columns inside [t_start, t_end].
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


def _detect_sync_windows(
    sync: np.ndarray,
    t: np.ndarray,
    min_duration_s: float = 0.0,
) -> list[tuple[float, float]]:
    """Find every sync window = [rising-edge, falling-edge).

    Threshold at midpoint(min, max) so digital and analog sync both
    work. For each rising edge, pair with the next falling edge.
    A trailing rising edge with no closing falling edge is dropped
    (incomplete trial — operator never released or recording stopped).

    `min_duration_s` filters phantom pulses (file-IO glitches that
    briefly toggle the sync line). Defaults to 0.0 here so existing
    direct callers see every detected pulse — the public endpoint
    `GET /api/inspector/{id}/syncs` applies a non-zero default itself.
    """
    if sync.size == 0 or not np.isfinite(sync).any():
        return []
    finite = sync[np.isfinite(sync)]
    if finite.size == 0:
        return []
    lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if hi - lo < 1e-9:
        return []  # constant signal
    threshold = (lo + hi) / 2.0
    high = (sync > threshold).astype(np.int8)
    diff = np.diff(high)
    rising  = np.where(diff ==  1)[0] + 1
    falling = np.where(diff == -1)[0] + 1
    if rising.size == 0 or falling.size == 0:
        return []

    windows: list[tuple[float, float]] = []
    used_falling = 0
    for r in rising:
        cands = falling[(falling > r) & (np.arange(falling.size) >= used_falling)]
        if cands.size == 0:
            break
        f = cands[0]
        used_falling = int(np.where(falling == f)[0][0]) + 1
        windows.append((float(t[r]), float(t[f])))
    if min_duration_s > 0:
        windows = [(s, e) for (s, e) in windows if (e - s) >= min_duration_s]
    return windows


# ============================================================
# Endpoints
# ============================================================

class SyncBoundary(BaseModel):
    index: int           # 0-based window number within recording
    t_start: float       # rising-edge time (seconds)
    t_end: float         # falling-edge time (seconds)
    duration: float      # t_end − t_start


class SyncsResponse(BaseModel):
    column: Optional[str]
    n_samples: int
    sample_rate_hz: Optional[float]
    cycles: list[SyncBoundary]
    """Every [rising, falling] sync window in the recording. Field
    name kept as `cycles` for frontend backwards-compat; the items
    are windows in the rising→falling sense."""


@router.get("/{ds_id}/syncs", response_model=SyncsResponse)
def list_syncs(
    ds_id: str,
    min_duration_s: float = 0.5,
) -> SyncsResponse:
    """List rising→falling sync windows in the recording.

    `min_duration_s` filters phantom pulses caused by file-IO toggling
    the sync GPIO line. Default 0.5 s drops anything shorter than half
    a stride. Pass 0.0 to see every detected edge for hardware
    debugging.
    """
    df = _read_df(ds_id)
    t = _time_axis(df)
    fs = float(1.0 / np.median(np.diff(t))) if len(t) > 1 else None

    if "Sync" not in df.columns:
        return SyncsResponse(
            column=None, n_samples=len(df), sample_rate_hz=fs, cycles=[],
        )

    sync = df["Sync"].to_numpy(dtype=float)
    windows = _detect_sync_windows(sync, t, min_duration_s=min_duration_s)
    return SyncsResponse(
        column="Sync",
        n_samples=len(df),
        sample_rate_hz=fs,
        cycles=[
            SyncBoundary(index=i, t_start=s, t_end=e, duration=e - s)
            for i, (s, e) in enumerate(windows)
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
