"""Multi-source sync alignment endpoints.

User-confirmed sync definition (2026-04-25):
    Rising edge = sync STARTS  (operator pressed → trial begins)
    Falling edge = sync ENDS   (operator released → trial ends)
    Window = [rising, falling] half-open interval

A recording can contain N sync windows. Each window is one trial.

Endpoints:

    GET  /api/sync/{ds_id}/windows
        List every [rising, falling] window in this dataset.

    POST /api/sync/align
        Align several datasets on the **same sync-window index**
        within each. Returns one common time grid + every numeric
        column from every source resampled to it. Caller picks
        which window (default 0 = first trial in recording).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services import sync_align


router = APIRouter(prefix="/api/sync", tags=["sync"])


def _read_df(ds_id: str) -> pd.DataFrame:
    from backend.services.dataset_registry import get_path  # lazy
    path = get_path(ds_id)
    if not path:
        raise HTTPException(status_code=404, detail=f"dataset '{ds_id}' not found")
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"CSV unreadable: {exc}") from exc


# ============================================================
# /windows — list all sync windows in this dataset
# ============================================================

class SyncWindowOut(BaseModel):
    index: int
    rising_t_s: float
    falling_t_s: float
    duration_s: float
    n_samples: int


class WindowsResponse(BaseModel):
    dataset_id: str
    sync_column: Optional[str]
    windows: list[SyncWindowOut]


@router.get("/{ds_id}/windows", response_model=WindowsResponse)
def list_windows(
    ds_id: str,
    min_duration_s: float = 0.5,
) -> WindowsResponse:
    """List operator-driven sync windows.

    `min_duration_s` (default 0.5 s) drops phantom pulses caused by
    file-IO toggling the sync line. Set to 0.0 to see every detected
    edge for hardware debugging.
    """
    df = _read_df(ds_id)
    sync_col = sync_align._find_sync_column(df)
    wins = sync_align.find_sync_windows(df, min_duration_s=min_duration_s)
    return WindowsResponse(
        dataset_id=ds_id,
        sync_column=sync_col,
        windows=[SyncWindowOut(**w.as_dict()) for w in wins],
    )


# ============================================================
# /align — align N sources on a chosen window index
# ============================================================

class AlignRequest(BaseModel):
    dataset_ids: list[str]
    window_idx: int = 0
    """Which sync window to align on (0-based within each recording).
    Defaults to the first trial in the recording."""
    target_fs_hz: float = 1000.0
    columns_per_source: Optional[dict[str, list[str]]] = None
    """Optional: per-dataset, only resample these columns. When
    omitted, every numeric column is resampled."""


class AlignedSeries(BaseModel):
    dataset_id: str
    columns: list[str]


class AlignResponse(BaseModel):
    target_fs_hz: float
    window_idx: int
    t_min_s: float
    t_max_s: float
    n_grid_samples: int
    t_aligned_s: list[float]
    series: dict[str, dict[str, list[float]]]
    """series[dataset_id][column_name] = values on common grid."""
    sources: list[AlignedSeries]
    sync_windows: dict[str, Optional[SyncWindowOut]]
    """Window picked from each source (None when missing)."""
    warnings: list[str]


@router.post("/align", response_model=AlignResponse)
def align(req: AlignRequest) -> AlignResponse:
    if len(req.dataset_ids) < 2:
        raise HTTPException(
            status_code=400, detail="sync align needs at least 2 dataset ids",
        )
    if req.target_fs_hz <= 0 or req.target_fs_hz > 10000:
        raise HTTPException(
            status_code=400,
            detail=f"target_fs_hz {req.target_fs_hz} out of [1, 10000]",
        )
    if req.window_idx < 0:
        raise HTTPException(status_code=400, detail="window_idx must be ≥ 0")

    sources_dfs: dict[str, pd.DataFrame] = {ds: _read_df(ds) for ds in req.dataset_ids}

    try:
        aligned = sync_align.align_sources_on_window(
            sources_dfs,
            window_idx=req.window_idx,
            target_fs=req.target_fs_hz,
            columns_per_source=req.columns_per_source,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    series_out: dict[str, dict[str, list[float]]] = {}
    sources_out: list[AlignedSeries] = []
    t_grid_ref: list[float] = []
    for src_id, gridded in aligned.grids.items():
        if not t_grid_ref:
            t_grid_ref = gridded["t_aligned"].tolist()
        per_ds = {c: gridded[c].tolist()
                  for c in gridded.columns if c != "t_aligned"}
        series_out[src_id] = per_ds
        sources_out.append(AlignedSeries(
            dataset_id=src_id, columns=list(per_ds.keys()),
        ))

    sync_windows_out: dict[str, Optional[SyncWindowOut]] = {}
    for src_id, win in aligned.window_per_source.items():
        sync_windows_out[src_id] = (
            SyncWindowOut(**win.as_dict()) if win is not None else None
        )

    return AlignResponse(
        target_fs_hz=aligned.target_fs_hz,
        window_idx=req.window_idx,
        t_min_s=aligned.t_min_s,
        t_max_s=aligned.t_max_s,
        n_grid_samples=aligned.n_grid_samples,
        t_aligned_s=t_grid_ref,
        series=series_out,
        sources=sources_out,
        sync_windows=sync_windows_out,
        warnings=aligned.warnings,
    )
