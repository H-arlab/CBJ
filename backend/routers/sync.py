"""Multi-source sync alignment endpoints.

User contract: when one experimental session has a Robot CSV, a
Motion CSV, and a Loadcell calibration CSV, they cannot be
analysed in lockstep until each is anchored to the **first sync
falling edge** and resampled onto a common high-rate grid. These
endpoints expose the building blocks for that workflow:

  GET  /api/sync/{ds_id}/edge
        → first-falling-edge time of this dataset's Sync column,
          in seconds (or null if no sync). Useful for the UI to
          show "anchor at 1.234 s" before alignment.

  POST /api/sync/align
        body: { dataset_ids: [str, ...], target_fs_hz: float }
        → returns the gridded time axis + every numeric column from
          every source on a single uniform time grid. Calls
          backend.services.sync_align.align_pair iteratively.
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


# ---------------------------------------------------------------
# /edge — show first sync falling-edge time
# ---------------------------------------------------------------

class EdgeResponse(BaseModel):
    dataset_id: str
    sync_column: Optional[str]
    first_falling_t_s: Optional[float]


@router.get("/{ds_id}/edge", response_model=EdgeResponse)
def edge(ds_id: str) -> EdgeResponse:
    df = _read_df(ds_id)
    sync_col = sync_align._find_sync_column(df)
    t0 = sync_align.find_first_sync_falling_t(df)
    return EdgeResponse(
        dataset_id=ds_id,
        sync_column=sync_col,
        first_falling_t_s=t0,
    )


# ---------------------------------------------------------------
# /align — bring N sources onto a common time grid
# ---------------------------------------------------------------

class AlignRequest(BaseModel):
    dataset_ids: list[str]
    target_fs_hz: float = 1000.0
    columns_per_source: Optional[dict[str, list[str]]] = None
    """Optional restriction: per-dataset, only resample these columns
    (everything else is dropped from the output to keep the payload
    small). When omitted, every numeric column is resampled."""


class AlignedSeries(BaseModel):
    dataset_id: str
    columns: list[str]


class AlignResponse(BaseModel):
    target_fs_hz: float
    t_min_s: float
    t_max_s: float
    n_grid_samples: int
    t_aligned_s: list[float]
    series: dict[str, dict[str, list[float]]]
    """series[dataset_id][column_name] = list of values on the common grid."""
    sources: list[AlignedSeries]
    sync_offsets_s: dict[str, Optional[float]]
    warnings: list[str]


@router.post("/align", response_model=AlignResponse)
def align(req: AlignRequest) -> AlignResponse:
    if len(req.dataset_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="sync align needs at least 2 dataset ids",
        )
    if req.target_fs_hz <= 0 or req.target_fs_hz > 10000:
        raise HTTPException(
            status_code=400,
            detail=f"target_fs_hz {req.target_fs_hz} out of [1, 10000]",
        )

    # Load every source, anchor each to its own sync falling-edge.
    raw: list[tuple[str, pd.DataFrame]] = []
    sync_offsets: dict[str, Optional[float]] = {}
    warnings: list[str] = []
    for ds_id in req.dataset_ids:
        df = _read_df(ds_id)
        t0 = sync_align.find_first_sync_falling_t(df)
        sync_offsets[ds_id] = float(t0) if t0 is not None else None
        if t0 is None:
            warnings.append(
                f"{ds_id}: no sync falling-edge found — anchored at 0 s "
                "(alignment may be off by this source's recording offset)"
            )
        aligned = sync_align.align_to_t0(df, t0)
        raw.append((ds_id, aligned))

    # Common overlap window across all aligned sources.
    t_min = max(df["t_aligned"].min() for _, df in raw)
    t_max = min(df["t_aligned"].max() for _, df in raw)
    if t_max <= t_min:
        raise HTTPException(
            status_code=409,
            detail=(
                "no temporal overlap after sync alignment. Each source "
                "covers a different post-sync window — check that sync "
                "edges actually correspond to the same physical event."
            ),
        )

    # Resample every source onto the common grid.
    series_out: dict[str, dict[str, list[float]]] = {}
    sources_out: list[AlignedSeries] = []
    t_grid_ref: Optional[list[float]] = None
    for ds_id, aligned in raw:
        cols_filter = (req.columns_per_source or {}).get(ds_id)
        gridded = sync_align.resample_to_grid(
            aligned, req.target_fs_hz, t_min, t_max,
            value_cols=cols_filter,
        )
        if t_grid_ref is None:
            t_grid_ref = gridded["t_aligned"].tolist()
        per_ds = {
            c: gridded[c].tolist()
            for c in gridded.columns if c != "t_aligned"
        }
        series_out[ds_id] = per_ds
        sources_out.append(AlignedSeries(
            dataset_id=ds_id, columns=list(per_ds.keys()),
        ))

    return AlignResponse(
        target_fs_hz=req.target_fs_hz,
        t_min_s=float(t_min),
        t_max_s=float(t_max),
        n_grid_samples=len(t_grid_ref or []),
        t_aligned_s=t_grid_ref or [],
        series=series_out,
        sources=sources_out,
        sync_offsets_s=sync_offsets,
        warnings=warnings,
    )
