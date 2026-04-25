"""
/api/stats — Phase 2A statistical-test endpoint.

Two request shapes supported:

1) Raw series:
    {
        "op": "ttest_paired" | "ttest_welch" | "anova1" | "pearson" | "cohens_d" | "shapiro",
        "a":  [float, ...],          # for paired/welch/pearson/cohens_d/shapiro
        "b":  [float, ...],          # for paired/welch/pearson/cohens_d
        "groups": [[...], [...]],    # for anova1
        "paired": false              # cohens_d only
    }

2) Dataset-backed (convenience): pulls columns from an uploaded CSV.
    {
        "op": "ttest_paired",
        "dataset_id": "ds_xxx",
        "a_col": "L_ActForce_N",
        "b_col": "R_ActForce_N",
        "groups_cols": ["L_ActForce_N", "R_ActForce_N"]   # for anova1
    }

Response is a uniform stats dict (see `stats_engine.py` module docstring).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.dataset_registry import get_path
from backend.services import stats_engine


router = APIRouter(prefix="/api/stats", tags=["stats"])


class DatasetMetricRef(BaseModel):
    id: str
    metric: str


class StatsRequest(BaseModel):
    op: str
    # raw series
    a: Optional[list[float]] = None
    b: Optional[list[float]] = None
    groups: Optional[list[list[float]]] = None
    paired: bool = False
    # dataset-backed convenience (single dataset + column names — legacy)
    dataset_id: Optional[str] = None
    a_col: Optional[str] = None
    b_col: Optional[str] = None
    groups_cols: Optional[list[str]] = None
    # Phase 3 · cross-dataset: each entry = one dataset contributing a
    # metric value (peak_force_L, cadence_L, stride_time_mean_R, …).
    datasets_a: Optional[list[DatasetMetricRef]] = None
    datasets_b: Optional[list[DatasetMetricRef]] = None
    datasets_groups: Optional[list[list[DatasetMetricRef]]] = None


def _col(df: pd.DataFrame, name: str) -> list[float]:
    if name not in df.columns:
        raise HTTPException(status_code=400, detail=f"column '{name}' not in dataset")
    arr = df[name].to_numpy(dtype=float)
    return arr[np.isfinite(arr)].tolist()


def _extract_from_datasets(refs: list[DatasetMetricRef]) -> list[float]:
    """Flatten [{id, metric}, …] into a list of values by running the
    metric extractor against each dataset's analyzer result.

    For scalar metrics (peak_force_L, cadence_L, …) this gives one value
    per dataset — the typical between-subject design.
    For array metrics (stride_times_L, peaks_per_stride_L, …) this gives
    the concatenation of all strides across all listed datasets — useful
    for within-subject or pooled variability comparisons.
    """
    from backend.routers.analyze import analyze_cached
    from backend.services.metric_extractor import extract as _extract

    out: list[float] = []
    for ref in refs:
        try:
            res, _ = analyze_cached(ref.id)
        except HTTPException as e:
            raise HTTPException(
                status_code=e.status_code,
                detail=f"{ref.id}: {e.detail}",
            ) from e
        if res is None:
            raise HTTPException(
                status_code=409,
                detail=f"dataset {ref.id} is in generic mode (not H-Walker format); "
                       "cross-file stats require H-Walker analysis.",
            )
        try:
            vals = _extract(ref.metric, res)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        out.extend(vals)
    return out


@router.post("")
def run_stats(req: StatsRequest) -> dict[str, Any]:
    if req.op not in stats_engine.OP_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown op '{req.op}'. "
                   f"Known: {sorted(stats_engine.OP_REGISTRY.keys())}",
        )

    payload: dict[str, Any] = {"paired": req.paired}

    # Phase 3 · cross-dataset path takes precedence when any datasets_*
    # array is populated.
    if req.datasets_a or req.datasets_b or req.datasets_groups:
        if req.op == "anova1":
            refs = req.datasets_groups or []
            if not refs or len(refs) < 2:
                raise HTTPException(status_code=400, detail="anova1 cross-file needs ≥ 2 groups")
            payload["groups"] = [_extract_from_datasets(g) for g in refs]
        elif req.op == "shapiro":
            refs = req.datasets_a or []
            if not refs:
                raise HTTPException(status_code=400, detail="shapiro cross-file needs datasets_a")
            payload["a"] = _extract_from_datasets(refs)
        else:
            if not req.datasets_a or not req.datasets_b:
                raise HTTPException(
                    status_code=400,
                    detail=f"{req.op} cross-file needs both datasets_a and datasets_b",
                )
            payload["a"] = _extract_from_datasets(req.datasets_a)
            payload["b"] = _extract_from_datasets(req.datasets_b)

    elif req.dataset_id:
        path = get_path(req.dataset_id)
        if not path:
            raise HTTPException(status_code=404, detail=f"dataset '{req.dataset_id}' not found")
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"CSV unreadable: {exc}") from exc

        if req.op == "anova1":
            if not req.groups_cols:
                raise HTTPException(status_code=400, detail="anova1 needs groups_cols")
            payload["groups"] = [_col(df, c) for c in req.groups_cols]
        elif req.op == "shapiro":
            if not req.a_col:
                raise HTTPException(status_code=400, detail="shapiro needs a_col")
            payload["a"] = _col(df, req.a_col)
        else:
            if not req.a_col or not req.b_col:
                raise HTTPException(status_code=400, detail=f"{req.op} needs a_col + b_col")
            payload["a"] = _col(df, req.a_col)
            payload["b"] = _col(df, req.b_col)
    else:
        if req.op == "anova1":
            payload["groups"] = req.groups or []
        elif req.op == "shapiro":
            payload["a"] = req.a or []
        else:
            payload["a"] = req.a or []
            payload["b"] = req.b or []

    try:
        return stats_engine.run(req.op, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"stats failed: {exc}") from exc


@router.get("/ops")
def list_ops() -> list[str]:
    return sorted(stats_engine.OP_REGISTRY.keys())


@router.get("/metrics")
def list_metrics() -> list[dict[str, Any]]:
    """Phase 3 · return every cross-file metric key with human-readable
    label, unit, side (L/R/-), and kind (kinetic/temporal/spatial/...)
    so the UI can group them in a dropdown."""
    from backend.services.metric_extractor import METRIC_EXTRACTORS, describe_metric
    return [describe_metric(k) for k in sorted(METRIC_EXTRACTORS.keys())]
