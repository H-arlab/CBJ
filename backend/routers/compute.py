"""
/api/compute — Phase 2A compute-metric endpoint.

Request body:
    {
        "dataset_id": "ds_xxx",
        "metric":     "per_stride" | "impulse" | "loading_rate" | "rom" | "cadence" | "target_dev",
        "options":    { ... }          # optional kwargs forwarded to the metric fn
    }

Response:
    { "label", "cols", "rows", "summary": { "mean": [...] }, "meta": { ... } }

Fallback: if the dataset is in generic mode (non-H-Walker CSV), returns an
explanatory 409 instead of crashing.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.routers.analyze import analyze_cached
from backend.services.dataset_registry import get_path
from backend.services import compute_engine


router = APIRouter(prefix="/api/compute", tags=["compute"])


class ComputeRequest(BaseModel):
    dataset_id: str
    metric: str
    options: Optional[dict[str, Any]] = None


@router.post("")
def compute_metric(req: ComputeRequest) -> dict[str, Any]:
    if req.metric not in compute_engine.METRIC_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown metric '{req.metric}'. "
                   f"Known: {sorted(compute_engine.METRIC_REGISTRY.keys())}",
        )

    res, _payload = analyze_cached(req.dataset_id)
    if res is None:
        raise HTTPException(
            status_code=409,
            detail="Dataset is in generic fallback mode (not H-Walker format). "
                   "Compute metrics require H-Walker CSV.",
        )

    path = get_path(req.dataset_id)
    if not path:
        raise HTTPException(status_code=404, detail=f"dataset '{req.dataset_id}' not found")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"CSV unreadable: {exc}") from exc

    opts = req.options or {}
    try:
        return compute_engine.compute(req.metric, df, res, **opts)
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=f"bad options: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"compute failed: {exc}") from exc


@router.get("/metrics")
def list_metrics() -> list[str]:
    return sorted(compute_engine.METRIC_REGISTRY.keys())
