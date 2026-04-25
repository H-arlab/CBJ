"""
/api/analyze/{ds_id} — Phase 2A gait analysis endpoint.

Wraps `tools.auto_analyzer.analyzer.analyze_file()` around an uploaded CSV.
Returns the rich AnalysisResult as JSON (via result_to_dict) plus GCP-normalized
force profiles so the frontend can bind real curves.

Uses a per-dataset cache so repeated calls don't re-run the analyzer. The
cache is invalidated automatically when the dataset is deleted (see
`datasets.delete_dataset`).

When the CSV doesn't match the H-Walker 67-column format (no L_/R_ prefixed
columns, or too few samples), returns a `fallback_mode: "generic"` payload
with descriptive stats only — no crash.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from backend.services.dataset_registry import get_path, _REGISTRY
from backend.services.analysis_engine import run_full_analysis
from tools.auto_analyzer.analyzer import result_to_dict, AnalysisResult


router = APIRouter(prefix="/api/analyze", tags=["analyze"])


# In-memory cache: ds_id → (AnalysisResult, payload_dict)
_CACHE: dict[str, tuple[AnalysisResult, dict[str, Any]]] = {}

# Phase 4 · disk cache for analyzer results. Keyed by CSV content hash
# (sha256 of first 1 MB + mtime + size) so identical files across
# sessions / datasets don't re-analyze.
_DISK_CACHE_DIR = Path(os.path.expanduser("~/.hw_graph/cache/analyze"))
_DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_VERSION = 1  # bump when AnalysisResult shape changes


def _cache_key(path: str) -> str:
    """Stable key from CSV content + mtime + size. Avoids re-hashing
    large files by taking the first 1 MB sample."""
    try:
        stat = os.stat(path)
        h = hashlib.sha256()
        h.update(f"v{_CACHE_VERSION}|size={stat.st_size}|mtime={int(stat.st_mtime)}|".encode())
        with open(path, "rb") as f:
            h.update(f.read(1024 * 1024))
        return h.hexdigest()[:24]
    except OSError:
        return ""


def _disk_load(path: str) -> tuple[AnalysisResult, dict[str, Any]] | None:
    key = _cache_key(path)
    if not key:
        return None
    cpath = _DISK_CACHE_DIR / f"{key}.pkl"
    if not cpath.exists():
        return None
    try:
        with open(cpath, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _disk_save(path: str, res: AnalysisResult, payload: dict[str, Any]) -> None:
    key = _cache_key(path)
    if not key:
        return
    cpath = _DISK_CACHE_DIR / f"{key}.pkl"
    try:
        with open(cpath, "wb") as f:
            pickle.dump((res, payload), f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass


def _is_hwalker_csv(df: pd.DataFrame) -> bool:
    """Heuristic: does this CSV look like H-Walker firmware output?"""
    cols = [str(c) for c in df.columns]
    has_l = any(c.startswith("L_") for c in cols)
    has_r = any(c.startswith("R_") for c in cols)
    has_force = any("Force" in c or "GCP" in c for c in cols)
    return has_l and has_r and has_force


def _generic_analysis(df: pd.DataFrame, filename: str) -> dict[str, Any]:
    """Fallback: descriptive stats per numeric column."""
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary = {}
    for c in cols:
        arr = df[c].to_numpy(dtype=float)
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            continue
        summary[c] = {
            "n": int(len(valid)),
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "median": float(np.median(valid)),
        }
    return {
        "fallback_mode": "generic",
        "filename": filename,
        "n_samples": int(len(df)),
        "n_columns": int(len(cols)),
        "columns": cols,
        "descriptive": summary,
        "note": "CSV does not match H-Walker 67-column format; gait metrics skipped.",
    }


def _profile_to_json(fp) -> dict[str, Any]:
    """Convert ForceProfileResult to JSON-friendly dict."""
    if fp.mean is None:
        return {"available": False}
    out = {
        "available": True,
        "n_points": int(len(fp.mean)),
        "mean": fp.mean.tolist(),
        "std": fp.std.tolist() if fp.std is not None else [],
    }
    if fp.des_mean is not None:
        out["des_mean"] = fp.des_mean.tolist()
        out["des_std"] = fp.des_std.tolist() if fp.des_std is not None else []
    return out


def _result_payload(res: AnalysisResult) -> dict[str, Any]:
    """Convert AnalysisResult to JSON, including force profiles."""
    d = result_to_dict(res)
    d["mode"] = "hwalker"
    d["profiles"] = {
        "left": _profile_to_json(res.left_force_profile),
        "right": _profile_to_json(res.right_force_profile),
    }
    # Per-stride arrays (truncated for payload size)
    for side_name, sr in [("left", res.left_stride), ("right", res.right_stride)]:
        d[side_name]["stride_times_list"] = sr.stride_times.tolist()
        d[side_name]["stride_lengths_list"] = (
            sr.stride_lengths[np.isfinite(sr.stride_lengths)].tolist()
            if len(sr.stride_lengths) else []
        )
    return d


def analyze_cached(ds_id: str) -> tuple[AnalysisResult | None, dict[str, Any]]:
    """Return (AnalysisResult, payload). Result may be None in fallback mode.

    Cache hierarchy:
      1. Per-session in-memory (fastest)
      2. Disk cache keyed by file content hash (survives restarts)
      3. Fresh analysis via auto_analyzer
    """
    if ds_id in _CACHE:
        res, payload = _CACHE[ds_id]
        return res, payload

    path = get_path(ds_id)
    if not path:
        raise HTTPException(status_code=404, detail=f"dataset '{ds_id}' not found")

    # Disk cache — skip the full pipeline when we've seen this file before
    cached = _disk_load(path)
    if cached is not None:
        _CACHE[ds_id] = cached
        return cached

    try:
        df = pd.read_csv(path, nrows=5)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"CSV unreadable: {exc}") from exc

    ds_name = _REGISTRY.get(ds_id, {}).get("name", "unknown.csv")

    if not _is_hwalker_csv(df):
        # Read full CSV for generic mode
        try:
            full = pd.read_csv(path)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"CSV unreadable: {exc}") from exc
        payload = _generic_analysis(full, ds_name)
        _CACHE[ds_id] = (None, payload)  # type: ignore[assignment]
        return None, payload

    try:
        res = run_full_analysis(path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"analyzer failed: {exc}") from exc

    payload = _result_payload(res)
    _CACHE[ds_id] = (res, payload)
    _disk_save(path, res, payload)
    return res, payload


def invalidate_cache(ds_id: str) -> None:
    _CACHE.pop(ds_id, None)


@router.get("/{ds_id}")
def analyze(ds_id: str) -> dict[str, Any]:
    """Run or fetch cached H-Walker analysis for a dataset."""
    _, payload = analyze_cached(ds_id)
    return payload


@router.delete("/{ds_id}/cache")
def drop_cache(ds_id: str) -> dict[str, Any]:
    existed = ds_id in _CACHE
    invalidate_cache(ds_id)
    return {"ds_id": ds_id, "invalidated": existed}


@router.get("/cache/stats")
def cache_stats() -> dict[str, Any]:
    """Phase 4 · inspect the disk cache size + entry count."""
    try:
        files = list(_DISK_CACHE_DIR.glob("*.pkl"))
        total_bytes = sum(f.stat().st_size for f in files)
        return {
            "memory_entries": len(_CACHE),
            "disk_entries": len(files),
            "disk_bytes": total_bytes,
            "disk_mb": round(total_bytes / 1024 / 1024, 2),
            "cache_dir": str(_DISK_CACHE_DIR),
        }
    except Exception as exc:
        return {"error": str(exc)}


@router.delete("/cache")
def clear_disk_cache() -> dict[str, Any]:
    """Wipe the entire disk cache (memory cache untouched)."""
    removed = 0
    for f in _DISK_CACHE_DIR.glob("*.pkl"):
        try:
            f.unlink()
            removed += 1
        except Exception:
            pass
    return {"removed": removed}
