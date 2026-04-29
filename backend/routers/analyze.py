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
from backend.services.analysis_engine import (
    run_full_analysis, run_per_window_analysis, count_sync_windows,
)
from tools.auto_analyzer.analyzer import result_to_dict, AnalysisResult


router = APIRouter(prefix="/api/analyze", tags=["analyze"])


# In-memory cache: (ds_id, window_idx) → (AnalysisResult, payload_dict)
_CACHE: dict[tuple[str, int], tuple[AnalysisResult, dict[str, Any]]] = {}

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


def _disk_load(path: str, window_idx: int) -> tuple[AnalysisResult, dict[str, Any]] | None:
    key = _cache_key(path)
    if not key:
        return None
    cpath = _DISK_CACHE_DIR / f"{key}_w{window_idx}.pkl"
    if not cpath.exists():
        return None
    try:
        with open(cpath, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _disk_save(path: str, window_idx: int, res: AnalysisResult, payload: dict[str, Any]) -> None:
    key = _cache_key(path)
    if not key:
        return
    cpath = _DISK_CACHE_DIR / f"{key}_w{window_idx}.pkl"
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


def _result_payload(res: AnalysisResult, n_windows: int = 1) -> dict[str, Any]:
    """Convert AnalysisResult to JSON, including force profiles + sync window meta.

    `n_windows` is the total trial count in the recording so the
    frontend can offer a window selector when there's more than one.
    Per-window slicing is enforced upstream by `run_full_analysis`,
    which reads only `[rising, falling)` samples — data outside the
    sync window never reaches the analyzer.
    """
    d = result_to_dict(res)
    d["mode"] = "hwalker"
    d["profiles"] = {
        "left": _profile_to_json(res.left_force_profile),
        "right": _profile_to_json(res.right_force_profile),
    }
    # Sync window provenance: which trial in the recording this is.
    d["sync"] = {
        "n_windows": int(n_windows),
        "window_idx": (int(res.sync_window_idx)
                        if res.sync_window_idx is not None else None),
        "t_rise_s": (float(res.sync_window_t_rise_s)
                      if res.sync_window_t_rise_s is not None else None),
        "t_fall_s": (float(res.sync_window_t_fall_s)
                      if res.sync_window_t_fall_s is not None else None),
        "duration_s": (
            float(res.sync_window_t_fall_s - res.sync_window_t_rise_s)
            if (res.sync_window_t_rise_s is not None
                and res.sync_window_t_fall_s is not None)
            else None
        ),
    }
    # Per-stride arrays (truncated for payload size)
    for side_name, sr in [("left", res.left_stride), ("right", res.right_stride)]:
        d[side_name]["stride_times_list"] = sr.stride_times.tolist()
        d[side_name]["stride_lengths_list"] = (
            sr.stride_lengths[np.isfinite(sr.stride_lengths)].tolist()
            if len(sr.stride_lengths) else []
        )
    return d


def analyze_cached(
    ds_id: str, window_idx: int = 0,
) -> tuple[AnalysisResult | None, dict[str, Any]]:
    """Return (AnalysisResult, payload) for a single sync window.

    Per the user's sync contract a recording with N rising/falling
    pulses is N trials and analysis must use **only data inside the
    [rising, falling) window**. `window_idx` selects which trial.

    Cache hierarchy:
      1. Per-session in-memory (fastest, keyed on (ds_id, window_idx))
      2. Disk cache keyed by file content hash + window
      3. Fresh sync-sliced analysis via auto_analyzer
    """
    cache_key = (ds_id, window_idx)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    path = get_path(ds_id)
    if not path:
        raise HTTPException(status_code=404, detail=f"dataset '{ds_id}' not found")

    # Disk cache — skip the full pipeline when we've seen this file before
    cached = _disk_load(path, window_idx)
    if cached is not None:
        _CACHE[cache_key] = cached
        return cached

    try:
        df = pd.read_csv(path, nrows=5)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"CSV unreadable: {exc}") from exc

    ds_name = _REGISTRY.get(ds_id, {}).get("name", "unknown.csv")

    if not _is_hwalker_csv(df):
        # Generic mode: no sync semantics, return descriptive stats only.
        try:
            full = pd.read_csv(path)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"CSV unreadable: {exc}") from exc
        payload = _generic_analysis(full, ds_name)
        _CACHE[cache_key] = (None, payload)  # type: ignore[assignment]
        return None, payload

    n_windows = count_sync_windows(path)
    if window_idx < 0 or window_idx >= n_windows:
        raise HTTPException(
            status_code=400,
            detail=(
                f"window_idx={window_idx} out of range; recording has "
                f"{n_windows} sync window(s)"
            ),
        )

    try:
        res = run_full_analysis(path, window_idx=window_idx)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"analyzer failed: {exc}") from exc

    payload = _result_payload(res, n_windows=n_windows)
    _CACHE[cache_key] = (res, payload)
    _disk_save(path, window_idx, res, payload)
    return res, payload


def invalidate_cache(ds_id: str) -> None:
    """Drop every per-window entry for this dataset."""
    for key in list(_CACHE.keys()):
        if key[0] == ds_id:
            _CACHE.pop(key, None)


@router.get("/{ds_id}")
def analyze(ds_id: str, window: int = 0) -> dict[str, Any]:
    """Run or fetch cached H-Walker analysis for one sync window.

    Default `window=0` returns the first trial. Use `?window=N` to
    pick a later trial in the same recording. The response always
    includes `sync.n_windows` so callers know how many trials are
    available without a separate probe call.
    """
    _, payload = analyze_cached(ds_id, window_idx=window)
    return payload


@router.get("/{ds_id}/windows")
def list_windows(
    ds_id: str,
    min_duration_s: float = 0.5,
    include_phantom: bool = False,
) -> dict[str, Any]:
    """List every sync window in a dataset with per-trial metadata.

    Light-weight probe — runs the rising/falling edge detector but
    not the full analyzer. Use this to populate a trial picker in the
    UI before deciding which window(s) to analyze.

    Phantom-pulse filtering:
      `min_duration_s` (default 0.5 s) drops sub-threshold pulses
      caused by file-IO toggling the sync line. Set to 0.0 to keep
      every detected pulse; set higher for noisier hardware.
      `include_phantom=true` returns BOTH the surviving windows and
      the dropped phantoms (in `phantoms[]`) so the inspector UI can
      surface what was filtered without losing visibility.
    """
    path = get_path(ds_id)
    if not path:
        raise HTTPException(status_code=404, detail=f"dataset '{ds_id}' not found")
    from backend.services.sync_align import find_sync_windows
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"CSV unreadable: {exc}") from exc

    def _serialize(w):
        return {
            "idx": w.index,
            "t_rise_s": float(w.rising_t_s),
            "t_fall_s": float(w.falling_t_s),
            "duration_s": float(w.duration_s),
            "n_samples": int(w.sample_falling - w.sample_rising),
        }

    windows = find_sync_windows(df, min_duration_s=min_duration_s)
    payload: dict[str, Any] = {
        "ds_id": ds_id,
        "min_duration_s": float(min_duration_s),
        "n_windows": len(windows),
        "windows": [_serialize(w) for w in windows],
    }
    if include_phantom:
        all_pulses = find_sync_windows(df, min_duration_s=0.0)
        survivor_keys = {(w.sample_rising, w.sample_falling) for w in windows}
        phantoms = [
            _serialize(w) for w in all_pulses
            if (w.sample_rising, w.sample_falling) not in survivor_keys
        ]
        payload["phantoms"] = phantoms
        payload["n_phantoms"] = len(phantoms)
    return payload


@router.delete("/{ds_id}/cache")
def drop_cache(ds_id: str) -> dict[str, Any]:
    existed = any(k[0] == ds_id for k in _CACHE)
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
