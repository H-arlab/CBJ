"""Analysis engine for H-Walker CSV data.

Wraps auto_analyzer's robust engine while keeping the simple API
that graph_quick.py and graph_publication.py expect.
"""
from __future__ import annotations

import sys
import os

import numpy as np
import pandas as pd

# Add project root so auto_analyzer can be imported
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tools.auto_analyzer.analyzer import (
    analyze_file as _analyze_file,
    analyze_file_windows as _analyze_file_windows,
    AnalysisResult,
    result_to_dict,
    compare_results,
    _detect_heel_strikes,
    _filter_strides_iqr,
)
from tools.graph_analyzer.data_manager import DataManager

from backend.models.schema import StatsResult


def load_csv(path: str) -> pd.DataFrame:
    """Load a Teensy CSV log with duplicate header removal."""
    dm = DataManager()
    lf = dm.load_csv(path)
    if lf is None:
        # Fallback to simple read
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        return df
    return lf.df


def resolve_gcp(df: pd.DataFrame, side: str) -> np.ndarray:
    """Return GCP values for *side* normalised to [0, 1]."""
    col = f"{side}_GCP"
    if col not in df.columns:
        return np.zeros(len(df))
    raw: np.ndarray = df[col].to_numpy(dtype=float)
    if len(raw) > 0 and np.nanmax(raw) > 10:
        return raw / 100.0
    elif len(raw) > 0 and np.nanmax(raw) > 1.5:
        return raw / np.nanmax(raw)
    return raw


def detect_heel_strikes(gcp_or_df, side: str = None, sample_rate: float = 111.0) -> np.ndarray:
    """Detect heel-strike indices. Accepts DataFrame (uses Event+GCP) or GCP array.

    For GCP arrays with independent L/R sawtooth patterns (0→peak during stance,
    0 during swing), returns the **start** of each active segment as a heel strike.
    """
    if isinstance(gcp_or_df, pd.DataFrame) and side is not None:
        return _detect_heel_strikes(gcp_or_df, side, sample_rate)

    # Legacy: accept GCP array directly
    gcp = np.asarray(gcp_or_df, dtype=float)
    if len(gcp) == 0:
        return np.array([], dtype=np.int64)

    # Find active segments: contiguous regions where GCP > threshold
    threshold = 0.01
    active = gcp > threshold
    # Rising edge: transition from inactive to active = start of stance
    edges = np.diff(active.astype(np.int8), prepend=0)
    starts = np.where(edges == 1)[0]

    if len(starts) >= 2:
        return starts.astype(np.int64)

    # Fallback to old drop-based detection if active-segment approach fails
    diff = np.diff(gcp, prepend=gcp[0])
    ptp = float(np.ptp(gcp))
    drop_threshold = -max(0.3, ptp * 0.4)
    return np.where(diff < drop_threshold)[0].astype(np.int64)


def normalize_to_gcp(
    signal: np.ndarray,
    hs_indices: np.ndarray,
    gcp: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample signal into 101-point strides and return (mean, std).

    When *gcp* is provided, each stride is trimmed to the active portion
    (GCP > 0.01) starting at each heel-strike index, avoiding the GCP=0
    gap between stance phases of independent L/R sawtooth signals.
    """
    gcp_axis = np.linspace(0, 1, 101)

    if len(hs_indices) < 2:
        return np.zeros(101), np.zeros(101)

    strides: list[np.ndarray] = []
    threshold = 0.01

    for i in range(len(hs_indices)):
        start = int(hs_indices[i])

        if gcp is not None:
            # Find end of the active segment (where GCP drops back to ~0)
            end = start + 1
            while end < len(signal) and end < len(gcp) and gcp[end] > threshold:
                end += 1
        else:
            # No GCP available: fall back to start-to-start span
            if i + 1 >= len(hs_indices):
                break
            end = int(hs_indices[i + 1])

        chunk = signal[start:end]
        if len(chunk) < 2:
            continue
        x_orig = np.linspace(0, 1, len(chunk))
        resampled = np.interp(gcp_axis, x_orig, chunk)
        strides.append(resampled)

    if not strides:
        return np.zeros(101), np.zeros(101)

    mat = np.vstack(strides)
    return mat.mean(axis=0), mat.std(axis=0)


def compute_stats(
    df: pd.DataFrame,
    columns: list[str],
    filename: str,
) -> list[StatsResult]:
    """Compute per-column descriptive statistics."""
    results: list[StatsResult] = []
    for col in columns:
        if col not in df.columns:
            continue
        arr = df[col].to_numpy(dtype=float)
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            continue
        results.append(
            StatsResult(
                column=col,
                file=filename,
                mean=float(np.mean(valid)),
                std=float(np.std(valid)),
                max_val=float(np.max(valid)),
                min_val=float(np.min(valid)),
            )
        )
    return results


def compute_symmetry_index(left: np.ndarray, right: np.ndarray) -> float:
    """Return the symmetry index (%) between left and right signals."""
    ml = float(np.mean(left))
    mr = float(np.mean(right))
    denominator = (ml + mr) / 2.0
    if abs(denominator) < 1e-12:
        return 0.0
    return abs(ml - mr) / abs(denominator) * 100.0


def run_full_analysis(
    filepath: str,
    analyses: list[str] = None,
    window_idx: int = 0,
) -> AnalysisResult:
    """Run full auto_analyzer analysis on a single sync window of a CSV.

    Per the user-confirmed sync contract (CLAUDE.md, 2026-04-25):
      one sync window = [rising, falling) = one trial. Analysis must
      use **only data inside the window** — outside is prep/rest.

    Default `window_idx=0` returns the first trial; pass a different
    index to pick a later trial in the same recording. If the file
    has no Sync column or no detectable windows, the analyzer falls
    back to whole-CSV analysis transparently.
    """
    results = _analyze_file_windows(filepath, analyses)
    if not results:
        # analyze_file_windows always returns ≥1 element by contract,
        # but guard anyway so we never silently return None.
        raise ValueError(f"analyzer produced 0 results for {filepath}")
    if window_idx < 0 or window_idx >= len(results):
        raise IndexError(
            f"window_idx={window_idx} out of range; recording has "
            f"{len(results)} window(s)"
        )
    return results[window_idx]


def run_per_window_analysis(
    filepath: str,
    analyses: list[str] = None,
) -> list[AnalysisResult]:
    """Return one AnalysisResult per sync window in the recording.

    Use this when the caller needs to compare or aggregate trials
    inside a single CSV (e.g. trial-pairing across Robot+Motion at
    the same window index). Single-element list when the file has
    no sync information.
    """
    return _analyze_file_windows(filepath, analyses)


def count_sync_windows(filepath: str) -> int:
    """How many sync windows (trials) are in this CSV.

    Cheap — only loads the CSV and runs the rising/falling edge
    detector, no full analysis. Returns 1 when there's no Sync
    column or no detectable windows (the analyzer treats that as a
    single-trial fallback, so callers can use this count directly).

    Phantom pulses (e.g. file-IO glitches narrower than 0.5 s) are
    filtered out by `find_sync_windows`'s default threshold so this
    count reflects real operator-driven trials only.
    """
    import pandas as pd
    from backend.services.sync_align import find_sync_windows
    try:
        df = pd.read_csv(filepath)
    except Exception:
        return 1
    n = len(find_sync_windows(df))
    return n if n > 0 else 1


def full_analysis_to_stats(result: AnalysisResult) -> list[StatsResult]:
    """Convert AnalysisResult into StatsResult list for LLM insights."""
    stats = []
    for side_name, sr, ft in [
        ('L', result.left_stride, result.left_force_tracking),
        ('R', result.right_stride, result.right_force_tracking),
    ]:
        if sr.n_strides > 0:
            stats.append(StatsResult(
                column=f"{side_name}_StrideTime", file=result.filename,
                mean=sr.stride_time_mean, std=sr.stride_time_std,
                max_val=float(np.max(sr.stride_times)) if len(sr.stride_times) else 0,
                min_val=float(np.min(sr.stride_times)) if len(sr.stride_times) else 0,
            ))
            stats.append(StatsResult(
                column=f"{side_name}_Cadence", file=result.filename,
                mean=sr.cadence, std=0, max_val=sr.cadence, min_val=sr.cadence,
            ))
            stats.append(StatsResult(
                column=f"{side_name}_StancePct", file=result.filename,
                mean=sr.stance_pct_mean, std=sr.stance_pct_std,
                max_val=sr.stance_pct_mean + sr.stance_pct_std,
                min_val=sr.stance_pct_mean - sr.stance_pct_std,
            ))
        if ft.rmse > 0:
            stats.append(StatsResult(
                column=f"{side_name}_ForceRMSE", file=result.filename,
                mean=ft.rmse, std=0, max_val=ft.peak_error, min_val=ft.mae,
            ))
    return stats
