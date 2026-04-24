"""
Compute metrics — Phase 2A real implementation.

Replaces the mock `computeMetrics.ts` tables with real values derived from
the H-Walker CSV + auto_analyzer `AnalysisResult`.

Each metric returns a payload shape-compatible with the frontend
`ComputeMetric` interface:

    {
        "label":   str,
        "cols":    list[str],
        "rows":    list[list[str]],
        "summary": { "mean": list[str] },
        "meta":    { "n_strides": int, ... }   # extra, tolerated
    }

`n_max_rows` caps table length (default 30). If more strides exist, an
ellipsis row "…" is inserted and the last stride appended.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from tools.auto_analyzer.analyzer import AnalysisResult


# ============================================================
# Helpers
# ============================================================


def _fmt_mean_std(x: np.ndarray, digits: int = 2) -> str:
    arr = x[np.isfinite(x)]
    if len(arr) == 0:
        return "—"
    return f"{np.mean(arr):.{digits}f} ± {np.std(arr, ddof=1) if len(arr) > 1 else 0:.{digits}f}"


def _truncate_rows(rows: list[list[str]], n_max: int = 30) -> list[list[str]]:
    """Keep the first n_max-2 rows, an ellipsis row, then the last row."""
    if len(rows) <= n_max:
        return rows
    width = len(rows[0])
    out = rows[: n_max - 2]
    out.append(["…"] * width)
    out.append(rows[-1])
    return out


def _peak_per_stride(df: pd.DataFrame, col: str,
                     hs: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if col not in df.columns:
        return np.array([])
    arr = df[col].to_numpy(dtype=float)
    out = []
    for i in range(len(valid)):
        if not valid[i] or i + 1 >= len(hs):
            continue
        chunk = arr[hs[i]: hs[i + 1]]
        chunk = chunk[np.isfinite(chunk)]
        out.append(float(np.max(chunk)) if len(chunk) else float("nan"))
    return np.array(out)


def _impulse_per_stride(df: pd.DataFrame, col: str,
                        hs: np.ndarray, valid: np.ndarray, fs: float) -> np.ndarray:
    if col not in df.columns or fs <= 0:
        return np.array([])
    arr = df[col].to_numpy(dtype=float)
    dt = 1.0 / fs
    out = []
    for i in range(len(valid)):
        if not valid[i] or i + 1 >= len(hs):
            continue
        chunk = arr[hs[i]: hs[i + 1]]
        chunk = chunk[np.isfinite(chunk)]
        # Simple rectangular integration; only positive force contributes
        chunk = np.clip(chunk, 0.0, None)
        out.append(float(np.sum(chunk) * dt))
    return np.array(out)


def _loading_rate_per_stride(df: pd.DataFrame, col: str,
                             hs: np.ndarray, valid: np.ndarray,
                             fs: float, window_ms: float = 50.0,
                             body_weight_n: float = 700.0) -> np.ndarray:
    """Initial loading rate (BW/s). Slope over first `window_ms` of the stride
    after heel-strike. BW defaults to 700 N if not given (fallback only)."""
    if col not in df.columns or fs <= 0:
        return np.array([])
    arr = df[col].to_numpy(dtype=float)
    window = max(2, int(round(window_ms / 1000.0 * fs)))
    out = []
    for i in range(len(valid)):
        if not valid[i] or i + 1 >= len(hs):
            continue
        s = hs[i]
        e = min(s + window, hs[i + 1])
        chunk = arr[s:e]
        chunk = chunk[np.isfinite(chunk)]
        if len(chunk) < 2:
            continue
        peak = float(np.max(chunk))
        initial = float(chunk[0])
        dt = (len(chunk) - 1) / fs
        if dt <= 0:
            continue
        bw_per_s = (peak - initial) / body_weight_n / dt
        out.append(bw_per_s)
    return np.array(out)


def _rom_per_stride(df: pd.DataFrame, col: str,
                    hs: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if col not in df.columns:
        return np.array([])
    arr = df[col].to_numpy(dtype=float)
    out = []
    for i in range(len(valid)):
        if not valid[i] or i + 1 >= len(hs):
            continue
        chunk = arr[hs[i]: hs[i + 1]]
        chunk = chunk[np.isfinite(chunk)]
        if len(chunk) < 2:
            continue
        out.append(float(np.max(chunk) - np.min(chunk)))
    return np.array(out)


def _asym_idx(l: float, r: float) -> float:
    if not (np.isfinite(l) and np.isfinite(r)):
        return float("nan")
    denom = (l + r) / 2.0
    if abs(denom) < 1e-12:
        return 0.0
    return abs(l - r) / abs(denom) * 100.0


def resample_column(df: pd.DataFrame, col: str,
                    hs: np.ndarray, valid: np.ndarray,
                    n_points: int = 101) -> tuple[np.ndarray, np.ndarray] | None:
    """Resample a generic column into per-stride profiles (n_strides x n_points).

    Used for any kinematic / kinetic signal that the analyzer doesn't
    already produce profiles for (joint angles, acceleration, etc.).

    Returns (mean(axis=0), std(axis=0)) or None if no valid strides.
    """
    if col not in df.columns or len(hs) < 2:
        return None
    arr = df[col].to_numpy(dtype=float)
    profiles: list[np.ndarray] = []
    n_strides = len(valid) if valid is not None else len(hs) - 1
    for i in range(n_strides):
        if valid is not None and not valid[i]:
            continue
        if i + 1 >= len(hs):
            break
        chunk = arr[hs[i]: hs[i + 1]]
        valid_chunk = chunk[np.isfinite(chunk)]
        if len(valid_chunk) < 10:
            continue
        # NaN interpolation
        if np.any(np.isnan(chunk)):
            nan_mask = np.isnan(chunk)
            valid_idx = np.where(~nan_mask)[0]
            if len(valid_idx) >= 2:
                chunk = chunk.copy()
                chunk[nan_mask] = np.interp(
                    np.where(nan_mask)[0], valid_idx, chunk[valid_idx])
            else:
                continue
        x_orig = np.linspace(0, 100, len(chunk))
        resampled = np.interp(np.linspace(0, 100, n_points), x_orig, chunk)
        profiles.append(resampled)
    if not profiles:
        return None
    mat = np.vstack(profiles)
    return mat.mean(axis=0), mat.std(axis=0)


# ============================================================
# Metric computations
# ============================================================


def per_stride(df: pd.DataFrame, res: AnalysisResult,
               n_max_rows: int = 30) -> dict[str, Any]:
    """Per-stride table: stride#, peak_L, peak_R, stride_T, asym_idx."""
    ls, rs = res.left_stride, res.right_stride
    # Use the *left* hs indices as the timeline reference (matching on index is
    # approximate — L/R strides aren't perfectly aligned, but this matches the
    # mockup's stride-index layout).
    n = min(
        len(ls.stride_times),
        len(rs.stride_times),
        ls.valid_mask.sum() if len(ls.valid_mask) else 0,
        rs.valid_mask.sum() if len(rs.valid_mask) else 0,
    )

    peaks_l = _peak_per_stride(df, "L_ActForce_N", ls.hs_indices, ls.valid_mask)
    peaks_r = _peak_per_stride(df, "R_ActForce_N", rs.hs_indices, rs.valid_mask)

    n = min(n, len(peaks_l), len(peaks_r), len(ls.stride_times), len(rs.stride_times))

    rows = []
    asym_vals = []
    for i in range(n):
        asym = _asym_idx(peaks_l[i], peaks_r[i])
        asym_vals.append(asym)
        rows.append([
            str(i + 1),
            f"{peaks_l[i]:.1f}",
            f"{peaks_r[i]:.1f}",
            f"{ls.stride_times[i]:.2f}",
            f"{asym:.1f}",
        ])

    summary_mean = [
        _fmt_mean_std(peaks_l[:n], digits=1),
        _fmt_mean_std(peaks_r[:n], digits=1),
        _fmt_mean_std(ls.stride_times[:n], digits=2),
        _fmt_mean_std(np.array(asym_vals), digits=1),
    ]

    return {
        "label": "Per-stride metrics",
        "cols": ["stride_#", "peak_L (N)", "peak_R (N)", "stride_T (s)", "asym_idx (%)"],
        "rows": _truncate_rows(rows, n_max_rows),
        "summary": {"mean": summary_mean},
        "meta": {"n_strides": int(n)},
    }


def impulse(df: pd.DataFrame, res: AnalysisResult,
            n_max_rows: int = 30) -> dict[str, Any]:
    ls, rs = res.left_stride, res.right_stride
    fs = res.sample_rate
    imp_l = _impulse_per_stride(df, "L_ActForce_N", ls.hs_indices, ls.valid_mask, fs)
    imp_r = _impulse_per_stride(df, "R_ActForce_N", rs.hs_indices, rs.valid_mask, fs)
    n = min(len(imp_l), len(imp_r))

    rows = []
    delta_vals = []
    for i in range(n):
        d = _asym_idx(imp_l[i], imp_r[i])
        delta_vals.append(d)
        rows.append([str(i + 1), f"{imp_l[i]:.1f}", f"{imp_r[i]:.1f}", f"{d:.1f}"])

    return {
        "label": "Impulse (N·s)",
        "cols": ["stride_#", "L impulse", "R impulse", "Δ (%)"],
        "rows": _truncate_rows(rows, n_max_rows),
        "summary": {"mean": [
            _fmt_mean_std(imp_l, digits=1),
            _fmt_mean_std(imp_r, digits=1),
            _fmt_mean_std(np.array(delta_vals), digits=1),
        ]},
        "meta": {"n_strides": int(n)},
    }


def loading_rate(df: pd.DataFrame, res: AnalysisResult,
                 body_weight_n: float = 700.0,
                 n_max_rows: int = 30) -> dict[str, Any]:
    ls, rs = res.left_stride, res.right_stride
    fs = res.sample_rate
    lr_l = _loading_rate_per_stride(df, "L_ActForce_N", ls.hs_indices, ls.valid_mask,
                                    fs, body_weight_n=body_weight_n)
    lr_r = _loading_rate_per_stride(df, "R_ActForce_N", rs.hs_indices, rs.valid_mask,
                                    fs, body_weight_n=body_weight_n)
    n = min(len(lr_l), len(lr_r))

    rows, deltas = [], []
    for i in range(n):
        d = lr_l[i] - lr_r[i]
        deltas.append(d)
        rows.append([str(i + 1), f"{lr_l[i]:.1f}", f"{lr_r[i]:.1f}", f"{d:+.1f}"])

    return {
        "label": f"Loading rate (BW/s, 0–50ms, BW≈{body_weight_n:.0f}N)",
        "cols": ["stride_#", "L rate", "R rate", "Δ"],
        "rows": _truncate_rows(rows, n_max_rows),
        "summary": {"mean": [
            _fmt_mean_std(lr_l, digits=1),
            _fmt_mean_std(lr_r, digits=1),
            _fmt_mean_std(np.array(deltas), digits=1),
        ]},
        "meta": {"n_strides": int(n), "body_weight_n": body_weight_n},
    }


def rom(df: pd.DataFrame, res: AnalysisResult,
        n_max_rows: int = 30) -> dict[str, Any]:
    """Per-stride range of motion for the pitch columns actually present.

    L_Pitch / R_Pitch are the *left* and *right* pitch channels (whatever
    the sensor is mounted on — shank, thigh, or foot is caller-determined).
    We intentionally do not label them as "shank" / "thigh" — that conflates
    bilateral comparison (L vs R) with joint comparison (shank vs thigh).
    """
    ls = res.left_stride
    l_col = "L_Pitch" if "L_Pitch" in df.columns else None
    r_col = "R_Pitch" if "R_Pitch" in df.columns else None

    l_rom = _rom_per_stride(df, l_col, ls.hs_indices, ls.valid_mask) if l_col else np.array([])
    r_rom = _rom_per_stride(df, r_col, ls.hs_indices, ls.valid_mask) if r_col else np.array([])
    n = min(len(l_rom), len(r_rom)) if (len(l_rom) and len(r_rom)) else max(len(l_rom), len(r_rom))

    cols = ["stride"]
    if len(l_rom):
        cols.append(f"{l_col} ROM (°)")
    if len(r_rom):
        cols.append(f"{r_col} ROM (°)")

    rows = []
    for i in range(n):
        row = [str(i + 1)]
        if len(l_rom):
            row.append(f"{l_rom[i]:.1f}" if i < len(l_rom) else "—")
        if len(r_rom):
            row.append(f"{r_rom[i]:.1f}" if i < len(r_rom) else "—")
        rows.append(row)

    summary = []
    if len(l_rom):
        summary.append(_fmt_mean_std(l_rom, digits=1))
    if len(r_rom):
        summary.append(_fmt_mean_std(r_rom, digits=1))

    return {
        "label": "ROM per stride",
        "cols": cols,
        "rows": _truncate_rows(rows, n_max_rows),
        "summary": {"mean": summary},
        "meta": {"n_strides": int(n), "l_col": l_col, "r_col": r_col},
    }


def cadence(df: pd.DataFrame, res: AnalysisResult,
            window_s: float = 4.0) -> dict[str, Any]:
    """Cadence per rolling window (average of L + R steps)."""
    fs = res.sample_rate
    total_dur = res.duration_s
    ls, rs = res.left_stride, res.right_stride

    n_windows = int(total_dur // window_s)
    rows = []
    cadences = []
    for w in range(n_windows):
        t0, t1 = w * window_s, (w + 1) * window_s
        i0, i1 = int(t0 * fs), int(t1 * fs)
        steps = 0
        if len(ls.hs_indices):
            steps += int(np.sum((ls.hs_indices >= i0) & (ls.hs_indices < i1)))
        if len(rs.hs_indices):
            steps += int(np.sum((rs.hs_indices >= i0) & (rs.hs_indices < i1)))
        spm = steps / window_s * 60.0
        cadences.append(spm)
        rows.append([f"{t0:.0f}-{t1:.0f}s", f"{spm:.0f}"])

    return {
        "label": "Cadence",
        "cols": ["window", "spm"],
        "rows": rows or [["0-0s", f"{(ls.cadence + rs.cadence) / 2:.0f}"]],
        "summary": {"mean": [_fmt_mean_std(np.array(cadences) if cadences else
                                            np.array([(ls.cadence + rs.cadence) / 2]),
                                            digits=0)]},
        "meta": {"window_s": window_s, "n_windows": n_windows},
    }


def target_dev(df: pd.DataFrame, res: AnalysisResult,
               n_max_rows: int = 30) -> dict[str, Any]:
    """Force tracking deviation per stride."""
    lft = res.left_force_tracking
    rft = res.right_force_tracking
    rmses_l = lft.rmse_per_stride
    rmses_r = rft.rmse_per_stride
    n = min(len(rmses_l), len(rmses_r)) if len(rmses_l) and len(rmses_r) else max(len(rmses_l), len(rmses_r))

    rows = []
    for i in range(n):
        l_val = float(rmses_l[i]) if i < len(rmses_l) else float("nan")
        r_val = float(rmses_r[i]) if i < len(rmses_r) else float("nan")
        pair = [v for v in (l_val, r_val) if np.isfinite(v)]
        rmse_avg = float(np.mean(pair)) if pair else float("nan")
        rows.append([
            f"S{i + 1}",
            f"{rmse_avg:.2f}" if np.isfinite(rmse_avg) else "—",
            "—",
        ])

    return {
        "label": "Target deviation (Des vs Act force)",
        "cols": ["stride", "RMSE (N)", "peak Δ (%)"],
        "rows": _truncate_rows(rows, n_max_rows),
        "summary": {"mean": [
            _fmt_mean_std(np.concatenate([rmses_l, rmses_r]) if len(rmses_l) or len(rmses_r)
                          else np.array([0.0]), digits=2),
            "improving" if len(rmses_l) > 1 and rmses_l[-1] < rmses_l[0] else "stable",
        ]},
        "meta": {"n_strides": int(n), "global_rmse_L": lft.rmse, "global_rmse_R": rft.rmse},
    }


def stride_length(df: pd.DataFrame, res: AnalysisResult,
                  n_max_rows: int = 30,
                  belt_speed_ms: Optional[float] = None,
                  is_treadmill: bool = False) -> dict[str, Any]:
    """Per-stride length (m).

    Two integration paths, chosen automatically:

      (1) ZUPT over-ground — integrates IMU global velocity between heel
          strikes. Valid when the robot physically travels through space.

      (2) Treadmill belt-speed × stride-time — valid for stationary walker
          on a moving belt. Triggered when (a) user flags the dataset as
          treadmill via PATCH /api/datasets/{id}/meta {is_treadmill:true,
          belt_speed_ms:X}, OR (b) the ZUPT median is implausibly small
          (< 0.15 m) and belt_speed_ms is provided as a fallback.

    The output `meta` carries diagnostic fields the UI surfaces as warnings.
    """
    ls, rs = res.left_stride, res.right_stride
    L_zupt = ls.stride_lengths[np.isfinite(ls.stride_lengths)]
    R_zupt = rs.stride_lengths[np.isfinite(rs.stride_lengths)]
    L_times = ls.stride_times
    R_times = rs.stride_times

    warnings: list[str] = []
    mode = "zupt"
    label = "Stride length (m, ZUPT)"

    median_zupt = None
    if len(L_zupt) + len(R_zupt) >= 3:
        median_zupt = float(np.median(np.concatenate([L_zupt, R_zupt])))

    # Decide mode
    if is_treadmill or (belt_speed_ms and median_zupt is not None and median_zupt < 0.15):
        if not belt_speed_ms:
            warnings.append("Dataset flagged treadmill but no belt_speed_ms set — "
                            "falling back to ZUPT (will likely be wrong).")
        else:
            mode = "treadmill"
            label = f"Stride length (m, belt {belt_speed_ms:.2f} m/s × stride_time)"
            # stride_length = stride_time × belt_speed (simplest + robust)
            L = np.asarray(L_times, dtype=float) * float(belt_speed_ms)
            R = np.asarray(R_times, dtype=float) * float(belt_speed_ms)
            L_zupt = L  # overwrite for output
            R_zupt = R

    L, R = L_zupt, R_zupt
    n = min(len(L), len(R)) if len(L) and len(R) else max(len(L), len(R))

    # If we have literally nothing to report (ZUPT empty AND no treadmill
    # belt speed configured), refuse to return zeros / fabricated means.
    # Tells the user WHY — either the CSV has no L_Ax/L_Ay IMU velocity
    # columns (so ZUPT can't run) or treadmill mode wasn't flagged.
    if n == 0 and not (is_treadmill and belt_speed_ms):
        has_ax = any(f"{s}_Ax" in df.columns and f"{s}_Ay" in df.columns
                      for s in ("L", "R"))
        if not has_ax and not belt_speed_ms:
            reason = (
                "Stride length cannot be computed: CSV has no L_Ax/L_Ay IMU "
                "velocity columns (ZUPT needs global velocity), and the dataset "
                "is not flagged as treadmill (TREADMILL toggle + belt_speed_ms)."
            )
        elif not has_ax:
            reason = (
                "Stride length needs L_Ax/L_Ay (IMU global velocity) for ZUPT, "
                "or treadmill mode enabled. Neither condition met."
            )
        else:
            reason = (
                "ZUPT produced no valid stride lengths (likely missing heel-strikes "
                "or all values rejected by sanity filters). Check the debug_ts "
                "figure to verify the signal, or flag as treadmill with belt_speed_ms."
            )
        return {
            "label": "Stride length (unavailable)",
            "cols": ["stride_#", "L (m)", "R (m)", "asym (%)"],
            "rows": [],
            "summary": {"mean": ["—", "—", "—"]},
            "meta": {
                "n_strides": 0,
                "mode": "unavailable",
                "belt_speed_ms": belt_speed_ms,
                "is_treadmill": bool(is_treadmill),
                "warnings": [reason],
                "unavailable_reason": reason,
            },
        }

    # Physiological warnings
    if mode == "zupt" and median_zupt is not None:
        if median_zupt < 0.15:
            warnings.append(
                f"Median ZUPT stride = {median_zupt:.3f} m — likely treadmill "
                f"data (ZUPT assumes over-ground travel). Open Settings · "
                f"Dataset and set belt_speed_ms."
            )
        elif median_zupt > 2.5:
            warnings.append(
                f"Median stride = {median_zupt:.3f} m — verify IMU frame (L_Ax/L_Ay "
                f"must carry global velocity, m/s)."
            )
        # CV check
        if len(L) >= 3:
            cv_l = float(np.std(L) / np.mean(L)) if np.mean(L) > 0 else 1.0
            if cv_l > 0.20:
                warnings.append(
                    f"L stride-length CV = {cv_l*100:.0f}% (normal ≤ 5%). "
                    f"Check Event column for chatter or consider treadmill mode."
                )

    rows = []
    for i in range(n):
        l_val = f"{L[i]:.3f}" if i < len(L) else "—"
        r_val = f"{R[i]:.3f}" if i < len(R) else "—"
        asym = (
            f"{_asym_idx(L[i], R[i]):.1f}"
            if i < len(L) and i < len(R) else "—"
        )
        rows.append([str(i + 1), l_val, r_val, asym])

    return {
        "label": label,
        "cols": ["stride_#", "L (m)", "R (m)", "asym (%)"],
        "rows": _truncate_rows(rows, n_max_rows),
        "summary": {"mean": [
            _fmt_mean_std(L, digits=3),
            _fmt_mean_std(R, digits=3),
            "—",
        ]},
        "meta": {
            "n_strides": int(n),
            "mode": mode,
            "belt_speed_ms": belt_speed_ms,
            "warnings": warnings,
        },
    }


def stance_time(df: pd.DataFrame, res: AnalysisResult,
                n_max_rows: int = 30) -> dict[str, Any]:
    """Per-stride stance-phase duration (s) = stride_time × stance%."""
    ls, rs = res.left_stride, res.right_stride
    n = min(len(ls.stride_times), len(rs.stride_times))

    l_st = ls.stride_times[:n] * (ls.stance_pct_mean / 100.0) if ls.stance_pct_mean else np.array([])
    r_st = rs.stride_times[:n] * (rs.stance_pct_mean / 100.0) if rs.stance_pct_mean else np.array([])

    rows = []
    for i in range(n):
        l_val = f"{l_st[i]:.3f}" if i < len(l_st) else "—"
        r_val = f"{r_st[i]:.3f}" if i < len(r_st) else "—"
        rows.append([str(i + 1), l_val, r_val])

    return {
        "label": "Stance time (s)",
        "cols": ["stride_#", "L stance (s)", "R stance (s)"],
        "rows": _truncate_rows(rows, n_max_rows),
        "summary": {"mean": [
            _fmt_mean_std(l_st, digits=3),
            _fmt_mean_std(r_st, digits=3),
        ]},
        "meta": {
            "n_strides": int(n),
            "L_stance_pct": ls.stance_pct_mean,
            "R_stance_pct": rs.stance_pct_mean,
        },
    }


def swing_time(df: pd.DataFrame, res: AnalysisResult,
               n_max_rows: int = 30) -> dict[str, Any]:
    """Per-stride swing-phase duration (s) = stride_time × swing%."""
    ls, rs = res.left_stride, res.right_stride
    n = min(len(ls.stride_times), len(rs.stride_times))

    l_sw = ls.stride_times[:n] * (ls.swing_pct_mean / 100.0) if ls.swing_pct_mean else np.array([])
    r_sw = rs.stride_times[:n] * (rs.swing_pct_mean / 100.0) if rs.swing_pct_mean else np.array([])

    rows = []
    for i in range(n):
        l_val = f"{l_sw[i]:.3f}" if i < len(l_sw) else "—"
        r_val = f"{r_sw[i]:.3f}" if i < len(r_sw) else "—"
        rows.append([str(i + 1), l_val, r_val])

    return {
        "label": "Swing time (s)",
        "cols": ["stride_#", "L swing (s)", "R swing (s)"],
        "rows": _truncate_rows(rows, n_max_rows),
        "summary": {"mean": [
            _fmt_mean_std(l_sw, digits=3),
            _fmt_mean_std(r_sw, digits=3),
        ]},
        "meta": {"n_strides": int(n)},
    }


def fatigue_index(df: pd.DataFrame, res: AnalysisResult,
                  **_kwargs) -> dict[str, Any]:
    """Fatigue = % change in stride time between first and last 10% of strides."""
    rows = [
        ["L", f"{res.left_fatigue:+.2f}%",
         "increasing (slower)" if res.left_fatigue > 2 else
         "stable" if abs(res.left_fatigue) <= 2 else
         "decreasing (faster)"],
        ["R", f"{res.right_fatigue:+.2f}%",
         "increasing (slower)" if res.right_fatigue > 2 else
         "stable" if abs(res.right_fatigue) <= 2 else
         "decreasing (faster)"],
    ]
    return {
        "label": "Fatigue index (stride time: first 10% vs last 10%)",
        "cols": ["side", "Δ%", "interpretation"],
        "rows": rows,
        "summary": {"mean": [
            f"{res.left_fatigue:+.2f}% / {res.right_fatigue:+.2f}%",
            "—",
        ]},
        "meta": {
            "L_fatigue_pct": res.left_fatigue,
            "R_fatigue_pct": res.right_fatigue,
        },
    }


def symmetry_summary(df: pd.DataFrame, res: AnalysisResult,
                     **_kwargs) -> dict[str, Any]:
    """Six-axis symmetry summary (0% = perfect, higher = more asymmetric)."""
    rows = [
        ["stride time",   f"{res.stride_time_symmetry:.2f}"],
        ["stride length", f"{res.stride_length_symmetry:.2f}" if res.stride_length_symmetry >= 0 else "—"],
        ["stance %",      f"{res.stance_symmetry:.2f}" if res.stance_symmetry >= 0 else "—"],
        ["force RMSE",    f"{res.force_symmetry:.2f}" if res.force_symmetry > 0 else "—"],
    ]
    # Peak-force asymmetry (derived)
    lft, rft = res.left_force_tracking, res.right_force_tracking
    lfp, rfp = res.left_force_profile, res.right_force_profile
    if lfp.mean is not None and rfp.mean is not None:
        peak_l = float(np.max(lfp.mean))
        peak_r = float(np.max(rfp.mean))
        rows.append(["peak GRF", f"{_asym_idx(peak_l, peak_r):.2f}"])

    values = [float(r[1].replace("%", "")) for r in rows if r[1] != "—"]
    return {
        "label": "Symmetry summary (% asymmetry · 0 = perfect)",
        "cols": ["metric", "Δ (%)"],
        "rows": rows,
        "summary": {"mean": [
            f"{np.mean(values):.2f}" if values else "—",
            "",
        ]},
        "meta": {
            "avg_asymmetry": float(np.mean(values)) if values else 0.0,
            "max_asymmetry_metric": rows[int(np.argmax([
                float(r[1].replace("%", "")) if r[1] != "—" else -1 for r in rows
            ]))][0] if values else None,
        },
    }


# ============================================================
# Dispatcher
# ============================================================


METRIC_REGISTRY = {
    "per_stride":        per_stride,
    "impulse":           impulse,
    "loading_rate":      loading_rate,
    "rom":               rom,
    "cadence":           cadence,
    "target_dev":        target_dev,
    # Phase 0: motion-data metrics
    "stride_length":     stride_length,
    "stance_time":       stance_time,
    "swing_time":        swing_time,
    "fatigue_index":     fatigue_index,
    "symmetry_summary":  symmetry_summary,
}


def compute(metric: str, df: pd.DataFrame, res: AnalysisResult,
            **kwargs) -> dict[str, Any]:
    if metric not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric '{metric}'. Known: {sorted(METRIC_REGISTRY.keys())}")
    return METRIC_REGISTRY[metric](df, res, **kwargs)
