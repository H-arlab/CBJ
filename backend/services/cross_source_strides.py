"""Cross-source per-stride table builder.

For one Trial (Robot + Motion ± Loadcell, all aligned on a single
sync window), produce a flat table with **one row per stride** and
metrics from every source side-by-side. This is the unit downstream
statistics operate on:

    stride_idx, side, t_start_s, t_end_s, stride_time_s,
    # Robot-source per-stride metrics
    cable_force_peak_<side>, cable_force_rmse_<side>,
    cable_force_impulse_<side>, shank_pitch_rom_<side>,
    # Motion-source per-stride metrics  (only when Motion is present)
    grf_fz_peak, grf_fz_impulse, grf_loading_rate, cop_path_length,
    rhip_flex_peak, rhip_rom, rknee_flex_peak, rknee_rom,
    rankle_dorsi_peak, rankle_rom,
    # EMG per-muscle (only when EMG channels detected)
    emg_<muscle>_rms, emg_<muscle>_peak, ...
    # Cross-source flag
    has_motion, has_loadcell

Stride boundaries are detected on the **Robot's L_GCP / R_GCP** in
the aligned grid (not the original CSV) — this is the per-side
ground-truth signal in the H-Walker firmware. Motion-source metrics
are sliced at the same sample indices (the grids share the time
axis after sync alignment).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from backend.services import motion_metrics


# ============================================================
# Stride detection on the aligned grid
# ============================================================

def detect_strides_on_grid(robot_grid: pd.DataFrame,
                            side: str,
                            min_stance_samples: int = 10,
                            ) -> list[tuple[int, int]]:
    """Heel-strike pairs on the **aligned grid**, using the firmware's
    `<side>_GCP` active-segment starts (per HANDOVER-2026-04-17 the
    GCP is the only reliable per-side cue).

    Returns [(start_idx, end_idx), ...] = consecutive HS pairs in the
    grid's sample space. Each tuple defines one stride window.
    """
    gcp_col = f"{side}_GCP"
    if gcp_col not in robot_grid.columns:
        return []
    gcp = robot_grid[gcp_col].to_numpy(dtype=np.float64)
    finite = np.isfinite(gcp)
    if finite.sum() < 20:
        return []
    # Normalize scale — GCP can come as 0..1 or 0..100 or 0..1.48.
    gcp_max = float(np.nanmax(gcp))
    if gcp_max > 10:
        gcp = gcp / 100.0
    elif gcp_max > 1.5:
        gcp = gcp / gcp_max
    active = (gcp > 0.01).astype(np.int8)
    edges = np.diff(active, prepend=0)
    hs = np.where(edges == 1)[0]
    if hs.size < 2:
        return []
    # Debounce: drop a duplicate HS within `min_stance_samples` of
    # the previous one (firmware noise at lift-off / re-strike).
    keep = [hs[0]]
    for h in hs[1:]:
        if h - keep[-1] >= min_stance_samples:
            keep.append(h)
    if len(keep) < 2:
        return []
    return [(keep[i], keep[i + 1]) for i in range(len(keep) - 1)]


# ============================================================
# Per-stride metric extraction (Robot)
# ============================================================

def _robot_stride_metrics(robot_grid: pd.DataFrame,
                           strides: list[tuple[int, int]],
                           side: str,
                           time_col: str = "t_aligned") -> dict[str, list]:
    """Per-stride scalars derived from Robot columns on the aligned grid."""
    sd = side
    des_col = f"{sd}_DesForce_N"
    act_col = f"{sd}_ActForce_N"
    pitch_col = f"{sd}_Pitch"

    n = len(strides)
    out: dict[str, list] = {
        f"cable_force_peak_{sd}": [float("nan")] * n,
        f"cable_force_rmse_{sd}": [float("nan")] * n,
        f"cable_force_impulse_{sd}": [float("nan")] * n,
        f"shank_pitch_rom_{sd}": [float("nan")] * n,
    }
    t = robot_grid[time_col].to_numpy(dtype=np.float64) \
        if time_col in robot_grid.columns else None

    for i, (s, e) in enumerate(strides):
        if act_col in robot_grid.columns:
            a = robot_grid[act_col].to_numpy(dtype=np.float64)[s:e]
            finite = a[np.isfinite(a)]
            if finite.size >= 5:
                out[f"cable_force_peak_{sd}"][i] = float(np.max(finite))
                if t is not None:
                    out[f"cable_force_impulse_{sd}"][i] = float(
                        np.trapezoid(np.where(np.isfinite(a), a, 0.0), t[s:e])
                    )
        if des_col in robot_grid.columns and act_col in robot_grid.columns:
            d = robot_grid[des_col].to_numpy(dtype=np.float64)[s:e]
            a = robot_grid[act_col].to_numpy(dtype=np.float64)[s:e]
            mask = np.isfinite(d) & np.isfinite(a)
            if mask.sum() >= 5:
                err = a[mask] - d[mask]
                out[f"cable_force_rmse_{sd}"][i] = float(
                    np.sqrt(np.mean(err ** 2))
                )
        if pitch_col in robot_grid.columns:
            p = robot_grid[pitch_col].to_numpy(dtype=np.float64)[s:e]
            finite = p[np.isfinite(p)]
            if finite.size >= 5:
                out[f"shank_pitch_rom_{sd}"][i] = float(
                    np.max(finite) - np.min(finite)
                )
    return out


# ============================================================
# Per-stride metric extraction (Motion: GRF + joints + EMG)
# ============================================================

def _motion_stride_metrics(motion_grid: pd.DataFrame,
                            strides: list[tuple[int, int]],
                            time_col: str = "t_aligned") -> dict[str, list]:
    out: dict[str, list] = {}

    # ---- Force plate (try Vicon then V3D conventions) ----
    fp_candidates: list[tuple[str, str, str]] = []
    for prefix in ("FP1", "FP2"):
        if f"{prefix}_Fz" in motion_grid.columns:
            fp_candidates.append((f"{prefix}_Fz",
                                   f"{prefix}_COPx",
                                   f"{prefix}_COPy"))
    for side_word in ("LeftFP", "RightFP", "LFP", "RFP"):
        if f"{side_word}_Force_Z" in motion_grid.columns:
            fp_candidates.append((f"{side_word}_Force_Z",
                                   f"{side_word}_COP_X",
                                   f"{side_word}_COP_Y"))

    for fz_col, copx_col, copy_col in fp_candidates:
        prefix_lower = fz_col.lower().replace("_fz", "")\
            .replace("_force_z", "").replace(".", "_").replace("__", "_")
        out[f"grf_fz_peak.{prefix_lower}"]      = motion_metrics.grf_fz_peak(
            motion_grid, strides, fz_col=fz_col)
        out[f"grf_fz_impulse.{prefix_lower}"]   = motion_metrics.grf_fz_impulse(
            motion_grid, strides, fz_col=fz_col, time_col=time_col)
        out[f"grf_loading_rate.{prefix_lower}"] = motion_metrics.grf_loading_rate(
            motion_grid, strides, fz_col=fz_col, time_col=time_col)
        out[f"cop_path_length.{prefix_lower}"]  = motion_metrics.cop_path_length(
            motion_grid, strides, copx_col=copx_col, copy_col=copy_col)

    # ---- Joint kinematics (V3D-style) ----
    joint_cols = [c for c in motion_grid.columns
                  if any(k in c.lower() for k in ("hipangle", "kneeangle",
                                                    "ankleangle"))]
    for c in joint_cols:
        key = c.lower()
        out[f"joint_peak.{key}"] = motion_metrics.joint_peak_per_stride(
            motion_grid, strides, angle_col=c)
        out[f"joint_rom.{key}"]  = motion_metrics.joint_rom_per_stride(
            motion_grid, strides, angle_col=c)

    moment_cols = [c for c in motion_grid.columns
                   if "moment" in c.lower() and any(j in c.lower()
                       for j in ("hip", "knee", "ankle"))]
    for c in moment_cols:
        key = c.lower()
        out[f"joint_moment_peak.{key}"] = \
            motion_metrics.joint_moment_peak_per_stride(
                motion_grid, strides, moment_col=c)

    # ---- EMG (any column that looks like EMG) ----
    from backend.services.source_kind import _EMG_RE
    emg_cols = [c for c in motion_grid.columns if _EMG_RE.match(c)]
    for c in emg_cols:
        key = c.lower()
        out[f"emg_rms.{key}"]  = motion_metrics.emg_rms_per_stride(
            motion_grid, strides, emg_col=c)
        out[f"emg_peak.{key}"] = motion_metrics.emg_peak_per_stride(
            motion_grid, strides, emg_col=c)

    return out


# ============================================================
# Top-level: build the per-stride DataFrame for one Trial
# ============================================================

@dataclass
class CrossSourceStrideResult:
    side: str
    n_strides: int
    table: pd.DataFrame                  # one row per stride
    has_motion: bool
    has_loadcell: bool
    notes: list[str] = field(default_factory=list)


def build_stride_table(
    robot_grid: pd.DataFrame,
    motion_grid: Optional[pd.DataFrame] = None,
    side: str = "L",
    time_col: str = "t_aligned",
) -> CrossSourceStrideResult:
    """Build the per-stride cross-source table for one Trial.

    Inputs are the **aligned grids** produced by
    `sync_align.align_sources_on_window` — i.e. each source has been
    anchored to its own sync window's rising edge, both share the same
    `t_aligned` axis, and `motion_grid` (if present) lines up sample-
    by-sample with `robot_grid`.

    Output: a DataFrame with one row per stride and every cross-source
    metric as a column. Strides are detected from Robot's
    `<side>_GCP` (per the firmware's per-side ground-truth cue).
    """
    notes: list[str] = []
    strides = detect_strides_on_grid(robot_grid, side)
    if not strides:
        return CrossSourceStrideResult(
            side=side, n_strides=0,
            table=pd.DataFrame(),
            has_motion=motion_grid is not None,
            has_loadcell=False,
            notes=[f"no strides detected on {side}_GCP active mask"],
        )

    cols: dict[str, list] = {
        "stride_idx": list(range(len(strides))),
        "side":       [side] * len(strides),
        "sample_start": [s for s, _ in strides],
        "sample_end":   [e for _, e in strides],
    }

    # Stride times from the aligned grid
    if time_col in robot_grid.columns:
        t = robot_grid[time_col].to_numpy(dtype=np.float64)
        cols["t_start_s"]    = [float(t[s]) for s, _ in strides]
        cols["t_end_s"]      = [float(t[e]) for _, e in strides]
        cols["stride_time_s"] = [float(t[e] - t[s]) for s, e in strides]

    # Robot metrics
    cols.update(_robot_stride_metrics(robot_grid, strides, side, time_col))

    # Motion metrics (only when Motion grid is present)
    has_motion = motion_grid is not None and len(motion_grid) > 0
    if has_motion:
        # Only use motion samples that line up with robot samples.
        # `align_sources_on_window` guarantees identical t_aligned, so
        # sample indices are interchangeable.
        n_align = min(len(robot_grid), len(motion_grid))
        clipped_strides = [(s, min(e, n_align)) for s, e in strides
                           if s < n_align]
        cols.update(_motion_stride_metrics(
            motion_grid.iloc[:n_align], clipped_strides, time_col))
    else:
        notes.append("motion grid not provided — GRF/joint/EMG columns omitted")

    # Tabulate
    table = pd.DataFrame(cols)
    return CrossSourceStrideResult(
        side=side,
        n_strides=len(strides),
        table=table,
        has_motion=has_motion,
        has_loadcell=False,    # set by caller when Loadcell is attached
        notes=notes,
    )
