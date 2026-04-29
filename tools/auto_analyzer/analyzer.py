"""
H-Walker Auto Analyzer — Core analysis engine.
Computes gait statistics, force tracking metrics, stride length (ZUPT), symmetry, fatigue.
"""

import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

# Add project root so we can import graph_analyzer
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tools.graph_analyzer.data_manager import DataManager


# Column groups for auto-detection
COLUMN_GROUPS = {
    'force_left': ['L_DesForce_N', 'L_ActForce_N'],
    'force_right': ['R_DesForce_N', 'R_ActForce_N'],
    'vel_left': ['L_Ax', 'L_Ay', 'L_Az'],       # EBIMU soa5: Global Velocity (m/s)
    'vel_right': ['R_Ax', 'R_Ay', 'R_Az'],
    'orient_left': ['L_Roll', 'L_Pitch', 'L_Yaw'],
    'orient_right': ['R_Roll', 'R_Pitch', 'R_Yaw'],
    'disp_left': ['L_Dx', 'L_Dy', 'L_Dz'],
    'disp_right': ['R_Dx', 'R_Dy', 'R_Dz'],
    'gait_events': ['L_Event', 'R_Event'],
    'gait_phase': ['L_Phase', 'R_Phase'],
    'gcp': ['L_GCP', 'R_GCP'],
}


@dataclass
class ForceTrackingResult:
    """Force tracking error metrics for one side."""
    rmse: float = 0.0
    mae: float = 0.0
    peak_error: float = 0.0
    rmse_per_stride: np.ndarray = field(default_factory=lambda: np.array([]))
    mae_per_stride: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class StrideResult:
    """Stride-level results for one side."""
    stride_times_raw: np.ndarray = field(default_factory=lambda: np.array([]))
    stride_times: np.ndarray = field(default_factory=lambda: np.array([]))
    stride_lengths: np.ndarray = field(default_factory=lambda: np.array([]))
    stride_time_mean: float = 0.0
    stride_time_std: float = 0.0
    stride_time_cv: float = 0.0
    stride_length_mean: float = 0.0
    stride_length_std: float = 0.0
    cadence: float = 0.0
    stance_pct_mean: float = 0.0
    stance_pct_std: float = 0.0
    swing_pct_mean: float = 0.0
    swing_pct_std: float = 0.0
    n_strides: int = 0
    hs_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    valid_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))


@dataclass
class ForceProfileResult:
    """Resampled force profiles (101 points per stride, 0-100% GCP)."""
    individual: Optional[np.ndarray] = None   # (n_strides, 101)
    mean: Optional[np.ndarray] = None         # (101,)
    std: Optional[np.ndarray] = None          # (101,)
    des_individual: Optional[np.ndarray] = None
    des_mean: Optional[np.ndarray] = None
    des_std: Optional[np.ndarray] = None


@dataclass
class AnalysisResult:
    """Complete analysis result for one CSV file."""
    filename: str = ""
    filepath: str = ""
    n_samples: int = 0
    duration_s: float = 0.0
    sample_rate: float = 111.0

    # Per-side gait results
    left_stride: StrideResult = field(default_factory=StrideResult)
    right_stride: StrideResult = field(default_factory=StrideResult)

    # Force tracking
    left_force_tracking: ForceTrackingResult = field(default_factory=ForceTrackingResult)
    right_force_tracking: ForceTrackingResult = field(default_factory=ForceTrackingResult)

    # Force profiles (GCP-normalized)
    left_force_profile: ForceProfileResult = field(default_factory=ForceProfileResult)
    right_force_profile: ForceProfileResult = field(default_factory=ForceProfileResult)

    # Symmetry indices (%)
    stride_time_symmetry: float = 0.0
    stride_length_symmetry: float = 0.0
    force_symmetry: float = 0.0
    stance_symmetry: float = 0.0

    # Fatigue indices (% change, first 10% vs last 10%)
    left_fatigue: float = 0.0
    right_fatigue: float = 0.0

    # Sync window provenance (set only by analyze_file_windows)
    sync_window_idx: int | None = None
    sync_window_t_rise_s: float | None = None
    sync_window_t_fall_s: float | None = None


def _has_columns(df: pd.DataFrame, cols: list[str]) -> bool:
    """Check if all columns exist in the DataFrame."""
    return all(c in df.columns for c in cols)


def _symmetry_index(left_val: float, right_val: float) -> float:
    """Compute symmetry index: |L-R| / ((L+R)/2) * 100. 0 = perfect symmetry.
    Returns -1 if one side has no data (cannot compute)."""
    if left_val <= 0 or right_val <= 0:
        return -1.0   # Cannot compute symmetry with missing side
    total = left_val + right_val
    return abs(left_val - right_val) / (total / 2) * 100


def _fatigue_index(values: np.ndarray, pct: float = 0.1) -> float:
    """Compare first pct vs last pct of values. Returns % change."""
    n = len(values)
    if n < 10:
        return 0.0
    k = max(2, int(n * pct))
    first = np.mean(values[:k])
    last = np.mean(values[-k:])
    if abs(first) < 1e-12:
        return 0.0
    return (last - first) / abs(first) * 100


def _detect_heel_strikes(df: pd.DataFrame, side: str, sample_rate: float) -> np.ndarray:
    """Heel-strike indices for one side, in samples.

    H-Walker firmware emits two per-side gait signals: `{side}_Event`
    and `{side}_GCP`. Field experience (and docs/HANDOVER-2026-04-17.md
    line 100) shows the **GCP sawtooth is the trustworthy per-side
    cue** — it ramps 0 → 1+ exactly during that side's stance phase
    and stays flat during swing. The Event column, in contrast, varies
    by firmware build: on some loadcell builds it pulses on every
    detected heel strike of either foot, which makes consecutive
    rising edges measure step time, not stride time, and mangles the
    downstream cadence + stance % calculations.

    Order of preference (changed 2026-04-25):
      1. `{side}_GCP` rising edges (active segment starts) — primary.
      2. `{side}_Event` rising edges — fallback when GCP is missing
         or flatlined; we also drop pairs of edges < 0.4 s apart so a
         step-pulsing Event column doesn't collapse into half-strides.
    """
    event_col = f'{side}_Event'
    gcp_col = f'{side}_GCP'

    # ---------- Primary: GCP active-segment starts ----------
    if gcp_col in df.columns:
        gcp = df[gcp_col].to_numpy(dtype=np.float64)
        finite = np.isfinite(gcp)
        if finite.sum() >= 10:
            gcp_range = float(np.ptp(gcp[finite]))
            if gcp_range >= 0.3:
                # Normalize so threshold logic doesn't depend on the
                # raw firmware scale (some builds emit 0..1, others 0..100).
                gcp_max = float(np.nanmax(gcp))
                if gcp_max > 10:
                    gcp = gcp / 100.0
                elif gcp_max > 1.5:
                    gcp = gcp / gcp_max
                active = (gcp > 0.01).astype(np.int8)
                edges = np.diff(active, prepend=0)
                hs_idx = np.where(edges == 1)[0]
                # Refractory: drop a duplicate start within 0.3 s of the
                # previous one (debounce GCP noise at lift-off / re-strike).
                if hs_idx.size > 1:
                    keep = np.concatenate([[True],
                                           np.diff(hs_idx) > max(20, int(sample_rate * 0.3))])
                    hs_idx = hs_idx[keep]
                if hs_idx.size >= 2:
                    return hs_idx

    # ---------- Fallback: Event rising edges ----------
    if event_col in df.columns:
        events = df[event_col].to_numpy(dtype=np.float64)
        raw = np.where(np.diff(events) > 0.5)[0] + 1
        if raw.size >= 2:
            # If consecutive events are tighter than a physiologically
            # plausible stride (~0.7 s for slow walking) we are almost
            # certainly looking at a per-step Event signal, not
            # per-stride. Keep only every other edge.
            min_stride_samples = max(int(sample_rate * 0.7), 10)
            gaps = np.diff(raw)
            if np.median(gaps) < min_stride_samples:
                raw = raw[::2]
            return raw

    return np.array([], dtype=int)


def _filter_strides_iqr(stride_times: np.ndarray, multiplier: float = 2.0
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Filter outlier strides using IQR method. Returns (filtered_times, valid_mask)."""
    if len(stride_times) < 4:
        return stride_times.copy(), np.ones(len(stride_times), dtype=bool)

    q1 = np.percentile(stride_times, 25)
    q3 = np.percentile(stride_times, 75)
    iqr = q3 - q1
    lower = max(q1 - multiplier * iqr, 0.3)
    upper = min(q3 + multiplier * iqr, 5.0)
    valid_mask = (stride_times >= lower) & (stride_times <= upper)
    return stride_times[valid_mask], valid_mask


def _detect_midstance_zupt(df: pd.DataFrame, side: str,
                            gyro_threshold: float = 50.0) -> np.ndarray:
    """
    Detect mid-stance (ZUPT) frames using gyro magnitude threshold.
    During mid-stance, the foot is flat on the ground and angular velocity is low.
    Returns boolean mask (True = ZUPT frame).

    Approach from MATLAB reference: find frames where gyro magnitude < threshold,
    then keep only the longest continuous segment per stride.
    """
    gx_col = f'{side}_Gx'
    gy_col = f'{side}_Gy'
    gz_col = f'{side}_Gz'
    phase_col = f'{side}_Phase'

    n = len(df)

    # Primary: gyro magnitude method
    if _has_columns(df, [gx_col, gy_col, gz_col]):
        gx = df[gx_col].values.astype(np.float64)
        gy = df[gy_col].values.astype(np.float64)
        gz = df[gz_col].values.astype(np.float64)
        gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        return gyro_mag < gyro_threshold

    # Fallback: Phase column (0 = stance)
    if phase_col in df.columns:
        phase = df[phase_col].values.astype(np.float64)
        return phase < 0.5

    return np.zeros(n, dtype=bool)


def _resolve_velocity_columns(df: pd.DataFrame, side: str) -> tuple[Optional[str], Optional[str]]:
    """Find this side's 2D global-velocity columns in the DataFrame.

    H-Walker / EBIMU firmware ships global-frame walking velocity under
    several names depending on firmware build. We try each in order:

      1. `<side>_Ax` / `<side>_Ay`            EBIMU soa5 (mislabelled
                                              accel — actually velocity)
      2. `<side>_Vx` / `<side>_Vy`            generic 2-axis vel
      3. `<side>_VelX` / `<side>_VelY`        explicit naming
      4. `<side>_GlobalVelX` / `<side>_GlobalVelY`  fully-qualified

    If only a scalar speed `<side>_Vel` is present we fall back to that
    in `_compute_stride_length_from_speed`. Returns (vx_col, vy_col) or
    (None, None) when no candidate set is complete.
    """
    candidates = [
        (f"{side}_Ax", f"{side}_Ay"),
        (f"{side}_Vx", f"{side}_Vy"),
        (f"{side}_VelX", f"{side}_VelY"),
        (f"{side}_GlobalVelX", f"{side}_GlobalVelY"),
        (f"{side}_globalVelX", f"{side}_globalVelY"),
    ]
    for vx, vy in candidates:
        if vx in df.columns and vy in df.columns:
            return vx, vy
    return None, None


def _compute_stride_length_from_speed(df: pd.DataFrame, side: str,
                                       hs_indices: np.ndarray,
                                       valid_mask: np.ndarray,
                                       sample_rate: float) -> np.ndarray:
    """Per-stride length from a scalar walking-speed column.

    When the firmware emits only `<side>_Vel` (m/s, magnitude), the
    integral of |v| over one stride window is the path length. This is
    less accurate than ZUPT-corrected 2D integration (it can't tell
    sideways vs forward motion) but it's what we get when the IMU
    rig only logs scalar speed.
    """
    speed_col = next((c for c in (f"{side}_Vel", f"{side}_Speed",
                                   f"{side}_walking_speed_mps")
                      if c in df.columns), None)
    if speed_col is None:
        return np.array([])
    v = df[speed_col].to_numpy(dtype=np.float64)
    dt = 1.0 / sample_rate
    out: list[float] = []
    for i in range(len(valid_mask)):
        if not valid_mask[i]:
            out.append(np.nan)
            continue
        s, e = hs_indices[i], hs_indices[i + 1]
        if e - s < 10:
            out.append(np.nan)
            continue
        seg = v[s:e]
        seg = seg[np.isfinite(seg)]
        out.append(float(np.sum(np.abs(seg)) * dt) if seg.size else np.nan)
    return np.asarray(out)


def _compute_stride_length_zupt(df: pd.DataFrame, side: str,
                                 hs_indices: np.ndarray, valid_mask: np.ndarray,
                                 sample_rate: float) -> np.ndarray:
    """
    Compute stride length using ZUPT (Zero Velocity Update).

    Reads global-frame walking velocity from whichever pair of columns
    the firmware exposes (see `_resolve_velocity_columns` for the
    accepted aliases). Single integration: velocity → displacement.

    ZUPT method (from MATLAB reference):
    - Accumulate velocity_error_offset during mid-stance (gyro mag < threshold)
    - Subtract offset from velocity at every frame → smooth drift correction
    - NOT hard-zeroing (which creates discontinuities)

    Stride length = norm of horizontal displacement between consecutive heel strikes.

    If 2D columns aren't available, falls back to scalar-speed
    integration via `_compute_stride_length_from_speed` (less precise
    but works on a `<side>_Vel` magnitude channel).
    """
    vx_col, vy_col = _resolve_velocity_columns(df, side)
    if vx_col is None or vy_col is None:
        # Scalar-speed fallback when 2D global-velocity columns aren't
        # logged. Some H-Walker builds only emit `<side>_Vel`.
        return _compute_stride_length_from_speed(
            df, side, hs_indices, valid_mask, sample_rate,
        )

    vx_full = df[vx_col].values.astype(np.float64)
    vy_full = df[vy_col].values.astype(np.float64)

    # Detect ZUPT (mid-stance) frames
    is_zupt = _detect_midstance_zupt(df, side)

    # Apply ZUPT with error-offset accumulation (MATLAB-style)
    # The velocity signal drifts over time. At ZUPT frames, the true velocity
    # should be 0, so the current measured velocity IS the accumulated error.
    # We record this error and subtract it from all subsequent frames.
    n = len(vx_full)
    vx_corrected = np.zeros(n)
    vy_corrected = np.zeros(n)
    offset_x, offset_y = 0.0, 0.0

    for j in range(n):
        vx_corrected[j] = vx_full[j] - offset_x
        vy_corrected[j] = vy_full[j] - offset_y
        if is_zupt[j]:
            offset_x = vx_full[j]
            offset_y = vy_full[j]

    # Now compute position by integrating corrected velocity
    dt = 1.0 / sample_rate
    pos_x = np.cumsum(vx_corrected) * dt
    pos_y = np.cumsum(vy_corrected) * dt

    # Extract stride lengths from position
    n_strides = len(valid_mask)
    stride_lengths = []

    for i in range(n_strides):
        if not valid_mask[i]:
            stride_lengths.append(np.nan)
            continue

        s = hs_indices[i]
        e = hs_indices[i + 1]
        if e - s < 10:
            stride_lengths.append(np.nan)
            continue

        # Horizontal displacement between heel strikes
        dx = pos_x[e] - pos_x[s]
        dy = pos_y[e] - pos_y[s]
        stride_len = np.sqrt(dx**2 + dy**2)
        stride_lengths.append(stride_len)

    result = np.array(stride_lengths)

    # Sanity check: flag unreasonable values
    valid = result[np.isfinite(result)]
    if len(valid) > 0:
        median_len = np.median(valid)
        if median_len < 0.05 or median_len > 3.0:
            print(f"  WARNING: {side} median stride length = {median_len:.3f} m "
                  f"(expected 0.3-2.0 m). Check IMU data frame/units.")

    return result


def _compute_force_tracking(df: pd.DataFrame, side: str,
                             hs_indices: np.ndarray, valid_mask: np.ndarray
                             ) -> ForceTrackingResult:
    """Compute force tracking error (Des vs Act) per stride."""
    des_col = f'{side}_DesForce_N'
    act_col = f'{side}_ActForce_N'

    if not _has_columns(df, [des_col, act_col]):
        return ForceTrackingResult()

    des = df[des_col].values.astype(np.float64)
    act = df[act_col].values.astype(np.float64)

    rmse_list, mae_list = [], []
    all_errors = []

    n_strides = len(valid_mask)
    for i in range(n_strides):
        if not valid_mask[i]:
            continue
        s, e = hs_indices[i], hs_indices[i + 1]
        if e - s < 10:
            continue

        d = des[s:e]
        a = act[s:e]
        err = a - d
        valid_err = err[np.isfinite(err)]
        if len(valid_err) == 0:
            continue

        rmse_list.append(float(np.sqrt(np.mean(valid_err**2))))
        mae_list.append(float(np.mean(np.abs(valid_err))))
        all_errors.extend(valid_err.tolist())

    all_errors = np.array(all_errors) if all_errors else np.array([0.0])
    return ForceTrackingResult(
        rmse=float(np.sqrt(np.mean(all_errors**2))),
        mae=float(np.mean(np.abs(all_errors))),
        peak_error=float(np.max(np.abs(all_errors))),
        rmse_per_stride=np.array(rmse_list),
        mae_per_stride=np.array(mae_list),
    )


def _compute_force_profiles(df: pd.DataFrame, side: str,
                             hs_indices: np.ndarray, valid_mask: np.ndarray,
                             n_points: int = 101) -> ForceProfileResult:
    """Extract GCP-normalized force profiles (resampled to n_points per stride)."""
    act_col = f'{side}_ActForce_N'
    des_col = f'{side}_DesForce_N'

    result = ForceProfileResult()

    for col, attr_ind, attr_mean, attr_std in [
        (act_col, 'individual', 'mean', 'std'),
        (des_col, 'des_individual', 'des_mean', 'des_std'),
    ]:
        if col not in df.columns:
            continue

        force = df[col].values.astype(np.float64)
        profiles = []

        n_strides = len(valid_mask)
        for i in range(n_strides):
            if not valid_mask[i]:
                continue
            s, e = hs_indices[i], hs_indices[i + 1]
            if e - s < 10:
                continue

            stride_f = force[s:e].copy()
            if np.all(np.isnan(stride_f)):
                continue

            # Fill NaN before resampling
            nan_mask = np.isnan(stride_f)
            if np.any(nan_mask):
                valid_idx = np.where(~nan_mask)[0]
                if len(valid_idx) >= 2:
                    stride_f[nan_mask] = np.interp(
                        np.where(nan_mask)[0], valid_idx, stride_f[valid_idx])
                else:
                    stride_f = np.nan_to_num(stride_f, nan=0.0)

            # Resample to n_points
            x_orig = np.linspace(0, 100, len(stride_f))
            resampled = np.interp(np.linspace(0, 100, n_points), x_orig, stride_f)
            profiles.append(resampled)

        if profiles:
            arr = np.array(profiles)
            setattr(result, attr_ind, arr)
            setattr(result, attr_mean, np.mean(arr, axis=0))
            setattr(result, attr_std, np.std(arr, axis=0))

    return result


def _compute_stance_swing(df: pd.DataFrame, side: str,
                           hs_indices: np.ndarray, valid_mask: np.ndarray
                           ) -> tuple[list[float], list[float]]:
    """Compute stance / swing % of each stride.

    Definition (per H-Walker firmware spec — see HANDOVER-2026-04-17):
      `{side}_GCP` ramps 0 → 1+ ONLY during that side's stance phase
      and is 0 (or NaN) during swing. So the stance fraction per
      stride = fraction of samples where GCP > 0.01.

    Earlier code preferred `{side}_Phase` (n_stance = sum(phase < 0.5))
    but the firmware's Phase encoding turned out to be inconsistent
    across builds — some emit 0=stance, others 0=swing — and the
    fallback `gcp < 0.6` (i.e. "first 60 % of the cycle is stance")
    was a hard-coded biomechanical assumption that has nothing to do
    with the actual signal. Both modes produced ~50/50 splits when
    the truth was 60/40. This routine now uses the GCP active mask
    only and reports nothing for sides where GCP is missing.
    """
    gcp_col = f'{side}_GCP'
    if gcp_col not in df.columns:
        return [], []
    gcp_full = df[gcp_col].values.astype(np.float64)

    stance_ratios, swing_ratios = [], []
    n_strides = len(valid_mask)
    for i in range(n_strides):
        if not valid_mask[i]:
            continue
        s, e = hs_indices[i], hs_indices[i + 1]
        if e - s < 5:
            continue
        stride_gcp = gcp_full[s:e]
        # Ignore NaN samples in the count
        finite = np.isfinite(stride_gcp)
        if finite.sum() < 5:
            continue
        active = stride_gcp[finite] > 0.01
        n_stance = int(np.sum(active))
        total = int(np.sum(finite))
        stance_ratios.append(n_stance / total * 100)
        swing_ratios.append((total - n_stance) / total * 100)

    return stance_ratios, swing_ratios


def _analyze_df(df: pd.DataFrame, fs: float, *, filename: str, filepath: str,
                run_force: bool, run_imu: bool, run_gait: bool,
                label_prefix: str = "") -> AnalysisResult:
    """Per-DataFrame analysis core.

    Used by both the legacy whole-CSV path (`analyze_file`) and the
    sync-window path (`analyze_file_windows`). The `df` passed here
    must already be the slice the caller wants analyzed — the function
    does not look at any sync columns or apply any windowing logic.
    """
    result = AnalysisResult(
        filename=filename,
        filepath=filepath,
        n_samples=len(df),
        duration_s=len(df) / fs,
        sample_rate=fs,
    )

    print(f"  {label_prefix}{filename} ({result.n_samples} samples, "
          f"{result.duration_s:.1f}s, {fs:.1f} Hz)")

    for side, stride_attr, ft_attr, fp_attr in [
        ('L', 'left_stride', 'left_force_tracking', 'left_force_profile'),
        ('R', 'right_stride', 'right_force_tracking', 'right_force_profile'),
    ]:
        # Detect heel strikes
        hs_idx = _detect_heel_strikes(df, side, fs)
        if len(hs_idx) < 2:
            print(f"    {side}: No strides detected")
            continue

        # Stride times
        n_raw = len(hs_idx) - 1
        stride_times_raw = np.array([(hs_idx[i+1] - hs_idx[i]) / fs for i in range(n_raw)])
        stride_times_filtered, valid_mask = _filter_strides_iqr(stride_times_raw)

        sr = StrideResult(
            stride_times_raw=stride_times_raw,
            stride_times=stride_times_filtered,
            stride_time_mean=float(np.mean(stride_times_filtered)) if len(stride_times_filtered) else 0,
            stride_time_std=float(np.std(stride_times_filtered)) if len(stride_times_filtered) else 0,
            n_strides=len(stride_times_filtered),
            hs_indices=hs_idx,
            valid_mask=valid_mask,
        )
        if sr.stride_time_mean > 0:
            sr.stride_time_cv = sr.stride_time_std / sr.stride_time_mean * 100
            # Whole-body cadence (steps/min) estimated from one side's
            # heel-strike series. `stride_time_mean` is the time between
            # two consecutive heel strikes on the SAME leg → one full
            # gait cycle → two steps (one L step + one R step). So:
            #   cadence [steps/min] = (60 / stride_time_s) * 2 = 120 / T
            # Do NOT "simplify" the * 2 away — removing it halves the
            # reported cadence. test_cadence_formula pins this.
            sr.cadence = 60.0 / sr.stride_time_mean * 2

        # Stance/Swing
        if run_gait:
            stance, swing = _compute_stance_swing(df, side, hs_idx, valid_mask)
            if stance:
                sr.stance_pct_mean = float(np.mean(stance))
                sr.stance_pct_std = float(np.std(stance))
                sr.swing_pct_mean = float(np.mean(swing))
                sr.swing_pct_std = float(np.std(swing))

        # Stride length (ZUPT)
        if run_imu:
            sr.stride_lengths = _compute_stride_length_zupt(df, side, hs_idx, valid_mask, fs)
            valid_lengths = sr.stride_lengths[np.isfinite(sr.stride_lengths)]
            if len(valid_lengths) > 0:
                sr.stride_length_mean = float(np.mean(valid_lengths))
                sr.stride_length_std = float(np.std(valid_lengths))

        setattr(result, stride_attr, sr)
        print(f"    {side}: {sr.n_strides} strides, "
              f"time={sr.stride_time_mean:.3f}±{sr.stride_time_std:.3f}s, "
              f"cadence={sr.cadence:.1f} steps/min")

        # Force tracking
        if run_force:
            ft = _compute_force_tracking(df, side, hs_idx, valid_mask)
            setattr(result, ft_attr, ft)
            if ft.rmse > 0:
                print(f"    {side} Force: RMSE={ft.rmse:.2f}N, MAE={ft.mae:.2f}N, "
                      f"Peak={ft.peak_error:.2f}N")

            # Force profiles
            fp = _compute_force_profiles(df, side, hs_idx, valid_mask)
            setattr(result, fp_attr, fp)
            if fp.individual is not None:
                print(f"    {side} Profiles: {fp.individual.shape[0]} strides resampled")

    # Symmetry indices
    ls, rs = result.left_stride, result.right_stride
    result.stride_time_symmetry = _symmetry_index(ls.stride_time_mean, rs.stride_time_mean)
    result.stride_length_symmetry = _symmetry_index(ls.stride_length_mean, rs.stride_length_mean)
    result.stance_symmetry = _symmetry_index(ls.stance_pct_mean, rs.stance_pct_mean)

    lf_t, rf_t = result.left_force_tracking, result.right_force_tracking
    if lf_t.rmse > 0 and rf_t.rmse > 0:
        result.force_symmetry = _symmetry_index(lf_t.rmse, rf_t.rmse)

    # Fatigue
    if len(ls.stride_times) >= 10:
        result.left_fatigue = _fatigue_index(ls.stride_times)
    if len(rs.stride_times) >= 10:
        result.right_fatigue = _fatigue_index(rs.stride_times)

    return result


def analyze_file(filepath: str, analyses: list[str] = None) -> AnalysisResult:
    """Run full analysis on the entire CSV (legacy behavior).

    For multi-trial recordings (one CSV with several sync windows),
    use `analyze_file_windows()` instead — that returns one result
    per window and is the correct unit for downstream aggregation.

    Args:
        filepath: Path to H-Walker CSV file.
        analyses: List of analysis types to run. Options: 'force',
                  'imu', 'gait', 'all'. Default is ['all'].

    Returns:
        AnalysisResult covering the whole CSV.
    """
    if analyses is None:
        analyses = ['all']
    run_all   = 'all'   in analyses
    run_force = run_all or 'force' in analyses
    run_imu   = run_all or 'imu'   in analyses
    run_gait  = run_all or 'gait'  in analyses

    dm = DataManager()
    lf = dm.load_csv(filepath)
    if lf is None:
        raise ValueError(f"Failed to load CSV: {filepath}")

    df = lf.df
    fs = DataManager.estimate_sample_rate(df)
    if not fs or fs <= 0:
        raise ValueError(
            f"Could not estimate sample rate for {filepath} (got {fs!r}). "
            "CSV is likely missing a monotonic timestamp column."
        )

    return _analyze_df(
        df, fs,
        filename=os.path.basename(filepath),
        filepath=filepath,
        run_force=run_force, run_imu=run_imu, run_gait=run_gait,
        label_prefix="Loaded: ",
    )


def analyze_file_windows(filepath: str, analyses: list[str] = None
                          ) -> list[AnalysisResult]:
    """Analyze each sync window of a recording independently.

    A sync window = [rising-edge → falling-edge] in the firmware's
    `Sync` column (operator pressed → released). Each window is one
    trial in the experiment. Returns one AnalysisResult per window.

    If the recording has no Sync column (or no detectable windows),
    falls back to whole-CSV analysis and returns a single-element list
    so callers can treat the no-sync case identically.
    """
    if analyses is None:
        analyses = ['all']
    run_all   = 'all'   in analyses
    run_force = run_all or 'force' in analyses
    run_imu   = run_all or 'imu'   in analyses
    run_gait  = run_all or 'gait'  in analyses

    dm = DataManager()
    lf = dm.load_csv(filepath)
    if lf is None:
        raise ValueError(f"Failed to load CSV: {filepath}")

    df = lf.df
    fs = DataManager.estimate_sample_rate(df)
    if not fs or fs <= 0:
        raise ValueError(
            f"Could not estimate sample rate for {filepath} (got {fs!r}). "
            "CSV is likely missing a monotonic timestamp column."
        )

    # Lazy import keeps the analyzer importable in environments that
    # don't have the FastAPI tree on the path.
    from backend.services.sync_align import find_sync_windows
    windows = find_sync_windows(df)
    fname = os.path.basename(filepath)

    if not windows:
        # No sync information → treat whole recording as one trial.
        return [_analyze_df(
            df, fs,
            filename=fname, filepath=filepath,
            run_force=run_force, run_imu=run_imu, run_gait=run_gait,
            label_prefix="No sync: ",
        )]

    results: list[AnalysisResult] = []
    for win in windows:
        slice_df = df.iloc[win.sample_rising:win.sample_falling].reset_index(drop=True)
        if len(slice_df) < 10:
            print(f"  Window {win.index}: only {len(slice_df)} samples — skipping")
            continue
        prefix = f"Window {win.index} ({win.duration_s:.1f}s) of "
        res = _analyze_df(
            slice_df, fs,
            filename=fname, filepath=filepath,
            run_force=run_force, run_imu=run_imu, run_gait=run_gait,
            label_prefix=prefix,
        )
        # Tag the result with its window so downstream aggregation
        # can pair it across sources.
        res.sync_window_idx = win.index
        res.sync_window_t_rise_s = win.rising_t_s
        res.sync_window_t_fall_s = win.falling_t_s
        results.append(res)
    return results


def compare_results(results: list[AnalysisResult]) -> dict:
    """
    Generate comparison statistics across multiple analysis results.
    Returns a dict summarizing differences.
    """
    if len(results) < 2:
        return {}

    comp = {
        'n_files': len(results),
        'filenames': [r.filename for r in results],
        'files': [],
    }

    for r in results:
        comp['files'].append({
            'filename': r.filename,
            'duration_s': r.duration_s,
            'l_stride_time': r.left_stride.stride_time_mean,
            'r_stride_time': r.right_stride.stride_time_mean,
            'l_stride_length': r.left_stride.stride_length_mean,
            'r_stride_length': r.right_stride.stride_length_mean,
            'l_cadence': r.left_stride.cadence,
            'r_cadence': r.right_stride.cadence,
            'l_stance_pct': r.left_stride.stance_pct_mean,
            'r_stance_pct': r.right_stride.stance_pct_mean,
            'l_force_rmse': r.left_force_tracking.rmse,
            'r_force_rmse': r.right_force_tracking.rmse,
            'stride_time_symmetry': r.stride_time_symmetry,
            'force_symmetry': r.force_symmetry,
            'l_n_strides': r.left_stride.n_strides,
            'r_n_strides': r.right_stride.n_strides,
        })

    return comp


def result_to_dict(r: AnalysisResult) -> dict:
    """Convert AnalysisResult to a JSON-serializable dict."""
    d = {
        'filename': r.filename,
        'filepath': r.filepath,
        'n_samples': r.n_samples,
        'duration_s': round(r.duration_s, 2),
        'sample_rate': round(r.sample_rate, 1),
        'symmetry': {
            'stride_time': round(r.stride_time_symmetry, 2),
            'stride_length': round(r.stride_length_symmetry, 2),
            'force': round(r.force_symmetry, 2),
            'stance': round(r.stance_symmetry, 2),
        },
        'fatigue': {
            'left_pct_change': round(r.left_fatigue, 2),
            'right_pct_change': round(r.right_fatigue, 2),
        },
    }

    for side_name, sr, ft in [
        ('left', r.left_stride, r.left_force_tracking),
        ('right', r.right_stride, r.right_force_tracking),
    ]:
        d[side_name] = {
            'n_strides': sr.n_strides,
            'stride_time_mean': round(sr.stride_time_mean, 4),
            'stride_time_std': round(sr.stride_time_std, 4),
            'stride_time_cv': round(sr.stride_time_cv, 2),
            'stride_length_mean': round(sr.stride_length_mean, 4),
            'stride_length_std': round(sr.stride_length_std, 4),
            'cadence': round(sr.cadence, 1),
            'stance_pct_mean': round(sr.stance_pct_mean, 1),
            'stance_pct_std': round(sr.stance_pct_std, 1),
            'swing_pct_mean': round(sr.swing_pct_mean, 1),
            'swing_pct_std': round(sr.swing_pct_std, 1),
            'force_tracking': {
                'rmse': round(ft.rmse, 3),
                'mae': round(ft.mae, 3),
                'peak_error': round(ft.peak_error, 3),
            },
        }

    return d
