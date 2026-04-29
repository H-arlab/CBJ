import numpy as np
import pandas as pd
import pytest

from backend.services.analysis_engine import (
    load_csv,
    resolve_gcp,
    detect_heel_strikes,
    normalize_to_gcp,
    compute_stats,
    compute_symmetry_index,
    run_full_analysis,
    run_per_window_analysis,
    count_sync_windows,
)
from backend.models.schema import StatsResult


class TestLoadCsv:
    def test_returns_dataframe(self, sample_csv):
        df = load_csv(sample_csv)
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self, sample_csv):
        df = load_csv(sample_csv)
        assert len(df) == 500

    def test_expected_columns_present(self, sample_csv):
        df = load_csv(sample_csv)
        for col in ["L_ActForce_N", "R_ActForce_N", "L_GCP", "R_GCP"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_column_names_are_stripped(self, tmp_path):
        """Columns with leading/trailing spaces must be stripped."""
        df = pd.DataFrame({" L_ActForce_N ": [1.0, 2.0], " R_ActForce_N ": [3.0, 4.0]})
        path = str(tmp_path / "spaced.csv")
        df.to_csv(path, index=False)
        loaded = load_csv(path)
        assert "L_ActForce_N" in loaded.columns
        assert "R_ActForce_N" in loaded.columns


class TestResolveGcp:
    def test_returns_array(self, sample_csv_path_and_df):
        path, df = sample_csv_path_and_df
        gcp = resolve_gcp(df, "L")
        assert isinstance(gcp, np.ndarray)

    def test_length_matches_dataframe(self, sample_csv_path_and_df):
        path, df = sample_csv_path_and_df
        gcp = resolve_gcp(df, "L")
        assert len(gcp) == len(df)

    def test_values_normalized_0_to_1(self, sample_csv_path_and_df):
        path, df = sample_csv_path_and_df
        gcp = resolve_gcp(df, "L")
        assert gcp.min() >= 0.0
        assert gcp.max() <= 1.0

    def test_both_sides_supported(self, sample_csv_path_and_df):
        path, df = sample_csv_path_and_df
        for side in ["L", "R"]:
            gcp = resolve_gcp(df, side)
            assert len(gcp) == len(df)

    def test_already_normalized_passthrough(self, sample_csv_path_and_df):
        """GCP values already in 0-1 should not be scaled again."""
        path, df = sample_csv_path_and_df
        df2 = df.copy()
        df2["L_GCP"] = df2["L_GCP"] / 100.0
        gcp = resolve_gcp(df2, "L")
        assert gcp.max() <= 1.0


class TestDetectHeelStrikes:
    def test_returns_indices_array(self):
        gcp = np.tile(np.linspace(0, 1, 110), 4)  # 4 strides
        hs = detect_heel_strikes(gcp)
        assert isinstance(hs, np.ndarray)
        assert hs.dtype in (np.int32, np.int64, int)

    def test_detects_correct_number_of_strides(self):
        """4 repetitions of 0→1 sawtooth = 4 heel strikes (at resets)."""
        gcp = np.tile(np.linspace(0, 1, 110), 4)
        hs = detect_heel_strikes(gcp)
        # Expect 3-5 crossings (boundary at start may or may not be detected)
        assert 2 <= len(hs) <= 5

    def test_hs_indices_within_bounds(self):
        gcp = np.tile(np.linspace(0, 1, 110), 4)
        hs = detect_heel_strikes(gcp)
        assert all(0 <= i < len(gcp) for i in hs)

    def test_hs_sorted_ascending(self):
        gcp = np.tile(np.linspace(0, 1, 110), 4)
        hs = detect_heel_strikes(gcp)
        assert list(hs) == sorted(hs)


class TestNormalizeToGcp:
    def test_returns_101_points(self):
        gcp = np.tile(np.linspace(0, 1, 110), 5)
        signal = np.sin(np.linspace(0, 10 * np.pi, len(gcp)))
        hs = detect_heel_strikes(gcp)
        mean_101, std_101 = normalize_to_gcp(signal, hs)
        assert len(mean_101) == 101
        assert len(std_101) == 101

    def test_mean_is_finite(self):
        gcp = np.tile(np.linspace(0, 1, 110), 5)
        signal = np.sin(np.linspace(0, 10 * np.pi, len(gcp)))
        hs = detect_heel_strikes(gcp)
        mean_101, std_101 = normalize_to_gcp(signal, hs)
        assert np.all(np.isfinite(mean_101))
        assert np.all(np.isfinite(std_101))

    def test_std_is_non_negative(self):
        gcp = np.tile(np.linspace(0, 1, 110), 5)
        signal = np.sin(np.linspace(0, 10 * np.pi, len(gcp)))
        hs = detect_heel_strikes(gcp)
        _, std_101 = normalize_to_gcp(signal, hs)
        assert np.all(std_101 >= 0)

    def test_fewer_than_2_strides_returns_zeros(self):
        signal = np.sin(np.linspace(0, np.pi, 100))
        hs = np.array([0], dtype=int)
        mean_101, std_101 = normalize_to_gcp(signal, hs)
        assert len(mean_101) == 101
        assert np.all(std_101 == 0)


class TestComputeStats:
    def test_returns_list_of_stats_results(self, sample_csv):
        df = load_csv(sample_csv)
        results = compute_stats(df, ["L_ActForce_N", "R_ActForce_N"], "trial_001.csv")
        assert len(results) == 2
        assert all(isinstance(r, StatsResult) for r in results)

    def test_column_names_in_results(self, sample_csv):
        df = load_csv(sample_csv)
        results = compute_stats(df, ["L_ActForce_N"], "trial_001.csv")
        assert results[0].column == "L_ActForce_N"

    def test_file_name_in_results(self, sample_csv):
        df = load_csv(sample_csv)
        results = compute_stats(df, ["L_ActForce_N"], "trial_001.csv")
        assert results[0].file == "trial_001.csv"

    def test_mean_within_expected_range(self, sample_csv):
        df = load_csv(sample_csv)
        results = compute_stats(df, ["L_ActForce_N"], "trial_001.csv")
        assert 0 < results[0].mean < 50

    def test_max_does_not_exceed_70(self, sample_csv):
        df = load_csv(sample_csv)
        results = compute_stats(df, ["L_ActForce_N"], "trial_001.csv")
        assert results[0].max_val <= 72  # allow tiny noise overshoot

    def test_min_is_non_negative_for_force(self, sample_csv):
        df = load_csv(sample_csv)
        results = compute_stats(df, ["L_ActForce_N"], "trial_001.csv")
        assert results[0].min_val >= 0

    def test_std_is_positive(self, sample_csv):
        df = load_csv(sample_csv)
        results = compute_stats(df, ["L_ActForce_N"], "trial_001.csv")
        assert results[0].std > 0


class TestComputeSymmetryIndex:
    def test_identical_signals_return_zero(self):
        sig = np.ones(100) * 25.0
        assert compute_symmetry_index(sig, sig) == pytest.approx(0.0)

    def test_returns_float(self):
        left = np.ones(100) * 30.0
        right = np.ones(100) * 20.0
        result = compute_symmetry_index(left, right)
        assert isinstance(result, float)

    def test_returns_positive_value(self):
        left = np.ones(100) * 30.0
        right = np.ones(100) * 20.0
        result = compute_symmetry_index(left, right)
        assert result > 0

    def test_known_value(self):
        # |30-20| / ((30+20)/2) * 100 = 10/25*100 = 40.0
        left = np.ones(100) * 30.0
        right = np.ones(100) * 20.0
        result = compute_symmetry_index(left, right)
        assert result == pytest.approx(40.0, rel=1e-5)


# ============================================================
# Sync-window slicing — locks the user-confirmed contract
# (CLAUDE.md, 2026-04-25): 1 sync pulse = 1 trial, and the analyzer
# must use only data inside [rising, falling). A regression that
# routes /api/analyze back through whole-file processing must turn
# this suite red.
# ============================================================

def _make_dual_window_robot_csv(tmp_path,
                                 fs: float = 111.0,
                                 dur_s: float = 12.0,
                                 stride_s: float = 1.05,
                                 peak_w0_n: float = 40.0,
                                 peak_w1_n: float = 60.0) -> str:
    """Two distinct sync windows in one recording, with different peak
    forces so per-window analysis is observably different from a
    whole-file mean."""
    n = int(dur_s * fs)
    t = np.arange(n) / fs
    sync = np.zeros(n)
    sync[(t >= 1.0) & (t < 5.0)] = 1.0     # window 0
    sync[(t >= 7.0) & (t < 11.0)] = 1.0    # window 1
    df = pd.DataFrame({
        "Time_ms": t * 1000.0,
        "Freq_Hz": np.full(n, fs),
        "Sync": sync,
    })
    for side, off in (("L", 0.0), ("R", stride_s / 2)):
        gcp = np.zeros(n); evt = np.zeros(n); des = np.zeros(n); act = np.zeros(n); pit = np.zeros(n)
        for i in range(int(dur_s / stride_s)):
            s = i * stride_s + off
            e = s + stride_s * 0.62
            peak = peak_w0_n if s < 6.0 else peak_w1_n
            sm = (t >= s) & (t < e)
            rm = (t >= s) & (t < s + stride_s)
            if sm.any():
                gcp[sm] = (t[sm] - s) / (stride_s * 0.62)
                hump = np.sin(np.pi * (t[sm] - s) / (stride_s * 0.62)) * peak
                act[sm] = hump
                des[sm] = hump * 1.05
            if rm.any():
                pit[rm] = 12.0 * np.sin(2 * np.pi * (t[rm] - s) / stride_s)
            idx = int(np.searchsorted(t, s))
            if 0 <= idx < n:
                evt[idx:idx + 2] = 1.0
        df[f"{side}_GCP"] = gcp
        df[f"{side}_Event"] = evt
        df[f"{side}_Phase"] = (gcp > 0.01).astype(float)
        df[f"{side}_DesForce_N"] = des
        df[f"{side}_ActForce_N"] = act
        df[f"{side}_ErrForce_N"] = act - des
        df[f"{side}_Pitch_deg"] = pit
    path = str(tmp_path / "Robot_dual_0.csv")
    df.to_csv(path, index=False)
    return path


class TestSyncSlicing:
    def test_count_matches_pulse_count(self, tmp_path):
        path = _make_dual_window_robot_csv(tmp_path)
        assert count_sync_windows(path) == 2

    def test_default_returns_first_window(self, tmp_path):
        path = _make_dual_window_robot_csv(tmp_path)
        res = run_full_analysis(path)
        assert res.sync_window_idx == 0
        assert res.sync_window_t_rise_s == pytest.approx(1.0, abs=0.05)
        assert res.sync_window_t_fall_s == pytest.approx(5.0, abs=0.05)

    def test_window_idx_selects_specific_trial(self, tmp_path):
        path = _make_dual_window_robot_csv(tmp_path)
        res1 = run_full_analysis(path, window_idx=1)
        assert res1.sync_window_idx == 1
        assert res1.sync_window_t_rise_s == pytest.approx(7.0, abs=0.05)
        assert res1.sync_window_t_fall_s == pytest.approx(11.0, abs=0.05)

    def test_per_window_force_is_independent(self, tmp_path):
        """The two windows have peak forces 40 N vs 60 N. Per-window
        analysis must surface those two distinct values, not a mean."""
        path = _make_dual_window_robot_csv(tmp_path,
                                           peak_w0_n=40.0, peak_w1_n=60.0)
        r0 = run_full_analysis(path, window_idx=0)
        r1 = run_full_analysis(path, window_idx=1)
        peak0 = float(np.max(r0.left_force_profile.mean))
        peak1 = float(np.max(r1.left_force_profile.mean))
        assert 35.0 < peak0 < 45.0, f"window 0 peak {peak0:.1f} N not near 40 N"
        assert 55.0 < peak1 < 65.0, f"window 1 peak {peak1:.1f} N not near 60 N"
        # And they must be obviously different — a whole-file regression
        # would collapse both to the same mid-value (~50 N).
        assert abs(peak1 - peak0) > 10.0

    def test_per_window_returns_one_per_pulse(self, tmp_path):
        path = _make_dual_window_robot_csv(tmp_path)
        results = run_per_window_analysis(path)
        assert len(results) == 2
        idxs = [r.sync_window_idx for r in results]
        assert idxs == [0, 1]

    def test_out_of_range_window_raises(self, tmp_path):
        path = _make_dual_window_robot_csv(tmp_path)
        with pytest.raises(IndexError):
            run_full_analysis(path, window_idx=5)

    def test_no_sync_falls_back_to_whole_file(self, tmp_path):
        """A CSV with no Sync column must still analyze (single
        synthetic 'window 0' covering the whole recording)."""
        path = _make_dual_window_robot_csv(tmp_path)
        df = pd.read_csv(path).drop(columns=["Sync"])
        no_sync_path = str(tmp_path / "Robot_nosync.csv")
        df.to_csv(no_sync_path, index=False)
        assert count_sync_windows(no_sync_path) == 1
        res = run_full_analysis(no_sync_path)
        # Whole-file analysis covers ~12 s, so we should see noticeably
        # more strides than a single 4 s window would yield (~3).
        assert res.left_stride.n_strides >= 8


def _make_poison_outside_csv(tmp_path,
                              fs: float = 111.0,
                              dur_s: float = 12.0,
                              stride_s: float = 1.05,
                              poison: float = 999.0) -> str:
    """Two sync windows with clean strides inside; everything outside
    every window is set to a POISON value. If the slicer leaks even
    one out-of-window sample into the analyzer, the profile peak
    will spike up to `poison`."""
    n = int(dur_s * fs)
    t = np.arange(n) / fs
    sync = np.zeros(n)
    sync[(t >= 1.0) & (t < 5.0)] = 1.0
    sync[(t >= 7.0) & (t < 11.0)] = 1.0
    in_window = sync > 0.5
    out_window = ~in_window
    df = pd.DataFrame({
        "Time_ms": t * 1000.0,
        "Freq_Hz": np.full(n, fs),
        "Sync": sync,
    })
    for side, off in (("L", 0.0), ("R", stride_s / 2)):
        gcp = np.zeros(n); evt = np.zeros(n)
        des = np.zeros(n); act = np.zeros(n); pit = np.zeros(n)
        gcp[out_window] = poison
        des[out_window] = poison
        act[out_window] = poison
        pit[out_window] = poison
        for i in range(int(dur_s / stride_s)):
            s = i * stride_s + off
            e = s + stride_s * 0.62
            idx = int(np.searchsorted(t, s))
            if idx >= n or not in_window[idx]:
                continue
            peak = 40.0 if s < 6.0 else 60.0
            sm = (t >= s) & (t < e)
            rm = (t >= s) & (t < s + stride_s)
            if sm.any():
                gcp[sm] = (t[sm] - s) / (stride_s * 0.62)
                hump = np.sin(np.pi * (t[sm] - s) / (stride_s * 0.62)) * peak
                act[sm] = hump
                des[sm] = hump * 1.05
            if rm.any():
                pit[rm] = 12.0 * np.sin(2 * np.pi * (t[rm] - s) / stride_s)
            evt[idx:idx + 2] = 1.0
        df[f"{side}_GCP"] = gcp
        df[f"{side}_Event"] = evt
        df[f"{side}_Phase"] = (gcp > 0.01).astype(float)
        df[f"{side}_DesForce_N"] = des
        df[f"{side}_ActForce_N"] = act
        df[f"{side}_ErrForce_N"] = act - des
        df[f"{side}_Pitch_deg"] = pit
    path = str(tmp_path / "Robot_poison_outside.csv")
    df.to_csv(path, index=False)
    return path


class TestSyncSlicingNoLeakage:
    """End-to-end: out-of-window samples must NEVER reach the analyzer.

    Builds a recording with a 999 N poison value everywhere outside
    the two sync windows. If the half-open `[rising, falling)` slice
    is honored, the analyzer never sees the poison and the per-stride
    force profile peaks cleanly at 40 N and 60 N.
    """

    def test_window_0_returns_clean_40n_peak(self, tmp_path):
        path = _make_poison_outside_csv(tmp_path)
        res = run_full_analysis(path, window_idx=0)
        peak = float(np.max(res.left_force_profile.mean))
        assert peak < 100.0, f"poison leaked into window 0 (peak={peak:.1f})"
        assert 35.0 < peak < 45.0, f"window 0 peak {peak:.1f} N not near 40 N"

    def test_window_1_returns_clean_60n_peak(self, tmp_path):
        path = _make_poison_outside_csv(tmp_path)
        res = run_full_analysis(path, window_idx=1)
        peak = float(np.max(res.left_force_profile.mean))
        assert peak < 100.0, f"poison leaked into window 1 (peak={peak:.1f})"
        assert 55.0 < peak < 65.0, f"window 1 peak {peak:.1f} N not near 60 N"

    def test_force_tracking_rmse_stays_small_in_each_window(self, tmp_path):
        """RMSE between Des and Act in either window should be a few
        Newtons (Des is 1.05 × Act). A leak would drive RMSE > 100."""
        path = _make_poison_outside_csv(tmp_path)
        for w in (0, 1):
            res = run_full_analysis(path, window_idx=w)
            rmse = float(res.left_force_tracking.rmse)
            assert rmse < 50.0, (
                f"window {w} RMSE {rmse:.1f} N — poison leaked into "
                f"force_tracking"
            )


def _make_phantom_mixed_csv(tmp_path,
                             fs: float = 111.0,
                             stride_s: float = 1.05) -> str:
    """Recording with two REAL trials interleaved with three short
    phantom pulses (the kind file-IO events produce on the H-Walker
    sync line). Real trials carry clean stride data; phantoms have
    POISON values everywhere. If phantom filtering works, only the
    two real trials reach the analyzer.

    Layout:
        t=0.10–0.12 s  : phantom (file-open)
        t=1.00–5.00 s  : REAL trial 0  (peak 40 N)
        t=5.20–5.22 s  : phantom (auto-save)
        t=7.00–11.00 s : REAL trial 1  (peak 60 N)
        t=11.50–11.51s : phantom (file-close)
    """
    dur_s = 12.0
    n = int(dur_s * fs)
    t = np.arange(n) / fs
    sync = np.zeros(n)
    # phantoms (must be < 0.5 s default threshold)
    sync[(t >= 0.10) & (t < 0.12)] = 1.0
    sync[(t >= 5.20) & (t < 5.22)] = 1.0
    sync[(t >= 11.50) & (t < 11.51)] = 1.0
    # REAL trials (≥ 0.5 s)
    real_a = (t >= 1.0) & (t < 5.0)
    real_b = (t >= 7.0) & (t < 11.0)
    sync[real_a] = 1.0
    sync[real_b] = 1.0

    in_real = real_a | real_b
    POISON = 999.0
    df = pd.DataFrame({
        "Time_ms": t * 1000.0,
        "Freq_Hz": np.full(n, fs),
        "Sync": sync,
    })
    for side, off in (("L", 0.0), ("R", stride_s / 2)):
        gcp = np.zeros(n); evt = np.zeros(n)
        des = np.zeros(n); act = np.zeros(n); pit = np.zeros(n)
        # Outside REAL trials: poison so any leak is observable.
        gcp[~in_real] = POISON
        des[~in_real] = POISON
        act[~in_real] = POISON
        pit[~in_real] = POISON
        for i in range(int(dur_s / stride_s)):
            s = i * stride_s + off
            e = s + stride_s * 0.62
            idx = int(np.searchsorted(t, s))
            if idx >= n or not in_real[idx]:
                continue
            peak = 40.0 if s < 6.0 else 60.0
            sm = (t >= s) & (t < e)
            rm = (t >= s) & (t < s + stride_s)
            if sm.any():
                gcp[sm] = (t[sm] - s) / (stride_s * 0.62)
                hump = np.sin(np.pi * (t[sm] - s) / (stride_s * 0.62)) * peak
                act[sm] = hump
                des[sm] = hump * 1.05
            if rm.any():
                pit[rm] = 12.0 * np.sin(2 * np.pi * (t[rm] - s) / stride_s)
            evt[idx:idx + 2] = 1.0
        df[f"{side}_GCP"] = gcp
        df[f"{side}_Event"] = evt
        df[f"{side}_Phase"] = (gcp > 0.01).astype(float)
        df[f"{side}_DesForce_N"] = des
        df[f"{side}_ActForce_N"] = act
        df[f"{side}_ErrForce_N"] = act - des
        df[f"{side}_Pitch_deg"] = pit
    path = str(tmp_path / "Robot_phantom_mixed.csv")
    df.to_csv(path, index=False)
    return path


class TestPhantomPulseEndToEnd:
    """End-to-end: phantoms must NEVER show up as analyzable trials."""

    def test_count_excludes_phantom_pulses(self, tmp_path):
        path = _make_phantom_mixed_csv(tmp_path)
        # Five raw pulses, two of which are real → count is 2.
        assert count_sync_windows(path) == 2

    def test_per_window_returns_only_real_trials(self, tmp_path):
        path = _make_phantom_mixed_csv(tmp_path)
        results = run_per_window_analysis(path)
        assert len(results) == 2, (
            "phantom pulses must not be returned as trials; got "
            f"{len(results)} results"
        )

    def test_default_window_is_real_trial_zero(self, tmp_path):
        path = _make_phantom_mixed_csv(tmp_path)
        res = run_full_analysis(path, window_idx=0)
        assert res.sync_window_t_rise_s == pytest.approx(1.0, abs=0.05), (
            f"default window 0 should be the first REAL trial (t=1.0s), "
            f"got t={res.sync_window_t_rise_s:.3f}s — phantom may have "
            f"slipped through"
        )
        peak = float(np.max(res.left_force_profile.mean))
        assert 35.0 < peak < 45.0
        assert peak < 100.0   # no poison leak

    def test_window_one_is_second_real_trial(self, tmp_path):
        path = _make_phantom_mixed_csv(tmp_path)
        res = run_full_analysis(path, window_idx=1)
        assert res.sync_window_t_rise_s == pytest.approx(7.0, abs=0.05)
        peak = float(np.max(res.left_force_profile.mean))
        assert 55.0 < peak < 65.0
        assert peak < 100.0   # no poison leak

    def test_addressing_phantom_index_raises(self, tmp_path):
        """Trying to address a phantom (e.g. index 2 in 5-pulse raw) is
        out of range after filtering, since count_sync_windows returns 2."""
        path = _make_phantom_mixed_csv(tmp_path)
        with pytest.raises(IndexError):
            run_full_analysis(path, window_idx=2)
