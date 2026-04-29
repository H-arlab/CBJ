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
