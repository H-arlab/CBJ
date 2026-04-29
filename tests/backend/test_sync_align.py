"""Tests for sync_align — rising-edge / falling-edge window detection.

User-confirmed sync contract (2026-04-25):
    Rising edge = sync STARTS  (operator pressed → trial begins)
    Falling edge = sync ENDS   (operator released → trial ends)
    Window = [rising, falling] half-open interval

A recording can contain N sync windows = N trials inside one CSV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.services import sync_align


# ============================================================
# Synthetic sync signal helpers
# ============================================================

def _sync_with_windows(n: int, fs: float,
                        windows: list[tuple[float, float]]) -> np.ndarray:
    """Return a sync signal of length n at sampling rate fs that is
    HIGH inside each (rise, fall) interval and LOW everywhere else.

    Default state = LOW (subject prep / between trials).
    """
    out = np.zeros(n, dtype=float)
    t = np.arange(n) / fs
    for rise, fall in windows:
        mask = (t >= rise) & (t < fall)
        out[mask] = 1.0
    return out


def _df(sync: np.ndarray, fs: float, **extra) -> pd.DataFrame:
    n = len(sync)
    data = {"Time_ms": np.arange(n) * (1000.0 / fs), "Sync": sync, **extra}
    return pd.DataFrame(data)


# ============================================================
# Sync window detection
# ============================================================

def test_finds_single_window():
    fs = 100.0
    sync = _sync_with_windows(500, fs, [(1.0, 3.0)])
    df = _df(sync, fs)
    wins = sync_align.find_sync_windows(df)
    assert len(wins) == 1
    w = wins[0]
    assert abs(w.rising_t_s - 1.0) < 0.02
    assert abs(w.falling_t_s - 3.0) < 0.02
    assert abs(w.duration_s - 2.0) < 0.02
    assert w.index == 0


def test_finds_multiple_windows_with_gaps():
    fs = 100.0
    sync = _sync_with_windows(1500, fs, [
        (1.0, 3.0),    # window 0: 2 s
        (5.0, 7.5),    # window 1: 2.5 s
        (9.0, 13.0),   # window 2: 4 s
    ])
    df = _df(sync, fs)
    wins = sync_align.find_sync_windows(df)
    assert len(wins) == 3
    assert [w.index for w in wins] == [0, 1, 2]
    expected_durations = [2.0, 2.5, 4.0]
    for w, expected in zip(wins, expected_durations):
        assert abs(w.duration_s - expected) < 0.02


def test_drops_trailing_unclosed_window():
    """If sync goes high and the recording ends before it goes low,
    that window is incomplete and must be dropped."""
    fs = 100.0
    n = 500
    sync = np.zeros(n, dtype=float)
    sync[100:300] = 1.0    # complete window 0: rise@1s, fall@3s
    sync[400:] = 1.0       # rises but never falls before EOF
    df = _df(sync, fs)
    wins = sync_align.find_sync_windows(df)
    assert len(wins) == 1


def test_constant_signal_yields_no_windows():
    fs = 100.0
    df = _df(np.zeros(300), fs)
    assert sync_align.find_sync_windows(df) == []
    df_high = _df(np.ones(300), fs)
    assert sync_align.find_sync_windows(df_high) == []


def test_no_sync_column_yields_no_windows():
    df = pd.DataFrame({"Time_ms": np.arange(100) * 10.0,
                       "L_ActForce_N": np.zeros(100)})
    assert sync_align.find_sync_windows(df) == []


def test_handles_analog_sync_via_threshold():
    """Sync that ramps (analog TTL) — threshold at midpoint must
    still produce one window per pulse. Pulses are 0.4 s here to
    cover the analog-detection mechanic; pass min_duration_s=0.0 so
    the phantom-pulse filter doesn't drop them — this test isn't
    about phantom rejection."""
    fs = 1000.0
    n = 5000
    t = np.arange(n) / fs
    # 3 pulses, each 0.4 s wide, separated by 0.6 s gaps
    sync = np.zeros(n)
    for start in (0.5, 1.5, 2.5):
        sync[(t >= start) & (t < start + 0.4)] = 5.0  # analog scale
    df = _df(sync, fs)
    wins = sync_align.find_sync_windows(df, min_duration_s=0.0)
    assert len(wins) == 3


# ============================================================
# Slicing + rebasing
# ============================================================

def test_extract_window_slice_returns_correct_rows():
    fs = 100.0
    sync = _sync_with_windows(1000, fs, [(1.0, 4.0)])
    df = _df(sync, fs, value=np.arange(1000, dtype=float))
    wins = sync_align.find_sync_windows(df)
    sliced = sync_align.extract_window_slice(df, wins[0])
    # Should cover ~3 s × 100 Hz = 300 rows
    assert 295 <= len(sliced) <= 305
    # First sample's t_window_s ≈ 0
    assert abs(sliced["t_window_s"].iloc[0]) < 0.02
    # Last sample's t_window_s ≈ window duration
    assert abs(sliced["t_window_s"].iloc[-1] - wins[0].duration_s) < 0.05


def test_align_to_window_start_zeroes_rising_edge():
    fs = 100.0
    sync = _sync_with_windows(800, fs, [(2.0, 5.0)])
    df = _df(sync, fs)
    wins = sync_align.find_sync_windows(df)
    aligned = sync_align.align_to_window_start(df, wins[0])
    assert "t_aligned" in aligned.columns
    # The sample AT the rising edge should have t_aligned ≈ 0
    rise_sample = wins[0].sample_rising
    assert abs(aligned["t_aligned"].iloc[rise_sample]) < 0.02


# ============================================================
# resample_to_grid
# ============================================================

def test_resample_preserves_count_and_endpoints():
    n = 500
    fs_src = 100.0
    t_src = np.arange(n) / fs_src
    df = pd.DataFrame({"t_aligned": t_src, "y": np.sin(2 * np.pi * t_src)})
    out = sync_align.resample_to_grid(df, target_fs=1000.0,
                                       t_min=0.0, t_max=4.0)
    assert len(out) == 4001
    assert abs(out["t_aligned"].iloc[0]) < 1e-9
    assert abs(out["t_aligned"].iloc[-1] - 4.0) < 1e-9


def test_resample_yields_nan_outside_source_range():
    df = pd.DataFrame({"t_aligned": [0.0, 1.0, 2.0],
                       "y": [0.0, 10.0, 20.0]})
    out = sync_align.resample_to_grid(df, target_fs=10.0,
                                       t_min=-1.0, t_max=3.0)
    assert np.isnan(out["y"].iloc[0])    # before source start
    assert np.isnan(out["y"].iloc[-1])   # after source end


# ============================================================
# Multi-source alignment on same window index
# ============================================================

def test_align_two_sources_on_same_window():
    """Robot @ 111 Hz with windows at 1.0–3.0 and 5.0–8.0;
    Motion @ 1000 Hz with windows at 1.5–3.5 and 5.5–8.5
    (different absolute clock — same physical events).

    Aligning on window 0 must anchor each source's first rising edge
    at t_aligned = 0 and produce a common grid in [0, ~min duration]."""
    fs_r = 111.0
    n_r = int(10.0 * fs_r)
    sync_r = _sync_with_windows(n_r, fs_r, [(1.0, 3.0), (5.0, 8.0)])
    robot = _df(sync_r, fs_r,
                L_ActForce_N=np.sin(2 * np.pi * np.arange(n_r) / fs_r) * 30 + 50)

    fs_m = 1000.0
    n_m = 10000
    sync_m = _sync_with_windows(n_m, fs_m, [(1.5, 3.5), (5.5, 8.5)])
    motion = pd.DataFrame({
        "Time": np.arange(n_m) / fs_m,
        "Sync": sync_m,
        "FP1_Fz": np.cos(2 * np.pi * np.arange(n_m) / fs_m) * 200 + 600,
    })

    aligned = sync_align.align_sources_on_window(
        {"robot": robot, "motion": motion},
        window_idx=0, target_fs=500.0,
    )
    # Both sources have window 0 → both grids present
    assert "robot" in aligned.grids
    assert "motion" in aligned.grids
    # Grid is the analysis window only: t = 0 at rising edge, t_max
    # = the shorter of the two window durations.
    grid_t = aligned.grids["robot"]["t_aligned"].to_numpy()
    assert len(grid_t) == aligned.n_grid_samples
    assert abs(grid_t[0] - 0.0) < 1e-9          # rising edge anchor
    # Both sources' grids share the exact same time axis
    np_grid_motion = aligned.grids["motion"]["t_aligned"].to_numpy()
    np.testing.assert_allclose(grid_t, np_grid_motion)
    # Grid duration = shorter window (here both are 2 s)
    assert 1.95 < (grid_t[-1] - grid_t[0]) < 2.05


def test_align_raises_when_window_missing_in_a_source():
    """Asking for window 5 when only 1 window exists in one source
    must raise — caller needs to know which source is short."""
    fs = 100.0
    short = _df(_sync_with_windows(500, fs, [(1.0, 3.0)]), fs)
    long  = _df(_sync_with_windows(2000, fs, [(1.0, 3.0), (5.0, 7.0),
                                              (10.0, 12.0)]), fs)
    with pytest.raises(ValueError):
        sync_align.align_sources_on_window(
            {"short": short, "long": long},
            window_idx=5, target_fs=200.0,
        )


def test_align_warns_for_unsync_source_but_continues_with_synced():
    """A loadcell-style source with no Sync column should be
    skipped with a warning, not crash the alignment of the rest."""
    fs = 100.0
    robot = _df(_sync_with_windows(800, fs, [(1.0, 5.0)]), fs)
    loadcell = pd.DataFrame({"time": np.linspace(0, 5, 50),
                              "applied_N": np.linspace(0, 50, 50)})
    aligned = sync_align.align_sources_on_window(
        {"robot": robot, "loadcell": loadcell},
        window_idx=0, target_fs=200.0,
    )
    assert "robot" in aligned.grids
    assert "loadcell" not in aligned.grids
    assert any("loadcell" in w for w in aligned.warnings)


def test_align_uses_shortest_window_duration():
    """Grid range = min(window_durations). Source with shorter window
    determines the analysis range, longer source's window gets
    truncated. Use `min_duration_s=0.0` so the 100 ms test window
    isn't filtered as a phantom — this test is about alignment
    range, not phantom rejection."""
    fs = 100.0
    a = _df(_sync_with_windows(500, fs, [(0.5, 0.6)]), fs)   # 100 ms window
    b = _df(_sync_with_windows(2000, fs, [(0.5, 5.0)]), fs)  # 4.5 s window
    aligned = sync_align.align_sources_on_window(
        {"a": a, "b": b},
        window_idx=0, target_fs=500.0,
        min_duration_s=0.0,
    )
    # t_max should be ~0.1 s (the shorter window), not 4.5 s.
    assert 0.05 < aligned.t_max_s < 0.15


def test_align_three_sources():
    """Robot + Motion + EMG-only motion (no Loadcell) — all three
    have sync, all three should land on the common grid."""
    fs1 = 111.0; fs2 = 1000.0; fs3 = 2000.0
    n1 = int(8 * fs1); n2 = int(8 * fs2); n3 = int(8 * fs3)
    a = _df(_sync_with_windows(n1, fs1, [(1.0, 4.0)]), fs1,
            x=np.zeros(n1))
    b = _df(_sync_with_windows(n2, fs2, [(1.5, 4.5)]), fs2,
            y=np.zeros(n2))
    c = _df(_sync_with_windows(n3, fs3, [(2.0, 5.0)]), fs3,
            z=np.zeros(n3))
    aligned = sync_align.align_sources_on_window(
        {"robot": a, "motion": b, "emg": c},
        window_idx=0, target_fs=500.0,
    )
    assert set(aligned.grids.keys()) == {"robot", "motion", "emg"}


# ============================================================
# Half-open boundary + multi-window edge cases
#   The contract is `[rising, falling)` — rising IS in the window,
#   falling sample is NOT. These tests pin every corner so a future
#   refactor that drops one sample at the start (off-by-one) or
#   includes the falling sample (closes the interval) turns red.
# ============================================================

def _df_from_sync(values: list[int], fs: float = 100.0) -> pd.DataFrame:
    n = len(values)
    return pd.DataFrame({"Time_s": np.arange(n) / fs,
                         "Sync": np.asarray(values, dtype=float)})


class TestHalfOpenBoundary:
    """These tests probe raw detection mechanics — half-open boundary,
    rising/falling sample inclusion. The synthetic signals are tens
    of samples long (well below the phantom-filter default of 0.5 s),
    so they call `find_sync_windows_raw` to bypass duration filtering.
    Phantom-pulse filtering itself is covered in TestPhantomPulseFilter.
    """
    def test_falling_sample_is_excluded(self):
        """The sample at sample_falling must hold a LOW value."""
        df = _df_from_sync([0, 0, 1, 1, 1, 0, 0])
        ws = sync_align.find_sync_windows_raw(df, "Sync")
        assert len(ws) == 1
        # Slice covers indices 2,3,4 — all HIGH; index 5 (falling) is LOW.
        assert ws[0].sample_rising == 2
        assert ws[0].sample_falling == 5
        assert df["Sync"].iat[5] == 0.0

    def test_rising_sample_is_included(self):
        """The sample at sample_rising must hold a HIGH value."""
        df = _df_from_sync([0, 0, 1, 1, 1, 0, 0])
        ws = sync_align.find_sync_windows_raw(df, "Sync")
        assert df["Sync"].iat[ws[0].sample_rising] == 1.0

    def test_extract_slice_contains_only_high_samples(self):
        df = _df_from_sync([0, 1, 1, 1, 0, 0, 1, 1, 0])
        ws = sync_align.find_sync_windows_raw(df, "Sync")
        for w in ws:
            sub = sync_align.extract_window_slice(df, w)
            assert (sub["Sync"] > 0.5).all(), (
                f"window {w.index} slice contains LOW sample(s) — "
                f"half-open contract violated"
            )

    def test_single_sample_window_supported(self):
        """A 1-sample HIGH pulse → 1-sample slice."""
        df = _df_from_sync([0, 0, 1, 0, 0])
        ws = sync_align.find_sync_windows_raw(df, "Sync")
        assert len(ws) == 1
        assert ws[0].sample_falling - ws[0].sample_rising == 1
        sub = sync_align.extract_window_slice(df, ws[0])
        assert len(sub) == 1
        assert sub["Sync"].iat[0] == 1.0


class TestMultiWindowDistinguishability:
    """Same disclaimer as TestHalfOpenBoundary — these tests verify
    rising-to-falling pairing logic, not duration filtering."""
    def test_three_windows_have_disjoint_slices(self):
        df = _df_from_sync(
            [0, 0, 1, 1, 0,  1, 1, 1, 1, 0,  0, 1, 1, 1, 1, 1, 0,  0, 0]
        )
        ws = sync_align.find_sync_windows_raw(df, "Sync")
        assert len(ws) == 3
        idxs = sorted([(w.sample_rising, w.sample_falling) for w in ws])
        assert idxs == [(2, 4), (5, 9), (11, 16)]
        # Every pair of windows has zero sample overlap.
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                a = set(range(ws[i].sample_rising, ws[i].sample_falling))
                b = set(range(ws[j].sample_rising, ws[j].sample_falling))
                assert not (a & b), (
                    f"windows {i} and {j} overlap at samples {a & b}"
                )

    def test_adjacent_windows_with_one_sample_gap(self):
        """Two pulses separated by a single LOW sample must still be
        emitted as two distinct windows (no merging)."""
        df = _df_from_sync([0, 1, 1, 0, 1, 1, 1, 0, 0])
        ws = sync_align.find_sync_windows_raw(df, "Sync")
        assert len(ws) == 2
        assert ws[0].sample_falling <= ws[1].sample_rising

    def test_unclosed_trailing_pulse_dropped(self):
        """Rising edge at the tail with no closing falling → that
        pulse is incomplete and must be dropped, never returned."""
        df = _df_from_sync([0, 1, 1, 0, 0, 1, 1, 1])
        ws = sync_align.find_sync_windows_raw(df, "Sync")
        assert len(ws) == 1, (
            "incomplete trailing pulse must be dropped per CLAUDE.md spec"
        )
        assert ws[0].sample_falling == 3

    def test_born_high_recording_drops_the_pre_window(self):
        """Recording that starts already HIGH → the leading 'window'
        with no rising edge must be dropped; only the proper
        rising→falling pulse later in the file is returned."""
        df = _df_from_sync([1, 1, 1, 0, 0, 1, 1, 0])
        ws = sync_align.find_sync_windows_raw(df, "Sync")
        assert len(ws) == 1
        assert ws[0].sample_rising == 5

    def test_each_falling_paired_only_once(self):
        """Two risings must each pair with their **own** falling, not
        share one. Used-falling guard prevents double-pairing."""
        df = _df_from_sync(
            [0, 1, 1, 0,  1, 1, 1, 0,  1, 1, 0]
        )
        ws = sync_align.find_sync_windows_raw(df, "Sync")
        assert len(ws) == 3
        falling_samples = [w.sample_falling for w in ws]
        assert len(set(falling_samples)) == 3, (
            f"duplicate falling-edge pairing: {falling_samples}"
        )


# ============================================================
# Phantom-pulse rejection
#   The sync GPIO line on the H-Walker DAQ is shared with file-IO
#   handlers in the recording software. New File / Save File events
#   briefly toggle the line for milliseconds, producing fake "trial"
#   pulses that must NOT reach the analyzer. find_sync_windows takes
#   a min_duration_s threshold (default 0.5 s) that drops sub-
#   threshold pulses.
# ============================================================

def _df_with_pulses_at_seconds(pulses_s: list[tuple[float, float]],
                               fs: float = 1000.0,
                               total_s: float = 12.0) -> pd.DataFrame:
    """Build a DataFrame with explicit pulse intervals in seconds."""
    n = int(total_s * fs)
    t = np.arange(n) / fs
    sync = np.zeros(n, dtype=float)
    for r, f in pulses_s:
        sync[(t >= r) & (t < f)] = 1.0
    return pd.DataFrame({"Time_s": t, "Sync": sync})


class TestPhantomPulseFilter:
    def test_default_drops_phantom_pulses(self):
        """One real 4 s trial + three 10 ms phantom glitches. Default
        threshold (0.5 s) keeps only the trial."""
        df = _df_with_pulses_at_seconds([
            (0.50, 0.51),    # phantom 10 ms — file IO
            (1.00, 5.00),    # REAL trial 4 s
            (6.00, 6.005),   # phantom 5 ms
            (7.50, 7.515),   # phantom 15 ms
        ])
        ws = sync_align.find_sync_windows(df, "Sync")
        assert len(ws) == 1
        assert ws[0].duration_s == pytest.approx(4.0, abs=0.01)

    def test_zero_threshold_keeps_every_pulse(self):
        """min_duration_s=0.0 → no filtering, returns every pulse."""
        df = _df_with_pulses_at_seconds([
            (0.50, 0.51), (1.00, 5.00), (6.00, 6.005), (7.50, 7.515),
        ])
        ws = sync_align.find_sync_windows(df, "Sync", min_duration_s=0.0)
        assert len(ws) == 4

    def test_indices_reassigned_after_filter(self):
        """When phantoms surround the real trial, the surviving
        window must be reindexed to 0 — downstream code uses .index
        as a contiguous trial id."""
        df = _df_with_pulses_at_seconds([
            (0.10, 0.12),   # phantom — would have idx 0 raw
            (1.00, 5.00),   # REAL — would have idx 1 raw
            (8.00, 8.05),   # phantom — would have idx 2 raw
        ])
        ws = sync_align.find_sync_windows(df, "Sync")
        assert len(ws) == 1
        assert ws[0].index == 0, (
            "surviving real trial must be reindexed to 0; got "
            f"index={ws[0].index}"
        )

    def test_threshold_higher_than_real_trial_returns_empty(self):
        """If the user sets min_duration_s above the real trial's
        width, the trial gets filtered too — explicit signal so the
        operator can debug an over-aggressive threshold."""
        df = _df_with_pulses_at_seconds([(1.0, 3.0)])  # 2 s
        ws = sync_align.find_sync_windows(df, "Sync", min_duration_s=5.0)
        assert ws == []

    def test_raw_helper_ignores_threshold(self):
        df = _df_with_pulses_at_seconds([
            (0.50, 0.51), (1.00, 5.00), (6.00, 6.005),
        ])
        raw = sync_align.find_sync_windows_raw(df, "Sync")
        assert len(raw) == 3

    def test_two_real_trials_with_phantoms_between(self):
        """The realistic case: two operator-driven trials with file-
        IO phantoms scattered between/around them."""
        df = _df_with_pulses_at_seconds([
            (0.10, 0.12),   # phantom (file open)
            (1.00, 5.00),   # REAL trial 0 (4 s)
            (5.20, 5.22),   # phantom (auto-save)
            (7.00, 11.00),  # REAL trial 1 (4 s)
            (11.50, 11.51), # phantom (file close)
        ])
        ws = sync_align.find_sync_windows(df, "Sync")
        assert len(ws) == 2
        assert [w.index for w in ws] == [0, 1]
        # Durations clearly > threshold, in expected ranges
        assert all(3.5 < w.duration_s < 4.5 for w in ws)

    def test_threshold_exactly_at_pulse_width_keeps_pulse(self):
        """Boundary: a pulse exactly at the threshold survives
        (`>=` comparison, not strict >)."""
        df = _df_with_pulses_at_seconds([(1.0, 1.5)], fs=1000.0)  # 0.5s
        ws = sync_align.find_sync_windows(df, "Sync", min_duration_s=0.5)
        assert len(ws) == 1, "0.5 s pulse must survive 0.5 s threshold"


# ============================================================
# Sync column auto-detection — the user's H-Walker firmware names
# its sync channel `A7` (analog input 7), not `Sync`. Newer rigs
# may use `sync_signal` etc. The detector must find all of them.
# ============================================================

class TestSyncColumnAutoDetect:
    def test_detects_A7_uppercase(self):
        df = _df_with_pulses_at_seconds([(1.0, 5.0)])
        df = df.rename(columns={"Sync": "A7"})
        ws = sync_align.find_sync_windows(df)
        assert len(ws) == 1
        assert sync_align._find_sync_column(df) == "A7"

    def test_detects_a7_lowercase(self):
        df = _df_with_pulses_at_seconds([(1.0, 5.0)])
        df = df.rename(columns={"Sync": "a7"})
        ws = sync_align.find_sync_windows(df)
        assert len(ws) == 1

    def test_A7_takes_precedence_over_Sync_when_both_present(self):
        """If both columns exist, A7 wins (firmware-native)."""
        df = _df_with_pulses_at_seconds([(1.0, 5.0)])
        df = df.copy()
        df["A7"] = df["Sync"]   # firmware-native column
        df["Sync"] = 0.0        # generic column with no pulses
        assert sync_align._find_sync_column(df) == "A7"
        ws = sync_align.find_sync_windows(df)
        assert len(ws) == 1, "should use A7's pulses, not Sync's flat zeros"

    def test_detects_sync_signal_alias(self):
        df = _df_with_pulses_at_seconds([(1.0, 5.0)])
        df = df.rename(columns={"Sync": "sync_signal"})
        assert sync_align._find_sync_column(df) == "sync_signal"
        ws = sync_align.find_sync_windows(df)
        assert len(ws) == 1

    def test_returns_empty_when_no_known_sync_column(self):
        df = _df_with_pulses_at_seconds([(1.0, 5.0)])
        df = df.rename(columns={"Sync": "random_channel_42"})
        assert sync_align._find_sync_column(df) is None
        assert sync_align.find_sync_windows(df) == []
