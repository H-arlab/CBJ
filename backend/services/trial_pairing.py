"""Pair Robot / Motion / Loadcell uploads into Trial objects.

A Trial is the smallest unit the analysis pipeline operates on:

    Trial = (subject, condition, trial_idx, sync_window_idx)
              + up to 3 paired source CSVs
              + the matching sync window in each source

A single uploaded CSV → 1+ Trials, depending on how many sync windows
the recording contains. Three matched recordings (Robot + Motion +
Loadcell, same filename modulo prefix) → one row per sync window
that's present in *both* Robot and Motion (Loadcell is unsynced and
attached at the trial-set level, not per-window).

Pairing key: `(subject, feat1, feat2, trial_idx, sync_window_idx)`.
Sources are looked up in the dataset registry by parsed filename.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from backend.services.filename_parser import ParsedFilename, parse
from backend.services.sync_align import SyncWindow, find_sync_windows


# --- Source descriptor (one CSV the registry knows about) ----

@dataclass
class SourceInfo:
    ds_id: str                          # registry id
    parsed: ParsedFilename
    sync_windows: list[SyncWindow] = field(default_factory=list)


# --- Trial dataclass ---------------------------------------

@dataclass
class Trial:
    """One trial = one sync window across paired Robot+Motion+Loadcell."""
    id: str
    subject: str
    feat1: str                          # prox / mid / dist / none
    feat2: str                          # axial / radial / normal / WB
    trial_idx: int                      # from filename
    sync_window_idx: int                # 0-based within recording
    # Paired dataset ids
    robot_ds_id: Optional[str] = None
    motion_ds_id: Optional[str] = None
    loadcell_ds_id: Optional[str] = None
    # Sync windows (None when source is missing or unsynced)
    robot_window: Optional[SyncWindow] = None
    motion_window: Optional[SyncWindow] = None
    # Validation flags
    warnings: list[str] = field(default_factory=list)

    @property
    def is_baseline(self) -> bool:
        return self.feat1 == "none"

    @property
    def condition_label(self) -> str:
        return f"{self.feat1}_{self.feat2}"

    @property
    def has_robot(self) -> bool:
        return self.robot_ds_id is not None

    @property
    def has_motion(self) -> bool:
        return self.motion_ds_id is not None

    @property
    def has_loadcell(self) -> bool:
        return self.loadcell_ds_id is not None

    @property
    def is_complete_dual(self) -> bool:
        """Robot + Motion both present, with the requested sync window
        in both. Loadcell absence is OK (it's optional per recording)."""
        return self.has_robot and self.has_motion \
            and self.robot_window is not None \
            and self.motion_window is not None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "subject": self.subject,
            "feat1": self.feat1,
            "feat2": self.feat2,
            "condition_label": self.condition_label,
            "trial_idx": self.trial_idx,
            "sync_window_idx": self.sync_window_idx,
            "robot_ds_id": self.robot_ds_id,
            "motion_ds_id": self.motion_ds_id,
            "loadcell_ds_id": self.loadcell_ds_id,
            "is_complete_dual": self.is_complete_dual,
            "warnings": self.warnings,
        }


# --- Pairing entry point ------------------------------------

def collect_sources(uploads: list[tuple[str, str, pd.DataFrame]]
                     ) -> dict[str, list[SourceInfo]]:
    """Parse filenames and detect sync windows for each upload.

    `uploads` = [(ds_id, filename, dataframe), ...]
    Returns a dict keyed by source_prefix (robot / motion / loadcell).
    Files whose names don't match the canonical convention are ignored.
    """
    by_kind: dict[str, list[SourceInfo]] = {
        "robot": [], "motion": [], "loadcell": [],
    }
    for ds_id, filename, df in uploads:
        parsed = parse(filename)
        if parsed is None:
            continue
        windows = find_sync_windows(df) if parsed.source_prefix != "loadcell" else []
        info = SourceInfo(ds_id=ds_id, parsed=parsed, sync_windows=windows)
        by_kind[parsed.source_prefix].append(info)
    return by_kind


def pair_into_trials(uploads: list[tuple[str, str, pd.DataFrame]]
                      ) -> list[Trial]:
    """Group uploads into Trials.

    Algorithm:
      1. Parse all filenames; skip non-canonical.
      2. Group by (subject, feat1, feat2, trial_idx) → recording set.
      3. For each set, take min(robot.windows, motion.windows) and
         emit one Trial per sync-window-index. Loadcell (if present
         in the same set) attaches at the trial-set level.
      4. Mismatched window counts produce a per-trial warning.
    """
    by_kind = collect_sources(uploads)
    by_recording: dict[tuple[str, str, str, int], dict[str, SourceInfo]] = {}
    for kind, infos in by_kind.items():
        for info in infos:
            key = info.parsed.cell_key
            by_recording.setdefault(key, {})[kind] = info

    trials: list[Trial] = []
    for (subject, f1, f2, trial_idx), kind_map in sorted(by_recording.items()):
        robot = kind_map.get("robot")
        motion = kind_map.get("motion")
        loadcell = kind_map.get("loadcell")

        n_windows_robot  = len(robot.sync_windows)  if robot  else 0
        n_windows_motion = len(motion.sync_windows) if motion else 0

        # We can only emit Trial objects for windows that appear in both
        # Robot and Motion. If only one source has windows we still emit
        # one Trial per its own windows but flag the missing partner.
        n_windows = max(n_windows_robot, n_windows_motion, 1)
        if n_windows_robot and n_windows_motion \
                and n_windows_robot != n_windows_motion:
            # Mismatched window counts — pair what we can; emit a
            # warning on every trial in this recording.
            n_windows = min(n_windows_robot, n_windows_motion)
            mismatch_warning = (
                f"sync window count mismatch: Robot has {n_windows_robot}, "
                f"Motion has {n_windows_motion}. Pairing first {n_windows} "
                f"only — verify operator pressed sync simultaneously across rigs."
            )
        else:
            mismatch_warning = None

        for win_idx in range(n_windows):
            t = Trial(
                id=f"{subject}_{f1}_{f2}_t{trial_idx}_w{win_idx}",
                subject=subject,
                feat1=f1, feat2=f2,
                trial_idx=trial_idx,
                sync_window_idx=win_idx,
                robot_ds_id=robot.ds_id if robot else None,
                motion_ds_id=motion.ds_id if motion else None,
                loadcell_ds_id=loadcell.ds_id if loadcell else None,
                robot_window=(robot.sync_windows[win_idx]
                              if robot and win_idx < len(robot.sync_windows)
                              else None),
                motion_window=(motion.sync_windows[win_idx]
                               if motion and win_idx < len(motion.sync_windows)
                               else None),
            )
            if not robot:
                t.warnings.append("missing Robot recording")
            if not motion:
                t.warnings.append("missing Motion recording")
            if mismatch_warning:
                t.warnings.append(mismatch_warning)
            if t.robot_window is None and robot is not None:
                t.warnings.append(f"Robot has no sync window {win_idx}")
            if t.motion_window is None and motion is not None:
                t.warnings.append(f"Motion has no sync window {win_idx}")
            trials.append(t)
    return trials


def summarize(trials: list[Trial]) -> dict:
    """Aggregate trial-set stats for the UI."""
    by_subj: dict[str, dict] = {}
    for t in trials:
        s = by_subj.setdefault(t.subject, {
            "n_trials": 0,
            "n_complete": 0,
            "conditions_seen": set(),
            "missing_motion": 0,
            "missing_robot": 0,
            "warnings": 0,
        })
        s["n_trials"] += 1
        if t.is_complete_dual:
            s["n_complete"] += 1
        s["conditions_seen"].add(t.condition_label)
        if not t.has_motion:
            s["missing_motion"] += 1
        if not t.has_robot:
            s["missing_robot"] += 1
        if t.warnings:
            s["warnings"] += 1
    for s in by_subj.values():
        s["conditions_seen"] = sorted(s["conditions_seen"])
    return {
        "n_trials_total": len(trials),
        "n_complete_dual": sum(1 for t in trials if t.is_complete_dual),
        "by_subject": by_subj,
    }
