"""Detect which of the three H-Walker experiment sources a CSV is.

User contract (file naming):
    Robot CBJ_4.csv     /  robot_high_0.csv      → kind = "robot"
    Loadcell CBJ_5.csv  /  loadcell_low_30.csv   → kind = "loadcell"
    Motion CBJ.csv      /  motion_trial_3.csv    → kind = "motion"

Filename match is the primary cue (the user owns the export naming).
When a file is renamed or comes in from a third party, fall back to
**column-signature** matching:

  Robot     : has L_ActForce_N + (L_GCP or Sync)  — H-Walker firmware
              CSV.
  Motion    : has at least one of {force-plate Fz, EMG_*, marker
              X/Y/Z triplet, joint-angle column}. MoCap exports are
              very heterogeneous; the detector only requires ONE of
              these signatures.
  Loadcell  : tiny calibration log — typically just `time` + a force
              column (often `Force_N`, `applied_N`, `Robot_F_N`),
              and crucially NO H-Walker per-side signals.

Returns one of: "robot" | "loadcell" | "motion" | "unknown".
A confidence score in [0, 1] and a list of matched cues are
returned alongside so the UI can explain *why* a kind was chosen.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


SourceKind = Literal["robot", "loadcell", "motion", "unknown"]


@dataclass
class SourceDetection:
    kind: SourceKind
    confidence: float           # 0 (guess) … 1 (multiple strong cues)
    matched_cues: list[str]     # human-readable explanation

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "confidence": self.confidence,
            "matched_cues": self.matched_cues,
        }


# ---------------------------------------------------------------
# Filename rules
# ---------------------------------------------------------------

_FILENAME_PATTERNS: list[tuple[re.Pattern[str], SourceKind]] = [
    (re.compile(r"^\s*robot[_\s\-]", re.IGNORECASE),    "robot"),
    (re.compile(r"^\s*loadcell[_\s\-]", re.IGNORECASE), "loadcell"),
    (re.compile(r"^\s*motion[_\s\-]", re.IGNORECASE),   "motion"),
    # Permissive trailing-anything: "Robot.csv", "Robot_high.csv", etc.
    (re.compile(r"^\s*robot\b", re.IGNORECASE),    "robot"),
    (re.compile(r"^\s*loadcell\b", re.IGNORECASE), "loadcell"),
    (re.compile(r"^\s*motion\b", re.IGNORECASE),   "motion"),
]


def detect_from_filename(filename: str) -> SourceKind | None:
    base = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    for pat, kind in _FILENAME_PATTERNS:
        if pat.search(base):
            return kind
    return None


# ---------------------------------------------------------------
# Column-signature rules
# ---------------------------------------------------------------

# Robot firmware: per-side controller channels.
_ROBOT_REQUIRED = {"L_ActForce_N"}
_ROBOT_BONUS = {"R_ActForce_N", "L_GCP", "R_GCP", "L_Phase", "R_Phase",
                "Sync", "L_Pitch", "R_Pitch"}

# Motion capture: any of these patterns are sufficient.
#
# Recognized export formats:
#   - Vicon Nexus (Plug-in Gait):
#       FP1_Fz, FP2_COPx, RHipAngle_X, REMGEnvelope (rare)
#   - Visual 3D:
#       LeftFP_Force_Z, RightFP_COP_X, RHipAngle_X, RKneeMoment_Y,
#       RKneePower
#   - Anybody Modeling System:
#       Vastuslateralis_R_Force, BicepsFemoris_L_Activity,
#       Knee_R_ReactionForce_z
#   - Qualisys QTM analog (raw):
#       FP1_Fx, EMG_VL, EMG.VL, Voltage_VL, IM_EMG_1
# Two distinct patterns:
#   Vicon-style: `FP1_Fz`, `Plate2_COPx` — terminal token has a leading
#               F/M/COP prefix glued to xyz
#   V3D-style:   `LeftFP_Force_Z`, `RightFP_COP_X` — `Force`/`Moment`/`COP`
#               and the axis are separated by `_`
_FORCE_PLATE_RE_VICON = re.compile(
    r"""^(?:fp\d? | plate\d? | forceplate\d?)
        [_\.\s]?
        (?:f[xyz] | m[xyz] | cop[xy])
        $""",
    re.IGNORECASE | re.VERBOSE,
)
_FORCE_PLATE_RE_V3D = re.compile(
    r"""^(?:
        (?:left|right|l|r)fp\d?     # LeftFP, RightFP, RFP1
        | fp[_\s]?(?:left|right)    # FP_Left
    )
    [_\.\s]?
    (?:force | moment | cop)
    [_\.\s]?
    [xyz]
    $""",
    re.IGNORECASE | re.VERBOSE,
)


def _is_force_plate_col(col: str) -> bool:
    return bool(_FORCE_PLATE_RE_VICON.match(col)
                or _FORCE_PLATE_RE_V3D.match(col))
_EMG_RE = re.compile(
    r"""^(?:
        emg[_\.\s]?[a-z]+           # EMG_VL, EMG.VL
        | [a-z]{2,5}[_\.\s]?emg     # VL_EMG, BF.EMG
        | voltage[_\.\s]?[a-z]+     # Voltage_VL
        | im[_\.\s]?emg[_\.\s]?\d+  # IM_EMG_1 (Delsys Trigno indexed)
    )""",
    re.IGNORECASE | re.VERBOSE,
)
_MARKER_TRIPLET  = re.compile(r"^[A-Z][A-Za-z0-9]{1,8}_[XYZ]$")
_JOINT_ANGLE_RE  = re.compile(
    r"""(hip|knee|ankle|pelvis|shoulder|elbow|trunk)
        [_\s]?angle[_\s]?[xyz]?""",
    re.IGNORECASE | re.VERBOSE,
)
_JOINT_MOMENT_RE = re.compile(
    r"""(hip|knee|ankle|pelvis|shoulder|elbow)
        [_\s]?moment[_\s]?[xyz]?""",
    re.IGNORECASE | re.VERBOSE,
)
_JOINT_POWER_RE  = re.compile(
    r"""(hip|knee|ankle)[_\s]?power""",
    re.IGNORECASE | re.VERBOSE,
)
# Anybody Modeling System patterns.
#  - Muscle force: e.g. `Vastuslateralis_R_Force`, `Bicepsfemoris_L`
#  - Muscle activity: `<MuscleName>_R_Activity`
#  - Joint reaction force: `Knee_R_ReactionForce_z`
_ANYBODY_MUSCLE_RE = re.compile(
    r"""^(?:
        vastus(?:lateralis|medialis|intermedius)
        | bicepsfemoris
        | semitendinosus
        | semimembranosus
        | rectusfemoris
        | gastrocnemius(?:medialis|lateralis)?
        | soleus
        | tibialisanterior
        | gluteus(?:maximus|medius|minimus)
    )
    _[lr]
    (?:_(?:force|activity))?
    $""",
    re.IGNORECASE | re.VERBOSE,
)
_ANYBODY_REACTION_RE = re.compile(
    r"""^(hip|knee|ankle|pelvis)_[lr]_reactionforce_[xyz]$""",
    re.IGNORECASE | re.VERBOSE,
)
_VICON_TRIGGER_RE = re.compile(r"^(sync|trigger|ttl|analog\d+)$", re.IGNORECASE)

# Loadcell calibration: simple manual-press log.
_LOADCELL_FORCE_RE = re.compile(
    r"^(applied|reference|hand|calib|robot)[_\s]?(force|n|f|load)",
    re.IGNORECASE,
)


def detect_from_columns(columns: list[str]) -> tuple[SourceKind, list[str]]:
    cols = set(columns)
    cues: list[str] = []

    # ---- Robot ----
    if _ROBOT_REQUIRED.issubset(cols):
        cues.append("L_ActForce_N (robot loadcell)")
        bonus = sorted(_ROBOT_BONUS & cols)
        if bonus:
            cues.append(f"+{len(bonus)} robot signals: {', '.join(bonus[:4])}…"
                        if len(bonus) > 4 else f"+robot signals: {', '.join(bonus)}")
        return "robot", cues

    # ---- Motion ----
    fp_cols       = [c for c in columns if _is_force_plate_col(c)]
    emg_cols      = [c for c in columns if _EMG_RE.match(c)]
    marker_cols   = [c for c in columns if _MARKER_TRIPLET.match(c)]
    angle_cols    = [c for c in columns if _JOINT_ANGLE_RE.search(c)]
    moment_cols   = [c for c in columns if _JOINT_MOMENT_RE.search(c)]
    power_cols    = [c for c in columns if _JOINT_POWER_RE.search(c)]
    muscle_cols   = [c for c in columns if _ANYBODY_MUSCLE_RE.match(c)]
    reaction_cols = [c for c in columns if _ANYBODY_REACTION_RE.match(c)]
    motion_signals = bool(
        fp_cols or emg_cols or marker_cols or angle_cols
        or moment_cols or power_cols or muscle_cols or reaction_cols
    )
    if motion_signals:
        if fp_cols:
            cues.append(f"force-plate: {', '.join(fp_cols[:3])}"
                        + (f" (+{len(fp_cols)-3} more)" if len(fp_cols) > 3 else ""))
        if emg_cols:
            cues.append(f"EMG: {', '.join(emg_cols[:3])}"
                        + (f" (+{len(emg_cols)-3} more)" if len(emg_cols) > 3 else ""))
        if marker_cols:
            cues.append(f"markers: {len(marker_cols)} XYZ columns")
        if angle_cols:
            cues.append(f"joint angles: {', '.join(angle_cols[:3])}")
        if moment_cols:
            cues.append(f"joint moments (V3D): {', '.join(moment_cols[:3])}")
        if power_cols:
            cues.append(f"joint powers (V3D): {', '.join(power_cols[:3])}")
        if muscle_cols:
            cues.append(f"Anybody muscles: {', '.join(muscle_cols[:3])}")
        if reaction_cols:
            cues.append(f"Anybody reaction forces: {', '.join(reaction_cols[:3])}")
        return "motion", cues

    # ---- Loadcell ----
    has_time = any(c.lower() in {"time", "time_ms", "time_s", "timestamp", "t"}
                   for c in columns)
    loadcell_force_cols = [c for c in columns if _LOADCELL_FORCE_RE.match(c)]
    if has_time and loadcell_force_cols and len(columns) <= 6:
        cues.append(f"calibration force columns: {', '.join(loadcell_force_cols)}")
        cues.append(f"only {len(columns)} columns total — looks like a hand-press log")
        return "loadcell", cues

    return "unknown", cues


# ---------------------------------------------------------------
# Combined entry-point
# ---------------------------------------------------------------

def detect_source_kind(filename: str, columns: list[str]) -> SourceDetection:
    """Detect source kind from filename + column signature.

    Filename match is high-confidence (1.0) when columns also agree
    with that kind. Filename without column agreement gets 0.6 (we
    trust the user but flag the mismatch as a cue). Column-only
    match gets 0.7. No match → unknown / 0.0.
    """
    fname_kind = detect_from_filename(filename)
    col_kind, col_cues = detect_from_columns(columns)

    if fname_kind and col_kind == fname_kind:
        return SourceDetection(
            kind=fname_kind,
            confidence=1.0,
            matched_cues=[f"filename starts with '{fname_kind}'", *col_cues],
        )
    if fname_kind and col_kind == "unknown":
        return SourceDetection(
            kind=fname_kind,
            confidence=0.6,
            matched_cues=[f"filename starts with '{fname_kind}'",
                          "no recognized column signature"],
        )
    if fname_kind and col_kind != fname_kind:
        # Disagreement — believe the columns, surface the conflict.
        return SourceDetection(
            kind=col_kind,
            confidence=0.5,
            matched_cues=[f"filename suggests '{fname_kind}'",
                          f"columns suggest '{col_kind}'",
                          *col_cues],
        )
    if col_kind != "unknown":
        return SourceDetection(
            kind=col_kind,
            confidence=0.7,
            matched_cues=col_cues,
        )
    return SourceDetection(kind="unknown", confidence=0.0, matched_cues=[])
