"""H-Walker 9-token filename parser.

Canonical format (user-confirmed 2026-04-25):

    {date}_{condition}_{terrain}_{speed}_{project}_{subject}_{f1}_{f2}_{trial}.csv

with optional source prefix:

    Loadcell_<base>.csv     →  source = "loadcell"
    Motion_<base>.csv       →  source = "motion"
    <base>.csv              →  source = "robot"  (no prefix)

Tokens:
    date       6-digit YYMMDD            260427
    condition  alphabetic                TD
    terrain    alphabetic                level
    speed      decimal in m/s            1.0
    project    word, may contain '-'     H-Walker
    subject    s + 1-2 digits            s01 .. s10
    f1         alphabetic                prox / mid / dist / none
    f2         alphabetic                axial / radial / normal / WB
    trial      digits                    1, 2, 3 ...

Eight valid (f1, f2) combinations for this study:

    (none, normal)   baseline 1: walking, no Weight Bearing
    (none, WB)       baseline 2: WB only, no robot assist
    (prox, axial)    assist condition 1
    (prox, radial)   assist condition 2
    (mid,  axial)    assist condition 3
    (mid,  radial)   assist condition 4
    (dist, axial)    assist condition 5
    (dist, radial)   assist condition 6

The parser returns None for filenames that don't match — caller
falls back to column-signature detection.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional


SourcePrefix = Literal["robot", "loadcell", "motion"]


# --- Whitelisted condition labels (for validation) -------------

VALID_POSITIONS  = {"prox", "mid", "dist", "none"}
VALID_DIRECTIONS = {"axial", "radial", "normal", "WB"}

VALID_CONDITION_PAIRS = {
    ("none", "normal"),
    ("none", "WB"),
    ("prox", "axial"),
    ("prox", "radial"),
    ("mid",  "axial"),
    ("mid",  "radial"),
    ("dist", "axial"),
    ("dist", "radial"),
}


# --- Regex --------------------------------------------------

_FILENAME_RE = re.compile(
    r"""^
    (?:(?P<source_prefix>Loadcell|Motion)_)?         # optional Robot=none
    (?P<date>\d{6})_                                  # 260427
    (?P<condition>[A-Za-z]+)_                         # TD
    (?P<terrain>[A-Za-z]+)_                           # level
    (?P<speed>\d+(?:\.\d+)?)_                         # 1.0
    (?P<project>[A-Za-z][A-Za-z0-9\-]*)_              # H-Walker
    (?P<subject>s\d{1,2})_                            # s01
    (?P<f1>[A-Za-z]+)_                                # prox / none
    (?P<f2>[A-Za-z]+)_                                # axial / WB / normal
    (?P<trial>\d+)                                    # 1
    \.csv$
    """,
    re.VERBOSE,
)


# --- Result ------------------------------------------------

@dataclass(frozen=True)
class ParsedFilename:
    source_prefix: SourcePrefix    # robot / loadcell / motion
    date: str                      # YYMMDD
    condition: str                 # TD
    terrain: str                   # level
    speed_mps: float               # 1.0
    project: str                   # H-Walker
    subject: str                   # s01
    feat1: str                     # prox / mid / dist / none
    feat2: str                     # axial / radial / normal / WB
    trial_idx: int                 # 1, 2, 3
    raw: str                       # original filename

    @property
    def is_baseline(self) -> bool:
        return self.feat1 == "none"

    @property
    def is_assist(self) -> bool:
        return self.feat1 in {"prox", "mid", "dist"}

    @property
    def condition_label(self) -> str:
        """Stable cell key for grouping. Same across the 3 sources of
        one trial. Example: 's01_prox_axial_1' (no source prefix)."""
        return f"{self.subject}_{self.feat1}_{self.feat2}_{self.trial_idx}"

    @property
    def cell_key(self) -> tuple[str, str, str, int]:
        """Tuple form usable as dict key for trial pairing."""
        return (self.subject, self.feat1, self.feat2, self.trial_idx)

    def as_dict(self) -> dict:
        return {
            "source_prefix": self.source_prefix,
            "date": self.date,
            "condition": self.condition,
            "terrain": self.terrain,
            "speed_mps": self.speed_mps,
            "project": self.project,
            "subject": self.subject,
            "feat1": self.feat1,
            "feat2": self.feat2,
            "trial_idx": self.trial_idx,
            "is_baseline": self.is_baseline,
            "condition_label": self.condition_label,
            "raw": self.raw,
        }


# --- Parser entry point ------------------------------------

def parse(filename: str) -> Optional[ParsedFilename]:
    """Parse a filename into its tokens, or return None on failure.

    Validates that:
      - the regex matches all 9 tokens (+ optional source prefix)
      - (feat1, feat2) is one of the 8 valid condition pairs
        for the H-Walker study
    """
    base = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    m = _FILENAME_RE.match(base)
    if not m:
        return None

    feat1 = m.group("f1")
    feat2 = m.group("f2")
    if (feat1, feat2) not in VALID_CONDITION_PAIRS:
        # Token shape OK but condition pair unknown — treat as invalid
        # so detector falls through to columns instead of trusting bad
        # metadata.
        return None

    return ParsedFilename(
        source_prefix=(m.group("source_prefix") or "robot").lower(),  # type: ignore[arg-type]
        date=m.group("date"),
        condition=m.group("condition"),
        terrain=m.group("terrain"),
        speed_mps=float(m.group("speed")),
        project=m.group("project"),
        subject=m.group("subject").lower(),
        feat1=feat1,
        feat2=feat2,
        trial_idx=int(m.group("trial")),
        raw=base,
    )


def is_canonical(filename: str) -> bool:
    """True if the filename matches the H-Walker canonical 9-token
    convention with a valid condition pair."""
    return parse(filename) is not None
