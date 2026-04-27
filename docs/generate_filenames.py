\
"""H-Walker 실험 파일명 generator.

사용자가 실험 디자인 (피험자 수, condition, trial 수) 바뀌었을 때
이 스크립트만 수정하고 다시 돌리면 FILENAMES.md / FILENAMES.txt 갱신.

Backend wire-up 없음 — 순수 참조용 리스트.
"""
SUBJECTS  = [f"s{i:02d}" for i in range(1, 11)]
CONDITIONS = [
    ("none",  "normal"),
    ("none",  "WB"),
    ("prox",  "axial"),  ("prox",  "radial"),
    ("mid",   "axial"),  ("mid",   "radial"),
    ("dist",  "axial"),  ("dist",  "radial"),
]
TRIALS  = [1, 2, 3]
SOURCES = ["Robot", "Loadcell", "Motion"]

DATE      = "YYMMDD"
CONDITION = "TD"
TERRAIN   = "level"
SPEED     = "1.0"
PROJECT   = "H-Walker"


def make_name(source, subj, feat1, feat2, trial):
    base = f"{DATE}_{CONDITION}_{TERRAIN}_{SPEED}_{PROJECT}_{subj}_{feat1}_{feat2}_{trial}.csv"
    return base if source == "Robot" else f"{source}_{base}"


if __name__ == "__main__":
    for subj in SUBJECTS:
        for feat1, feat2 in CONDITIONS:
            for trial in TRIALS:
                for src in SOURCES:
                    print(make_name(src, subj, feat1, feat2, trial))
