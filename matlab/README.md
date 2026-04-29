# H-Walker MATLAB Analysis Toolbox

Stand-alone MATLAB port of the v3bWQ Python pipeline. Single-researcher
workflow: point at a folder of CSVs, get publication-grade figures +
per-stride tables + statistics out the other end.

## Why MATLAB

- Direct access to the data — no upload, no cache, no rebuild.
- `dbstop if error` + variable inspector beats juggling browser
  console + server stdout + localStorage.
- Signal Processing / Statistics toolboxes already do most of the
  heavy lifting; we just glue them together for H-Walker semantics.
- `exportgraphics` produces journal-grade PDF / EPS / PNG at exact
  mm + DPI — same quality bar the Python `publication_engine` was
  hitting.

## Install

Add the toolbox to your path. From the repo root:

```matlab
addpath(genpath(fullfile(pwd, 'matlab')));
```

Or copy the `matlab/+hwalker` folder anywhere on your MATLAB path.

## Quick start — folder-based (recommended)

This is the path that fixes the "uploading CSVs in MATLAB is annoying"
problem. Point at a folder containing one trial-set worth of CSVs and
the toolbox handles classification + pairing automatically.

```matlab
% UI prompt — opens uigetdir
hwalker.analyzeFolder();

% Or pass the path directly
result = hwalker.analyzeFolder('/Users/cbj/H-Walker/Pilot03');
```

The folder may contain any mix of Robot / Motion / Loadcell exports.
The classifier looks at:
1. The filename prefix (`Robot_`, `Motion_`, `Loadcell_`).
2. If that's ambiguous, the column signature
   (`L_ActForce_N` → robot, `FP1_Fz` → motion, `applied_N` → loadcell).

Trial pairing matches on `(subject, feat1, feat2, trial_idx)` parsed
from the canonical filename. Zero-padding is normalized so `s1` and
`s01` are the same subject.

Output (written to `<folder>/analysis_output/`):

```
analysis_output/
├── per_stride_table.csv         all strides × all metrics
├── stats_summary.txt            paired t / Welch / Cohen's d
├── Fig1_force_tracking_L.pdf    journal-ready figure
├── Fig2_force_tracking_R.pdf
├── Fig3_stride_length_trend.pdf
├── Fig4_force_avg_L_R.pdf
└── README.txt                   what each file contains
```

## Single-file usage

When you only want to inspect one recording:

```matlab
res = hwalker.analyzeFile('/path/to/Robot_high_0.CSV');
disp(res.left.strideLengths);       % per-stride lengths (m)
disp(res.left.forceTracking.rmse);  % cable force tracking RMSE
hwalker.plot.forceTracking(res, 'side', 'L');
```

## Architecture

```
+hwalker/
├── analyzeFolder.m         top-level, batch over a folder
├── analyzeFile.m           top-level, single CSV
│
├── +io/                    ingestion + classification
│   ├── loadCSV.m           readtable wrapper, header sanity
│   ├── detectSourceKind.m  filename + columns → robot|motion|loadcell
│   └── parseFilename.m     <subject>_<feat1>_<feat2>_<trial>
│
├── +sync/                  rising→falling window detection
│   ├── findColumn.m        A7 / Sync / sync_signal / Trigger / TTL
│   ├── findWindows.m       [rising, falling) + min_duration filter
│   └── extractWindow.m     slice rows by window, rebase t to 0
│
├── +stride/                gait cycle decomposition
│   ├── detectFromGCP.m     L_GCP / R_GCP active-segment starts
│   ├── lengthZUPT.m        ZUPT-corrected 2D velocity integration
│   └── lengthFromSpeed.m   scalar |v| fallback
│
├── +force/                 controller tracking
│   ├── tracking.m          per-stride RMSE / MAE / peak error
│   └── perStrideMetrics.m  peak / impulse / loading rate
│
├── +stats/                 hypothesis tests + effect sizes
│   ├── pairedT.m           Shapiro-fail → Wilcoxon fallback
│   ├── welchT.m
│   ├── anova1.m
│   ├── pearson.m
│   └── cohensD.m
│
├── +trial/                 multi-source pairing
│   ├── pairSources.m       (subject, feat1, feat2, idx) → Trial
│   └── crossSourceTable.m  per-stride DataFrame across sources
│
└── +plot/                  publication-grade figures
    ├── journalPresets.m    IEEE / Nature / APA / Elsevier / MDPI / JNER
    ├── forceTracking.m     Des vs Act on GCP axis
    ├── strideLengthTrend.m per-stride length + linear fit
    ├── forceAvg.m          stride-averaged L/R mean ± SD
    └── exportFigure.m      exact mm + DPI write to disk
```

## Sync contract (user-confirmed, 2026-04-25)

A sync window is `[rising-edge, falling-edge)` — half-open. The rising
edge is when the operator pressed the trigger button (trial start), the
falling edge is when they released (trial end). One recording can hold
N windows = N trials. Analysis uses **only data inside the window**;
samples outside are prep / rest and must not pollute results.

Phantom pulses (file-IO toggling the sync GPIO) shorter than 0.5 s are
filtered out by default. Override per-call with the `MinDurationS`
name-value argument.

## Testing

```matlab
runtests('matlab/tests')
```

The tests cover the same regressions the Python suite does — sync
window detection edge cases, half-open boundary, phantom filtering,
zero-pad robustness, velocity-column alias detection, force tracking
math.
