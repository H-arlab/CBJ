function windows = findWindows(T, opts)
%FINDWINDOWS  Detect every rising→falling sync window in a recording.
%
%   windows = hwalker.sync.findWindows(T)
%   windows = hwalker.sync.findWindows(T, 'SyncColumn', 'A7')
%   windows = hwalker.sync.findWindows(T, 'MinDurationS', 0.5)
%
%   `windows` is a struct array (possibly empty) with fields:
%       index           0-based trial id (contiguous, after filtering)
%       sampleRising    1-based row index where the window opens
%       sampleFalling   1-based row index where the window closes
%       risingTimeS     time at the rising sample (seconds)
%       fallingTimeS    time at the falling sample (seconds)
%       durationS       fallingTimeS - risingTimeS
%
%   Sync contract (operator-confirmed, 2026-04-25):
%       A sync window is `[rising, falling)` half-open. The rising
%       edge is when the operator pressed the button (trial start),
%       the falling edge is when they released (trial end). Analysis
%       must use only data inside that interval — outside is prep /
%       rest and would pollute results.
%
%   Phantom-pulse rejection:
%       The H-Walker DAQ shares its sync line with the recording
%       software's file-IO handlers, so New File / Save File events
%       briefly toggle the line for a few milliseconds. Pulses
%       narrower than `MinDurationS` (default 0.5 s) are dropped as
%       phantoms, then surviving windows are re-indexed 0..N-1 so
%       downstream code sees a contiguous trial-id space.
%
%       Pass `MinDurationS=0` to keep every detected pulse — useful
%       for debugging the recording line itself, never the right
%       setting for analysis.
%
%   Trailing rising-without-falling pulses (operator never released,
%   or recording stopped mid-trial) are dropped — incomplete windows.

arguments
    T table
    opts.SyncColumn (1,1) string = ""
    opts.MinDurationS (1,1) double = 0.5
end

windows = emptyWindowStruct();

% --- Resolve sync column ----------------------------------------
columns = string(T.Properties.VariableNames);
if opts.SyncColumn == ""
    syncCol = hwalker.sync.findColumn(columns);
else
    syncCol = opts.SyncColumn;
end
if syncCol == "" || ~any(columns == syncCol)
    return
end

% --- Resolve a seconds time axis --------------------------------
t = secondsAxis(T);
if numel(t) < 4
    return
end

% --- Detect rising / falling edges ------------------------------
s = double(T.(syncCol));
finiteMask = isfinite(s);
if nnz(finiteMask) < 4
    return
end
sFinite = s(finiteMask);
tFinite = t(finiteMask);
finiteIdx = find(finiteMask);

lo = min(sFinite); hi = max(sFinite);
if (hi - lo) < 1e-9
    return  % constant signal — no edges
end

threshold = (lo + hi) / 2;
high = sFinite > threshold;
d = diff(int8(high));
risingLocal  = find(d ==  1) + 1;
fallingLocal = find(d == -1) + 1;
if isempty(risingLocal) || isempty(fallingLocal)
    return
end

% --- Pair each rising with the next available falling -----------
%   used_falling guards against a single falling being shared by
%   two risings (which would happen with chattery edges).
raw = emptyWindowStruct();
usedFalling = 0;
for r = risingLocal(:)'
    candidates = fallingLocal(fallingLocal > r);
    candIdxInFalling = find(ismember(fallingLocal, candidates));
    candIdxInFalling = candIdxInFalling(candIdxInFalling > usedFalling);
    if isempty(candIdxInFalling)
        break
    end
    fIdx = candIdxInFalling(1);
    f = fallingLocal(fIdx);
    usedFalling = fIdx;

    raw(end+1) = struct( ...
        'index',         numel(raw), ...
        'sampleRising',  finiteIdx(r), ...
        'sampleFalling', finiteIdx(f), ...
        'risingTimeS',   tFinite(r), ...
        'fallingTimeS',  tFinite(f), ...
        'durationS',     tFinite(f) - tFinite(r));    %#ok<AGROW>
end

% --- Phantom-pulse rejection + re-indexing ----------------------
if opts.MinDurationS <= 0
    windows = raw;
    return
end
keep = arrayfun(@(w) w.durationS >= opts.MinDurationS, raw);
survivors = raw(keep);
for k = 1:numel(survivors)
    survivors(k).index = k - 1;
end
windows = survivors;

end


% =====================================================================
function w = emptyWindowStruct()
w = struct( ...
    'index', {}, ...
    'sampleRising', {}, ...
    'sampleFalling', {}, ...
    'risingTimeS', {}, ...
    'fallingTimeS', {}, ...
    'durationS', {});
end


% =====================================================================
function t = secondsAxis(T)
%SECONDSAXIS  Best-effort seconds axis from the table.
%   Prefer explicit time columns; fall back to a 1 kHz sample axis
%   so callers always get something workable.

cols = string(T.Properties.VariableNames);
candidates = struct( ...
    'Time_ms',   1e-3, ...
    'time_ms',   1e-3, ...
    'Time_s',    1.0, ...
    'time_s',    1.0, ...
    'timestamp', 1.0, ...
    'Timestamp', 1.0, ...
    'Time',      1.0, ...
    'time',      1.0, ...
    't',         1.0);
fields = fieldnames(candidates);
for i = 1:numel(fields)
    name = fields{i};
    if any(cols == name)
        col = double(T.(name));
        if numel(col) > 1 && any(isfinite(col))
            t = col * candidates.(name);
            return
        end
    end
end
% Last-resort fallback so detection still works on a CSV with no
% explicit time column.
t = (0:height(T)-1)' / 1000;
end
