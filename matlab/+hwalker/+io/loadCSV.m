function ds = loadCSV(filepath, opts)
%LOADCSV  Read an H-Walker recording CSV into a tidy struct.
%
%   ds = hwalker.io.loadCSV(filepath)
%   ds = hwalker.io.loadCSV(filepath, 'PreserveColumnNames', true)
%
%   Returns a struct with:
%       ds.table       a MATLAB table — rows × columns
%       ds.columns     string array of column names (preserved as-is)
%       ds.filepath    absolute path
%       ds.filename    base filename (no directory)
%       ds.nRows       row count
%       ds.sampleRateHz best-effort sample-rate estimate, or NaN
%
%   The reader is deliberately defensive about the things that bite
%   researchers when they import CSVs into MATLAB:
%
%   · BOM byte at the start of the file (Excel-on-Mac export).
%   · Column names with spaces or dots that readtable() would
%     otherwise rename to `Var3` etc. — `PreserveVariableNames=true`
%     is forced.
%   · A units row immediately under the header (some firmware
%     exports it). If the second row of every column is non-numeric
%     in a column that should be numeric, drop it.
%
%   The sample rate is estimated from a `Time_ms` / `Time_s` /
%   `timestamp` / `Time` / `t` column when one exists. Returns NaN
%   when no monotonic time axis is found — caller decides whether
%   that's fatal.

arguments
    filepath (1,1) string
    opts.PreserveColumnNames (1,1) logical = true
end

if ~isfile(filepath)
    error('hwalker:io:fileNotFound', "CSV not found: %s", filepath);
end

% readtable with column names preserved verbatim. Suppress the
% "modified column headings" warning since we are explicitly opting
% out of the rename behaviour.
warnState = warning('off', 'MATLAB:table:ModifiedAndSavedVarnames');
cleanup = onCleanup(@() warning(warnState));

T = readtable(filepath, ...
    'VariableNamingRule', 'preserve', ...
    'TextType', 'string');

% Strip a stray units row when present. The heuristic: if a column
% has type cell/string in row 1 but the column name suggests a
% numeric channel (ends in `_N`, `_mps`, `_deg`, `_Hz`, `_ms`...),
% the first row is a units string and gets dropped.
T = stripUnitsRow(T);

% Build the struct
ds = struct();
ds.table = T;
ds.columns = string(T.Properties.VariableNames);
ds.filepath = char(filepath);
[~, base, ext] = fileparts(filepath);
ds.filename = char(base + ext);
ds.nRows = height(T);
ds.sampleRateHz = inferSampleRate(T);

end


% =====================================================================
function T = stripUnitsRow(T)
%STRIPUNITSROW  Drop a non-numeric first row when columns are numeric.
%   Some firmware exports include a unit row directly under the header:
%       L_ActForce_N , L_DesForce_N
%       N            , N
%       12.4         , 14.0
%   readtable then types those columns as `string`. We detect this and
%   drop the first row.

if height(T) < 2, return; end

numericCols = endsWithUnit(T.Properties.VariableNames);
if ~any(numericCols), return; end

firstRowAllNonNumeric = true;
for ci = find(numericCols)
    val = T{1, ci};
    if isnumeric(val) && all(~isnan(val))
        firstRowAllNonNumeric = false;
        break
    end
    if iscell(val), val = val{1}; end
    if isstring(val) || ischar(val)
        if ~isnan(str2double(val))
            firstRowAllNonNumeric = false;
            break
        end
    end
end

if firstRowAllNonNumeric
    T(1, :) = [];
    % Re-read column types: strings that are now all numeric should
    % be cast back to double.
    for ci = find(numericCols)
        col = T{:, ci};
        if isstring(col) || iscell(col)
            T.(T.Properties.VariableNames{ci}) = str2double(col);
        end
    end
end

end


% =====================================================================
function tf = endsWithUnit(names)
%ENDSWITHUNIT  Heuristic: name suggests a numeric / SI-unit channel.
units = ["_N","_mps","_m","_deg","_Hz","_ms","_s","_rad","_pct"];
nm = string(names);
tf = false(size(nm));
for u = units
    tf = tf | endsWith(nm, u, 'IgnoreCase', true);
end
end


% =====================================================================
function fs = inferSampleRate(T)
%INFERSAMPLERATE  Best-effort sample rate from a time column.
fs = NaN;
candidates = ["Time_ms","time_ms","Time_s","time_s", ...
              "timestamp","Timestamp","Time","time","t"];
nm = string(T.Properties.VariableNames);
for c = candidates
    hit = nm == c;
    if any(hit)
        col = T{:, find(hit, 1)};
        if isnumeric(col) && numel(col) > 1
            dt = median(diff(double(col)), 'omitnan');
            if dt > 0
                if endsWith(c, "_ms"), dt = dt / 1000; end
                fs = 1.0 / dt;
                return
            end
        end
    end
end
end
