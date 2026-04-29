function parsed = parseFilename(filename)
%PARSEFILENAME  Extract subject / condition / trial idx from a CSV name.
%
%   parsed = hwalker.io.parseFilename(filename)
%
%   Two-stage parser:
%     1. Strict canonical 9-token shape (used by trial pairing):
%          {date}_{cond}_{terrain}_{speed}_{project}_{subject}_
%          {feat1}_{feat2}_{trial}.csv
%        Optional source prefix `Robot_` / `Motion_` / `Loadcell_`.
%     2. If the canonical pattern doesn't match, a tolerant heuristic
%        guesses (subject_id, condition) for arbitrary names.
%
%   Output struct:
%       parsed.canonical    logical — did the strict pattern match?
%       parsed.sourcePrefix "robot" | "motion" | "loadcell"
%       parsed.subject      string, normalized to "s##" two-digit form
%       parsed.feat1        string ("prox" | "mid" | "dist" | "none")
%       parsed.feat2        string ("axial" | "radial" | "normal" | "WB")
%       parsed.trialIdx     numeric (zero-pad collapsed via int)
%       parsed.condition    string  (heuristic: "Pre", "Treadmill", ...)
%       parsed.cellKey      string — `<subject>_<feat1>_<feat2>_<idx>`
%                           used as the join key when pairing sources
%       parsed.raw          original basename
%
%   Fields that couldn't be inferred are "" (empty string) — never
%   silently invented. The operator can always type the correct value
%   in the dataset card; better blank than wrong.

arguments
    filename (1,1) string
end

% Strip directory + normalize extension casing.
[~, base, ext] = fileparts(filename);
raw = string(base) + string(ext);

parsed = struct( ...
    'canonical',    false, ...
    'sourcePrefix', "robot", ...
    'subject',      "", ...
    'feat1',        "", ...
    'feat2',        "", ...
    'trialIdx',     NaN, ...
    'condition',    "", ...
    'cellKey',      "", ...
    'raw',          raw);

% --- Stage 1: canonical 9-token ---------------------------------
canon = "^(?:(?<source>Loadcell|Motion)_)?" + ...
        "(?<date>\d{6})_" + ...
        "(?<cond>[A-Za-z]+)_" + ...
        "(?<terrain>[A-Za-z]+)_" + ...
        "(?<speed>\d+(?:\.\d+)?)_" + ...
        "(?<project>[A-Za-z][A-Za-z0-9\-]*)_" + ...
        "(?<subject>s\d{1,2})_" + ...
        "(?<f1>[A-Za-z]+)_" + ...
        "(?<f2>[A-Za-z]+)_" + ...
        "(?<trial>\d+)\.csv$";

m = regexpi(raw, canon, 'names', 'once');
if ~isempty(m)
    src = lower(string(m.source));
    if src == "", src = "robot"; end
    subjectRaw = lower(string(m.subject));
    digits = regexp(subjectRaw, "\d+", "match", "once");
    if digits ~= ""
        subject = "s" + sprintf("%02d", str2double(digits));
    else
        subject = subjectRaw;
    end
    parsed.canonical    = true;
    parsed.sourcePrefix = src;
    parsed.subject      = subject;
    parsed.feat1        = lower(string(m.f1));
    parsed.feat2        = string(m.f2);
    parsed.trialIdx     = str2double(m.trial);
    parsed.condition    = string(m.cond);
    parsed.cellKey      = sprintf("%s_%s_%s_%d", ...
        subject, parsed.feat1, parsed.feat2, parsed.trialIdx);
    return
end

% --- Stage 2: heuristic guess for arbitrary names ---------------
%   Conservative: bare 1-3 digit numbers are subject_id ONLY when
%   explicitly marked with `s` / `subj` / `subject`. Any trailing
%   number is a trial index, not a subject — `robot_high_0.CSV` and
%   `loadcell_low_30.CSV` end up with no auto-populated subject.
heuristics = [
    "(?<subject>s\d+)[_\-.]+(?<cond>pre|post|control|experimental|baseline|treatment|treadmill|overground|[a-z]{2,15})", ...
    "^(?<cond>pre|post|control|experimental|baseline|treatment|fast|slow|natural|[a-z]{3,10})[_\-.]+(?:subj(?:ect)?[_-]?)(?<subject>\d{1,3}|s\d+)", ...
    "^(?<cond>[a-z]{3,15})[_\-.]+(?<subject>s\d+)", ...
    "^(?<subject>\d{1,3})[_\-.]+(?<cond>[a-z]{3,15})$"
];
for pat = heuristics
    m = regexpi(raw, pat, 'names', 'once');
    if ~isempty(m)
        subj = lower(string(m.subject));
        digits = regexp(subj, "\d+", "match", "once");
        if startsWith(subj, "s") && digits ~= ""
            parsed.subject = "s" + sprintf("%02d", str2double(digits));
        else
            parsed.subject = subj;
        end
        parsed.condition = capitalize(string(m.cond));
        return
    end
end

% trial_N style — keep for recipe behaviour
m = regexpi(raw, "(?<trial>trial[_\-]?\d+)", 'names', 'once');
if ~isempty(m)
    parsed.subject = lower(string(m.trial));
end

end


function s = capitalize(s)
if strlength(s) == 0, return; end
s = upper(extractBefore(s, 2)) + lower(extractAfter(s, 1));
end
