function detection = detectSourceKind(filename, columns)
%DETECTSOURCEKIND  Classify a CSV as Robot / Motion / Loadcell / Unknown.
%
%   detection = hwalker.io.detectSourceKind(filename, columns)
%
%   Filename-first matching is the primary cue (the operator owns the
%   export naming). When the prefix is missing or the file came from a
%   third-party rig, the function falls back to column-signature
%   matching:
%
%     Robot     — has L_ActForce_N + (L_GCP or A7/Sync) — H-Walker
%                 firmware CSV.
%     Motion    — has at least one of force-plate (FP1_Fz, LeftFP_Force_Z),
%                 EMG channel (EMG_*), joint angle (RHipAngle_X), or
%                 marker triplet.
%     Loadcell  — small calibration log: time + force column
%                 (`applied_N`, `robot_reported_N`, `Force_N`), and
%                 NO H-Walker per-side signals.
%
%   Output struct:
%       detection.kind        "robot" | "motion" | "loadcell" | "unknown"
%       detection.confidence  0 (guess) to 1 (multiple strong cues)
%       detection.matchedCues string array — human-readable
%
%   Inputs:
%       filename — string or char, base name only
%       columns  — string array of column names

arguments
    filename (1,1) string
    columns (1,:) string
end

% Strip path so callers don't have to.
[~, base, ext] = fileparts(filename);
base = string(base) + string(ext);
columns = string(columns);

% Filename pass first.
[kind, cue] = matchFilename(base);
if kind ~= ""
    detection = struct( ...
        'kind', kind, ...
        'confidence', 1.0, ...
        'matchedCues', string(cue));
    return
end

% Column-signature pass.
robotCues   = matchRobotColumns(columns);
motionCues  = matchMotionColumns(columns);
loadcellCues = matchLoadcellColumns(columns);

scores = struct( ...
    'robot',    numel(robotCues), ...
    'motion',   numel(motionCues), ...
    'loadcell', numel(loadcellCues));

[topScore, topIdx] = max([scores.robot, scores.motion, scores.loadcell]);
labels = ["robot", "motion", "loadcell"];
allCues = {robotCues, motionCues, loadcellCues};

if topScore == 0
    detection = struct( ...
        'kind', "unknown", ...
        'confidence', 0.0, ...
        'matchedCues', string.empty);
    return
end

detection = struct( ...
    'kind', labels(topIdx), ...
    'confidence', min(1.0, topScore / 3), ...
    'matchedCues', allCues{topIdx});

end


% =====================================================================
function [kind, cue] = matchFilename(base)
%MATCHFILENAME  Filename prefix → source kind.
kind = ""; cue = "";
patterns = struct( ...
    'robot',    ["^robot[_\s\-]","^robot\b"], ...
    'loadcell', ["^loadcell[_\s\-]","^loadcell\b"], ...
    'motion',   ["^motion[_\s\-]","^motion\b"]);
for name = ["robot","loadcell","motion"]
    for pat = patterns.(name)
        if ~isempty(regexpi(base, pat, 'once'))
            kind = name;
            cue = sprintf("filename starts with '%s'", name);
            return
        end
    end
end
end


% =====================================================================
function cues = matchRobotColumns(cols)
cues = string.empty;
required = "L_ActForce_N";
if ~any(cols == required), return; end
cues(end+1) = required + " (robot loadcell)";

bonus = ["R_ActForce_N","L_GCP","R_GCP","L_Phase","R_Phase", ...
         "Sync","A7","L_Pitch","R_Pitch","L_DesForce_N","R_DesForce_N"];
hits = bonus(ismember(bonus, cols));
if ~isempty(hits)
    if numel(hits) <= 3
        cues(end+1) = sprintf("+%d robot signals: %s", ...
            numel(hits), strjoin(hits, ", "));
    else
        cues(end+1) = sprintf("+%d robot signals: %s…", ...
            numel(hits), strjoin(hits(1:3), ", "));
    end
end
end


% =====================================================================
function cues = matchMotionColumns(cols)
cues = string.empty;

% Force-plate channels (Vicon / V3D / Qualisys flavours).
fpRe = "^(?:fp\d?|plate\d?|forceplate\d?)[_\.]?(?:f[xyz]|m[xyz]|cop[xy])$";
v3dRe = "^(?:(?:left|right|l|r)fp\d?|fp[_\s]?(?:left|right))[_\.]?" + ...
        "(?:force|moment|cop)[_\.]?[xyz]$";
fpHits = cols(~cellfun('isempty', regexpi(cols, fpRe, 'once')) | ...
              ~cellfun('isempty', regexpi(cols, v3dRe, 'once')));
if ~isempty(fpHits)
    cues(end+1) = sprintf("force plate (%d ch): %s", ...
        numel(fpHits), strjoin(fpHits(1:min(2,numel(fpHits))), ", "));
end

% EMG envelope.
emgRe = "^(?:emg|im_emg|voltage)[_\.]";
emgHits = cols(~cellfun('isempty', regexpi(cols, emgRe, 'once')));
if ~isempty(emgHits)
    cues(end+1) = sprintf("EMG (%d ch)", numel(emgHits));
end

% Joint angles (Vicon Plug-in-Gait / V3D / Anybody).
jointRe = "^(?:[lr]?(?:hip|knee|ankle|pelvis|trunk|shoulder|elbow))" + ...
          "(?:angle|moment|power|reactionforce)?[_\.]?[xyz]?$";
jointHits = cols(~cellfun('isempty', regexpi(cols, jointRe, 'once')));
if ~isempty(jointHits)
    cues(end+1) = sprintf("joint kinematics (%d ch): %s", ...
        numel(jointHits), strjoin(jointHits(1:min(2,numel(jointHits))), ", "));
end
end


% =====================================================================
function cues = matchLoadcellColumns(cols)
cues = string.empty;
appliedAliases = ["applied_N","applied","reference_N","ref_N", ...
                  "hand_N","hand_applied_N"];
robotAliases   = ["robot_reported_N","robot_N","robot_force_N", ...
                  "loadcell_N","Force_N"];
hasApplied = any(ismember(appliedAliases, cols));
hasRobotF  = any(ismember(robotAliases,   cols));
hasHWPair  = any(cols == "L_ActForce_N") && any(cols == "R_ActForce_N");

if hasApplied
    cues(end+1) = "applied force column (calibration log)";
end
if hasRobotF && ~hasHWPair
    cues(end+1) = "robot-reported force without per-side H-Walker channels";
end
end
