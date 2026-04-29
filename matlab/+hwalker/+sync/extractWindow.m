function sub = extractWindow(T, window)
%EXTRACTWINDOW  Slice the rows of T that fall inside one sync window.
%
%   sub = hwalker.sync.extractWindow(T, window)
%
%   Returns a table containing rows [sampleRising, sampleFalling),
%   half-open per the sync contract — the rising sample is included,
%   the falling sample is excluded. Adds a `t_window_s` column with
%   time rebased so the rising edge sits at t = 0, which is what
%   downstream stride / force code expects.
%
%   `window` is one element of the struct array returned by
%   `hwalker.sync.findWindows`.

arguments
    T table
    window (1,1) struct
end

if window.sampleFalling <= window.sampleRising
    sub = T([], :);
    return
end

rows = window.sampleRising : (window.sampleFalling - 1);
rows(rows < 1 | rows > height(T)) = [];
sub = T(rows, :);

% Rebase time. Prefer an explicit time column when present so the
% caller can mix this rebased axis with absolute timestamps if they
% want; otherwise emit a synthetic axis based on the row index.
cols = string(T.Properties.VariableNames);
rebased = (0:height(sub)-1)' / inferFs(T);
for name = ["Time_ms","time_ms"]
    if any(cols == name)
        col = double(T.(name)(rows));
        rebased = (col - col(1)) / 1000;
        break
    end
end
for name = ["Time_s","time_s","Timestamp","timestamp","Time","time","t"]
    if any(cols == name)
        col = double(T.(name)(rows));
        rebased = col - col(1);
        break
    end
end

sub.t_window_s = rebased;

end


% =====================================================================
function fs = inferFs(T)
%INFERFS  Fallback sample-rate guess.
fs = 1000;   % final fallback
cols = string(T.Properties.VariableNames);
candidates = struct('Time_ms',1e-3, 'Time_s',1, 'timestamp',1, 't',1);
fields = fieldnames(candidates);
for i = 1:numel(fields)
    if any(cols == fields{i})
        col = double(T.(fields{i}));
        if numel(col) > 1
            dt = median(diff(col), 'omitnan') * candidates.(fields{i});
            if dt > 0
                fs = 1.0 / dt;
                return
            end
        end
    end
end
end
