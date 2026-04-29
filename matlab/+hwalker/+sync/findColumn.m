function name = findColumn(columns)
%FINDCOLUMN  Locate the sync / trigger column in a column list.
%
%   name = hwalker.sync.findColumn(columns)
%
%   Returns the first matching column name, or "" if none found.
%   Order of preference (first match wins):
%
%       A7          — H-Walker firmware (current — analog input 7)
%       a7
%       Sync        — generic / older builds
%       sync
%       sync_signal
%       Sync_Signal
%       Trigger
%       trigger
%       TTL
%       ttl
%
%   `A7` is checked first because that's what the operator's actual
%   hardware writes. If both `A7` and `Sync` exist (rare but possible
%   on hybrid rigs) the firmware-native `A7` wins — its pulse train
%   is what the operator-confirmed sync contract refers to.

arguments
    columns (1,:) string
end

candidates = ["A7", "a7", ...
              "Sync", "sync", ...
              "sync_signal", "Sync_Signal", ...
              "Trigger", "trigger", ...
              "TTL", "ttl"];

name = "";
for c = candidates
    hit = columns == c;
    if any(hit)
        name = columns(find(hit, 1));
        return
    end
end

end
