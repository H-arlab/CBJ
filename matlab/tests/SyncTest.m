classdef SyncTest < matlab.unittest.TestCase
    %SYNCTEST  Regressions for hwalker.sync.* matching the Python
    %suite's contract:
    %   · half-open `[rising, falling)` slicing
    %   · phantom-pulse rejection at 0.5 s default
    %   · A7 column auto-detection (operator's hardware uses `A7`)
    %   · multi-window distinguishability + reindexing

    methods (Test)

        % ------------------------------------------------------------
        % Column detection
        % ------------------------------------------------------------
        function detectsA7_uppercase(tc)
            T = makeSyncTable(["A7"], [(0:99)'/100, ...
                double((0:99)' >= 20 & (0:99)' < 80)]);
            T.Properties.VariableNames = {'Time_s', 'A7'};
            tc.verifyEqual(hwalker.sync.findColumn( ...
                string(T.Properties.VariableNames)), "A7");
        end

        function A7_takes_precedence_over_Sync(tc)
            cols = ["Time_s","Sync","A7"];
            tc.verifyEqual(hwalker.sync.findColumn(cols), "A7");
        end

        function returnsEmpty_when_no_known_column(tc)
            cols = ["Time_s","random_channel_42"];
            tc.verifyEqual(hwalker.sync.findColumn(cols), "");
        end

        % ------------------------------------------------------------
        % Window detection — half-open boundary
        % ------------------------------------------------------------
        function detects_one_window(tc)
            % 4 s pulse from t=1 to t=5 in a 10 s recording at 100 Hz
            T = pulseTable(10, 100, [1.0 5.0]);
            ws = hwalker.sync.findWindows(T, 'MinDurationS', 0);
            tc.verifyEqual(numel(ws), 1);
            tc.verifyEqual(ws(1).risingTimeS,  1.00, 'AbsTol', 0.02);
            tc.verifyEqual(ws(1).fallingTimeS, 5.00, 'AbsTol', 0.02);
        end

        function falling_sample_is_LOW(tc)
            T = pulseTable(2, 100, [0.5 1.5]);
            ws = hwalker.sync.findWindows(T, 'MinDurationS', 0);
            tc.verifyEqual(numel(ws), 1);
            % Sample at sampleFalling must hold the LOW value.
            tc.verifyLessThan(T.Sync(ws(1).sampleFalling), 0.5);
        end

        function detects_three_disjoint_windows(tc)
            T = pulseTable(15, 100, [1 3], [4 7], [9 13]);
            ws = hwalker.sync.findWindows(T, 'MinDurationS', 0);
            tc.verifyEqual(numel(ws), 3);
            tc.verifyEqual([ws.index], [0 1 2]);
        end

        function unclosed_trailing_pulse_is_dropped(tc)
            % 100 Hz × 5 s = 500 samples; rise at sample 50, never
            % falls before the recording ends.
            n = 500; fs = 100;
            T = table((0:n-1)'/fs, zeros(n,1), ...
                'VariableNames', {'Time_s', 'Sync'});
            T.Sync(50:end) = 1;
            ws = hwalker.sync.findWindows(T, 'MinDurationS', 0);
            tc.verifyEmpty(ws);
        end

        % ------------------------------------------------------------
        % Phantom-pulse filter
        % ------------------------------------------------------------
        function default_drops_phantoms(tc)
            % One real 4 s trial + three short phantoms.
            T = pulseTable(13, 1000, ...
                [0.50 0.51], ...   % phantom 10 ms
                [1.00 5.00], ...   % REAL trial
                [6.00 6.005], ...  % phantom
                [7.50 7.515]);     % phantom
            ws = hwalker.sync.findWindows(T);  % default 0.5 s
            tc.verifyEqual(numel(ws), 1);
            tc.verifyEqual(ws(1).durationS, 4.0, 'AbsTol', 0.01);
        end

        function zero_threshold_keeps_every_pulse(tc)
            T = pulseTable(13, 1000, ...
                [0.50 0.51], [1.00 5.00], [6.00 6.005], [7.50 7.515]);
            ws = hwalker.sync.findWindows(T, 'MinDurationS', 0);
            tc.verifyEqual(numel(ws), 4);
        end

        function indices_are_reassigned_after_filter(tc)
            % phantoms surround the real trial — the survivor must
            % come back at idx == 0 (trial-id space stays contiguous).
            T = pulseTable(13, 1000, ...
                [0.10 0.12], [1.00 5.00], [8.00 8.05]);
            ws = hwalker.sync.findWindows(T);
            tc.verifyEqual(numel(ws), 1);
            tc.verifyEqual(ws(1).index, 0);
        end

        % ------------------------------------------------------------
        % Window slicing
        % ------------------------------------------------------------
        function extract_returns_only_HIGH_samples(tc)
            T = pulseTable(10, 100, [1 5]);
            ws = hwalker.sync.findWindows(T, 'MinDurationS', 0);
            sub = hwalker.sync.extractWindow(T, ws(1));
            tc.verifyTrue(all(sub.Sync > 0.5));
        end

        function extract_rebases_t_to_zero(tc)
            T = pulseTable(10, 100, [1 5]);
            ws = hwalker.sync.findWindows(T, 'MinDurationS', 0);
            sub = hwalker.sync.extractWindow(T, ws(1));
            tc.verifyEqual(sub.t_window_s(1), 0, 'AbsTol', 1e-9);
            tc.verifyLessThan(sub.t_window_s(end), 4.01);
        end
    end
end


% =====================================================================
function T = makeSyncTable(~, data)
T = array2table(data);
end


function T = pulseTable(durationS, fs, varargin)
% Build a table with Time_s and Sync columns; varargin is a list
% of [rise fall] intervals.
n = round(durationS * fs);
t = (0:n-1)' / fs;
sync = zeros(n, 1);
for k = 1:numel(varargin)
    iv = varargin{k};
    sync(t >= iv(1) & t < iv(2)) = 1;
end
T = table(t, sync, 'VariableNames', {'Time_s','Sync'});
end
