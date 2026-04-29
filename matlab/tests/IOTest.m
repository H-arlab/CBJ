classdef IOTest < matlab.unittest.TestCase
    %IOTEST  Regressions for hwalker.io.* — source detection,
    %filename parsing, robustness against the ways a real CSV
    %makes it into MATLAB.

    methods (Test)

        % ------------------------------------------------------------
        % Source detection — matches the Python TestSourceKind suite
        % ------------------------------------------------------------
        function filename_robot_prefix(tc)
            d = hwalker.io.detectSourceKind("Robot_S07_prox_axial_3.csv", ...
                ["Time_ms","L_ActForce_N","L_GCP"]);
            tc.verifyEqual(d.kind, "robot");
            tc.verifyGreaterThanOrEqual(d.confidence, 0.9);
        end

        function filename_motion_prefix(tc)
            d = hwalker.io.detectSourceKind("Motion_trial_3.csv", ...
                ["time","FP1_Fz","RHipAngle_X"]);
            tc.verifyEqual(d.kind, "motion");
        end

        function filename_loadcell_prefix(tc)
            d = hwalker.io.detectSourceKind("Loadcell_calib.csv", ...
                ["timestamp","applied_N","robot_reported_N"]);
            tc.verifyEqual(d.kind, "loadcell");
        end

        function columns_robot_signature(tc)
            % No filename prefix — must fall through to columns.
            d = hwalker.io.detectSourceKind("anonymous.csv", ...
                ["Time_ms","L_ActForce_N","R_ActForce_N", ...
                 "L_GCP","R_GCP","Sync"]);
            tc.verifyEqual(d.kind, "robot");
        end

        function columns_motion_force_plate(tc)
            d = hwalker.io.detectSourceKind("rec.csv", ...
                ["time","FP1_Fx","FP1_Fz","RHipAngle_X"]);
            tc.verifyEqual(d.kind, "motion");
        end

        function columns_loadcell_calib(tc)
            d = hwalker.io.detectSourceKind("calib.csv", ...
                ["timestamp","applied_N","robot_reported_N"]);
            tc.verifyEqual(d.kind, "loadcell");
        end

        function unknown_falls_through(tc)
            d = hwalker.io.detectSourceKind("random.csv", ...
                ["x","y","z"]);
            tc.verifyEqual(d.kind, "unknown");
            tc.verifyEqual(d.confidence, 0.0);
        end

        % ------------------------------------------------------------
        % Filename parsing — canonical
        % ------------------------------------------------------------
        function canonical_robot(tc)
            p = hwalker.io.parseFilename( ...
                "260427_TD_level_1.0_H-Walker_s01_prox_axial_3.csv");
            tc.verifyTrue(p.canonical);
            tc.verifyEqual(p.subject,  "s01");
            tc.verifyEqual(p.feat1,    "prox");
            tc.verifyEqual(p.feat2,    "axial");
            tc.verifyEqual(p.trialIdx, 3);
            tc.verifyEqual(p.sourcePrefix, "robot");
        end

        function canonical_motion_prefix(tc)
            p = hwalker.io.parseFilename( ...
                "Motion_260427_TD_level_1.0_H-Walker_s01_prox_axial_3.csv");
            tc.verifyEqual(p.sourcePrefix, "motion");
        end

        function zero_pad_subject_normalizes(tc)
            a = hwalker.io.parseFilename( ...
                "260427_TD_level_1.0_H-Walker_s1_prox_axial_3.csv");
            b = hwalker.io.parseFilename( ...
                "260427_TD_level_1.0_H-Walker_s01_prox_axial_3.csv");
            tc.verifyEqual(a.subject, b.subject);
            tc.verifyEqual(a.subject, "s01");
        end

        function zero_pad_trial_collapses(tc)
            a = hwalker.io.parseFilename( ...
                "260427_TD_level_1.0_H-Walker_s01_prox_axial_3.csv");
            b = hwalker.io.parseFilename( ...
                "260427_TD_level_1.0_H-Walker_s01_prox_axial_03.csv");
            tc.verifyEqual(a.trialIdx, b.trialIdx);
        end

        function cell_key_matches_across_padding(tc)
            r = hwalker.io.parseFilename( ...
                "260427_TD_level_1.0_H-Walker_s1_prox_axial_3.csv");
            m = hwalker.io.parseFilename( ...
                "Motion_260427_TD_level_1.0_H-Walker_s01_prox_axial_03.csv");
            tc.verifyEqual(r.cellKey, m.cellKey);
        end

        % ------------------------------------------------------------
        % Filename parsing — heuristic must not invent subject ids
        % ------------------------------------------------------------
        function robot_high_0_does_not_invent_subject(tc)
            p = hwalker.io.parseFilename("robot_high_0.CSV");
            tc.verifyEqual(p.subject, "");
        end

        function loadcell_low_30_does_not_invent_subject(tc)
            p = hwalker.io.parseFilename("loadcell_low_30.CSV");
            tc.verifyEqual(p.subject, "");
        end

        function explicit_s_prefix_still_parses(tc)
            p = hwalker.io.parseFilename("s07_pre_2024_05_01.csv");
            tc.verifyEqual(p.subject, "s07");
            tc.verifyEqual(p.condition, "Pre");
        end
    end
end
