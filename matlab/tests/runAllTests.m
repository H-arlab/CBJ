function results = runAllTests()
%RUNALLTESTS  Execute every MATLAB unit test under matlab/tests.
%
%   Usage from the repo root:
%       addpath(genpath('matlab'));
%       results = runAllTests();
%
%   Returns the standard `matlab.unittest.TestResult` array. Throws
%   on failure so this can be wired into CI / pre-commit hooks.

import matlab.unittest.TestSuite
import matlab.unittest.TestRunner
import matlab.unittest.plugins.TestRunProgressPlugin

thisDir  = fileparts(mfilename('fullpath'));
matlabDir = fileparts(thisDir);
addpath(genpath(matlabDir));

suite = TestSuite.fromFolder(thisDir, 'IncludingSubfolders', true);
runner = TestRunner.withTextOutput;
runner.addPlugin(TestRunProgressPlugin.withVerbosity(1));
results = runner.run(suite);

nFail = nnz([results.Failed]);
fprintf('\nMATLAB tests: %d passed, %d failed (%d total)\n', ...
    nnz([results.Passed]), nFail, numel(results));
if nFail > 0
    error('hwalker:tests:failed', '%d MATLAB test(s) failed.', nFail);
end
end
