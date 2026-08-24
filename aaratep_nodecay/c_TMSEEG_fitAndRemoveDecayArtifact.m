function [EEG, misc] = c_TMSEEG_fitAndRemoveDecayArtifact(varargin)
% DROP-IN NO-OP replacement for AARATEP's decay fitting and removal stage.
%
% WHY THIS EXISTS
%   AARATEP's real c_TMSEEG_fitAndRemoveDecayArtifact fits an exponential with fit() from
%   the Curve Fitting Toolbox. Without that toolbox MATLAB resolves fit() to an unrelated
%   function and the pipeline dies with
%
%       Incorrect number or types of inputs or outputs for function fit.
%
%   after epoching, filtering, the early eye ICA and SOUND have already run. This file
%   returns the data untouched so the remaining stages (artifact re-interpolation, notch,
%   ICA, ICLabel, low-pass, average reference) can complete.
%
% HOW IT IS USED
%   Set SKIP_DECAY_REMOVAL = true in AARATEP_implementation.m. That addpath's this folder
%   with '-begin', so this function shadows the real one for the rest of the session.
%   Nothing else on AARATEP's path carries this name, so the shadowing is unambiguous --
%   but it IS shadowing, so `which c_TMSEEG_fitAndRemoveDecayArtifact` is worth a glance
%   if behaviour ever surprises you. rmpath this folder to go back.
%
% WHAT IT COSTS
%   Decay removal is one of AARATEP's signature steps and a run without it is NOT full
%   AARATEP output -- do not report it as such. How much it costs on this dataset is an
%   open question worth checking rather than assuming: TEP_sensorspace.ipynb established
%   that the raw post-pulse deflection here is not an exponential decay (a single
%   exponential fits it at R2 = 0.07), which suggests the stage may have little to remove.
%   But AARATEP fits the decay in a projected component space, after SOUND, not on the raw
%   channel data, so that is a hint and not a substitute for measuring it. The honest fix
%   is to install the toolbox and compare a run with and without.
%
% The real function returns misc.hf (a QC figure the pipeline saves and closes) and
% misc.didRemoveDecay. Both are provided.

assert(nargin >= 1 && isstruct(varargin{1}), ...
    'First argument must be an EEG struct.');
EEG = varargin{1};

warning('AARATEP:decaySkipped', ...
    ['Decay fitting and removal SKIPPED -- the no-op shim in %s is shadowing AARATEP''s ' ...
     'real function. This run is not full AARATEP output.'], fileparts(mfilename('fullpath')));

hf = figure('Name', 'Decay removal skipped', 'Color', 'w', ...
            'Position', [50 50 620 220], 'Visible', 'on');
ha = axes('Parent', hf, 'Visible', 'off');
text(ha, 0.5, 0.55, 'DECAY FITTING AND REMOVAL SKIPPED', ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 13, ...
    'Interpreter', 'none');
text(ha, 0.5, 0.30, ['Curve Fitting Toolbox unavailable; no-op shim in use.' newline ...
                     'This run is not full AARATEP output.'], ...
    'HorizontalAlignment', 'center', 'FontSize', 10, 'Interpreter', 'none');

misc = struct('hf', hf, 'didRemoveDecay', false);
end
