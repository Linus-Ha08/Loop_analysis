%% AARATEP_implementation.m -- sensor-space TEPs via the AARATEP pipeline
%
% An independent alternative to TEP_sensorspace.ipynb. Same recordings, same blocks,
% same ROI, same figures and the same output tables -- but the cleaning is done by
% Cline et al.'s AARATEP pipeline (v2.1.1) instead of the notebook's own chain, so the
% two can be read against each other and a TEP feature that only survives in one of them
% can be treated with suspicion.
%
%   organize_neurone.m  ->  THIS SCRIPT  ->  per-block TEP curves + figures
%    (per-block .set +      (prepare, AARATEP,
%     events.tsv)            average)
%
% ========================================================================================
% WHAT FORMAT AARATEP EXPECTS, AND WHAT organize_neurone.m ACTUALLY WRITES
% ========================================================================================
% Checked against the pipeline source rather than the README, because the README's
% one-line example hides four incompatibilities. In short: the data is fine, the
% documented ENTRY POINT is not, and four preparation steps have to happen first.
%
%   1. c_TMSEEG_prepareForPreprocessing CANNOT READ .set FILES.
%      Its 'inputFilePath' switch handles exactly two extensions -- '.vhdr'
%      (BrainVision) and '.mat' -- and anything else hits `error('Unsupported input
%      type')`. organize_neurone.m writes EEGLAB .set/.fdt. So the file path route in the
%      README is closed to us. The same function does accept an ALREADY-LOADED struct via
%      'inputEEGs', which is the route taken below: pop_loadset ourselves, hand over the
%      struct. Nothing is lost -- 'inputEEGs' reaches the identical trimming and
%      validation code.
%
%   2. AUTOMATIC PULSE-EVENT DETECTION WOULD THROW. With 'pulseEvent','auto' the helper
%      takes the most frequent event type and then asserts it matches the regex
%      '[RST][ 0-9]*' or 'Pulse' -- BrainVision marker names. This recording's real TMS
%      trigger is 'A - Stimulation', which fails that assert. It also would not be the
%      most frequent type anyway: these are closed-loop recordings that also carry
%      'A - Out' (NeurOne's own trigger output to the stimulator) plus 'B - Mute' /
%      'B - Out' housekeeping. The pulse event is therefore always passed explicitly.
%
%   3. THE EMG CHANNELS WOULD SURVIVE. The helper only strips EMG when chanlocs are
%      missing AND the labels match the regex 'EMG.*'. These are called 'FDIr' and
%      'APBr', so neither condition holds and they would go into SOUND, ICA and the
%      average reference. They are dropped explicitly below, as in the notebook.
%
%   4. CHANNEL LOCATIONS HAVE TO BE RE-LOOKED-UP. AARATEP leans on chanlocs far harder
%      than the notebook does: SOUND builds its lead field from them, bad-channel
%      interpolation and ICLabel's topography features both need them. The .set files
%      inherit NeurOne's positions, which TEP_sensorspace already judged unreliable and
%      overwrote with standard_1005. The same lookup is applied here.
%
%   Everything else lines up: continuous (un-epoched) data, which is what the pipeline
%   wants -- it epochs internally and warns that data must reach it BEFORE downsampling,
%   since the first artifact interpolation has to run at the native rate. events.tsv
%   gives one row per real TMS trigger with all the PRIME columns attached, which is what
%   the block and quartile splits below are built from.
%
%   One thing the .set does NOT contain: the online reference electrode. See REF_CHANNEL.
%
% ========================================================================================
% HOW THIS DIFFERS FROM TEP_sensorspace.ipynb
% ========================================================================================
%   * Cleaning is per RECORDING, not per block. AARATEP runs SOUND and two FastICA
%     decompositions per call; running it 20 times, once per block, would give the
%     20-trial predetermined blocks an ICA with ~50 000 samples for 60 channels, which is
%     badly under-determined, and would clean the two halves of every quartile contrast
%     with different unmixing matrices. So each recording is cleaned once, over all its
%     single pulses, and the blocks are formed afterwards by averaging subsets of those
%     epochs. Consequence to keep in mind when comparing: here intervention_all_prime is
%     exactly the mean of the same epochs that make up int1..4_prime, whereas in the
%     notebook those were five independent cleanings.
%
%   * The artifact window is AARATEP's, not the notebook's. Default artifactTimespan is
%     [-2, 12] ms against the notebook's CUT_MS of [-5, 15] ms, and the band-pass is
%     1-200 Hz against 1-100 Hz. Left at AARATEP's values on purpose: its narrower blank
%     window is earned by the decay fitting and SOUND that run either side of it, so
%     keeping them makes this a genuine second opinion rather than the notebook's
%     choices re-run under another name. Both are named constants in section 1 if you
%     want to align them.
%
%   * Only three things are changed from AARATEP's defaults, and none is a method
%     choice: lineNoiseFreq 60 -> 50 Hz (this is Germany), onOverRejection 'pause' ->
%     'warn' (the default drops into `keyboard` mid-loop, which would hang an unattended
%     batch), and the epoch window.
%
% ========================================================================================
% RUNNING IT
% ========================================================================================
% Budget roughly 15-40 min per recording, so 1-3 h for a full subject. AARATEP writes its
% own QC PNGs and intermediate .mat files (pre-SOUND, pre-decay-removal, pre-IC-rejection)
% into aaratep/<task>/ -- note that it MOVES any existing output dir to <dir>_old# rather
% than overwriting, so repeated runs accumulate. The largest recording
% (intervention-all, 3.4 GB continuous) peaks at roughly 8-10 GB while epoching.
%
% Requires: EEGLAB with the TESA and ICLabel plugins, and the AARATEP repo. FastICA ships
% inside AARATEP itself, so it does not need installing separately.

clear; clc; close all;

%% 1. Paths and parameters
%  The only cell that needs editing. subject / session move it to another subject;
%  everything below them is the analysis definition.

% ---- dependencies ----------------------------------------------------------------------
EEGLAB_DIR  = 'D:\Linus\MATLAB_applications\eeglab2026.0.0';
AARATEP_DIR = 'D:\Linus\MATLAB_applications\AARATEPPipeline-master';

addpath(EEGLAB_DIR);
eeglab nogui;                                    % also puts TESA and ICLabel on the path
addpath(AARATEP_DIR);
addpath(fullfile(AARATEP_DIR, 'Common'));
addpath(fullfile(AARATEP_DIR, 'Common', 'EEGAnalysisCode'));

% AARATEP calls c_EEG_openEEGLabIfNeeded() internally, which looks for a figure tagged
% 'EEGLAB' and, not finding one after `eeglab nogui`, opens the EEGLAB window once on the
% first pipeline call. Expected -- leave it open, it is a no-op from then on. That same
% function also prepends AARATEP's Common/ThirdParty/FromEEGLab to the path, so a handful
% of EEGLAB functions (topoplot, timtopo, epoch) are shadowed by AARATEP's copies for the
% rest of the session. Also expected.
assert(exist('tesa_sound', 'file') > 0, ...
    'TESA not found. Install the TESA plugin in EEGLAB (%s\\plugins).', EEGLAB_DIR);
assert(exist('iclabel', 'file') > 0, ...
    'ICLabel not found. Install the ICLabel plugin in EEGLAB (%s\\plugins).', EEGLAB_DIR);
assert(exist('c_TMSEEG_Preprocess_AARATEPPipeline', 'file') > 0, ...
    'AARATEP pipeline not found under %s', AARATEP_DIR);

% ---- Curve Fitting Toolbox -------------------------------------------------------------
% AARATEP's decay-fitting stage (c_TMSEEG_fitAndRemoveDecayArtifact) calls fit() from the
% Curve Fitting Toolbox. Without that toolbox MATLAB resolves fit() to an unrelated
% function and the pipeline dies with "Incorrect number or types of inputs or outputs for
% function fit" -- but only AFTER epoching, filtering, the eye ICA and SOUND have all run,
% i.e. ~20 minutes into the first recording. Checked here instead, before anything loads.
%
% SKIP_DECAY_REMOVAL puts a shim on the path that turns the decay stage into a no-op, so
% the rest of the pipeline can run without the toolbox. Only for getting unblocked: decay
% fitting is one of AARATEP's signature steps, and a run with it skipped is not full
% AARATEP output. Every figure and table from such a run is stamped accordingly.
SKIP_DECAY_REMOVAL = false;

has_curvefit = ~isempty(ver('curvefit')) && license('test', 'Curve_Fitting_Toolbox') == 1;
if SKIP_DECAY_REMOVAL
    nodecay_dir = fullfile(fileparts(mfilename('fullpath')), 'aaratep_nodecay');
    assert(isfolder(nodecay_dir), ...
        'SKIP_DECAY_REMOVAL is on but the shim folder is missing: %s', nodecay_dir);
    addpath(nodecay_dir, '-begin');
    warning('AARATEP:decaySkipped', ...
        'Decay fitting and removal is SKIPPED for this run. Results are not full AARATEP.');
elseif ~has_curvefit
    error(['The Curve Fitting Toolbox is not available, and AARATEP''s decay-removal ' ...
           'stage needs its fit(). Check with:\n' ...
           '    ver(''curvefit'')\n    license(''test'', ''Curve_Fitting_Toolbox'')\n' ...
           '    which -all fit\n' ...
           'Install it from the Home tab -> Add-Ons -> Get Add-Ons, or set ' ...
           'SKIP_DECAY_REMOVAL = true above to run without it.']);
end

% ---- subject / session -----------------------------------------------------------------
bids_root = 'D:\Linus\Loop\BIDS';
subject   = 'Pilot002';
session   = 'prime';
run       = '01';

eeg_dir     = fullfile(bids_root, ['sub-' subject], ['ses-' session], 'eeg');
out_dir     = fullfile(bids_root, 'derivatives', 'TEP_AARATEP', ['sub-' subject], ['ses-' session]);
fig_dir     = fullfile(out_dir, 'figures');
aaratep_dir = fullfile(out_dir, 'aaratep');      % the pipeline's own outputs and QC plots
if ~exist(fig_dir, 'dir'),     mkdir(fig_dir);     end
if ~exist(aaratep_dir, 'dir'), mkdir(aaratep_dir); end

% ---- what counts as a trial ------------------------------------------------------------
TRIGGER_TYPE = 'A - Stimulation';        % the real TMS trigger in these recordings
EXCLUDE_COND = {'prime_triplet'};        % 100 Hz triplets: not a single-pulse TEP
EMG_CHANNELS = {'FDIr', 'APBr'};         % dropped -- this is an EEG-only analysis
PULSE_EVENT  = 'TMS_sel';                % the marker this script inserts for the trials it
                                         % wants AARATEP to epoch around. Deliberately not
                                         % 'A - Stimulation': that type also marks the
                                         % triplet pulses, which must not be epoched.

% The recording is referenced to a physical electrode that is NOT in the file: 60 channels
% with no FCz and no AFz, the standard actiCAP/EasyCap arrangement of FCz = online
% reference, AFz = ground. AARATEP re-references to average at the end, and SOUND builds a
% lead field over whatever channels it is given, so leaving FCz out biases both. It is
% added back and average-referenced in prepare_recording below -- see the comment there
% for why the obvious pop_reref 'refloc' call does not work on these files. VERIFY against
% this session's NeurOne Protocol.xml; set to '' if the online reference was something
% else.
REF_CHANNEL  = 'FCz';
ADD_REF_BACK = true;

% ---- epoching and AARATEP settings -----------------------------------------------------
% The epoch has to be shorter than the shortest inter-pulse interval, or AARATEP's burst
% check (c_TMSEEG_handleBurstEvents, called with burstMaxIPI = max(abs(epochTimespan)))
% errors out. Shortest interval between trials in this dataset is 2.510 s, so 1.5 s of
% post-pulse epoch is the practical ceiling. -1 s comfortably covers the default
% [-0.5 -0.01] baseline.
% The other end of this trade-off: a pulse whose epoch would run past the end of the
% recording has to be dropped, because AARATEP's bundled epoch.m errors on out-of-range
% epochs instead of skipping them. evaluation-t30 loses its last trial at 1.5 s. Narrowing
% to [-1, 1.2] would keep it, at the cost of a shorter epoch everywhere.
EPOCH_TIMESPAN    = [-1, 1.5];           % s
ARTIFACT_TIMESPAN = [-0.002, 0.012];     % s. AARATEP default. The notebook blanks
                                         % [-0.005, 0.015] -- set that here to align them.
BANDPASS          = [1, 200];            % Hz. AARATEP default. Notebook uses [1, 100].
DOWNSAMPLE_TO     = 1000;                % Hz, applied after the first interpolation
LINE_FREQ         = 50;                  % Hz -- changed from AARATEP's 60 Hz default
LINE_HARMONICS    = 1;                   % AARATEP default. 2 also notches 100 Hz, which
                                         % is in band given the 200 Hz low-pass.
ON_OVER_REJECTION = 'warn';              % NOT the 'pause' default: that calls keyboard

% Stamped into every figure title and recorded per block in the summary table, so a run
% made without decay removal can never be mistaken later for a full AARATEP run.
RUN_TAG = ternary(SKIP_DECAY_REMOVAL, '  [NO DECAY REMOVAL]', '');

% ---- automatic rejection ---------------------------------------------------------------
% Bad channels and ICs are AARATEP's business, not this script's -- it uses PREP_deviation
% plus TESA's DDWiener per trial for channels, and ICLabel plus a TMS-muscle rule for
% components, and reports what it dropped in the summary table below. There is no trial
% rejection stage in AARATEP at all, which is one more difference from the notebook's
% peak-to-peak MAD rule.
MIN_TRIALS = 30;                         % below this a block average is flagged

% ---- PRIME prediction split ------------------------------------------------------------
% Same columns and same rule as TEP_sensorspace, so the two pipelines split on identical
% trials. Only prime_single_pulse trials carry a prediction -- predetermined and
% calibration pulses have none by definition.
PRED_COL          = 'prediction_probability';
AMP_COL           = 'tep_amplitude';
PRED_Q            = 0.25;                % quartiles
PRED_WITHIN_BLOCK = true;                % quartiles within each intervention block before
                                         % pooling, so "high prediction" cannot just mean
                                         % "whichever block ran higher predictions"

% ---- region of interest and plotting ---------------------------------------------------
ROI       = {'C3', 'FC3', 'CP3', 'C1', 'C5'};
ROI_NAME  = 'M1l';
PLOT_XLIM = [-100, 200];                 % ms
TOPO_WIN  = [40, 50];                    % ms, the N45 the intervention targets
WINDOWS   = {[35 55], 'N45 35-55'; [55 80], '55-80'; [85 140], '85-140'};

% ---- the standard montage to look channel positions up from ----------------------------
% Standard-10-5-Cap385_witheog.elp is pop_chanedit's own default lookup table and carries
% every label this montage uses, FCz included. The alternatives are there because the
% file has moved between EEGLAB releases.
elp_candidates = { ...
    fullfile(EEGLAB_DIR, 'functions', 'supportfiles', 'Standard-10-5-Cap385_witheog.elp'), ...
    fullfile(EEGLAB_DIR, 'functions', 'supportfiles', 'Standard-10-5-Cap385.sfp'), ...
    fullfile(EEGLAB_DIR, 'functions', 'resources',    'Standard-10-5-Cap385_witheog.elp'), ...
    fullfile(EEGLAB_DIR, 'sample_locs', 'Standard-10-5-Cap385.sfp'), ...
    fullfile(EEGLAB_DIR, 'plugins', 'dipfit', 'standard_BEM', 'elec', 'standard_1005.elc')};
ELP_PATH = '';
for iC = 1:numel(elp_candidates)
    if isfile(elp_candidates{iC}), ELP_PATH = elp_candidates{iC}; break; end
end
assert(~isempty(ELP_PATH), ['No standard channel-location file found under %s. ' ...
    'Point ELP_PATH at one by hand.'], EEGLAB_DIR);
fprintf('Channel locations from: %s\n', ELP_PATH);

% bundle everything the local functions need
cfg = struct('eeg_dir', eeg_dir, 'subject', subject, 'session', session, 'run', run, ...
    'EMG_CHANNELS', {EMG_CHANNELS}, 'REF_CHANNEL', REF_CHANNEL, 'ADD_REF_BACK', ADD_REF_BACK, ...
    'ELP_PATH', ELP_PATH, 'PULSE_EVENT', PULSE_EVENT, 'EPOCH_TIMESPAN', EPOCH_TIMESPAN, ...
    'ARTIFACT_TIMESPAN', ARTIFACT_TIMESPAN, 'BANDPASS', BANDPASS, ...
    'DOWNSAMPLE_TO', DOWNSAMPLE_TO, 'LINE_FREQ', LINE_FREQ, ...
    'LINE_HARMONICS', LINE_HARMONICS, 'ON_OVER_REJECTION', ON_OVER_REJECTION, ...
    'aaratep_dir', aaratep_dir);


%% 2. Trial table
%  Reads each block's events.tsv, drops the triplets, and builds one block_label column
%  with the same values the rest of this project uses -- the intervention-all recording
%  contains five blocks, told apart by its own stage column. Identical to section 2 of
%  TEP_sensorspace, including the block list at the bottom, so the two pipelines average
%  over exactly the same trials.
%
%  row_in_task is assigned AFTER the triplet filter: it is the index this script uses to
%  tie a cleaned epoch back to its trial, and only these rows get a pulse marker.

TASKS = {'baseline', 'intervention-all', ...
         'evaluation-t0', 'evaluation-t15', 'evaluation-t30', 'evaluation-t60'};

events = containers.Map();
for iT = 1:numel(TASKS)
    task = TASKS{iT};
    stem = sprintf('sub-%s_ses-%s_task-%s_run-%s', subject, session, task, run);
    tsv  = fullfile(eeg_dir, [stem '_events.tsv']);
    st   = fullfile(eeg_dir, [stem '_eeg.set']);
    if ~isfile(tsv) || ~isfile(st)
        fprintf('!! not found, skipping: task-%s\n', task);
        continue
    end

    T = readtable(tsv, 'FileType', 'text', 'Delimiter', '\t', 'TextType', 'string');
    T = T(strcmp(T.trigger_type_neurone, TRIGGER_TYPE), :);

    if ~ismember('condition', T.Properties.VariableNames)
        T.condition = repmat("", height(T), 1);
    end
    T.condition = string(T.condition);
    T.condition(ismissing(T.condition)) = "";

    % one file, five blocks: intervention-all carries calibration + the four blocks
    if strcmp(task, 'intervention-all')
        assert(ismember('stage', T.Properties.VariableNames), ...
            ['task-intervention-all needs a "stage" column to tell calibration from the ' ...
             'four intervention blocks, and %s has none.'], tsv);
        T.block_label = string(T.stage);
    else
        T.block_label = repmat(string(task), height(T), 1);
    end

    T = T(~ismember(T.condition, EXCLUDE_COND), :);
    T.row_in_task = (1:height(T))';
    events(task) = T;

    [labs, counts] = countlabels(T.block_label);
    fprintf('task-%-16s %4d single pulses  [%s]\n', task, height(T), ...
        strjoin(arrayfun(@(k) sprintf('%s:%d', labs(k), counts(k)), 1:numel(labs), ...
        'UniformOutput', false), ', '));
end

% ---- the blocks to analyse: (label, task, block_label(s), condition, split) -------------
int_labels = arrayfun(@(b) sprintf('intervention_block_%d', b), 1:4, 'UniformOutput', false);

blocks = mkblock('baseline',    'baseline',         {'baseline'},    '', {});
blocks = [blocks, mkblock('calibration', 'intervention-all', {'calibration'}, '', {})];
for b = 1:4
    blocks = [blocks, ...
        mkblock(sprintf('int%d_prime',  b), 'intervention-all', int_labels(b), 'prime_single_pulse',   {}), ...
        mkblock(sprintf('int%d_predet', b), 'intervention-all', int_labels(b), 'predetermined_single', {})]; %#ok<AGROW>
end
for iT = 1:numel(TASKS)
    task = TASKS{iT};
    if startsWith(task, 'evaluation-') && isKey(events, task)
        blocks = [blocks, mkblock(strrep(task, '-', '_'), task, {task}, '', {})]; %#ok<AGROW>
    end
end
blocks = [blocks, ...
    mkblock('intervention_all_prime',  'intervention-all', int_labels, 'prime_single_pulse',   {}), ...
    mkblock('intervention_all_predet', 'intervention-all', int_labels, 'predetermined_single', {})];

% The same PRIME single pulses again, split two ways: by what PRIME predicted beforehand,
% and by the TEP amplitude that actually resulted. Split is {column, side, within-block}.
blocks = [blocks, ...
    mkblock('prime_pred_high', 'intervention-all', int_labels, 'prime_single_pulse', {PRED_COL, 'high', PRED_WITHIN_BLOCK}), ...
    mkblock('prime_pred_low',  'intervention-all', int_labels, 'prime_single_pulse', {PRED_COL, 'low',  PRED_WITHIN_BLOCK}), ...
    mkblock('prime_tep_high',  'intervention-all', int_labels, 'prime_single_pulse', {AMP_COL,  'high', PRED_WITHIN_BLOCK}), ...
    mkblock('prime_tep_low',   'intervention-all', int_labels, 'prime_single_pulse', {AMP_COL,  'low',  PRED_WITHIN_BLOCK})];

% The top quartile of predictions taken across ALL FOUR intervention blocks at once. The
% prediction distribution drifts over the session, so this is a different set of trials
% from prime_pred_high -- it also carries a block/time contrast. Composition is printed.
blocks = [blocks, ...
    mkblock('prime_pred_high_global', 'intervention-all', int_labels, 'prime_single_pulse', {PRED_COL, 'high', false})];

blocks = blocks(arrayfun(@(b) isKey(events, b.task), blocks));
fprintf('\n%d blocks to analyse.\n', numel(blocks));


%% 3. Clean each recording once with AARATEP
%  One pipeline call per recording, over every single pulse it contains. Everything that
%  has to happen before AARATEP sees the data is in prepare_recording() at the bottom of
%  this file: drop EMG, re-look-up channel locations, add the online reference back, and
%  insert one PULSE_EVENT marker per retained trial at the events.tsv onset -- which is
%  exactly how epoch_block builds its events in TEP_sensorspace.
%
%  cleaned(task).epochRows maps epoch number -> row_in_task, recovered from the marker
%  rather than assumed from ordering, so a pulse dropped for sitting too close to a data
%  boundary cannot silently shift every later trial's label.

tasks_to_clean = unique({blocks.task});
cleaned = containers.Map();

for iT = 1:numel(tasks_to_clean)
    task = tasks_to_clean{iT};
    T = events(task);
    fprintf('\n=== AARATEP: task-%s (%d single pulses) ===\n', task, height(T));

    EEGc = prepare_recording(task, T, cfg);

    % 'drop' rather than the default 'ignore' for pulses whose epoch would run off the end
    % of the recording. AARATEP ships its own copy of EEGLAB's epoch.m that shadows the
    % real one, and where EEGLAB skips an out-of-range epoch with a warning, that copy
    % raises `error('Not implemented')` -- so a single trial too near a boundary kills the
    % whole recording. evaluation-t30 is one: its last pulse sits ~1.46 s before the end of
    % the recording and the epoch needs 1.5 s. 'drop' removes such pulses beforehand, which
    % costs that one trial. (Narrowing EPOCH_TIMESPAN would keep it, at the cost of a
    % shorter epoch for every block.) The count is reported below, and the epoch-to-trial
    % mapping is read back off the markers, so a dropped trial cannot shift any labels.
    n_marked = height(T);
    [EEGc, misc] = c_TMSEEG_prepareForPreprocessing( ...
        'inputEEGs', EEGc, ...
        'pulseEvent', cfg.PULSE_EVENT, ...
        'epochTimespan', cfg.EPOCH_TIMESPAN, ...
        'ifPulseTooCloseToBoundary', 'drop');
    n_kept_markers = sum(strcmp({EEGc.event.type}, cfg.PULSE_EVENT));
    if n_kept_markers < n_marked
        fprintf(['  !! %d of %d pulses dropped for sitting within the epoch window of a ' ...
                 'data boundary\n'], n_marked - n_kept_markers, n_marked);
    end

    outDir = fullfile(cfg.aaratep_dir, task);
    prefix = sprintf('sub-%s_ses-%s_task-%s', subject, session, task);

    EEGc = c_TMSEEG_Preprocess_AARATEPPipeline(EEGc, ...
        'pulseEvent',        misc.pulseEvent, ...
        'epochTimespan',     misc.epochTimespan, ...
        'outputDir',         outDir, ...
        'outputFilePrefix',  prefix, ...
        'artifactTimespan',  cfg.ARTIFACT_TIMESPAN, ...
        'bandpassFreqSpan',  cfg.BANDPASS, ...
        'downsampleTo',      cfg.DOWNSAMPLE_TO, ...
        'lineNoiseFreq',     cfg.LINE_FREQ, ...
        'lineNoiseNumHarmonics', cfg.LINE_HARMONICS, ...
        'onOverRejection',   cfg.ON_OVER_REJECTION);

    % the pipeline returns only EEG; its own bookkeeping is saved beside the data
    md = struct();
    mdPath = fullfile(outDir, [prefix '.mat']);
    if isfile(mdPath)
        S = load(mdPath, 'md');
        if isfield(S, 'md'), md = S.md; end
    end

    epochRows = epoch_rows(EEGc, cfg.PULSE_EVENT);
    fprintf('  kept %d/%d epochs, %d channels, %.0f Hz\n', ...
        EEGc.trials, height(T), EEGc.nbchan, EEGc.srate);

    cleaned(task) = struct('EEG', EEGc, 'md', md, 'epochRows', epochRows);
    clear EEGc
end


%% 4. Average into blocks
%  One row per block definition, formed by averaging the subset of that recording's
%  already-cleaned epochs. The summary table is the thing to read afterwards: a block
%  whose recording lost an unusual number of channels or components is the one to be
%  suspicious of.

results = struct('label', {}, 'task', {}, 'times', {}, 'evoked', {}, 'roi', {}, ...
                 'sem', {}, 'gmfa', {}, 'n', {}, 'split_median', {}, 'chanlocs', {});
summary = table();

for iB = 1:numel(blocks)
    B = blocks(iB);
    T = events(B.task);

    m = ismember(T.block_label, string(B.blabels));
    if ~isempty(B.cond)
        m = m & strcmp(T.condition, B.cond);
    end
    rows = T(m, :);

    split_median = NaN;
    split_on = '';
    if ~isempty(B.split)                        % keep only one quartile of the chosen column
        scol   = B.split{1};
        side   = B.split{2};
        within = B.split{3};
        split_on = scol;

        assert(ismember(scol, rows.Properties.VariableNames), ...
            'Column "%s" is not in task-%s''s events.tsv.', scol, B.task);
        v = rows.(scol);
        if ~isnumeric(v), v = str2double(string(v)); end    % blank cells arrive as ""
        v = double(v);
        keep = ~isnan(v);
        rows = rows(keep, :);
        v = v(keep);

        q = PRED_Q;
        if strcmp(side, 'high'), q = 1 - PRED_Q; end
        if within
            thr = nan(size(v));
            ublk = unique(rows.block_label);
            for iU = 1:numel(ublk)
                sel = rows.block_label == ublk(iU);
                thr(sel) = pquantile(v(sel), q);   % numpy/pandas 'linear' quantile, so the
            end                                    % same trials land in the split as in the
        else                                       % notebook
            thr = repmat(pquantile(v, q), size(v));
        end
        if strcmp(side, 'high'), sel = v >= thr; else, sel = v <= thr; end
        rows = rows(sel, :);
        v = v(sel);
        split_median = median(v);

        [l_, c_] = countlabels(rows.block_label);
        fprintf('  %-24s %s %-4s (%s blocks): %d trials, range %.3f-%.3f\n', ...
            B.label, scol, side, ternary(within, 'within', 'across'), numel(v), min(v), max(v));
        fprintf('  %-24s from %s\n', '', strjoin(arrayfun(@(k) ...
            sprintf('%s:%d', l_(k), c_(k)), 1:numel(l_), 'UniformOutput', false), ', '));
    end

    if isempty(rows)
        fprintf('  %-24s no trials, skipped\n', B.label);
        continue
    end

    C = cleaned(B.task);
    [tf, epochIdx] = ismember(rows.row_in_task, C.epochRows);
    if ~all(tf)
        warning('%s: %d/%d trials had no surviving epoch and were dropped.', ...
            B.label, sum(~tf), numel(tf));
    end
    epochIdx = epochIdx(tf);
    if isempty(epochIdx)
        fprintf('  %-24s no surviving epochs, skipped\n', B.label);
        continue
    end

    X = double(C.EEG.data(:, :, epochIdx));      % channels x time x trials, in uV
    nTr = size(X, 3);
    roiPick = find(ismember({C.EEG.chanlocs.labels}, ROI));
    roiTrials = permute(mean(X(roiPick, :, :), 1), [3 2 1]);   % trials x time

    r = struct();
    r.label  = B.label;
    r.task   = B.task;
    r.times  = C.EEG.times(:)';                  % ms
    r.evoked = mean(X, 3);
    r.roi    = mean(roiTrials, 1);
    if nTr > 1
        r.sem = std(roiTrials, 0, 1) / sqrt(nTr);
    else
        r.sem = nan(1, numel(r.times));
    end
    r.gmfa   = std(r.evoked, 0, 1);
    r.n      = nTr;
    r.split_median = split_median;
    r.chanlocs = C.EEG.chanlocs;
    results(end+1) = r; %#ok<SAGROW>

    badLabels = '';
    if isfield(C.md, 'earlyRejectedChannels') && ~isempty(C.md.earlyRejectedChannels)
        idx = C.md.earlyRejectedChannels;
        idx = idx(idx >= 1 & idx <= numel(C.EEG.chanlocs));
        badLabels = strjoin({C.EEG.chanlocs(idx).labels}, ' ');
    end
    summary = [summary; table(string(B.label), string(B.task), height(rows), nTr, ...
        string(badLabels), getfielddef(C.md, 'ICA_numRejComp', NaN), ...
        getfielddef(C.md, 'ICA_numComp', NaN), ...
        getfielddef(C.md, 'eyeICA_numRejComp', NaN), ...
        getfielddef(C.md, 'didRemoveDecay', NaN), ...
        string(split_on), ...
        split_median, string(ternary(nTr >= MIN_TRIALS, '', 'few trials')), ...
        'VariableNames', {'block', 'task', 'n_selected', 'n_kept', 'bad_channels', ...
        'n_ICs_rejected', 'n_components', 'n_eyeICs_rejected', 'decay_removed', ...
        'split_on', 'split_median', 'flag'})]; %#ok<AGROW>

    fprintf('  %-24s %3d trials from task-%s\n', B.label, nTr, B.task);
end

writetable(summary, fullfile(out_dir, ...
    sprintf('sub-%s_ses-%s_desc-TEPaaratep_summary.tsv', subject, session)), ...
    'FileType', 'text', 'Delimiter', '\t');
disp(summary);


%% 5. One figure per block
%  Four panels each, the same four as the notebook: butterfly of all channels, the left-M1
%  ROI with its 95% CI across trials, the global mean field amplitude, and the scalp
%  topography over the N45 window. The interpolated stretch is shaded grey -- nothing
%  inside it is real data.

for iR = 1:numel(results)
    r = results(iR);
    t = r.times;
    s = t >= PLOT_XLIM(1) & t <= PLOT_XLIM(2);
    ci = 1.96 * r.sem;

    hf = figure('Position', [50 50 1200 750], 'Color', 'w');

    ax1 = subplot(2, 2, 1);
    plot(t(s), r.evoked(:, s)', 'LineWidth', 0.5);
    title(sprintf('Butterfly, %d channels', size(r.evoked, 1)));
    xlabel('Time (ms)'); ylabel('Amplitude (uV)');

    ax2 = subplot(2, 2, 2);
    fill([t(s), fliplr(t(s))], [r.roi(s) - ci(s), fliplr(r.roi(s) + ci(s))], ...
        [0 0.45 0.74], 'FaceAlpha', 0.25, 'EdgeColor', 'none'); hold on
    plot(t(s), r.roi(s), 'LineWidth', 2, 'Color', [0 0.45 0.74]);
    title(sprintf('ROI %s (%s), mean +/- 95%% CI', ROI_NAME, strjoin(ROI, ' ')));
    xlabel('Time (ms)'); ylabel('Amplitude (uV)');

    ax3 = subplot(2, 2, 3);
    plot(t(s), r.gmfa(s), 'k', 'LineWidth', 1.5);
    title('Global mean field amplitude');
    xlabel('Time (ms)'); ylabel('GMFA (uV)');

    for ax = [ax1 ax2 ax3]
        mark_artifact_window(ax, ARTIFACT_TIMESPAN * 1000);
    end

    subplot(2, 2, 4);
    w = t >= TOPO_WIN(1) & t <= TOPO_WIN(2);
    topoplot(mean(r.evoked(:, w), 2), r.chanlocs, 'electrodes', 'on');
    title(sprintf('Topography %d-%d ms', TOPO_WIN(1), TOPO_WIN(2)));

    sgtitle(sprintf('sub-%s ses-%s  |  %s  |  %d trials  |  AARATEP%s', ...
        subject, session, strrep(r.label, '_', '\_'), r.n, RUN_TAG));
    print(hf, fullfile(fig_dir, sprintf('sub-%s_ses-%s_block-%s_desc-tep.png', ...
        subject, session, r.label)), '-dpng', '-r150');
end


%% 6. Across blocks
%  Same four views as section 6 of the notebook: every block's ROI waveform on one axis,
%  the prediction and achieved-amplitude quartile contrasts, and each intervention
%  condition averaged across the four blocks.

labels = {results.label};

hf = figure('Position', [50 50 1100 500], 'Color', 'w');
ax = axes('Parent', hf); hold(ax, 'on');
for iR = 1:numel(results)
    r = results(iR);
    if startsWith(r.label, 'intervention_all'), continue; end
    s = r.times >= PLOT_XLIM(1) & r.times <= PLOT_XLIM(2);
    plot(ax, r.times(s), r.roi(s), 'LineWidth', 1.3, ...
        'DisplayName', sprintf('%s (n=%d)', strrep(r.label, '_', '\_'), r.n));
end
mark_artifact_window(ax, ARTIFACT_TIMESPAN * 1000);
xlabel(ax, 'Time (ms)'); ylabel(ax, 'Amplitude (uV)');
title(ax, sprintf('sub-%s: %s ROI, every block (AARATEP)%s', subject, ROI_NAME, RUN_TAG));
legend(ax, 'FontSize', 7, 'NumColumns', 2, 'Location', 'northeast');
print(hf, fullfile(fig_dir, sprintf('sub-%s_ses-%s_desc-TEPaaratep_allblocks.png', ...
    subject, session)), '-dpng', '-r150');

compare_quartiles(results, 'prime_pred_high', 'prime_pred_low', ...
    ['PRIME single pulses by PREDICTED probability' RUN_TAG], 'predictionQuartiles', ...
    subject, session, ROI_NAME, PLOT_XLIM, ARTIFACT_TIMESPAN, WINDOWS, fig_dir);
compare_quartiles(results, 'prime_tep_high', 'prime_tep_low', ...
    ['PRIME single pulses by ACHIEVED TEP amplitude' RUN_TAG], 'tepAmplitudeQuartiles', ...
    subject, session, ROI_NAME, PLOT_XLIM, ARTIFACT_TIMESPAN, WINDOWS, fig_dir);

groups = {'PRIME single pulses',         arrayfun(@(b) sprintf('int%d_prime',  b), 1:4, 'UniformOutput', false); ...
          'Predetermined single pulses', arrayfun(@(b) sprintf('int%d_predet', b), 1:4, 'UniformOutput', false)};

grand = struct('condition', {}, 'times', {}, 'mean', {}, 'sem', {}, 'n_trials', {});
hf = figure('Position', [50 50 1200 450], 'Color', 'w');
for iG = 1:size(groups, 1)
    labs = groups{iG, 2};
    labs = labs(ismember(labs, labels));
    if isempty(labs), continue; end
    idx = cellfun(@(L) find(strcmp(labels, L), 1), labs);
    stack = vertcat(results(idx).roi);
    t = results(idx(1)).times;
    s = t >= PLOT_XLIM(1) & t <= PLOT_XLIM(2);
    mu = mean(stack, 1);
    se = std(stack, 0, 1) / sqrt(numel(labs));

    ax = subplot(1, 2, iG); hold(ax, 'on');
    plot(ax, t(s), stack(:, s)', 'Color', [0.75 0.75 0.75], 'LineWidth', 0.7);
    fill(ax, [t(s), fliplr(t(s))], [mu(s) - se(s), fliplr(mu(s) + se(s))], ...
        [0 0.45 0.74], 'FaceAlpha', 0.25, 'EdgeColor', 'none');
    plot(ax, t(s), mu(s), 'LineWidth', 2, 'Color', [0 0.45 0.74]);
    mark_artifact_window(ax, ARTIFACT_TIMESPAN * 1000);
    nTot = sum([results(idx).n]);
    title(ax, sprintf('%s\n%d blocks, %d trials', groups{iG, 1}, numel(labs), nTot));
    xlabel(ax, 'Time (ms)'); ylabel(ax, 'Amplitude (uV)');

    grand(end+1) = struct('condition', groups{iG, 1}, 'times', t, 'mean', mu, ...
        'sem', se, 'n_trials', nTot); %#ok<SAGROW>
end
sgtitle(sprintf('sub-%s: %s ROI averaged across intervention blocks (AARATEP)%s', ...
    subject, ROI_NAME, RUN_TAG));
print(hf, fullfile(fig_dir, sprintf('sub-%s_ses-%s_desc-TEPaaratep_interventionGrandAverage.png', ...
    subject, session)), '-dpng', '-r150');


%% 7. Save the waveforms
%  One tidy CSV of every block's ROI waveform (plus its CI and GMFA), one for the
%  across-block grand averages, and the block averages themselves as a .mat holding all
%  channels and the chanlocs -- the analogue of the notebook's -ave.fif, so a later
%  topography or peak measurement can start from this instead of re-running the pipeline.

wf = table();
for iR = 1:numel(results)
    r = results(iR);
    wf = [wf; table(repmat(string(r.label), numel(r.times), 1), r.times(:), r.roi(:), ...
        1.96 * r.sem(:), r.gmfa(:), repmat(r.n, numel(r.times), 1), ...
        'VariableNames', {'block', 'time_ms', 'roi_uV', 'ci95_uV', 'gmfa_uV', 'n_trials'})]; %#ok<AGROW>
end
writetable(wf, fullfile(out_dir, ...
    sprintf('sub-%s_ses-%s_desc-TEPaaratep_waveforms.csv', subject, session)));

ga = table();
for iG = 1:numel(grand)
    g = grand(iG);
    ga = [ga; table(repmat(string(g.condition), numel(g.times), 1), g.times(:), ...
        g.mean(:), g.sem(:), ...
        'VariableNames', {'condition', 'time_ms', 'roi_uV', 'sem_uV'})]; %#ok<AGROW>
end
writetable(ga, fullfile(out_dir, ...
    sprintf('sub-%s_ses-%s_desc-TEPaaratep_interventionGrandAverage.csv', subject, session)));

tep = results; %#ok<NASGU>
save(fullfile(out_dir, sprintf('sub-%s_ses-%s_desc-TEPaaratep_ave.mat', subject, session)), ...
    'tep', '-v7.3');

fprintf('\nSaved %d block averages to:\n  %s\n', numel(results), out_dir);


%% ========================================================================================
%%  Local functions
%% ========================================================================================

function b = mkblock(label, task, blabels, cond, split)
% One entry of the block list: which trials of which recording get averaged together.
    b = struct('label', label, 'task', task, 'blabels', {blabels}, 'cond', cond, ...
               'split', {split});
end


function EEG = prepare_recording(task, T, cfg)
% Everything AARATEP needs done to a raw organize_neurone.m .set before it will accept it.
% See the format notes at the top of this file for why each of these is necessary.
    stem = sprintf('sub-%s_ses-%s_task-%s_run-%s', cfg.subject, cfg.session, task, cfg.run);
    EEG = pop_loadset('filename', [stem '_eeg.set'], 'filepath', cfg.eeg_dir);

    % --- drop EMG: this is an EEG-only analysis, and AARATEP's own EMG rule ('EMG.*')
    %     does not match these labels
    drop = find(ismember({EEG.chanlocs.labels}, cfg.EMG_CHANNELS));
    if ~isempty(drop)
        fprintf('  dropping %d EMG channel(s): %s\n', numel(drop), ...
            strjoin({EEG.chanlocs(drop).labels}, ', '));
        EEG = pop_select(EEG, 'nochannel', drop);
    end

    % --- channel locations from the standard montage. SOUND's lead field, bad-channel
    %     interpolation and ICLabel's topography features all depend on these.
    EEG = pop_chanedit(EEG, 'lookup', cfg.ELP_PATH);
    missing = find(cellfun(@isempty, {EEG.chanlocs.X}));
    assert(isempty(missing), 'No position found for channel(s): %s', ...
        strjoin({EEG.chanlocs(missing).labels}, ', '));
    [EEG.chanlocs.type] = deal('EEG');

    % --- add the online reference back, then average reference over all of it
    %
    % Appended as a channel of zeros and immediately average-referenced, which is
    % mne.add_reference_channels + set_eeg_reference('average') in the notebook, and
    % pop_reref's "refloc" behaviour in EEGLAB. Done by hand rather than by calling
    % pop_reref(EEG, [], 'refloc', ...) because that route does NOT work on these files:
    % reref() copies every field of the existing chanlocs off the refloc struct (so a
    % bare struct('labels','FCz') errors on the first missing field), and pop_reref then
    % requires the reference to already be sitting in EEG.chaninfo.nodatchans, which a
    % NeurOne import does not populate -- it raises 'Missing reference channel
    % information. Edit channels and add reference first.'
    %
    % Referencing here rather than leaving it to AARATEP's own final pop_reref matters
    % for one reason: after averaging, FCz carries the negated mean, i.e. real signal. A
    % flat channel handed to AARATEP would be caught by its bad-channel detection and
    % interpolated away, which is the opposite of what adding it back is for.
    if cfg.ADD_REF_BACK && ~isempty(cfg.REF_CHANNEL) && ...
            ~ismember(cfg.REF_CHANNEL, {EEG.chanlocs.labels})
        refloc = EEG.chanlocs(1);
        fn = fieldnames(refloc);
        for iF = 1:numel(fn), refloc.(fn{iF}) = []; end
        refloc.labels = cfg.REF_CHANNEL;

        EEG.chanlocs(end+1) = refloc;
        EEG.data(end+1, :)  = 0;
        EEG.nbchan = size(EEG.data, 1);
        EEG = eeg_checkset(EEG);

        EEG = pop_chanedit(EEG, 'lookup', cfg.ELP_PATH);
        [EEG.chanlocs.type] = deal('EEG');
        assert(~isempty(EEG.chanlocs(end).X), ...
            'No position found for the reference channel %s in %s', ...
            cfg.REF_CHANNEL, cfg.ELP_PATH);

        EEG = pop_reref(EEG, []);
        fprintf('  added the online reference back as %s and average referenced (%d channels)\n', ...
            cfg.REF_CHANNEL, EEG.nbchan);
    end

    % --- one pulse marker per retained trial, placed from the events.tsv onset. Built
    %     fresh rather than reusing the 'A - Stimulation' events, because those also mark
    %     the 100 Hz triplet pulses, which must not be epoched around. tepRow carries the
    %     trial index through epoching so epochs can be tied back to trials afterwards.
    [EEG.event.tepRow] = deal(NaN);
    n0 = numel(EEG.event);
    lat = round(T.onset * EEG.srate) + 1;
    for k = 1:height(T)
        EEG.event(n0 + k).type     = cfg.PULSE_EVENT;
        EEG.event(n0 + k).latency  = lat(k);
        EEG.event(n0 + k).duration = 0;
        EEG.event(n0 + k).urevent  = [];
        EEG.event(n0 + k).tepRow   = T.row_in_task(k);
    end
    EEG = eeg_checkset(EEG, 'eventconsistency');

    nPulse = sum(strcmp({EEG.event.type}, cfg.PULSE_EVENT));
    assert(nPulse == height(T), 'Inserted %d pulse markers but expected %d', nPulse, height(T));

    % AARATEP's burst check (c_TMSEEG_handleBurstEvents, method 'error') refuses any two
    % pulse events closer than max(abs(epochTimespan)). Fail here instead, with a message
    % that says what to do about it.
    ipi = diff(sort(lat)) / EEG.srate;
    if isempty(ipi)
        fprintf('  1 pulse marker inserted\n');
    else
        fprintf('  %d pulse markers inserted, shortest interval %.3f s (epoch spans %.2f s)\n', ...
            nPulse, min(ipi), max(abs(cfg.EPOCH_TIMESPAN)));
        assert(min(ipi) > max(abs(cfg.EPOCH_TIMESPAN)), ...
            ['Two selected pulses are %.3f s apart, closer than the epoch window -- ' ...
             'AARATEP''s burst check will error. Narrow EPOCH_TIMESPAN.'], min(ipi));
    end
end


function rows = epoch_rows(EEG, pulseEvent)
% Epoch number -> row_in_task, read back off the marker that time-locked each epoch rather
% than assumed from ordering, so a pulse dropped at a data boundary cannot silently shift
% every later trial's label.
    rows = nan(EEG.trials, 1);
    for j = 1:EEG.trials
        types = tocell(EEG.epoch(j).eventtype);
        lats  = tocell(EEG.epoch(j).eventlatency);
        tags  = tocell(EEG.epoch(j).eventtepRow);
        isPulse = strcmp(types, pulseEvent);
        atZero  = cellfun(@(x) ~isempty(x) && abs(x) < 1e-6, lats);
        hit = find(isPulse & atZero, 1);
        if ~isempty(hit) && ~isempty(tags{hit})
            rows(j) = tags{hit};
        end
    end
    assert(~any(isnan(rows)), ...
        '%d epoch(s) could not be tied back to a trial -- check the %s markers.', ...
        sum(isnan(rows)), pulseEvent);
    assert(numel(unique(rows)) == numel(rows), 'Two epochs map to the same trial.');
end


function c = tocell(x)
    if iscell(x), c = x; else, c = {x}; end
end


function q = pquantile(x, p)
% numpy/pandas 'linear' quantile. MATLAB's own quantile() uses a different convention, and
% on a 60-trial block the two disagree about which trials sit in the top quartile -- this
% keeps the split identical to the one TEP_sensorspace makes.
    x = sort(x(:));
    n = numel(x);
    if n == 0, q = NaN; return; end
    if n == 1, q = x(1); return; end
    h  = (n - 1) * p + 1;
    lo = floor(h); hi = ceil(h);
    q  = x(lo) + (h - lo) * (x(hi) - x(lo));
end


function mark_artifact_window(ax, span_ms)
% Shade the stretch AARATEP replaced with interpolated values, and mark t = 0.
    hold(ax, 'on');
    yl = ylim(ax);
    hp = patch(ax, [span_ms(1) span_ms(2) span_ms(2) span_ms(1)], [yl(1) yl(1) yl(2) yl(2)], ...
        [0.85 0.85 0.85], 'EdgeColor', 'none', 'HandleVisibility', 'off');
    uistack(hp, 'bottom');                              % shading behind the traces
    xline(ax, 0, 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
    yline(ax, 0, 'k:', 'LineWidth', 0.8, 'HandleVisibility', 'off');
    ylim(ax, yl);
end


function [labs, counts] = countlabels(x)
% Sorted unique values of a string array with their counts. Plain unique() rather than
% groupcounts() so this does not depend on a particular MATLAB release.
    labs = unique(x);
    counts = arrayfun(@(v) sum(x == v), labs);
end


function compare_quartiles(results, hiLab, loLab, ttl, fname, subject, session, ...
                           roiName, xlimMs, artifactSpan, windows, figDir)
% Top vs bottom quartile on one axis, plus the printed window table. Same three windows as
% the notebook, so the numbers can be put side by side.
    labels = {results.label};
    iHi = find(strcmp(labels, hiLab), 1);
    iLo = find(strcmp(labels, loLab), 1);
    if isempty(iHi) || isempty(iLo), return; end

    hf = figure('Position', [50 50 900 450], 'Color', 'w');
    ax = axes('Parent', hf); hold(ax, 'on');
    cols = {[0.5 0.5 0.5], [0.84 0.15 0.16]};
    for k = 1:2
        r = results(ternary(k == 1, iLo, iHi));
        s = r.times >= xlimMs(1) & r.times <= xlimMs(2);
        ci = 1.96 * r.sem;
        fill(ax, [r.times(s), fliplr(r.times(s))], ...
            [r.roi(s) - ci(s), fliplr(r.roi(s) + ci(s))], cols{k}, ...
            'FaceAlpha', 0.20, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        plot(ax, r.times(s), r.roi(s), 'Color', cols{k}, 'LineWidth', 2, ...
            'DisplayName', sprintf('%s (n=%d, median %.3f)', ...
            strrep(r.label, '_', '\_'), r.n, r.split_median));
    end
    mark_artifact_window(ax, artifactSpan * 1000);
    xlim(ax, xlimMs);
    xlabel(ax, 'Time (ms)'); ylabel(ax, 'Amplitude (uV)');
    title(ax, sprintf('sub-%s: %s ROI, %s (AARATEP)', subject, roiName, ttl));
    legend(ax, 'FontSize', 9);
    print(hf, fullfile(figDir, sprintf('sub-%s_ses-%s_desc-TEPaaratep_%s.png', ...
        subject, session, fname)), '-dpng', '-r150');

    t  = results(iHi).times;
    hi = results(iHi).roi;   lo = results(iLo).roi;
    se = sqrt(results(iHi).sem.^2 + results(iLo).sem.^2);
    fprintf('\n%s\n%14s%12s%12s%16s\n', ttl, 'window', 'top', 'bottom', 'difference');
    for iW = 1:size(windows, 1)
        w = t >= windows{iW, 1}(1) & t <= windows{iW, 1}(2);
        fprintf('%14s%7.2f uV%7.2f uV%9.2f +/- %.2f uV\n', windows{iW, 2}, ...
            mean(hi(w)), mean(lo(w)), mean(hi(w)) - mean(lo(w)), mean(se(w)));
    end
    fprintf('\n');
end


function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end


function v = getfielddef(s, name, default)
    if isstruct(s) && isfield(s, name), v = s.(name); else, v = default; end
end
