% quick_raw_TEP_check.m
% -------------------------------------------------------------------------
% Load one raw NeurOne block from the BIDS tree, scroll it, and plot a
% stimulus-locked average over one electrode.
%
% No preprocessing: no filtering, no artifact handling, no interpolation,
% no channel/trial rejection. The only things done to the data are the ones
% needed for the plot to mean anything:
%   - EMG channels (FDIr/APBr) dropped, so they cannot enter the reference
%   - FCz added back as a zero channel (it was the online reference and is
%     not in the file), so the average is over all 61 electrodes
%   - standard 10-05 montage written over the NeurOne positions
%   - average reference
%   - (optional, on by default) baseline removal -- set BASELINE = [] to skip
%
% Nothing under prime/ or BIDS/ is written or modified.
% -------------------------------------------------------------------------

clear; close all; clc;

%% ----------------------------- choose here ------------------------------
BIDS_ROOT = 'D:/Linus/Loop/BIDS';

SUBJECT   = 'pilot003';          % sub-<SUBJECT>
SESSION   = 'prime';             % ses-<SESSION>
BLOCK     = 'baseline';  % task-<BLOCK>; leave '' to list what exists
                                 % baseline, intervention-all,
                                 % evaluation-t0

CHANNEL   = 'Cz';                % electrode for the average
PULSE_EVT = 'A - Stimulation';   % the only real TMS trigger (not A - Out / B - *)

EPOCH     = [-0.500  0.500];     % s, epoch window
BASELINE  = [-500   -10];        % ms, [] to disable
ZOOM      = [-50     250];       % ms, window for the zoomed panel

SCROLL_RAW = false;              % true = scroll as loaded, false = scroll avg-ref
%% -------------------------------------------------------------------------

if ~exist('pop_loadset', 'file')
    eeglab nogui;
end

%% 1. find the .set ---------------------------------------------------------
eegDir = fullfile(BIDS_ROOT, ['sub-' SUBJECT], ['ses-' SESSION], 'eeg');
assert(isfolder(eegDir), 'No such folder: %s', eegDir);

if isempty(BLOCK)
    listBlocks(eegDir);
    error('Set BLOCK to one of the task labels listed above.');
end

hits = dir(fullfile(eegDir, sprintf('sub-%s_ses-%s_task-%s*_eeg.set', ...
                                    SUBJECT, SESSION, BLOCK)));
if isempty(hits)
    listBlocks(eegDir);
    error('No .set matching task-%s in %s -- see the blocks listed above.', BLOCK, eegDir);
elseif numel(hits) > 1
    fprintf('%d files match task-%s:\n', numel(hits), BLOCK);
    fprintf('   %s\n', hits.name);
    warning('Taking the first one. Make BLOCK more specific to pick another.');
end

setFile = hits(1).name;
fprintf('\nLoading %s\n', fullfile(eegDir, setFile));
EEG = pop_loadset('filename', setFile, 'filepath', eegDir);
EEG = eeg_checkset(EEG);
fprintf('  %d channels, %.1f s, %g Hz\n', EEG.nbchan, EEG.xmax - EEG.xmin, EEG.srate);

EEG_raw = EEG;   % kept untouched for scrolling / comparison

%% 2. events ----------------------------------------------------------------
% The NeurOne triggers should already be in the .set. If they are not (or the
% pulse type is absent), fall back to the _events.tsv sidecar, which has one
% row per real TMS pulse.
haveEvents = ~isempty(EEG.event) && ...
             any(strcmp(cellfun(@num2str, {EEG.event.type}, 'uni', 0), PULSE_EVT));

if haveEvents
    types = cellfun(@num2str, {EEG.event.type}, 'uni', 0);
    [u, ~, k] = unique(types);
    fprintf('\nEvent types in the .set:\n');
    for i = 1:numel(u)
        fprintf('  %-22s %5d\n', u{i}, sum(k == i));
    end
else
    tsv = fullfile(eegDir, strrep(setFile, '_eeg.set', '_events.tsv'));
    assert(isfile(tsv), ...
        ['"%s" not found among the .set events and no sidecar at\n  %s\n' ...
         'Check PULSE_EVT against the event list printed above.'], PULSE_EVT, tsv);
    fprintf('\n"%s" not in the .set events -- importing from %s\n', PULSE_EVT, tsv);

    T = readtable(tsv, 'FileType', 'text', 'Delimiter', '\t');
    assert(ismember('onset', T.Properties.VariableNames), 'No onset column in the sidecar.');

    EEG.event = [];
    for i = 1:height(T)
        EEG.event(i).type    = PULSE_EVT;
        EEG.event(i).latency = T.onset(i) * EEG.srate + 1;   % s -> 1-based samples
        EEG.event(i).duration = 1;
    end
    EEG = eeg_checkset(EEG, 'eventconsistency');
    fprintf('  %d events imported\n', numel(EEG.event));
end

nPulse = sum(strcmp(cellfun(@num2str, {EEG.event.type}, 'uni', 0), PULSE_EVT));
fprintf('\n%d "%s" triggers\n', nPulse, PULSE_EVT);
assert(nPulse > 0, 'No "%s" triggers found.', PULSE_EVT);

%% 3. EMG out, FCz back in, montage, average reference ----------------------
% EMG: labelled FDIr / APBr, and typed in chanlocs.type. AARATEP's own EMG
% rule does not catch these labels, so they are dropped by name here.
emg = find(ismember(upper({EEG.chanlocs.labels}), {'FDIR', 'APBR'}));
if ~isempty(emg)
    fprintf('Dropping EMG: %s\n', strjoin({EEG.chanlocs(emg).labels}, ', '));
    EEG = pop_select(EEG, 'nochannel', emg);
end

% FCz was the online reference (AFz = ground), so it is not in the file.
% Without it the "average" reference is an average over 60 of 61 electrodes
% and every topography tilts. pop_reref(...,'refloc',...) does not work on
% these files -- reref() copies every field off the refloc struct and
% pop_reref wants the channel in chaninfo.nodatchans, which a NeurOne import
% does not populate. Appending a zero channel and calling pop_reref(EEG,[])
% gives the identical result.
if ~any(strcmpi({EEG.chanlocs.labels}, 'FCz'))
    EEG.data(end+1, :)          = 0;
    EEG.nbchan                  = EEG.nbchan + 1;
    EEG.chanlocs(end+1).labels  = 'FCz';
    EEG.chanlocs(end).type      = 'EEG';
    fprintf('Added FCz back as a zero channel (online reference)\n');
end

% NeurOne's own positions are not reliable -- overwrite with standard 10-05.
montage = which('standard_1005.elc');
if isempty(montage), montage = which('standard-10-5-cap385.elp'); end
assert(~isempty(montage), 'No standard 10-05 montage on the path (dipfit plugin missing?).');
EEG = pop_chanedit(EEG, 'lookup', montage);
EEG = eeg_checkset(EEG);

noPos = find(cellfun(@isempty, {EEG.chanlocs.X}));
if ~isempty(noPos)
    warning('No position for: %s', strjoin({EEG.chanlocs(noPos).labels}, ', '));
end

EEG.data = EEG.data - mean(EEG.data, 2);   % per-channel DC, whole recording
EEG = pop_reref(EEG, []);
EEG.setname = sprintf('%s %s task-%s | avg ref, unprocessed', SUBJECT, SESSION, BLOCK);
fprintf('Average reference over %d electrodes\n', EEG.nbchan);

%% 4. scroll ----------------------------------------------------------------
if SCROLL_RAW
    pop_eegplot(EEG_raw, 1, 1, 1);   % as loaded: 62 ch, online reference
else
    pop_eegplot(EEG, 1, 1, 1);       % avg ref, FCz restored
end

%% 5. epoch and average one electrode --------------------------------------
ch = find(strcmpi({EEG.chanlocs.labels}, CHANNEL), 1);
assert(~isempty(ch), 'Channel "%s" not found. Available: %s', ...
       CHANNEL, strjoin({EEG.chanlocs.labels}, ' '));

EP = pop_epoch(EEG, {PULSE_EVT}, EPOCH, 'epochinfo', 'yes');
if ~isempty(BASELINE)
    EP = pop_rmbase(EP, BASELINE);
end
fprintf('%d epochs, %g to %g ms\n', EP.trials, EP.xmin*1000, EP.xmax*1000);

t    = EP.times;
trls = squeeze(EP.data(ch, :, :));      % time x trials
avg  = mean(trls, 2);
sem  = std(trls, 0, 2) ./ sqrt(size(trls, 2));

figure('Color', 'w', 'Position', [100 100 1000 720], ...
       'Name', sprintf('%s %s task-%s | %s', SUBJECT, SESSION, BLOCK, CHANNEL));

% full window -- the pulse artifact dominates, that is the point of this panel
subplot(2, 1, 1);
plot(t, trls, 'Color', [0.8 0.8 0.8 0.35]); hold on;
plot(t, avg, 'k', 'LineWidth', 1.4);
xline(0, 'r'); yline(0, 'Color', [0.6 0.6 0.6]);
xlim(EPOCH * 1000); box off;
xlabel('ms'); ylabel('\muV');
title(sprintf('%s -- %d trials, single trials in grey (artifact on scale)', ...
              CHANNEL, EP.trials));

% zoom -- y-limits from the data outside the pulse, or the artifact flattens it
subplot(2, 1, 2);
fill([t fliplr(t)], [avg'-sem' fliplr(avg'+sem')], [0.2 0.4 0.8], ...
     'FaceAlpha', 0.25, 'EdgeColor', 'none'); hold on;
plot(t, avg, 'Color', [0.1 0.2 0.6], 'LineWidth', 1.6);
xline(0, 'r'); yline(0, 'Color', [0.6 0.6 0.6]);
clean = avg(t >= 20 & t <= ZOOM(2));
if ~isempty(clean)
    m = max(abs(clean)) * 1.6;
    if m > 0, ylim([-m m]); end
end
xlim(ZOOM); box off;
xlabel('ms'); ylabel('\muV');
title(sprintf('%s -- average \\pm SEM, y scaled to the post-20 ms range', CHANNEL));

fprintf('\nDone. Scroll window and figure are open.\n');

%% 5.2. resample to 1 kHz, epoch, average one electrode ----------------------
% pop_resample calls MATLAB's resample, i.e. a Kaiser-windowed polyphase FIR
% with the defaults n = 10, beta = 5 -> 2*10*max(p,q)+1 = 101 taps at 5->1 kHz,
% cutoff at the new Nyquist. Those are the same design parameters scipy's
% resample_poly uses by default, which is the resample inside the online
% preprocess_post. What differs is the edges: MATLAB zero-pads and compensates
% the delay, MNE reflect-pads, and that reaches ~50 raw samples (10 ms) from a
% segment boundary. Resampling the continuous data and epoching afterwards
% keeps those edges at the recording boundary; online the resample runs on the
% already-cropped [-30, 100] ms slice, so its padded ends sit inside the epoch.
% To mirror online instead, epoch to [-0.030 0.100] first and resample that.
RESAMPLE_TO = 1000;
 
if EEG.srate ~= RESAMPLE_TO
    fprintf('\nResampling %g -> %g Hz (anti-alias cutoff %g Hz)\n', ...
            EEG.srate, RESAMPLE_TO, RESAMPLE_TO/2);
    EEG = pop_resample(EEG, RESAMPLE_TO);
    EEG = eeg_checkset(EEG);
end
 
ch = find(strcmpi({EEG.chanlocs.labels}, CHANNEL), 1);
assert(~isempty(ch), 'Channel "%s" not found. Available: %s', ...
       CHANNEL, strjoin({EEG.chanlocs.labels}, ' '));
 
EP = pop_epoch(EEG, {PULSE_EVT}, EPOCH, 'epochinfo', 'yes');
if ~isempty(BASELINE)
    EP = pop_rmbase(EP, BASELINE);
end
fprintf('%d epochs, %g to %g ms\n', EP.trials, EP.xmin*1000, EP.xmax*1000);
 
t    = EP.times;
trls = squeeze(EP.data(ch, :, :));      % time x trials
avg  = mean(trls, 2);
sem  = std(trls, 0, 2) ./ sqrt(size(trls, 2));
 
figure('Color', 'w', 'Position', [100 100 1000 720], ...
       'Name', sprintf('%s %s task-%s | %s', SUBJECT, SESSION, BLOCK, CHANNEL));
 
% full window -- the pulse artifact dominates, that is the point of this panel
subplot(2, 1, 1);
plot(t, trls, 'Color', [0.8 0.8 0.8 0.35]); hold on;
plot(t, avg, 'k', 'LineWidth', 1.4);
xline(0, 'r'); yline(0, 'Color', [0.6 0.6 0.6]);
xlim(EPOCH * 1000); box off;
xlabel('ms'); ylabel('\muV');
title(sprintf('%s -- %d trials, single trials in grey (artifact on scale)', ...
              CHANNEL, EP.trials));
 
% zoom -- y-limits from the data outside the pulse, or the artifact flattens it
subplot(2, 1, 2);
fill([t fliplr(t)], [avg'-sem' fliplr(avg'+sem')], [0.2 0.4 0.8], ...
     'FaceAlpha', 0.25, 'EdgeColor', 'none'); hold on;
plot(t, avg, 'Color', [0.1 0.2 0.6], 'LineWidth', 1.6);
xline(0, 'r'); yline(0, 'Color', [0.6 0.6 0.6]);
clean = avg(t >= 20 & t <= ZOOM(2));
if ~isempty(clean)
    m = max(abs(clean)) * 1.6;
    if m > 0, ylim([-m m]); end
end
xlim(ZOOM); box off;
xlabel('ms'); ylabel('\muV');
title(sprintf('%s -- average \\pm SEM, y scaled to the post-20 ms range', CHANNEL));
 
fprintf('\nDone. Scroll window and figure are open.\n');
 
%% -------------------------------------------------------------------------
function listBlocks(eegDir)
    f = dir(fullfile(eegDir, '*_eeg.set'));
    if isempty(f)
        fprintf('\nNo _eeg.set files in %s\n', eegDir);
        return;
    end
    fprintf('\nBlocks available in %s:\n', eegDir);
    for i = 1:numel(f)
        tok = regexp(f(i).name, '_task-(.+?)_eeg\.set$', 'tokens', 'once');
        if ~isempty(tok), fprintf('  %s\n', tok{1}); end
    end
end