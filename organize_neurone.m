%% organize_neurone.m
% Converts one subject's NeurOne recordings (60 EEG + 2 EMG channels) into BIDS-style
% files, one block at a time:
%   <bids_root>/sub-<subID>/ses-<sesID>/eeg/sub-<subID>_ses-<sesID>_task-<task>_run-01_eeg.set
%   ..._events.tsv  (TMS blocks only: one row per NeurOne TMS trigger, with every column
%                    of the matching trials_*.csv row attached, matched by pulse time)
% Resting-state blocks (rsEEG-*) have no TMS and no CSV and get only the _eeg.set.
% Nothing is filtered or removed here; everything downstream reads these files.
%
% Per subject, set section 1: subID, the .ses files, each block's NeurOne phase number
% (phases are numbered in recording order within each .ses), and the CSV folder.
% Requires EEGLAB with the NeurOne plugin.

clear; clc;
addpath('D:\Linus\MATLAB_applications\eeglab2026.0.0');
eeglab nogui;
nogui_dir = 'D:\Linus\Loop\scripts_analysis\neurone_nogui';  % stops the channel-editor pop-ups

%% 1. Subject and blocks
subID = 'Loop002';
sesID = 'prime';
bids_root = 'D:\Linus\Loop\BIDS';
overwrite = false;   % true = replace existing output files

ses_1 = 'E:\Loop002\NeurOne-2026-11-06T175923.ses';
ses_2 = 'E:\Loop002\NeurOne-2026-11-06T201424.ses';

csv_dir = 'D:\Linus\Loop\BIDS\sub-Loop002\ses-prime\2\prime';

% .ses file, NeurOne phase, task, trials CSV in csv_dir ('' = resting state)
blocks = {
    ses_1, 1, 'baseline',          'trials_baseline.csv';
    ses_1, 2, 'rsEEG-baseline',    '';
    ses_1, 3, 'intervention-all',  'trials_intervention.csv';
    ses_1, 4, 'evaluation-t0',     'trials_evaluation_1.csv';
    ses_1, 5, 'rsEEG-eval-t0',     '';
    ses_1, 6, 'evaluation-t15',    'trials_evaluation_2.csv';
    ses_1, 7, 'rsEEG-eval-t15',    '';
    ses_1, 8, 'evaluation-t30',    'trials_evaluation_3.csv';
    ses_1, 9, 'rsEEG-eval-t30',    '';
    ses_2, 1, 'evaluation-t60',    'trials_evaluation_4.csv';
    ses_2, 2, 'rsEEG-eval-t60',    '';
};

emg_channels = {'FDIr', 'APBr'};
trigger_type = 'A - Stimulation';  % TMS pulse as received by NeurOne ('A - Out', 'B - Mute',
                                   % 'B - Out' are trigger outputs / housekeeping)
tolerance = 0.25;                  % s; max NeurOne-vs-CSV pulse time difference for a match

%% 2. Checks before anything is written
out_dir = fullfile(bids_root, ['sub-' subID], ['ses-' sesID], 'eeg');
name = @(task) sprintf('sub-%s_ses-%s_task-%s_run-01', subID, sesID, task);

% A wrong subID silently overwrites another subject: Windows treats sub-Pilot003 and
% sub-pilot003 as the same folder. So subID must appear in every path.
key = @(s) lower(regexprep(s, '[^A-Za-z0-9]', ''));   % 'E:\Loop_Pilot_001' -> 'elooppilot001'
assert(all(cellfun(@(p) contains(key(p), key(subID)), [blocks(:,1); {csv_dir}])), ...
    'subID "%s" is not in every .ses path and csv_dir -- wrong subID or wrong path?', subID);

existing = blocks(cellfun(@(t) isfile(fullfile(out_dir, [name(t) '_eeg.set'])), blocks(:,3)), 3);
assert(overwrite || isempty(existing), ...
    'Output already exists for task(s) %s -- set overwrite = true to replace.', strjoin(existing', ', '));
if ~isfolder(out_dir), mkdir(out_dir); end

%% 3. Convert each block
for b = 1:size(blocks, 1)
    [ses_path, phase, task, csv_name] = blocks{b, :};
    fprintf('\n=== task-%s  (%s, phase %d) ===\n', task, ses_path, phase);

    addpath(nogui_dir);
    EEG = pop_readneurone(ses_path, phase, '');  % also runs eeg_checkset ('eventconsistency', 'makeur')
    rmpath(nogui_dir);

    % Channel types (labels only, nothing removed)
    is_emg = ismember({EEG.chanlocs.labels}, emg_channels);
    [EEG.chanlocs(is_emg).type] = deal('EMG');
    [EEG.chanlocs(~is_emg).type] = deal('EEG');

    % TMS triggers. EEG.event is [] (not a struct) when the phase has no events at all.
    if isempty(EEG.event)
        ev = struct('type', {}, 'latency', {});
    else
        ev = EEG.event(strcmp({EEG.event.type}, trigger_type));
    end
    onset = reshape([ev.latency] - 1, [], 1) / EEG.srate;   % s from recording start
    fprintf('  %.1f min, %d TMS triggers\n', EEG.pnts / EEG.srate / 60, numel(onset));

    % Resting state: save the EEG only, after making sure it really has no TMS
    if isempty(csv_name)
        if ~isempty(onset)
            warning('task-%s has %d TMS triggers -- wrong phase number? NOT saved.', task, numel(onset));
            continue
        end
        pop_saveset(EEG, 'filename', [name(task) '_eeg.set'], 'filepath', out_dir);
        continue
    end

    % TMS block: match each trigger to its CSV row by time. The two clocks start at
    % different moments but run at the same rate, so their offset is constant: it is
    % the most frequent trigger-minus-pulse_time difference over all pairs.
    trials = readtable(fullfile(csv_dir, csv_name));
    assert(height(trials) > 0, '%s is empty -- wrong CSV for task-%s?', csv_name, task);
    pulse = trials.pulse_time;
    offset = mode(round(onset - pulse', 2), 'all');
    [err, row] = min(abs(onset - offset - pulse'), [], 2);
    matched = err < tolerance;

    d = onset(matched) - pulse(row(matched));
    fprintf('  %d/%d triggers matched to CSV rows (clock offset %.3f s, jitter %.1f ms)\n', ...
        nnz(matched), numel(onset), median(d), 1000 * max(abs(d - median(d))));
    n_rows = numel(unique(row(matched)));
    if nnz(matched) < numel(onset) || n_rows < height(trials) || n_rows < nnz(matched)
        warning(['task-%s: %d of %d triggers matched, %d of %d CSV rows matched ' ...
                 '-- check this block before using it.'], ...
                 task, nnz(matched), numel(onset), n_rows, height(trials));
    end

    % events.tsv: NeurOne timing + the matched CSV row (blank where nothing matched)
    events = table(onset, zeros(size(onset)), reshape({ev.type}, [], 1), matched, ...
        'VariableNames', {'onset', 'duration', 'trigger_type_neurone', 'csv_match'});
    for v = trials.Properties.VariableNames
        col = trials.(v{1});
        if iscell(col), out = repmat({''}, size(onset)); else, out = nan(size(onset)); end
        out(matched) = col(row(matched));
        events.(v{1}) = out;
    end

    pop_saveset(EEG, 'filename', [name(task) '_eeg.set'], 'filepath', out_dir);
    writetable(events, fullfile(out_dir, [name(task) '_events.tsv']), ...
        'FileType', 'text', 'Delimiter', '\t');
end
disp('Done.');
