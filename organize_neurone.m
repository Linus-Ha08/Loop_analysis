%% organize_neurone_recordings.m
%
% Purpose:
%   Convert per-block NeurOne recordings (EEG + EMG, all in one file per
%   NeurOne "phase") into a BIDS-style structure, with trigger/PRIME
%   performance info from the matching CSV attached as an events.tsv
%   sidecar. This is meant to run ONCE per subject, before any filtering/
%   epoching/artifact rejection -- everything downstream should read from
%   the files this script produces, not from the raw NeurOne data directly.
%
% Output, per block:
%   <bids_root>/sub-<subID>/ses-<sesID>/eeg/
%       sub-<subID>_ses-<sesID>_task-<task>_run-<run>_eeg.set
%       sub-<subID>_ses-<sesID>_task-<task>_run-<run>_events.tsv
%
%   events.tsv has one row per REAL NeurOne trigger (not one row per CSV
%   trial): "onset" / "duration" / "trigger_type_neurone" always come
%   from NeurOne itself, and every column from the matching trials_*.csv
%   (condition, is_forced, tep_amplitude, ...) is attached by matching
%   pulse times -- see match_neurone_to_csv_by_time() at the bottom of
%   this file. "csv_match" is false for the rare NeurOne trigger that had
%   no close-enough CSV row (watch the command window for warnings).
%
% Requires EEGLAB + the NeurOne plugin already on the MATLAB path.
%
% ============================ STATUS ============================
% Everything below was a placeholder ("FIXME") until checked directly
% against this project's raw data -- the raw NeurOne events.bin files
% (parsed by hand against the NeurOne Data Format spec), Protocol.xml,
% and the trials_*.csv files. All four are now confirmed for Pilot001:
%   1. evaluation-t30 -> trials_evaluation_4.csv (section 3).
%      trials_evaluation_3.csv exists but is empty (0 trial rows) --
%      presumably a recording that was started, stopped, and immediately
%      restarted as "_4". trials_evaluation_4.csv has exactly as many
%      rows (100) as that recording has TMS triggers, with a rock-steady
%      timing offset between the two (see point 4) -- there's no
%      realistic way this is the wrong file.
%   2. EMG channels are 'FDIr' and 'APBr' (section 4) -- these names are
%      taken straight from this recording's own Protocol.xml montage.
%   3. The real TMS trigger is event type 'A - Stimulation' (section 5).
%      These recordings also contain 'A - Out' (NeurOne's own trigger
%      OUTPUT to the stimulator -- this is a closed-loop setup, so
%      NeurOne both sends and receives trigger pulses), plus 'B - Mute'
%      / 'B - Out' (unrelated SyncBox/headbox housekeeping). All of these
%      must be excluded before matching against the CSV, or the counts
%      below won't line up.
%   4. NeurOne trigger order does equal CSV row order for every block
%      checked in this dataset -- event counts match the CSV row counts
%      exactly, and the timing offset between the two is constant to
%      within ~0.1 ms across a 52-minute, 925-trial recording. Section 5
%      below still matches by TIME rather than leaning on this, though:
%      a future subject/session with one dropped or extra trigger would
%      otherwise silently mislabel every trial after it, and time-based
%      matching catches that instead (and says so loudly) rather than
%      hiding it.
% ==================================================================


%% to confirm per subject: 
% subID, sesID 
% paths NeurOne session files
% trial csv per block (specifically evaluation might be off if stopped and started recording)


clear; clc;
addpath("D:\Linus\MATLAB_applications\eeglab2026.0.0")
eeglab nogui;


%% 1. Subject / session identifiers and output root
subID = 'Pilot002';
sesID = 'prime';

bids_root = 'D:\Linus\Loop\BIDS';

%% 2. NeurOne session files, and which phase/task/run each block is
% SINGLE QUOTES 'path'
path_1 = 'D:\Linus\Loop\Import\Loop_Pilot002\Loop_Pilot002\NeurOne-2026-08-06T112547.ses';
path_2 = 'D:\Linus\Loop\Import\Loop_Pilot002\Loop_Pilot002\NeurOne-2026-08-06T122258.ses';
path_3 = 'D:\Linus\Loop\Import\Loop_Pilot002\Loop_Pilot002\NeurOne-2026-08-06T134706.ses';
path_4 = 'D:\Linus\Loop\Import\Loop_Pilot002\Loop_Pilot002\NeurOne-2026-08-06T141352.ses';

% columns: SubID, SesID, .ses path, NeurOne phase number, task, run
recordings = {
    subID, sesID, path_1, 1, 'baseline',          '01';
    subID, sesID, path_2, 1, 'intervention-all',  '01';
    subID, sesID, path_2, 2, 'evaluation-t0',     '01';
    subID, sesID, path_2, 3, 'evaluation-t15',    '01';
    subID, sesID, path_3, 1, 'evaluation-t30',    '01';
    subID, sesID, path_4, 2, 'evaluation-t60',    '01';  
};


%% 3. Trigger / PRIME-performance CSVs, one per task
trigger_csv_root = fullfile('D:\Linus\Loop\Import\Loop_Pilot002\102\prime');

trigger_csv_map = containers.Map();
trigger_csv_map('baseline')         = fullfile(trigger_csv_root, 'trials_baseline.csv');
trigger_csv_map('intervention-all') = fullfile(trigger_csv_root, 'trials_intervention.csv');
trigger_csv_map('evaluation-t0')    = fullfile(trigger_csv_root, 'trials_evaluation_1.csv');
trigger_csv_map('evaluation-t15')   = fullfile(trigger_csv_root, 'trials_evaluation_2.csv');
trigger_csv_map('evaluation-t30')   = fullfile(trigger_csv_root, 'trials_evaluation_4.csv');
trigger_csv_map('evaluation-t60')   = fullfile(trigger_csv_root, 'trials_evaluation_5.csv'); 


%% 4. EMG channel identification
%    62 channels total, 60 EEG
%    (standard 10-05 montage) + 2 EMG
%    FDIr (right first dorsal interosseous) and APBr (right abductor pollicis brevis)
%    EEG+EMG stay in one continuous file
%    EMG tagged for downstream 
emg_channel_names = {'FDIr', 'APBr'};
 
%% 5. Main loop: read each block, tag channels, build events, save

% The real TMS trigger events use this NeurOne event type (see STATUS
% note at the top of this file for how that was confirmed, and what the
% other event types in these recordings turned out to be). Everything
% else gets dropped before matching against the CSV in step 5c below.
selected_trigger_type = 'A - Stimulation';

% How close (in seconds, after removing this recording's constant clock
% offset -- see match_neurone_to_csv_by_time() at the bottom of this
% file) a NeurOne trigger and a CSV "pulse_time" must be to count as the
% same pulse. 250 ms is generous headroom above the offset's observed
% jitter (well under 1 ms in this dataset) while staying far below the
% smallest real gap between consecutive trials (several seconds here),
% so there is no realistic way for this to pair up the wrong trial.
match_tolerance_sec = 0.25;
 
for r = 1:size(recordings,1)
    subID_now = recordings{r,1};
    sesID_now = recordings{r,2};
    ses_path  = recordings{r,3};
    phase_num = recordings{r,4};
    task_now  = recordings{r,5};
    run_now   = recordings{r,6};
 
    fprintf('\n=== sub-%s ses-%s task-%s run-%s (NeurOne phase %d) ===\n', ...
        subID_now, sesID_now, task_now, run_now, phase_num);
 
    %% 5a. Read the NeurOne phase (all channels, EEG+EMG together)
    EEG = pop_readneurone(ses_path, phase_num, '');
    EEG = eeg_checkset(EEG);
    EEG = eeg_checkset(EEG, 'eventconsistency');
    EEG = eeg_checkset(EEG, 'makeur');  % stable urevent IDs, same as at REFTEP's import stage
 
    %% 5b. Tag EMG vs EEG channels (labels only, nothing removed)
    chan_labels = {EEG.chanlocs.labels};
    is_emg = ismember(chan_labels, emg_channel_names);
    for c = 1:EEG.nbchan
        if is_emg(c)
            EEG.chanlocs(c).type = 'EMG';
        else
            EEG.chanlocs(c).type = 'EEG';
        end
    end
    fprintf('Tagged %d EMG channel(s): %s\n', sum(is_emg), strjoin(chan_labels(is_emg), ', '));
 
    %% 5c. Optionally restrict to a specific trigger type before matching
    if ~isempty(selected_trigger_type)
        keep_mask = strcmp({EEG.event.type}, selected_trigger_type);
    else
        keep_mask = true(1, length(EEG.event));
    end
    event_subset = EEG.event(keep_mask);
    n_events = length(event_subset);
    onsets_sec = ([event_subset.latency] - 1) / EEG.srate;
 
    %% 5d. Load the matching trigger / PRIME-performance CSV
    if isKey(trigger_csv_map, task_now)
        csv_path = trigger_csv_map(task_now);
        if isfile(csv_path)
            trial_info = readtable(csv_path);
        else
            warning('Trigger CSV not found for task "%s": %s', task_now, csv_path);
            trial_info = table();
        end
    else
        warning('No trigger CSV registered for task "%s"', task_now);
        trial_info = table();
    end
 
    %% 5e. Build the events table: NeurOne timing + CSV trial info,
    %     matched by TIME (see match_neurone_to_csv_by_time() at the
    %     bottom of this file), not by assuming row order lines up.
    events_table = table();
    events_table.onset = onsets_sec(:);
    events_table.duration = zeros(n_events,1);
    events_table.trigger_type_neurone = {event_subset.type}';

    if isempty(trial_info)
        % No CSV available for this task (missing file / no mapping --
        % already warned about in step 5d). events_table stays NeurOne-only.

    elseif ~ismember('pulse_time', trial_info.Properties.VariableNames)
        warning(['CSV for task "%s" has no "pulse_time" column -- cannot ' ...
                 'time-match against NeurOne events. Saving NeurOne-only events.tsv.'], ...
                 task_now);

    else
        % row_for_event(i) = which row of trial_info the i-th NeurOne
        % trigger was matched to in time, or NaN if none was close enough.
        [row_for_event, match_report] = match_neurone_to_csv_by_time( ...
            onsets_sec, trial_info.pulse_time, match_tolerance_sec);

        fprintf(['  Time-matched %d/%d NeurOne triggers to CSV rows ' ...
                 '(clock offset %.3f s, worst residual %.1f ms).\n'], ...
                 match_report.n_matched, n_events, match_report.offset_sec, ...
                 1000 * max([0; abs(match_report.residual_sec)]));

        if ~isempty(match_report.unmatched_event_idx)
            warning(['%d NeurOne trigger(s) in task "%s" had no CSV row within ' ...
                     '%.0f ms (possible extra/spurious trigger). Onset time(s) [s]: %s'], ...
                     numel(match_report.unmatched_event_idx), task_now, ...
                     1000 * match_tolerance_sec, ...
                     mat2str(round(onsets_sec(match_report.unmatched_event_idx), 3)));
        end
        if ~isempty(match_report.unmatched_trial_idx)
            warning(['%d CSV row(s) in task "%s" had no NeurOne trigger within ' ...
                     '%.0f ms (that trial may not have actually fired). CSV row number(s): %s'], ...
                     numel(match_report.unmatched_trial_idx), task_now, ...
                     1000 * match_tolerance_sec, ...
                     mat2str(match_report.unmatched_trial_idx));
        end

        % Attach every CSV column, one output row per NeurOne trigger:
        % matched triggers get that CSV row's values; any unmatched
        % trigger (csv_match = false) is left blank/NaN rather than
        % silently dropped, so it's still visible in the saved .tsv.
        matched_mask = ~isnan(row_for_event);
        events_table.csv_match = matched_mask;

        csv_columns = trial_info.Properties.VariableNames;
        for ci = 1:numel(csv_columns)
            col_name = csv_columns{ci};
            src_col = trial_info.(col_name);
            if iscell(src_col)
                out_col = repmat({''}, n_events, 1);
            elseif isstring(src_col)
                out_col = repmat("", n_events, 1);
            else
                out_col = nan(n_events, 1);  % also covers logical columns (True/False -> 1/0/NaN)
            end
            out_col(matched_mask) = src_col(row_for_event(matched_mask));
            events_table.(col_name) = out_col;
        end
    end
 
    %% 5f. Save into BIDS-style folders
    out_dir = fullfile(bids_root, ['sub-' subID_now], ['ses-' sesID_now], 'eeg');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end
 
    base_name = sprintf('sub-%s_ses-%s_task-%s_run-%s', subID_now, sesID_now, task_now, run_now);
 
    pop_saveset(EEG, 'filename', [base_name '_eeg.set'], 'filepath', out_dir);
    writetable(events_table, fullfile(out_dir, [base_name '_events.tsv']), ...
        'FileType', 'text', 'Delimiter', '\t');
 
    fprintf('Saved:\n  %s\n  %s\n', ...
        fullfile(out_dir, [base_name '_eeg.set']), ...
        fullfile(out_dir, [base_name '_events.tsv']));
end
 
disp('Done -- all blocks organized into BIDS structure.');


%% ========================================================================
%%  Local function: time-based matching between NeurOne and the CSV
%% ========================================================================
function [row_for_event, report] = match_neurone_to_csv_by_time(event_onsets_sec, trial_pulse_times_sec, tolerance_sec)
% MATCH_NEURONE_TO_CSV_BY_TIME  Pair real NeurOne trigger events to CSV
% trial rows using elapsed time, instead of assuming the two lists simply
% line up row-for-row.
%
% WHY THIS EXISTS
%   An earlier version of this script just glued the CSV onto the NeurOne
%   events side by side, on the assumption "NeurOne event #k is always
%   CSV row #k". For this project's Pilot001 data that assumption happens
%   to be exactly true (see the STATUS note at the top of this file), but
%   it is a dangerous thing to assume silently: a single dropped trigger,
%   one extra spurious event, or a trial that failed to fire anywhere in
%   a block would shift every later row by one, silently mislabelling
%   potentially hundreds of trials afterwards (e.g. tagging a 100 Hz
%   triplet as a lone single pulse, or vice versa) with no error or
%   warning at all. Matching by actual pulse time catches this instead of
%   hiding it -- for this subject, and for any future one where the
%   counts might not line up so neatly.
%
% THE IDEA
%   Both lists are already in chronological order: NeurOne events by
%   recording time, CSV rows by trial_in_stage. The CSV's own pulse_time
%   column is "seconds since that CSV's own clock started", which is NOT
%   the same zero-point as "seconds since this NeurOne recording started"
%   -- but the two clocks tick at the same rate, so across one recording
%   they differ by a single constant offset. (Checked directly against
%   this project's raw events.bin files: that offset is constant to
%   within ~0.1 ms over a 52-minute, 925-trial recording -- i.e. for all
%   practical purposes it never drifts.)
%
%   So: find that one constant offset (step 1), then walk both time-sorted
%   lists together (like riffling two sorted decks of cards), pairing up
%   whichever trial is closest to each event, as long as it's within
%   `tolerance_sec` (step 2, in the local function two_pointer_match()
%   below). Anything without a close-enough partner on the other side is
%   left unmatched and reported, rather than forced into a wrong pairing.
%
% INPUTS
%   event_onsets_sec       Nx1  Real NeurOne trigger times, in seconds
%                                relative to THIS recording's start.
%   trial_pulse_times_sec  Mx1  The CSV's "pulse_time" column, in seconds
%                                relative to THAT CSV's own start.
%   tolerance_sec           1x1  How close an event and a trial must be
%                                (after removing the constant offset) to
%                                count as the same pulse. Pick something
%                                well above the offset's real-world
%                                jitter (< 1 ms here) and well below the
%                                smallest realistic gap between
%                                consecutive trials (several seconds in
%                                this experiment), so there is no
%                                realistic way to pair the wrong trial.
%
% OUTPUTS
%   row_for_event   Nx1  For each NeurOne event, the row of
%                         trial_pulse_times_sec it was matched to, or NaN
%                         if nothing was close enough.
%   report          struct with, for logging/inspection by the caller:
%                     .offset_sec           estimated constant clock offset
%                     .n_matched             how many events got a match
%                     .unmatched_event_idx   events with no matching trial
%                     .unmatched_trial_idx   trials with no matching event
%                     .residual_sec          matched pairs' leftover timing
%                                             error after removing the
%                                             offset (should be tiny)
%
% KNOWN LIMITATION
%   Step 1 needs at least one of the trials tried as a candidate (see
%   below) to be a genuinely correct NeurOne-event/CSV-row pair. That
%   fails only if the very first trial(s) of the block itself never fired
%   AND nothing later in the block happens to sit at a clean offset
%   either -- vanishingly unlikely in practice, and not something this
%   dataset's blocks (checked directly, see STATUS note) ever does, but
%   worth knowing about if a whole future block comes back unmatched.

    event_onsets_sec = event_onsets_sec(:);
    trial_pulse_times_sec = trial_pulse_times_sec(:);
    n_events = numel(event_onsets_sec);
    n_trials = numel(trial_pulse_times_sec);

    row_for_event = nan(n_events, 1);
    report = struct('offset_sec', NaN, 'n_matched', 0, ...
        'unmatched_event_idx', (1:n_events)', ...
        'unmatched_trial_idx', (1:n_trials)', ...
        'residual_sec', []);

    if n_events == 0 || n_trials == 0
        return  % nothing to match on one side or the other
    end

    % --- Step 1: find the constant clock offset ---
    % Pair events to trials naively by position (1st with 1st, 2nd with
    % 2nd, ...) over whatever overlap the two lists have -- this gives a
    % list of CANDIDATE offsets, one of which is very likely correct even
    % if most of the others aren't (a dropped or extra event anywhere
    % before it would throw naive position-pairing off for everything
    % after it, which is exactly the failure mode this whole function
    % exists to avoid).
    %
    % Rather than guess which candidate to trust (a plain average/median
    % over all of them can be fooled if the true one happens to be a
    % minority -- e.g. an anomaly early in the block), just TRY EVERY
    % CANDIDATE: actually run the two-pointer match (below) with each one,
    % and keep whichever candidate matches the most trials. A wrong
    % candidate is off by roughly one random inter-trial gap (seconds), so
    % it misaligns nearly everything and matches almost nothing; the true
    % offset matches nearly everything. That gap between "almost nothing"
    % and "almost everything" is what makes the best-scoring candidate
    % reliable. This costs at most n_overlap full match attempts (a few
    % hundred for the largest block here), which runs in well under a
    % second -- entirely fine for a script that runs once per subject.
    n_overlap = min(n_events, n_trials);
    candidate_offsets = event_onsets_sec(1:n_overlap) - trial_pulse_times_sec(1:n_overlap);

    best_offset = candidate_offsets(1);
    best_n_matched = -1;
    for k = 1:n_overlap
        trial_candidate_row = two_pointer_match(event_onsets_sec, trial_pulse_times_sec, candidate_offsets(k), tolerance_sec);
        n_ok = sum(~isnan(trial_candidate_row));
        if n_ok > best_n_matched
            best_n_matched = n_ok;
            best_offset = candidate_offsets(k);
        end
    end

    % --- Step 2: refine and do the real match ---
    % The winning candidate above came from a single naive pair, so it can
    % be off by the same tiny (sub-millisecond) jitter every real pair
    % has. Re-estimate it as the median over ALL the matches it actually
    % found (much more precise), then match once more with that refined
    % offset. Two passes is enough: this project's data is already
    % essentially perfect after the first, and the second is cheap
    % insurance for a noisier future recording.
    offset_sec = best_offset;
    for pass = 1:2
        row_for_event = two_pointer_match(event_onsets_sec, trial_pulse_times_sec, offset_sec, tolerance_sec);
        matched = ~isnan(row_for_event);
        if ~any(matched)
            break  % nothing matched at all -- another pass won't fix that
        end
        offset_sec = median(event_onsets_sec(matched) - trial_pulse_times_sec(row_for_event(matched)));
    end

    matched = ~isnan(row_for_event);
    report.offset_sec = offset_sec;
    report.n_matched = sum(matched);
    report.unmatched_event_idx = find(~matched);
    report.unmatched_trial_idx = setdiff((1:n_trials)', row_for_event(matched));
    report.residual_sec = event_onsets_sec(matched) - (trial_pulse_times_sec(row_for_event(matched)) + offset_sec);
end


function row_for_event = two_pointer_match(event_onsets_sec, trial_pulse_times_sec, offset_sec, tolerance_sec)
% TWO_POINTER_MATCH  For ONE given constant clock offset, walk both
% time-sorted lists together (like riffling two sorted decks of cards)
% and pair up whichever trial is closest to each event, as long as it's
% within tolerance_sec. Used by match_neurone_to_csv_by_time() above, both
% to score candidate offsets and to compute the final match.
    n_events = numel(event_onsets_sec);
    n_trials = numel(trial_pulse_times_sec);
    row_for_event = nan(n_events, 1);
    j = 1;  % next not-yet-used row of trial_pulse_times_sec to try
    for i = 1:n_events
        % Express this event's time on the CSV's own clock, so it can be
        % compared directly to trial_pulse_times_sec.
        target = event_onsets_sec(i) - offset_sec;

        % Advance j forward past any trial that's a worse match than the
        % very next one -- since both lists are sorted, the distance to
        % `target` decreases monotonically down to the true nearest trial
        % and then increases again, so this correctly skips over any
        % trial(s) with no real event (e.g. one that failed to fire).
        while j < n_trials && ...
                abs(trial_pulse_times_sec(j+1) - target) <= abs(trial_pulse_times_sec(j) - target)
            j = j + 1;
        end

        if j <= n_trials && abs(trial_pulse_times_sec(j) - target) <= tolerance_sec
            row_for_event(i) = j;
            j = j + 1;  % that trial is now used up; the next event can't reuse it
        end
        % (if it's not within tolerance, j is left where it is, so an
        % extra/spurious event just gets skipped without consuming a
        % trial that a later, genuine event still needs)
    end
end