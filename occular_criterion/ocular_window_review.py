#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ocular_window_review.py - the ocular gate for the N45: which window (15-100, 30-100, 30-60, 20-60, 20-70 ms), which
reference (the window itself, or the ICA window before the pulse) and which cutoff?

Follow-up of ocular_threshold_review.py (same folder). It imports that script's machinery (git export, the override of
_detect_ocular_trials, the replay recorder, the N45 fit) and builds on its results in Loop/offline_ocular_window/replay
(the decider as it runs, 'current', and the 15-100 ms window): run ocular_threshold_review.py first.

The gate: a pulse is rejected when, for any eye IC, the median of z(t) = (|s(t)| - mean) / SD over the gate window
exceeds the cutoff z. Two things are chosen here:
  - the window: the samples the score is the median of (the decider's trial_reject_opts.ocular.post_timerange, now
    [15 ms, end]);
  - the reference (mean, SD of |s|): the window references take them from the calibration block's post epochs over the
    same window (detect_ocular_trials_ref, reference "gate"); the reference before the pulse (design PRE = 'pre20-60')
    takes them from every sample of the ICA window - the extended pre-stimulus window the decider fits the ICA on
    (-1100 to -5 ms of each calibration trial, 1 kHz, band-passed 1-100 Hz, average reference) - and applies them to
    the 20-60 ms gate window of the post epoch.
What has to be protected is the N45 (dipole window 38-50 ms); eye activity after it does not matter.

Why: in pilot006 an eye IC carries the decay of the TMS artifact right after the blank (median |s| 8.8 at 15-30 ms,
0.8 later). Over 15-100 ms it triples that IC's reference SD (4.46 vs 1.55 over 30-100 ms), so the IC can hardly
reject a pulse. Windows starting at 20 or 30 ms leave (part of) the transient out of the window reference; the
reference before the pulse leaves it out altogether.

How it runs (nothing in prime/ is written)
------------------------------------------
Child processes per session, importing the same decider export as the first review (HEAD, 007b932):
  part "windows"
  1. Calibrations: for every window W and cutoff z in Z_GRID, calibrate() on the session's 125 calibration trials with
     post_timerange = W, reference = W and z_threshold = z - the ocular drops, the trials left for PRIME, SOUND /
     SSP-SIR, the post-MAD statistics and the dipole fit, each as the decider computes them (calibrations/*.npy).
  2. Pulses: for every window except 15-100 ms one full variant at z = Z_ANCHOR (2.5): every pulse through that
     calibration's preprocess_post (decision, cleaned epoch, N45, post-MAD z) with the recorder on its ICA calls.
     15-100 ms at 2.5 is the first review's window_z2.5 variant, and 'current' is the decider as it runs.
     A window added to WINDOWS later gets its calibrations and its own pass over the pulses; its rows and arrays are
     appended to the block files (checked: the same epoch, frontal measures and scalp maps as the earlier pass).
  part "pre" (the reference before the pulse; after part "windows")
  1. Calibrations at every cutoff of PRE_Z_GRID with post_timerange = 20-60 ms: get_ica() is wrapped to take mean and
     SD of |s| of each eye IC over the ICA epochs it was fitted on, and _detect_ocular_trials() is swapped for a copy
     with that reference (checked on the data: the copy with the decider's own reference returns what the decider's
     function returns). The eye-IC sources of the ICA epochs are kept (pre_sources.npz).
  2. Pulses: every pulse through the PRE_Z_ANCHOR (3.0) calibration's preprocess_post, recording the decision, the
     cleaned epoch and N45, and the gate's epoch in the N45 window (frontal channels, scalp maps before ICA, what the
     eye ICs removed).
A design's score (max over the eye ICs of the median z over the window) depends on its reference, not on the cutoff,
so the ocular decision at any cutoff is score > z. Exact: checked against preprocess_post's own decisions at the
anchor cutoff of every design, for 15-100 ms at z 2, 3 and 3.5 (the first review's variants), and against every
calibration's ocular drops. The reference before the pulse on the other windows ('pre15-100', 'pre30-60', ...) is
derived exactly from each window's median |s| (median |s| = mean_W + score_W x SD_W per eye IC; checked on 20-60 ms
against the exact pass).

Analysis: what each design and cutoff costs, and how well it catches ocular pulses
----------------------------------------------------------------------------------
- Decisions: ocular(W, z) = score_W > z; kept(W, z) = not ocular and the post-MAD gates of 'current' passed (the same
  MAD gate everywhere, so only the ocular gate differs). Calibration numbers come from the (W, z) calibrations.
  'now' is the decider as it runs (15-100 ms, reference the whole epoch, z 3).
- Ocular evidence, the same for every design and independent of the gate (frontal channels before ICA, no ICA):
  'n45' - Fp1/Fpz/Fp2 mean (VEOG) or F7 - F8 (HEOG) averaged over 38-50 ms of the gate's epoch (baseline -15 to -5 ms)
  more than EVIDENCE_K robust SD from the session's median, either direction (the eyes moved between the baseline and
  the N45 window); 'pre' - VEOG or HEOG peak-to-peak in the raw pre window -210 to -10 ms above median + EVIDENCE_K
  robust SD (eye activity going on when the pulse came). Clean: a single pulse with neither.
- Per design and cutoff: % rejected (singles, triplets, intervention singles), false alarms (% of the clean pulses
  rejected), hit rate (% of the pulses with evidence that are rejected), precision (% of the rejected with evidence),
  kept pulses with strong evidence, calibration trials for PRIME, fewest pulses kept in an open-loop block, and the
  outcome: evoked N45 change from baseline to t0-t60 with one fixed processing ('current') and each (W, z)'s kept
  pulses; at the anchor cutoff also each design's own decider (its own gates, cleaning and dipole), with bootstrap CIs.
- Equal cost: each design's cutoff set so that it rejects the same q % of single pulses; precision per evidence, mean
  of the sessions with bootstrap CIs, and each design minus the 20-60 ms window reference on the same draws.
- The reference before the pulse: a fine sweep of its cutoff (pre_sweep), the candidate cutoffs per session
  (pre_cutoffs: the same cost as now, the clean pulses' 97.5 / 99th percentiles, the z above which the precision stays
  >= 90 %), and per band of its score what the pulses carry in the N45 window before ICA and after the decider's
  cleaning, each pulse minus the typical pulses of its block (pre_bands; scalp maps in p4 / p5).
- The pulses whose decision changes with the window (z 2.5) and with the reference (before the pulse at 3 vs 20-60 ms
  at 2.5), with their evidence (table and example cards).

Run
---
    python ocular_window_review.py                    # extract what is missing (part windows about 9 min per
                                                      # session, part pre about 4 min), then tables and figures
    python ocular_window_review.py --plots-only       # tables and figures from what is extracted (about 2 min)
    python ocular_window_review.py --plots-only --figures pre_cutoff pre_maps
    python ocular_window_review.py --extract --part pre --subject pilot006 [--budget 150]     # one child, resumable

Outputs (Loop/offline_ocular_window/windows/)
    sub-<ID>/calibrations/<W>_z<z>.npy   bundle, dipole fitting info and override record of every (W, z) calibration
    sub-<ID>/scores.csv                  one row per pulse x new window (z 2.5 variant): score per eye IC, decision,
                                         N45, post-MAD z; frontal measures over every window
    sub-<ID>/blocks/<block>.npz          each new window's dipole-window epochs; scalp maps over every window
    sub-<ID>/meta.json, fitting.npz
    sub-<ID>/pre/                        the reference before the pulse: calibrations/pre20-60_z<z>.npy, pre_sources.npz
                                         (eye-IC sources of the ICA epochs), scores.csv (one row per pulse: score per
                                         eye IC, decision at z 3, N45, frontal measures in the N45 window), blocks/
                                         (cleaned dipole-window epochs, N45-window maps before ICA / removed by the eye
                                         ICs), meta.json, fitting.npz
    tables/      decisions (per design and cutoff, and 'now'; 'pooled' = mean of the sessions), matched and
                 matched_pooled (equal cost, with CIs), disagreement, marginal (evidence per band of the score),
                 evoked (per block), calibration, reference (eye-IC statistics per design), evidence, pre_sweep,
                 pre_cutoffs, pre_bands, checks, pulses_<ID> (one row per pulse, every design)
    figures/     w1 reference per window · w2 summary · w3 per session · w4 cost curves · w5 scores against 15-100 ms ·
                 w6 cards per session · w7 outcome
                 p1 the reference before the pulse (eye-IC maps, |s| before and after the pulse) · p2 its scores by
                 evidence · p3 its cutoff (cost, false alarms, precision, catch, PRIME trials, outcome) · p4 N45-window
                 scalp maps by band of its score (per session and all) · p5 single pulses ordered by its score, with
                 maps · p6 its score against the 20-60 ms window reference's
"""

# %% 0 · Parameters ---------------------------------------------------------------------------------------------
import argparse
import contextlib
import hashlib
import io
import json
import platform
import subprocess
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    _HERE = Path(__file__).resolve().parent
except NameError:                                   # interactive window: the scripts folder is the working directory
    _HERE = Path.cwd()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import ocular_threshold_review as otr               # noqa: E402  (the first review: machinery and its results)

SUBJECTS = ["pilot003", "pilot004", "pilot005", "pilot006"]
WINDOWS = {"15-100": (0.015, None),                 # the decider's post_timerange; None = to the end (100 ms)
           "30-100": (0.030, None),
           "30-60": (0.030, 0.060),
           "20-60": (0.020, 0.060),
           "20-70": (0.020, 0.070)}                 # a window added later is extracted on its own (blocks are appended)
BASE_WINDOW = "15-100"                              # the first review's window (its window_z2.5 variant is reused)
Z_GRID = [2.0, 2.25, 2.5, 2.75, 3.0]
Z_ANCHOR = 2.5                                      # the cutoff of the full per-window variants

# The pre-stimulus reference (design PRE): the eye ICs' |s| reference (mean, SD) from the ICA window - the extended
# pre-stimulus window the decider fits the ICA on (-1100 to -5 ms of every calibration trial, 1 kHz, band-passed
# 1-100 Hz, average reference) - applied to the 20-60 ms gate window of the post epoch. Its own cutoff grid: its z is
# on another scale than the window references'.
PRE = "pre20-60"
PRE_GATE = "20-60"
PRE_Z_GRID = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
PRE_Z_ANCHOR = 3.0                                  # the cutoff of its full per-pulse variant
N45_S = (0.038, 0.050)                              # the N45 dipole window: what the gate has to protect

# Ocular evidence, the same for every design and independent of the gate (frontal channels before ICA, no ICA):
#   n45: Fp1/Fpz/Fp2 mean (VEOG) or F7 - F8 (HEOG), mean over 38-50 ms of the epoch the gate sees (baseline -15 to
#        -5 ms), more than EVIDENCE_K robust SD away from the session's median (two-sided): the eyes moved between the
#        baseline and the N45 window. Blinks after the N45 window do not count.
#   pre: VEOG or HEOG peak-to-peak in the raw pre window (-210 to -10 ms) above median + EVIDENCE_K robust SD: eye
#        activity going on when the pulse came.
# STRONG_K: strong evidence. Lines from the session's single pulses.
EVIDENCE_K = 3.0
STRONG_K = 6.0
EVIDENCE = {"n45": dict(label="frontal offset 38–50 ms", cols=("veog_mean_n45", "heog_mean_n45"), two_sided=True),
            "pre": dict(label="eye activity before the pulse", cols=("veog_pre_ptp", "heog_pre_ptp"), two_sided=False)}
MATCHED_PCT = [2, 3, 4, 5, 6, 8]                    # equal-cost comparison: each design's cutoff set to reject q %
MARGINAL_EDGES = [1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5]
PRE_MARGINAL_EDGES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]
FALSE_ALARM_PCT = [97.5, 99.0]                      # cutoffs that clean pulses (no evidence) exceed 2.5 % / 1 % of the time
PROFILE_SPANS = [(15, 20), (20, 30), (30, 60), (60, 100)]
N_BOOT = 1000
CARDS_PER_ROW = 5
SWEEP = np.round(np.arange(-1.0, 8.0001, 0.05), 2)
WINDOW_COLOR = {"15-100": "#52514e",                # the decider's window: neutral; the others: categorical slots 1-4
                "30-100": "#2a78d6", "30-60": "#eb6834", "20-60": "#1baf7a", "20-70": "#eda100"}
WINDOW_MARKER = {"15-100": "o", "30-100": "s", "30-60": "D", "20-60": "v", "20-70": "^"}

OUT = otr.OUT / "windows"
SCRIPT = (_HERE / "ocular_window_review.py") if (_HERE / "ocular_window_review.py").is_file() else Path(sys.argv[0])
NEW_WINDOWS = [w for w in WINDOWS if w != BASE_WINDOW]


def wz(window, z):
    return f"{window}_z{float(z):.2f}"


def pre_name(z):
    return wz(PRE, z)


def sub_dir(subject):
    return OUT / f"sub-{subject}"


def pre_dir(subject):
    return sub_dir(subject) / "pre"


# %% 1 · Extraction (child process: one session) ----------------------------------------------------------------

def set_gate(pp, window):
    """The gate window: post_timerange, which calibrate() reads (preprocess_post then takes it from the bundle)."""
    t0, t1 = WINDOWS[window]
    pp._cfg.trial_reject_opts.ocular.post_timerange = [float(t0), None if t1 is None else float(t1)]
    pp._opts = pp._cfg.to_dicts()
    return pp


def calibrate_windows(P, sp, trials, out, say, t_start, budget):
    """Every (window, cutoff) calibration, cached in calibrations/; False if the budget ran out first."""
    cal_dir = out / "calibrations"
    cal_dir.mkdir(parents=True, exist_ok=True)
    todo = [(w, z) for w in WINDOWS for z in Z_GRID if not (cal_dir / f"{wz(w, z)}.npy").is_file()]
    if not todo:
        return True
    acc = P.Preprocessor(otr.FORWARD_PATH)
    t_add = otr.add_calibration_trials(P, acc, sp, trials)
    per_cal = 20.0                                       # seconds, until measured
    for w, z in todo:
        if budget is not None and time.time() - t_start + 1.3 * per_cal > budget:
            return False
        tic0 = time.time()
        set_gate(acc, w)
        otr.set_z(acc, z)
        info, log, tic = {}, io.StringIO(), time.perf_counter()
        with contextlib.redirect_stdout(log), otr.ocular_reference(P, "gate", info):
            _, dipole_buffers = acc.calibrate()
        if info["calls"] != 1:
            raise RuntimeError(f"{wz(w, z)}: the gate function was called {info['calls']} times, expected once")
        fitter = otr.make_fitter(P, otr.post_rate(acc, P))
        with contextlib.redirect_stdout(io.StringIO()):
            fitter.calibrate(dipole_buffers)
        b = acc.calibration_params
        np.save(cal_dir / f"{wz(w, z)}.npy",
                dict(bundle=b, fitting_info=fitter.fitting_info, gate=info, window=w, window_s=WINDOWS[w], z_threshold=z,
                     log=log.getvalue(), seconds_add_trial=t_add, seconds_calibrate=time.perf_counter() - tic),
                allow_pickle=True)
        dr = b["trials_dropped"]
        per_cal = time.time() - tic0
        say(f"  calibrated {wz(w, z):12s} (gate and reference {info['ref_ms'][0]:.1f}-{info['ref_ms'][1]:.1f} ms, "
            f"{info['n_ref']} samples): dropped ocular {len(dr['ocular'])}, qc {len(dr['qc'])}, post {len(dr['post'])}; "
            f"kept {len(b['trials_kept'])}  ({per_cal:.0f} s)")
    return True


def extract_windows(subject, budget=None, verbose=True):
    t_start = time.time()
    say = print if verbose else (lambda *a, **k: None)
    if not otr.is_complete("variants", subject):
        raise RuntimeError(f"{subject}: run ocular_threshold_review.py first (its replay is the base of this review)")
    first = otr.part_dir("variants", subject)
    first_meta = json.loads((first / "meta.json").read_text(encoding="utf-8"))
    sp = otr.session_paths(subject)
    od = sp["online_dir"]
    commit = first_meta["commit"]
    code_dir, manifest = otr.ensure_export(commit)
    P = otr.import_prime(code_dir)
    mne = P.mne
    from scipy.stats import median_abs_deviation, zscore
    out = sub_dir(subject)
    (out / "blocks").mkdir(parents=True, exist_ok=True)
    say(f"[windows / {subject}] decider {commit[:10]} ({manifest['commit_subject']}); windows {', '.join(WINDOWS)}; "
        f"cutoffs {', '.join(f'{z:g}' for z in Z_GRID)}")
    online_bundle = np.load(sp["bundle"], allow_pickle=True).item()
    trials = otr.trial_table(sp, online_bundle)
    raw_sfreq = P.cfg.get_raw_sfreq()
    post_i_tmin = P.cfg.get_post_initial_time_range()[0]
    post_tmin = P.cfg.get_post_time_range()[0]
    dip_tmin, dip_tmax = P.cfg.get_dipole_time_range()

    if not calibrate_windows(P, sp, trials, out, say, t_start, budget):
        say(f"[windows / {subject}] stopped after {time.time() - t_start:.0f} s (budget, calibrations); run again")
        return False
    cal = {(w, z): np.load(out / "calibrations" / f"{wz(w, z)}.npy", allow_pickle=True).item()
           for w in WINDOWS for z in Z_GRID}
    skip_blocks = budget is not None and time.time() - t_start > 0.5 * budget

    runs = {}
    for w in NEW_WINDOWS:
        c = cal[(w, Z_ANCHOR)]
        # preprocess_post takes the gate samples from the bundle; the config is set too, as a decider configured with
        # this window would have it
        pp = otr.set_z(set_gate(P.Preprocessor.from_bundle(c["bundle"], otr.FORWARD_PATH), w), Z_ANCHOR)
        pp.keep_full_epochs = True
        fitter = otr.make_fitter(P, otr.post_rate(pp, P))
        fitter._fitting_info = c["fitting_info"]
        stats = c["bundle"]["ocular_threshold_post"]
        runs[w] = types.SimpleNamespace(pp=pp, fitter=fitter, cal=c, stats=stats, eye_ics=sorted(int(k) for k in stats),
                                        gts=c["bundle"]["good_trial_stats_post"], rec=None)
    R0 = runs[NEW_WINDOWS[0]]
    pp0 = R0.pp
    rate = otr.post_rate(pp0, P)
    eye_ics = R0.eye_ics
    step = int(round(rate / otr.STORE_SFREQ))
    reject_range = [float(v) for v in pp0._cfg.reject_range]
    info_post = getattr(pp0, "_info_post", None) or pp0._info_processed
    ch_names = list(pp0._info["ch_names"])
    d0 = int(round((dip_tmin - post_tmin) * rate))
    dip_idx = slice(d0, int(round((dip_tmax - post_tmin) * rate)) + 1)
    veog_idx = [ch_names.index(c) for c in otr.VEOG_CHANNELS]
    heog_idx = [ch_names.index(c) for c in otr.HEOG_PAIR]
    win_idx = {w: np.asarray(cal[(w, Z_ANCHOR)]["bundle"]["ocular_threshold_post"][eye_ics[0]]["time_indices_of_interest"])
               for w in WINDOWS}
    for w, R in runs.items():
        if R.eye_ics != eye_ics:
            raise RuntimeError(f"{w}: other eye ICs than {NEW_WINDOWS[0]}")
    first_src = {}                                       # the first review's eye-IC sources (1 kHz), per block

    def one_trial(stem, todo):
        x = np.load(od / f"{stem}_post_raw.npy")
        tt = post_i_tmin + np.arange(len(x)) / raw_sfreq
        recs, arrays, ref = [], {}, None
        for w in todo:
            R = runs[w]
            R.rec.reset()
            ret = R.pp.preprocess_post(x, tt)
            if R.rec.pre is None or R.rec.removed is None:
                raise RuntimeError("the recorder did not see the ICA calls: is this the decider's preprocess_post?")
            final = np.asarray(R.pp.last_post_full)
            z_eye, z_max = otr.gate_value(R.stats, R.eye_ics, R.rec.src)
            dip = np.ascontiguousarray(final[:, dip_idx])
            ep = mne.EpochsArray(final[np.newaxis], info_post, tmin=post_tmin, verbose=False)
            rd = ep.crop(reject_range[0], reject_range[1]).get_data(copy=True)
            if ref is None:
                ref = R.rec
            row = dict(stem=stem, window=w, z_threshold=Z_ANCHOR, z_max=z_max,
                       z_argmax_ic=R.eye_ics[int(np.argmax(z_eye))], rejected_ocular=bool(z_max > Z_ANCHOR),
                       reason=R.pp.last_post_reject_reason or "", amp=float(R.fitter.fit_trial(dip)),
                       z_mad_global=float((median_abs_deviation(rd, axis=(1, 2))[0] - R.gts["mads_mean"])
                                          / R.gts["mads_std"]),
                       z_mad_local_max=float(np.max(np.abs(zscore(median_abs_deviation(rd, axis=2), axis=1)))),
                       same_as_returned=None if ret is None else bool(np.array_equal(np.asarray(ret), dip)),
                       same_inputs=R.rec is ref or bool(np.array_equal(R.rec.pre, ref.pre) and
                                                         np.array_equal(R.rec.src, ref.src) and
                                                         np.array_equal(R.rec.removed, ref.removed)))
            for j, z in enumerate(z_eye):
                row[f"z_eye{j + 1}"] = z
            recs.append(row)
            arrays[f"dip_{w}"] = dip
        pre, src = ref.pre, ref.src
        veog, heog = pre[veog_idx].mean(axis=0), pre[heog_idx[0]] - pre[heog_idx[1]]
        common = {}
        for w, idx in win_idx.items():                    # frontal measures and scalp map over every window
            common[f"veog_ptp__{w}"] = float(np.ptp(veog[idx]))
            common[f"heog_ptp__{w}"] = float(np.ptp(heog[idx]))
            common[f"veog_absmean__{w}"] = float(np.abs(veog[idx].mean()))
            arrays[f"map_{w}"] = pre[:, idx].mean(axis=1)
        s1 = first_src.get(stem)                          # the same sources as in the first review?
        common["src_rel_diff_first_review"] = (float(np.max(np.abs(src[eye_ics][:, ::step] - s1)) / np.max(np.abs(s1)))
                                               if s1 is not None else np.nan)
        for r in recs:
            r.update(common)
        return recs, arrays

    blocks = [b for b in otr.BLOCK_ORDER if b in set(trials.block)] + sorted(set(trials.block) - set(otr.BLOCK_ORDER))
    complete = False
    if not skip_blocks:
        for f in sorted((first / "blocks").glob("*.npz")):
            with np.load(f) as z:
                first_src.update(zip([str(s) for s in z["stems"]], z["src"]))
        for R in runs.values():
            R.rec = otr.IcaRecorder(R.cal["bundle"]["ica"])
        complete = run_window_blocks(blocks, trials, out, NEW_WINDOWS, one_trial, t_start, budget, say)
        for R in runs.values():
            R.rec.remove()

    # ---- calibration summaries, fitting, checks, meta (every run)
    fwd = mne.read_forward_solution(str(otr.FORWARD_PATH), verbose=False)
    L = fwd["sol"]["data"] - np.mean(fwd["sol"]["data"], axis=0)
    cur = np.load(first / "calibration_current.npy", allow_pickle=True).item()["bundle"]
    fit_arrays, cal_meta, checks = {}, {}, {}
    for (w, z), c in cal.items():
        b, fi, g = c["bundle"], c["fitting_info"], c["gate"]
        p = int(fi["position_index"])
        v = wz(w, z)
        fit_arrays[f"L_pos_{v}"] = L[:, 3 * p:3 * p + 3]
        fit_arrays[f"index_range_{v}"] = np.asarray(fi["index_range"], dtype=int)
        fit_arrays[f"rr_mm_{v}"] = 1e3 * np.asarray(fwd["source_rr"][p], dtype=float)
        st = b["ocular_threshold_post"]
        cal_meta[v] = dict(window=w, z_threshold=z, gate_ms=g["gate_ms"], n_gate=g["n_gate"], ref_ms=g["ref_ms"],
                           n_ref=g["n_ref"], eye_stats={str(i): dict(mean=float(st[i]["mean"]), std=float(st[i]["std"]))
                                                        for i in eye_ics},
                           dropped={k: list(map(int, np.asarray(b["trials_dropped"][k]).ravel()))
                                    for k in ("ocular", "qc", "post")},
                           n_kept=int(np.asarray(b["trials_kept"]).size), position_index=p,
                           index_range=list(map(int, fi["index_range"])),
                           post_mad_mean_std=[float(b["good_trial_stats_post"][k]) for k in ("mads_mean", "mads_std")],
                           sound_filter_rel_max_diff_vs_current=float(np.max(np.abs(b["sound_filter"] - cur["sound_filter"]))
                                                                      / np.max(np.abs(cur["sound_filter"]))))
        checks[f"{v}: copy of _detect_ocular_trials with the whole epoch == the decider's function"] = g["copy_equals_decider"]
        checks[f"{v}: reference mean / SD == mean / SD of |s| over the window"] = dict(
            ok=bool(g["ref_stats_rel_diff"] < 1e-12), detail=f"max rel. diff {g['ref_stats_rel_diff']:.1e}")
        cmp = otr.compare_bundles(b, cur)
        checks[f"{v}: same ICA, eye ICs, bad channels as the decider ('current', first review)"] = {
            k: cmp[k] for k in ("bad channels", "eye ICs", "ICA unmixing, max |diff|")}
    for w in WINDOWS:                                    # the reference does not depend on the cutoff
        s0 = cal_meta[wz(w, Z_GRID[0])]["eye_stats"]
        same = all(cal_meta[wz(w, z)]["eye_stats"] == s0 for z in Z_GRID)
        checks[f"{w}: the same reference mean / SD at every cutoff"] = dict(ok=bool(same), detail="identical" if same else "differ")
    for z in Z_GRID:                                     # 15-100 ms: the first review's calibrations
        f = first / f"calibration_{otr.variant_name(z)}.npy"
        if any(abs(z - z1) < 1e-9 for z1 in otr.Z_GRID) and f.is_file():
            a = np.load(f, allow_pickle=True).item()["bundle"]
            b = cal[(BASE_WINDOW, z)]["bundle"]
            same_drop = all(sorted(map(int, np.asarray(a["trials_dropped"][k]).ravel())) ==
                            sorted(map(int, np.asarray(b["trials_dropped"][k]).ravel())) for k in ("ocular", "qc", "post"))
            dd = max(max(abs(a["ocular_threshold_post"][i]["mean"] - b["ocular_threshold_post"][i]["mean"]),
                         abs(a["ocular_threshold_post"][i]["std"] - b["ocular_threshold_post"][i]["std"])) for i in eye_ics)
            checks[f"{wz(BASE_WINDOW, z)} == the first review's {otr.variant_name(z)} calibration"] = dict(
                ok=bool(same_drop and dd < 1e-8), detail=f"drops {'identical' if same_drop else 'differ'}; "
                                                         f"reference max |diff| {dd:.1e}")
    np.savez(out / "fitting.npz", **fit_arrays)
    if complete:
        scores = pd.concat([pd.read_csv(out / "blocks" / f"{b}.csv", keep_default_na=False, na_values=[""])
                            for b in blocks], ignore_index=True)
        for c in ("reason", "online_reason"):
            scores[c] = scores[c].fillna("").astype(str) if c in scores else ""
        scores.insert(0, "subject", subject)
        scores.to_csv(out / "scores.csv", index=False)
        for w, R in runs.items():
            s_w = scores[scores.window == w]
            checks.update(otr.internal_checks(s_w, tag=f"{wz(w, Z_ANCHOR)}: "))
            cal_rows = s_w[s_w.block == "calibration"]
            drop = set(map(int, np.asarray(R.cal["bundle"]["trials_dropped"]["ocular"]).ravel()))
            mism = int((cal_rows.rejected_ocular != cal_rows.trial_in_stage.isin(drop)).sum())
            checks[f"{wz(w, Z_ANCHOR)}: calibration pulses, score > cutoff vs its calibration's ocular drops"] = dict(
                ok=mism == 0, detail=f"{len(cal_rows) - mism}/{len(cal_rows)} agree")
            for z in Z_GRID:                             # the drops of every cutoff follow from the same score
                dz = set(map(int, np.asarray(cal[(w, z)]["bundle"]["trials_dropped"]["ocular"]).ravel()))
                mism = int(((cal_rows.z_max > z) != cal_rows.trial_in_stage.isin(dz)).sum())
                checks[f"{wz(w, z)}: calibration pulses, score > {z:g} vs that calibration's ocular drops"] = dict(
                    ok=mism == 0, detail=f"{len(cal_rows) - mism}/{len(cal_rows)} agree")
        si = scores.same_inputs.map({"True": True, "False": False, True: True, False: False}).astype(bool)
        checks["every window's gate saw the same epoch, eye-IC sources and ICA removal"] = dict(
            ok=bool(si.all()), detail=f"{int(si.sum())}/{len(si)} rows")
        if "same_epoch_as_earlier_pass_max_abs_diff" in scores:
            dd = scores.same_epoch_as_earlier_pass_max_abs_diff.astype(float)
            added = sorted(set(scores.window[dd.notna()]))
            checks["windows added later (" + ", ".join(added) + "): the same epoch as the earlier pass "
                   "(frontal measures, scalp maps), max |diff|"] = dict(
                ok=bool(dd.max() < 1e-6), detail=f"{dd.max():.1e} over {int(dd.notna().sum())} rows")
        n_rows = scores.groupby("stem").window.nunique()
        checks["every pulse has a row for every window"] = dict(
            ok=bool((n_rows == len(NEW_WINDOWS)).all() and len(n_rows) == len(trials)),
            detail=f"{len(n_rows)} pulses x {sorted(set(n_rows))} windows")
        rd = scores.src_rel_diff_first_review.astype(float)
        checks["eye-IC sources == the first review's (1 kHz, float32), max rel. diff"] = dict(
            ok=bool(rd.notna().all() and rd.max() < 1e-5), detail=f"{rd.max():.1e} over {rd.notna().sum()} rows")
    meta = dict(part="windows", subject=subject, complete=complete, commit=commit,
                commit_subject=manifest["commit_subject"], windows={w: list(t) for w, t in WINDOWS.items()},
                window_samples={w: dict(first_ms=cal_meta[wz(w, Z_ANCHOR)]["gate_ms"][0],
                                        last_ms=cal_meta[wz(w, Z_ANCHOR)]["gate_ms"][1],
                                        n=cal_meta[wz(w, Z_ANCHOR)]["n_gate"]) for w in WINDOWS},
                z_grid=Z_GRID, z_anchor=Z_ANCHOR, base_window=BASE_WINDOW, eye_ics=eye_ics, calibrations=cal_meta,
                ch_names=ch_names, times_dip_ms=list(np.round(1e3 * (dip_tmin + np.arange(dip_idx.stop - dip_idx.start)
                                                                     / rate), 3)),
                forward_sha256=hashlib.sha256(otr.FORWARD_PATH.read_bytes()).hexdigest()[:16],
                versions=dict(python=platform.python_version(), numpy=np.__version__, mne=mne.__version__,
                              pandas=pd.__version__, scipy=__import__("scipy").__version__),
                checks=checks, written=datetime.now().isoformat(timespec="seconds"))
    (out / "meta.json").write_text(json.dumps(otr.to_jsonable(meta), indent=1), encoding="utf-8")
    otr.report(f"windows / {subject}", complete, checks, t_start, say)
    return complete


COMMON_PREFIXES = ("veog_ptp__", "heog_ptp__", "veog_absmean__")     # per pulse, the same in every window's rows


def run_window_blocks(blocks, trials, out, windows, one_trial, t_start, budget, say):
    """Block by block, the windows a block's csv does not have yet (all of them the first time). A window added later
    is run on its own and appended: the earlier rows and arrays stay, the frontal measures and maps of the new pass
    are compared with the earlier ones (same epoch), the csv is written last (done marker). False: budget ran out."""
    per_trial_s = None
    for block in blocks:
        f_csv, f_npz = out / "blocks" / f"{block}.csv", out / "blocks" / f"{block}.npz"
        old = pd.read_csv(f_csv, keep_default_na=False, na_values=[""], low_memory=False) if f_csv.is_file() else None
        todo = [w for w in windows if old is None or w not in set(old.window)]
        if not todo:
            continue
        rows = trials[trials.block == block]
        if budget is not None and per_trial_s is not None and \
                time.time() - t_start + per_trial_s * len(rows) * len(todo) * 1.2 > budget:
            break
        tic = time.time()
        recs, arrs = [], {}
        for r in rows.itertuples():
            rec, arrays = one_trial(r.stem, todo)
            recs += rec
            for k, v in arrays.items():
                arrs.setdefault(k, []).append(np.asarray(v, dtype=np.float32))
        per_trial_s = max(per_trial_s or 0.0, (time.time() - tic) / max(len(rows) * len(todo), 1))
        new = rows.merge(pd.DataFrame(recs), on="stem", how="left")
        arrs = {k: np.stack(v) for k, v in arrs.items()}
        if old is not None:
            with np.load(f_npz) as z:
                prev = {k: z[k] for k in z.files}
            if [str(s) for s in prev["stems"]] != list(rows.stem):
                raise RuntimeError(f"{f_npz}: other pulses than the trial table")
            o = old.drop_duplicates("stem").set_index("stem")
            n = new.drop_duplicates("stem").set_index("stem").loc[o.index]
            diff = 0.0
            for c in n.columns:
                if c.startswith(COMMON_PREFIXES):
                    if c in o.columns:
                        diff = max(diff, float(np.nanmax(np.abs(o[c].astype(float) - n[c].astype(float)))))
                    else:                                # a measure over the new window: fill in the earlier rows
                        old[c] = old.stem.map(n[c])
            for k, v in arrs.items():
                if k in prev:
                    diff = max(diff, float(np.max(np.abs(prev[k].astype(float) - v.astype(float)))))
            new["same_epoch_as_earlier_pass_max_abs_diff"] = diff
            arrs = {**{k: v for k, v in prev.items() if k != "stems"}, **{k: v for k, v in arrs.items() if k not in prev}}
            res = pd.concat([old, new], ignore_index=True)
        else:
            res = new
        np.savez(f_npz, stems=np.asarray(rows.stem, dtype=str), **arrs)
        res.to_csv(f_csv, index=False)
        say(f"  {block:12s} {len(rows):4d} pulses  {time.time() - tic:5.1f} s  ocular at z {Z_ANCHOR:g}: " + " ".join(
            f"{w} {int(((new.window == w) & (new.reason == 'ocular')).sum())}" for w in todo)
            + ("" if old is None else f"  (added to {', '.join(sorted(set(old.window)))}; same epoch: max |diff| "
                                      f"{new.same_epoch_as_earlier_pass_max_abs_diff.iloc[0]:.1e})"))
        if budget is not None and time.time() - t_start > budget:
            break
    for block in blocks:
        f_csv = out / "blocks" / f"{block}.csv"
        if not f_csv.is_file() or set(windows) - set(pd.read_csv(f_csv, usecols=["window"]).window):
            return False
    return True


def run_child(subject, budget=None, part="windows"):
    cmd = [sys.executable, str(SCRIPT), "--extract", "--part", part, "--subject", subject]
    if budget:
        cmd += ["--budget", str(budget)]
    for _ in range(50):
        r = subprocess.run(cmd)
        if r.returncode == 0:
            return True
        if r.returncode != 3:
            raise RuntimeError(f"extraction failed for {part} / {subject} (exit code {r.returncode})")
    raise RuntimeError(f"extraction of {part} / {subject} did not finish")


def is_complete(subject):
    d = sub_dir(subject)
    f = d / "meta.json"
    if not (f.is_file() and (d / "scores.csv").is_file()):
        return False
    m = json.loads(f.read_text(encoding="utf-8"))
    return bool(m.get("complete", False)) and set(WINDOWS) <= set(m.get("windows", {})) and \
        all(abs(z - z1) < 1e-9 for z, z1 in zip(Z_GRID, m.get("z_grid", []))) and len(m.get("z_grid", [])) == len(Z_GRID)


# ---------------------------------------------------------------------- part "pre": the pre-stimulus reference

def detect_ocular_trials_given(ica, epochs, tmin, tmax, ocular_component_indices, z_threshold, ref_stats):
    """The decider's _detect_ocular_trials (prime_core/preprocessing/preprocessor.py) with the reference mean / SD of
    |s| given per eye IC (ref_stats, here from the ICA window) instead of taken from the post epoch itself. The score
    (median z over the gate samples [tmin, tmax]) and the rule (> z_threshold for any eye IC) are the decider's.
    Changed lines are marked CHANGED."""
    sources = ica.get_sources(epochs)
    source_data = sources.get_data(copy=True)

    if tmax is None:
        tmax = np.inf
    if tmin is None:
        tmin = -np.inf

    time_mask = np.where((sources.times >= tmin) & (sources.times <= tmax))[0]
    bad_trials = np.array([])
    thresholds = {}

    for component_idx in ocular_component_indices:
        st = ref_stats[int(component_idx)]                                                  # CHANGED: given reference
        z_gate = (np.abs(source_data[:, component_idx, time_mask]) - st["mean"]) / st["std"]  # CHANGED: z of the gate
        median_z = np.median(z_gate, axis=1)                                                #   samples against it
        bad_now = np.where(median_z > z_threshold)[0]
        thresholds[component_idx] = {
            'std': st["std"],                                                               # CHANGED
            'mean': st["mean"],                                                             # CHANGED
            'time_indices_of_interest': time_mask,
        }
        bad_trials = np.union1d(bad_trials, bad_now)

    return list(bad_trials.astype(int)), thresholds


@contextlib.contextmanager
def pre_reference(P, info, keep_sources=False):
    """Inside the block, the decider's calibration builds the ocular gate's reference from the ICA window: get_ica() is
    wrapped to take mean and SD of |s| of each eye IC over every sample of every calibration trial's ICA epoch (the
    epochs the ICA is fitted on: -1100 to -5 ms, 1 kHz, band-passed 1-100 Hz, average reference), and
    _detect_ocular_trials() is swapped for detect_ocular_trials_given() with that reference. Checked on this very
    data: the copy of the decider's function with its own reference returns what the decider's function returns.
    keep_sources: info['pre_src'] receives the eye-IC sources of the ICA epochs (float32, for the figures)."""
    orig_get_ica, orig_detect = P.pp_module.get_ica, P.pp_module._detect_ocular_trials
    info.update(calls_get_ica=0, calls_detect=0)

    def get_ica(epochs_or_raw, *args, **kwargs):
        out = orig_get_ica(epochs_or_raw, *args, **kwargs)
        ica, excluded, _ = out
        info["calls_get_ica"] += 1
        src = ica.get_sources(epochs_or_raw)
        data, times = src.get_data(copy=True), np.asarray(src.times, dtype=float)
        eye = sorted(int(i) for i in excluded["eye blink"])
        info["pre_stats"] = {ic: dict(mean=float(np.mean(np.abs(data[:, ic]))), std=float(np.std(np.abs(data[:, ic]))))
                             for ic in eye}
        info.update(pre_eye_ics=eye, pre_ms=[1e3 * times[0], 1e3 * times[-1]], n_pre=int(len(times)),
                    n_pre_trials=int(data.shape[0]), pre_sfreq=float(epochs_or_raw.info["sfreq"]))
        if keep_sources:
            info.update(pre_src=data[:, eye, :].astype(np.float32), pre_times=times)
        return out

    def gate(ica, epochs, tmin, tmax, ocular_component_indices, z_threshold):
        info["calls_detect"] += 1
        if info["calls_get_ica"] != 1 or "pre_stats" not in info:
            raise RuntimeError("the ocular gate ran before the ICA was fitted: the ICA-window reference is missing")
        if sorted(int(i) for i in ocular_component_indices) != info["pre_eye_ics"]:
            raise RuntimeError("the gate's eye ICs differ from ICLabel's eye-blink components")
        decider = orig_detect(ica, epochs, tmin, tmax, ocular_component_indices, z_threshold)
        copy = otr.detect_ocular_trials_ref(ica, epochs, tmin, tmax, ocular_component_indices, z_threshold, None)
        info["copy_equals_decider"] = otr.same_gate_output(decider, copy)
        if not info["copy_equals_decider"]["ok"]:
            raise RuntimeError("the copy of _detect_ocular_trials differs from the decider's: compare with its code")
        out = detect_ocular_trials_given(ica, epochs, tmin, tmax, ocular_component_indices, z_threshold,
                                         info["pre_stats"])
        times = np.asarray(ica.get_sources(epochs).times, dtype=float)
        gate_idx = out[1][ocular_component_indices[0]]["time_indices_of_interest"]
        info.update(gate_ms=[1e3 * times[gate_idx[0]], 1e3 * times[gate_idx[-1]]], n_gate=int(len(gate_idx)),
                    z_threshold=float(z_threshold), n_trials=int(len(epochs)))
        return out

    P.pp_module.get_ica, P.pp_module._detect_ocular_trials = get_ica, gate
    try:
        yield info
    finally:
        P.pp_module.get_ica, P.pp_module._detect_ocular_trials = orig_get_ica, orig_detect


def calibrate_pre(P, sp, trials, out, say, t_start, budget):
    """Every cutoff's calibration with the ICA-window reference and the 20-60 ms gate (calibrations/), and the
    eye-IC sources of the ICA epochs (pre_sources.npz); False if the budget ran out first."""
    cal_dir = out / "calibrations"
    cal_dir.mkdir(parents=True, exist_ok=True)
    f_src = out / "pre_sources.npz"
    todo = [z for z in PRE_Z_GRID if not (cal_dir / f"{pre_name(z)}.npy").is_file()]
    if not todo and f_src.is_file():
        return True
    todo = todo or [PRE_Z_GRID[0]]                       # only the sources are missing
    acc = P.Preprocessor(otr.FORWARD_PATH)
    t_add = otr.add_calibration_trials(P, acc, sp, trials)
    per_cal = 20.0
    for z in todo:
        if budget is not None and time.time() - t_start + 1.3 * per_cal > budget:
            return False
        tic0 = time.time()
        set_gate(acc, PRE_GATE)
        otr.set_z(acc, z)
        info, log, tic = {}, io.StringIO(), time.perf_counter()
        with contextlib.redirect_stdout(log), pre_reference(P, info, keep_sources=not f_src.is_file()):
            _, dipole_buffers = acc.calibrate()
        if info["calls_get_ica"] != 1 or info["calls_detect"] != 1:
            raise RuntimeError(f"{pre_name(z)}: get_ica called {info['calls_get_ica']}, the gate {info['calls_detect']} "
                               f"times; expected once each")
        if "pre_src" in info:
            np.savez(f_src, src=info.pop("pre_src"), times=info.pop("pre_times"), eye_ics=np.asarray(info["pre_eye_ics"]))
        fitter = otr.make_fitter(P, otr.post_rate(acc, P))
        with contextlib.redirect_stdout(io.StringIO()):
            fitter.calibrate(dipole_buffers)
        b = acc.calibration_params
        f = cal_dir / f"{pre_name(z)}.npy"
        if not f.is_file():
            np.save(f, dict(bundle=b, fitting_info=fitter.fitting_info, gate=info, design=PRE, window=PRE_GATE,
                            reference="ICA window", z_threshold=z, log=log.getvalue(), seconds_add_trial=t_add,
                            seconds_calibrate=time.perf_counter() - tic), allow_pickle=True)
        dr = b["trials_dropped"]
        per_cal = time.time() - tic0
        say(f"  calibrated {pre_name(z):16s} (reference {info['pre_ms'][0]:.0f} to {info['pre_ms'][1]:.0f} ms, "
            f"{info['n_pre']} samples x {info['n_pre_trials']} trials; gate {info['gate_ms'][0]:.1f}-"
            f"{info['gate_ms'][1]:.1f} ms): dropped ocular {len(dr['ocular'])}, qc {len(dr['qc'])}, post "
            f"{len(dr['post'])}; kept {len(b['trials_kept'])}  ({per_cal:.0f} s)")
    return True


def extract_pre(subject, budget=None, verbose=True):
    """The pre-stimulus reference on one session: the calibrations of PRE_Z_GRID, then every pulse through the
    PRE_Z_ANCHOR calibration's preprocess_post, recording the decision, the cleaned epoch and N45, and what the
    gate's epoch looks like in the N45 window (frontal channels, scalp maps before ICA and what the ICA removed)."""
    t_start = time.time()
    say = print if verbose else (lambda *a, **k: None)
    if not (otr.is_complete("variants", subject) and is_complete(subject)):
        raise RuntimeError(f"{subject}: run the first review and the windows extraction first (cross-checks use them)")
    first = otr.part_dir("variants", subject)
    first_meta = json.loads((first / "meta.json").read_text(encoding="utf-8"))
    win_meta = json.loads((sub_dir(subject) / "meta.json").read_text(encoding="utf-8"))
    sp = otr.session_paths(subject)
    od = sp["online_dir"]
    commit = first_meta["commit"]
    code_dir, manifest = otr.ensure_export(commit)
    P = otr.import_prime(code_dir)
    mne = P.mne
    from scipy.stats import median_abs_deviation, zscore
    out = pre_dir(subject)
    (out / "blocks").mkdir(parents=True, exist_ok=True)
    say(f"[pre / {subject}] decider {commit[:10]} ({manifest['commit_subject']}); reference: ICA window; gate "
        f"{PRE_GATE} ms; cutoffs {', '.join(f'{z:g}' for z in PRE_Z_GRID)}")
    online_bundle = np.load(sp["bundle"], allow_pickle=True).item()
    trials = otr.trial_table(sp, online_bundle)
    raw_sfreq = P.cfg.get_raw_sfreq()
    post_i_tmin = P.cfg.get_post_initial_time_range()[0]
    post_tmin = P.cfg.get_post_time_range()[0]
    dip_tmin, dip_tmax = P.cfg.get_dipole_time_range()

    if not calibrate_pre(P, sp, trials, out, say, t_start, budget):
        say(f"[pre / {subject}] stopped after {time.time() - t_start:.0f} s (budget, calibrations); run again")
        return False
    cal = {z: np.load(out / "calibrations" / f"{pre_name(z)}.npy", allow_pickle=True).item() for z in PRE_Z_GRID}
    skip_blocks = budget is not None and time.time() - t_start > 0.5 * budget

    c = cal[PRE_Z_ANCHOR]
    pp = otr.set_z(set_gate(P.Preprocessor.from_bundle(c["bundle"], otr.FORWARD_PATH), PRE_GATE), PRE_Z_ANCHOR)
    pp.keep_full_epochs = True
    fitter = otr.make_fitter(P, otr.post_rate(pp, P))
    fitter._fitting_info = c["fitting_info"]
    stats = c["bundle"]["ocular_threshold_post"]
    eye_ics = sorted(int(k) for k in stats)
    gts = c["bundle"]["good_trial_stats_post"]
    rate = otr.post_rate(pp, P)
    reject_range = [float(v) for v in pp._cfg.reject_range]
    info_post = getattr(pp, "_info_post", None) or pp._info_processed
    ch_names = list(pp._info["ch_names"])
    d0 = int(round((dip_tmin - post_tmin) * rate))
    dip_idx = slice(d0, int(round((dip_tmax - post_tmin) * rate)) + 1)
    veog_idx = [ch_names.index(ch) for ch in otr.VEOG_CHANNELS]
    heog_idx = [ch_names.index(ch) for ch in otr.HEOG_PAIR]
    gate_idx = np.asarray(stats[eye_ics[0]]["time_indices_of_interest"])
    # the windows pass's 20-60 ms variant: same epoch and ICA, so the same median |s| per pulse (from its score)
    st_win = win_meta["calibrations"][wz(PRE_GATE, Z_ANCHOR)]["eye_stats"]
    win_rows = pd.read_csv(sub_dir(subject) / "scores.csv", low_memory=False)
    win_rows = win_rows[win_rows.window == PRE_GATE].set_index("stem")
    run = types.SimpleNamespace(rec=None, n45=None)

    def one_trial(stem):
        x = np.load(od / f"{stem}_post_raw.npy")
        rec = run.rec
        rec.reset()
        ret = pp.preprocess_post(x, post_i_tmin + np.arange(len(x)) / raw_sfreq)
        if rec.pre is None or rec.removed is None:
            raise RuntimeError("the recorder did not see the ICA calls: is this the decider's preprocess_post?")
        if run.n45 is None:
            run.n45 = np.where((rec.times >= N45_S[0] - 1e-9) & (rec.times <= N45_S[1] + 1e-9))[0]
        n45 = run.n45
        final = np.asarray(pp.last_post_full)
        z_eye, z_max = otr.gate_value(stats, eye_ics, rec.src)
        dip = np.ascontiguousarray(final[:, dip_idx])
        ep = mne.EpochsArray(final[np.newaxis], info_post, tmin=post_tmin, verbose=False)
        rd = ep.crop(reject_range[0], reject_range[1]).get_data(copy=True)
        pre, rem = rec.pre, rec.removed
        veog, heog = pre[veog_idx].mean(axis=0), pre[heog_idx[0]] - pre[heog_idx[1]]
        fin_veog, fin_heog = final[veog_idx].mean(axis=0), final[heog_idx[0]] - final[heog_idx[1]]
        med = [float(np.median(np.abs(rec.src[ic, gate_idx]))) for ic in eye_ics]
        row = dict(stem=stem, design=PRE, z_threshold=PRE_Z_ANCHOR, z_max=z_max,
                   z_argmax_ic=eye_ics[int(np.argmax(z_eye))], rejected_ocular=bool(z_max > PRE_Z_ANCHOR),
                   reason=pp.last_post_reject_reason or "", amp=float(fitter.fit_trial(dip)),
                   z_mad_global=float((median_abs_deviation(rd, axis=(1, 2))[0] - gts["mads_mean"]) / gts["mads_std"]),
                   z_mad_local_max=float(np.max(np.abs(zscore(median_abs_deviation(rd, axis=2), axis=1)))),
                   same_as_returned=None if ret is None else bool(np.array_equal(np.asarray(ret), dip)),
                   # the gate's epoch (before ICA, baseline -15 to -5 ms) in the N45 window and over the gate window
                   veog_mean_n45=float(veog[n45].mean()), heog_mean_n45=float(heog[n45].mean()),
                   veog_ptp_n45=float(np.ptp(veog[n45])), heog_ptp_n45=float(np.ptp(heog[n45])),
                   veog_mean_gate=float(veog[gate_idx].mean()), heog_mean_gate=float(heog[gate_idx].mean()),
                   veog_ptp_gate=float(np.ptp(veog[gate_idx])), heog_ptp_gate=float(np.ptp(heog[gate_idx])),
                   gfp_n45=float(pre[:, n45].std(axis=0).mean()),
                   # what the ICA took out in the N45 window, and the cleaned epoch there (this design's processing)
                   removed_veog_n45=float(rem[veog_idx][:, n45].mean()),
                   removed_rms_n45=float(np.sqrt(np.mean(rem[:, n45] ** 2))),
                   final_veog_n45=float(fin_veog[dip_idx].mean()), final_heog_n45=float(fin_heog[dip_idx].mean()))
        for j, (z, m) in enumerate(zip(z_eye, med)):
            row[f"z_eye{j + 1}"] = z
            row[f"med_abs_s{j + 1}"] = m
        if stem in win_rows.index:
            r2 = win_rows.loc[stem]
            row["med_abs_s_rel_diff_windows_pass"] = max(
                abs(m - (st_win[str(ic)]["mean"] + float(r2[f"z_eye{j + 1}"]) * st_win[str(ic)]["std"])) / max(m, 1e-12)
                for j, (ic, m) in enumerate(zip(eye_ics, med)))
            row["veog_ptp_gate_diff_windows_pass"] = abs(row["veog_ptp_gate"] - float(r2[f"veog_ptp__{PRE_GATE}"]))
        arrays = dict(dip_pre=dip, map_n45_pre=pre[:, n45].mean(axis=1), map_n45_removed=rem[:, n45].mean(axis=1),
                      map_gate_pre=pre[:, gate_idx].mean(axis=1))
        return [row], arrays

    blocks = [b for b in otr.BLOCK_ORDER if b in set(trials.block)] + sorted(set(trials.block) - set(otr.BLOCK_ORDER))
    complete = False
    if not skip_blocks:
        run.rec = otr.IcaRecorder(c["bundle"]["ica"])
        complete = otr.run_blocks(blocks, trials, out, one_trial, t_start, budget, say,
                                  lambda r: f"ocular at z {PRE_Z_ANCHOR:g}: {int((r.reason == 'ocular').sum())}")
        run.rec.remove()

    # ---- calibration summaries, fitting, checks, meta (every run)
    fwd = mne.read_forward_solution(str(otr.FORWARD_PATH), verbose=False)
    L = fwd["sol"]["data"] - np.mean(fwd["sol"]["data"], axis=0)
    cur = np.load(first / "calibration_current.npy", allow_pickle=True).item()["bundle"]
    with np.load(out / "pre_sources.npz") as zf:
        S_abs = np.abs(zf["src"].astype(float))
        src_eye = [int(i) for i in zf["eye_ics"]]
    fit_arrays, cal_meta, checks = {}, {}, {}
    if src_eye != eye_ics:
        raise RuntimeError("the saved ICA-window sources have other eye ICs")
    for z, cz in cal.items():
        b, fi, g = cz["bundle"], cz["fitting_info"], cz["gate"]
        p = int(fi["position_index"])
        v = pre_name(z)
        fit_arrays[f"L_pos_{v}"] = L[:, 3 * p:3 * p + 3]
        fit_arrays[f"index_range_{v}"] = np.asarray(fi["index_range"], dtype=int)
        fit_arrays[f"rr_mm_{v}"] = 1e3 * np.asarray(fwd["source_rr"][p], dtype=float)
        st = b["ocular_threshold_post"]
        cal_meta[v] = dict(design=PRE, window=PRE_GATE, z_threshold=z, gate_ms=g["gate_ms"], n_gate=g["n_gate"],
                           pre_ms=g["pre_ms"], n_pre=g["n_pre"], n_pre_trials=g["n_pre_trials"], pre_sfreq=g["pre_sfreq"],
                           eye_stats={str(i): dict(mean=float(st[i]["mean"]), std=float(st[i]["std"])) for i in eye_ics},
                           dropped={k: list(map(int, np.asarray(b["trials_dropped"][k]).ravel()))
                                    for k in ("ocular", "qc", "post")},
                           n_kept=int(np.asarray(b["trials_kept"]).size), position_index=p,
                           index_range=list(map(int, fi["index_range"])),
                           post_mad_mean_std=[float(b["good_trial_stats_post"][k]) for k in ("mads_mean", "mads_std")],
                           sound_filter_rel_max_diff_vs_current=float(np.max(np.abs(b["sound_filter"] - cur["sound_filter"]))
                                                                      / np.max(np.abs(cur["sound_filter"]))))
        checks[f"{v}: copy of _detect_ocular_trials with the decider's reference == the decider's function"] = \
            g["copy_equals_decider"]
        checks[f"{v}: the calibration fitted the ICA once and called the gate once"] = dict(
            ok=g["calls_get_ica"] == 1 and g["calls_detect"] == 1,
            detail=f"get_ica {g['calls_get_ica']}, gate {g['calls_detect']}")
        cmp = otr.compare_bundles(b, cur)
        checks[f"{v}: same ICA, eye ICs, bad channels as the decider ('current', first review)"] = {
            k: cmp[k] for k in ("bad channels", "eye ICs", "ICA unmixing, max |diff|")}
        rel = max(max(abs(S_abs[:, j].mean() - st[ic]["mean"]) / st[ic]["mean"],
                      abs(S_abs[:, j].std() - st[ic]["std"]) / st[ic]["std"]) for j, ic in enumerate(eye_ics))
        checks[f"{v}: reference == mean / SD of |s| over the ICA epochs (from the saved sources)"] = dict(
            ok=bool(rel < 1e-5), detail=f"max rel. diff {rel:.1e} (sources saved as float32)")
    s0 = cal_meta[pre_name(PRE_Z_GRID[0])]["eye_stats"]
    same = all(cal_meta[pre_name(z)]["eye_stats"] == s0 for z in PRE_Z_GRID)
    checks["the same ICA-window reference at every cutoff"] = dict(ok=bool(same), detail="identical" if same else "differ")
    np.savez(out / "fitting.npz", **fit_arrays)
    if complete:
        scores = pd.concat([pd.read_csv(out / "blocks" / f"{b}.csv", keep_default_na=False, na_values=[""])
                            for b in blocks], ignore_index=True)
        for col in ("reason", "online_reason"):
            scores[col] = scores[col].fillna("").astype(str) if col in scores else ""
        scores.insert(0, "subject", subject)
        scores.to_csv(out / "scores.csv", index=False)
        checks.update(otr.internal_checks(scores, tag=f"{pre_name(PRE_Z_ANCHOR)}: "))
        cal_rows = scores[scores.block == "calibration"]
        for z in PRE_Z_GRID:                             # the drops of every cutoff follow from the same score
            dz = set(map(int, np.asarray(cal[z]["bundle"]["trials_dropped"]["ocular"]).ravel()))
            mism = int(((cal_rows.z_max > z) != cal_rows.trial_in_stage.isin(dz)).sum())
            checks[f"{pre_name(z)}: calibration pulses, score > {z:g} vs that calibration's ocular drops"] = dict(
                ok=mism == 0, detail=f"{len(cal_rows) - mism}/{len(cal_rows)} agree")
        rd = scores.med_abs_s_rel_diff_windows_pass.astype(float)
        checks["median |s| over 20-60 ms == the windows pass's (same epoch and ICA), max rel. diff"] = dict(
            ok=bool(rd.notna().all() and rd.max() < 1e-9), detail=f"{rd.max():.1e} over {int(rd.notna().sum())} pulses")
        dd = scores.veog_ptp_gate_diff_windows_pass.astype(float)
        checks["Fp peak-to-peak over 20-60 ms == the windows pass's, max |diff|"] = dict(
            ok=bool(dd.notna().all() and dd.max() < 1e-6), detail=f"{dd.max():.1e} µV")
    meta = dict(part="pre", subject=subject, complete=complete, commit=commit, commit_subject=manifest["commit_subject"],
                design=PRE, gate_window=PRE_GATE, gate_window_s=list(WINDOWS[PRE_GATE]), reference="ICA window",
                pre_ms=cal_meta[pre_name(PRE_Z_ANCHOR)]["pre_ms"], z_grid=PRE_Z_GRID, z_anchor=PRE_Z_ANCHOR,
                n45_ms=[1e3 * N45_S[0], 1e3 * N45_S[1]], eye_ics=eye_ics, calibrations=cal_meta, ch_names=ch_names,
                times_dip_ms=list(np.round(1e3 * (dip_tmin + np.arange(dip_idx.stop - dip_idx.start) / rate), 3)),
                versions=dict(python=platform.python_version(), numpy=np.__version__, mne=mne.__version__,
                              pandas=pd.__version__, scipy=__import__("scipy").__version__),
                checks=checks, written=datetime.now().isoformat(timespec="seconds"))
    (out / "meta.json").write_text(json.dumps(otr.to_jsonable(meta), indent=1), encoding="utf-8")
    otr.report(f"pre / {subject}", complete, checks, t_start, say)
    return complete


def is_complete_pre(subject):
    d = pre_dir(subject)
    f = d / "meta.json"
    if not (f.is_file() and (d / "scores.csv").is_file()):
        return False
    m = json.loads(f.read_text(encoding="utf-8"))
    zg = m.get("z_grid", [])
    return bool(m.get("complete", False)) and len(zg) == len(PRE_Z_GRID) and \
        all(abs(z - z1) < 1e-9 for z, z1 in zip(PRE_Z_GRID, zg)) and m.get("gate_window") == PRE_GATE


# %% 2 · Load the extracted results --------------------------------------------------------------------------------

DESIGNS = list(WINDOWS) + [PRE]                     # the simulated gates (calibrations and a full pass each)
DERIVED = [f"pre{W}" for W in WINDOWS if f"pre{W}" != PRE]     # the ICA-window reference on the other windows:
#                                                     decisions only, derived exactly from the median |s| per window
ZGRID = {**{W: Z_GRID for W in WINDOWS}, PRE: PRE_Z_GRID, **{d: PRE_Z_GRID for d in DERIVED}}
ANCHOR = {**{W: Z_ANCHOR for W in WINDOWS}, PRE: PRE_Z_ANCHOR}
N45_COLS = ["veog_mean_n45", "heog_mean_n45", "veog_ptp_n45", "heog_ptp_n45", "veog_mean_gate", "heog_mean_gate",
            "gfp_n45", "removed_veog_n45", "removed_rms_n45"]


def wl(window):
    if window == PRE or window in DERIVED:
        return f"{window[3:].replace('-', '–')} ms, reference before the pulse"
    return window.replace("-", "–") + " ms"


def short(design):
    """Compact label: '20–60' (window reference), '20–60 pre' (ICA-window reference)."""
    return design[3:].replace("-", "–") + " pre" if design.startswith("pre") else design.replace("-", "–")


def load_session(s):
    """One row per pulse (the first review's wide table) plus every design's score, decision at its anchor cutoff,
    N45 and the frontal measures (N45 window from the pre pass); the ocular evidence flags; the calibrations (meta)
    and dipole fits of every (design, cutoff)."""
    res = otr.load_subject(s)
    d = sub_dir(s)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    sc = pd.read_csv(d / "scores.csv", keep_default_na=False, na_values=[""], low_memory=False)
    sc["reason"] = sc["reason"].fillna("").astype(str)
    eye = [int(i) for i in meta["eye_ics"]]
    if eye != [int(i) for i in res["meta"]["eye_ics"]]:
        raise RuntimeError(f"{s}: other eye ICs than in the first review")
    w = res["w"].set_index("stem").copy()                # (copies keep the frame in few blocks as columns are added)
    bv = otr.variant_name(Z_ANCHOR)                      # the first review's 15-100 ms window variant at z 2.5
    if f"z_max__{bv}" not in w:
        raise RuntimeError(f"{s}: the first review has no {bv} variant")
    for c, c1 in (("score", "z_max"), ("argmax", "z_argmax_ic"), ("reason", "reason"), ("amp", "amp")):
        w[f"{c}__{BASE_WINDOW}"] = w[f"{c1}__{bv}"]
    for j in range(len(eye)):
        w[f"zeye{j + 1}__{BASE_WINDOW}"] = w[f"z_eye{j + 1}__{bv}"]
    for W in NEW_WINDOWS:
        x = sc[sc.window == W].set_index("stem").reindex(w.index)
        if x.z_max.isna().any():
            raise RuntimeError(f"{s}: pulses without a {W} row")
        w[f"score__{W}"] = x.z_max.astype(float)
        w[f"argmax__{W}"] = x.z_argmax_ic.astype(int)
        w[f"reason__{W}"] = x.reason.fillna("").astype(str)
        w[f"amp__{W}"] = x.amp.astype(float)
        for j in range(len(eye)):
            w[f"zeye{j + 1}__{W}"] = x[f"z_eye{j + 1}"].astype(float)
    x0 = sc[sc.window == NEW_WINDOWS[0]].set_index("stem").reindex(w.index)
    for c in x0.columns:
        if c.startswith(COMMON_PREFIXES):
            w[c] = x0[c].astype(float)
    w = w.copy()                                         # one block again after the many added columns
    # ---- the pre-stimulus reference (exact pass) and the N45-window measures of every pulse
    pdir = pre_dir(s)
    pmeta = json.loads((pdir / "meta.json").read_text(encoding="utf-8"))
    pr = pd.read_csv(pdir / "scores.csv", keep_default_na=False, na_values=[""], low_memory=False)
    pr = pr.set_index("stem").reindex(w.index)
    if pr.z_max.isna().any() or [int(i) for i in pmeta["eye_ics"]] != eye:
        raise RuntimeError(f"{s}: the pre pass is incomplete or has other eye ICs")
    w[f"score__{PRE}"] = pr.z_max.astype(float)
    w[f"argmax__{PRE}"] = pr.z_argmax_ic.astype(int)
    w[f"reason__{PRE}"] = pr.reason.fillna("").astype(str)
    w[f"amp__{PRE}"] = pr.amp.astype(float)
    for j in range(len(eye)):
        w[f"zeye{j + 1}__{PRE}"] = pr[f"z_eye{j + 1}"].astype(float)
        w[f"med_abs_s{j + 1}__{PRE_GATE}"] = pr[f"med_abs_s{j + 1}"].astype(float)
    for c in N45_COLS:
        w[c] = pr[c].astype(float)
    w[f"final_veog_n45__{PRE}"] = pr.final_veog_n45.astype(float)
    w = w.copy()
    # the ICA-window reference applied to the other windows: median |s| over W from that window's score and reference
    pst = pmeta["calibrations"][pre_name(PRE_Z_ANCHOR)]["eye_stats"]
    for W in WINDOWS:
        st = meta["calibrations"][wz(W, Z_ANCHOR)]["eye_stats"]
        zz = np.column_stack([((st[str(ic)]["mean"] + w[f"zeye{j + 1}__{W}"] * st[str(ic)]["std"]) - pst[str(ic)]["mean"])
                              / pst[str(ic)]["std"] for j, ic in enumerate(eye)])
        if f"pre{W}" == PRE:                             # simulated exactly: keep the derived score as a check
            w[f"score__{PRE}_derived"] = zz.max(axis=1)
            continue
        w[f"score__pre{W}"] = zz.max(axis=1)
        w[f"argmax__pre{W}"] = np.asarray(eye)[zz.argmax(axis=1)]
    w = w.copy()                                         # one block again after the many added columns
    for W in DESIGNS:
        w[f"kept_own__{W}"] = w[f"reason__{W}"].eq("")
    w = w.reset_index()
    single = w.single.to_numpy(dtype=bool)
    lines = {}
    for k, spec in EVIDENCE.items():
        v, h = w[spec["cols"][0]].astype(float), w[spec["cols"][1]].astype(float)
        cv, ch = float(v[single].median()), float(h[single].median())
        sv, sh = otr.robust_sd(v[single]), otr.robust_sd(h[single])
        L = dict(veog_median_uV=cv, heog_median_uV=ch, veog_robust_sd_uV=sv, heog_robust_sd_uV=sh,
                 two_sided=spec["two_sided"])
        for tag, kk in (("", EVIDENCE_K), ("_strong", STRONG_K)):
            if spec["two_sided"]:                        # away from the session's typical value, either direction
                flag = ((v - cv).abs() > kk * sv) | ((h - ch).abs() > kk * sh)
            else:
                flag = (v > cv + kk * sv) | (h > ch + kk * sh)
            w[f"ev_{k}{tag}"] = flag.to_numpy()
        L["pct_singles"] = 100 * float(w.loc[single, f"ev_{k}"].mean())
        L["pct_singles_strong"] = 100 * float(w.loc[single, f"ev_{k}_strong"].mean())
        lines[k] = L
    w["clean"] = single & ~np.any([w[f"ev_{k}"].to_numpy(dtype=bool) for k in EVIDENCE], axis=0)
    S = w.loc[single, [f"score__{W}" for W in DESIGNS]]
    w["calm_all"] = False                                # single pulses at or below the median of every design
    w.loc[single, "calm_all"] = (S <= S.median()).all(axis=1).to_numpy()
    for k in EVIDENCE:
        lines[k]["pct_calm_pulses"] = 100 * float(w.loc[w.calm_all, f"ev_{k}"].mean())
    return dict(subject=s, res=res, meta=meta, w=w, dir=d, eye_ics=eye, evidence=lines,
                fit=dict(np.load(d / "fitting.npz")), arr=dict(dir=d, w=w),
                pre_meta=pmeta, pre_fit=dict(np.load(pdir / "fitting.npz")), pre_arr=dict(dir=pdir, w=w),
                pre_sources=dict(np.load(pdir / "pre_sources.npz")))


def load_all(subjects):
    D = {}
    for s in subjects:
        if otr.is_complete("variants", s) and is_complete(s) and is_complete_pre(s):
            D[s] = load_session(s)
        else:
            print(f"not extracted yet: {s}")
    return D


def calibration_meta(R, W, z):
    """The calibration record of design W at cutoff z (W 'now': the decider as it runs)."""
    if W == "now":
        return R["res"]["meta"]["variants"]["current"]
    if W == PRE:
        return R["pre_meta"]["calibrations"][pre_name(z)]
    return R["meta"]["calibrations"][wz(W, z)]


def now_z(R):
    return float(R["res"]["thr"]["current"])


def rejected(R, W, z):
    """Ocular decision per pulse: window W at cutoff z; W 'now': the decider as it runs (its own reason)."""
    w = R["w"]
    if W == "now":
        return w["reason__current"].eq("ocular").to_numpy()
    return (w[f"score__{W}"] > z).to_numpy()


def proc_spec(R, W=None):
    """(arrays, kind, (leadfield, index range)) of a processing: None or 'now' = the decider as it runs ('current',
    the fixed processing), a window = that window's own calibration at Z_ANCHOR (SOUND, SSP-SIR, dipole)."""
    res = R["res"]
    if W is None or W == "now":
        return res, "dip_current", otr.fit_of(res, "current")
    if W == BASE_WINDOW:
        v = otr.variant_name(Z_ANCHOR)
        return res, f"dip_{v}", otr.fit_of(res, v)
    if W == PRE:
        v = pre_name(PRE_Z_ANCHOR)
        return R["pre_arr"], "dip_pre", (R["pre_fit"][f"L_pos_{v}"], R["pre_fit"][f"index_range_{v}"])
    v = wz(W, Z_ANCHOR)
    return R["arr"], f"dip_{W}", (R["fit"][f"L_pos_{v}"], R["fit"][f"index_range_{v}"])


def evoked(spec, stems, n_boot=0, rng=None):
    """N45 of the average of `stems` (the decider's dipole fit on the averaged epoch); with n_boot, bootstrap
    amplitudes (pulses resampled with replacement, drawn as multinomial counts: one matrix product)."""
    arrays, kind, (L, ir) = spec
    stems = list(stems)
    if len(stems) < 2:
        return np.nan, None
    a, b = int(ir[0]), int(ir[1])
    X = otr.load_arrays(arrays, kind, stems)[:, :, a:b].astype(float)
    if not np.all(np.isfinite(X)):
        raise ValueError(f"missing dipole-window epochs ({kind})")
    amp = otr.n45_fit(X.mean(axis=0), L, (0, b - a))
    if not n_boot:
        return amp, None
    n = len(X)
    C = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot)
    return amp, otr.n45_fit((C @ X.reshape(n, -1) / n).reshape(n_boot, X.shape[1], b - a), L, (0, b - a))


OUTCOME_BLOCKS = ["baseline"] + list(otr.EVAL_BLOCKS)


def outcome(R, kept, spec, n_boot=0, rng=None):
    """Evoked N45 per open-loop block of the kept single pulses and its change from baseline (t0 ... t60)."""
    w = R["w"]
    single, block = w.single.to_numpy(dtype=bool), w.block.to_numpy()
    amps, boots, ns = {}, {}, {}
    for b in OUTCOME_BLOCKS:
        m = kept & single & (block == b)
        ns[b] = int(m.sum())
        amps[b], boots[b] = evoked(spec, w.stem[m], n_boot, rng)
    ev_blocks = [b for b in otr.EVAL_BLOCKS if np.isfinite(amps[b])]
    o = dict(n_kept=ns, evoked=amps, change={b: 100 * (amps[b] / amps["baseline"] - 1) for b in ev_blocks})
    o["mean_change"] = float(np.mean(list(o["change"].values())))
    if n_boot:
        ch = {b: 100 * (boots[b] / boots["baseline"] - 1) for b in ev_blocks}
        o["ci"] = {b: tuple(float(x) for x in np.percentile(ch[b], [2.5, 97.5])) for b in ev_blocks}
        o["mean_draws"] = np.mean([ch[b] for b in ev_blocks], axis=0)
        o["mean_ci"] = tuple(float(x) for x in np.percentile(o["mean_draws"], [2.5, 97.5]))
    return o


def evoked_rows(s, W, z, processing, o):
    rows = []
    for b, amp in o["evoked"].items():
        r = dict(subject=s, window=W, z=z, processing=processing, block=b, n_kept_single=o["n_kept"][b],
                 evoked_n45=amp, change_pct=o["change"].get(b, np.nan))
        if "ci" in o and b in o["ci"]:
            r["change_ci_lo"], r["change_ci_hi"] = o["ci"][b]
        rows.append(r)
    return rows


# %% 3 · Analysis: tables --------------------------------------------------------------------------------------------

EQUAL_COST_BASE = PRE_GATE                          # equal-cost differences: against the 20-60 ms window reference
SWEEP_Z = np.round(np.arange(0.5, 8.0001, 0.25), 2)  # decisions-only sweep of the ICA-window reference designs


def analyze(D):
    T = {k: [] for k in ("checks", "evidence", "reference", "decisions", "matched", "marginal", "disagreement", "evoked",
                         "calibration", "pre_cutoffs", "pre_sweep", "pre_bands")}
    band_data = {}                                       # session -> the pulses of the score bands (pre_band_data)
    draws = {}                                           # (design, z, processing) -> per session: mean-change draws
    mboot = {}                                           # (design, q, evidence) -> per session: equal-cost precision draws
    for s, R in D.items():
        w, meta, res = R["w"], R["meta"], R["res"]
        single = w.single.to_numpy(dtype=bool)
        clean = w.clean.to_numpy(dtype=bool)
        rng = np.random.default_rng(1)
        # ---- checks: extraction (this review, the first review) and analysis
        for part, m in (("windows", meta), ("pre", R["pre_meta"]), ("first review: replay", res["meta"]),
                        ("first review: online", res["online_meta"])):
            for name, v in otr.flatten_checks((m or {}).get("checks", {})):
                T["checks"].append(dict(part=part, subject=s, check=name, ok=bool(v["ok"]), detail=v["detail"]))

        def check(name, ok, detail):
            T["checks"].append(dict(part="analysis", subject=s, check=name, ok=bool(ok), detail=detail))
        for z in otr.Z_GRID:                             # derived decisions == the decider's own (15-100 ms)
            v = otr.variant_name(z)
            if f"reason__{v}" in w:
                a = (w[f"score__{BASE_WINDOW}"] > z).to_numpy() == w[f"reason__{v}"].eq("ocular").to_numpy()
                check(f"{BASE_WINDOW} ms: score > {z:g} == preprocess_post's own ocular decision (first review, {v})",
                      a.all(), f"{int(a.sum())}/{len(a)} agree")
        for W in NEW_WINDOWS + [PRE]:
            a = (w[f"score__{W}"] > ANCHOR[W]).to_numpy() == w[f"reason__{W}"].eq("ocular").to_numpy()
            check(f"{W}: score > {ANCHOR[W]:g} == preprocess_post's own ocular decision", a.all(),
                  f"{int(a.sum())}/{len(a)} agree")
            arrays, kind, (L, ir) = proc_spec(R, W)
            rd = np.abs(otr.n45_fit(otr.load_arrays(arrays, kind), L, ir) / w[f"amp__{W}"].to_numpy() - 1)
            check(f"{W}: n45_fit() == the decider's fit_trial (float32 epochs), max rel. diff",
                  np.nanmax(rd) < 1e-4, f"{np.nanmax(rd):.1e}")
        dd = max(float(np.max(np.abs(w[f"veog_ptp__{BASE_WINDOW}"] - w.veog_gate_ptp))),
                 float(np.max(np.abs(w[f"heog_ptp__{BASE_WINDOW}"] - w.heog_gate_ptp))))
        check("frontal peak-to-peak over 15-100 ms == the first review's (the same epoch before ICA)", dd < 1e-6,
              f"max |diff| {dd:.1e} µV")
        dd = float(np.max(np.abs(w[f"score__{PRE}_derived"] - w[f"score__{PRE}"])))
        check(f"{PRE}: score derived from the 20-60 ms window pass == the exact pass (so the other windows' "
              f"ICA-window scores are exact too)", dd < 1e-9, f"max |diff| {dd:.1e}")
        # ---- the evidence
        for k, L in R["evidence"].items():
            T["evidence"].append(dict(subject=s, evidence=k, label=EVIDENCE[k]["label"], columns=" / ".join(EVIDENCE[k]["cols"]),
                                      **L, pct_clean_singles=100 * float(clean[single].mean())))
        # ---- reference statistics per design and eye IC
        prof = {}
        src = res["sources"]
        if src is not None:
            t = 1e3 * np.asarray(src["times"], dtype=float)
            Sabs = np.abs(np.asarray(src["src"], dtype=float))
            for j, ic in enumerate(int(i) for i in src["eye_ics"]):
                prof[ic] = {f"median_abs_s_{a}_{b}ms": float(np.median(Sabs[:, j][:, (t >= a) & (t < b)]))
                            for a, b in PROFILE_SPANS}
        pre_src = np.abs(np.asarray(R["pre_sources"]["src"], dtype=float))
        pre_eye = [int(i) for i in R["pre_sources"]["eye_ics"]]
        for j, ic in enumerate(R["eye_ics"]):
            stc = res["meta"]["variants"]["current"]["eye_stats"][str(ic)]
            T["reference"].append(dict(subject=s, ic=ic, window="now (whole epoch)", first_ms=-30.0, last_ms=100.0,
                                       ref_mean=stc["mean"], ref_sd=stc["std"],
                                       level_at_cutoff=stc["mean"] + now_z(R) * stc["std"], cutoff=now_z(R),
                                       pct_singles_decided_by_ic=100 * float((w.loc[single, "z_argmax_ic__current"]
                                                                               == ic).mean()), **prof.get(ic, {})))
            for W in WINDOWS:
                st = meta["calibrations"][wz(W, Z_ANCHOR)]["eye_stats"][str(ic)]
                ws = meta["window_samples"][W]
                T["reference"].append(dict(subject=s, ic=ic, window=W, first_ms=ws["first_ms"], last_ms=ws["last_ms"],
                                           n_samples=ws["n"], ref_mean=st["mean"], ref_sd=st["std"],
                                           level_at_cutoff=st["mean"] + Z_ANCHOR * st["std"], cutoff=Z_ANCHOR,
                                           pct_singles_decided_by_ic=100 * float((w.loc[single, f"argmax__{W}"]
                                                                                   == ic).mean()), **prof.get(ic, {})))
            pm = R["pre_meta"]
            st = pm["calibrations"][pre_name(PRE_Z_ANCHOR)]["eye_stats"][str(ic)]
            a = pre_src[:, pre_eye.index(ic)].ravel()
            med_clean = float(np.median(w.loc[clean, f"med_abs_s{j + 1}__{PRE_GATE}"]))
            T["reference"].append(dict(
                subject=s, ic=ic, window=PRE, first_ms=pm["pre_ms"][0], last_ms=pm["pre_ms"][1],
                n_samples=int(pre_src.shape[0] * pre_src.shape[2]), ref_mean=st["mean"], ref_sd=st["std"],
                level_at_cutoff=st["mean"] + PRE_Z_ANCHOR * st["std"], cutoff=PRE_Z_ANCHOR,
                pct_singles_decided_by_ic=100 * float((w.loc[single, f"argmax__{PRE}"] == ic).mean()),
                ref_median=float(np.median(a)), ref_robust_sd=otr.robust_sd(a),
                pct_ref_samples_above_mean_plus_3sd=100 * float(np.mean(a > st["mean"] + 3 * st["std"])),
                clean_median_abs_s_gate=med_clean, clean_score_of_ic=(med_clean - st["mean"]) / st["std"],
                **prof.get(ic, {})))
        # ---- decisions: cost, catch, outcome per design and cutoff (and the decider now)
        ev = {k: w[f"ev_{k}"].to_numpy(dtype=bool) & single for k in EVIDENCE}
        strong = {k: w[f"ev_{k}_strong"].to_numpy(dtype=bool) & single for k in EVIDENCE}
        mad_ok = w["mad_ok__current"].to_numpy(dtype=bool)
        groups = {"singles": single, "open_loop": (w.category == "open_loop").to_numpy(),
                  "triplets": (w.category == "prime_triplet").to_numpy(),
                  "intervention_singles": single & w.block.str.startswith("iv").to_numpy()}
        blocks_ol = [b for b in otr.OPEN_LOOP_BLOCKS if b in set(w.block)]
        combos = [(W, z) for W in DESIGNS for z in ZGRID[W]] + [("now", now_z(R))]
        for W, z in combos:
            rej = rejected(R, W, z)
            kept = ~rej & mad_ok
            cm = calibration_meta(R, W, z)
            row = dict(subject=s, window=W, z=z)
            for g, m in groups.items():
                row[f"pct_{g}_rejected"] = 100 * float(rej[m].mean())
            row["pct_clean_rejected"] = 100 * float(rej[clean].mean())
            row["n_singles_rejected"] = int((rej & single).sum())
            row["n_clean_rejected"] = int((rej & clean).sum())
            for k, e in ev.items():
                row[f"n_evidence_{k}"] = int(e.sum())
                row[f"n_rejected_with_{k}"] = int((rej & e).sum())
                row[f"hit_{k}"] = 100 * float(rej[e].mean()) if e.any() else np.nan
                row[f"precision_{k}"] = 100 * float(e[rej & single].mean()) if (rej & single).any() else np.nan
                row[f"n_kept_strong_{k}"] = int((kept & strong[k]).sum())
            row["calibration_dropped_ocular"] = len(cm["dropped"]["ocular"])
            row["calibration_trials_for_prime"] = int(cm["n_kept"])
            nb = {b: int((kept & single & (w.block == b).to_numpy()).sum()) for b in blocks_ol}
            b_min = min(nb, key=nb.get)
            row["fewest_kept_open_loop_block"], row["fewest_kept_block"] = nb[b_min], b_min
            boot = N_BOOT if (W == "now" or abs(z - ANCHOR[W]) < 1e-9) else 0
            o = outcome(R, kept, proc_spec(R), boot, rng)
            row["evoked_change_mean_fixed_pct"] = o["mean_change"]
            row["evoked_change_t0_fixed_pct"] = o["change"].get("t0", np.nan)
            T["evoked"] += evoked_rows(s, W, z, "fixed", o)
            if boot:
                row["evoked_change_mean_fixed_ci_lo"], row["evoked_change_mean_fixed_ci_hi"] = o["mean_ci"]
                draws.setdefault((W, z, "fixed"), []).append(o["mean_draws"])
                ko = w["reason__current"].eq("").to_numpy() if W == "now" else w[f"kept_own__{W}"].to_numpy(dtype=bool)
                oo = outcome(R, ko, proc_spec(R, W), N_BOOT, rng)
                row["evoked_change_mean_own_pct"] = oo["mean_change"]
                row["evoked_change_mean_own_ci_lo"], row["evoked_change_mean_own_ci_hi"] = oo["mean_ci"]
                draws.setdefault((W, z, "own"), []).append(oo["mean_draws"])
                T["evoked"] += evoked_rows(s, W, z, "own", oo)
            T["decisions"].append(row)
        # ---- the ICA-window reference on every window, decisions only, on a fine sweep of the cutoff
        sg = w[w.single]
        for W in [PRE] + DERIVED:
            sco = sg[f"score__{W}"].to_numpy()
            for z in SWEEP_Z:
                rj = sco > z
                row = dict(subject=s, design=W, z=float(z), pct_singles_rejected=100 * float(rj.mean()),
                           pct_clean_rejected=100 * float(rj[sg.clean.to_numpy(dtype=bool)].mean()))
                for k in EVIDENCE:
                    e = sg[f"ev_{k}"].to_numpy(dtype=bool)
                    row[f"hit_{k}"] = 100 * float(rj[e].mean()) if e.any() else np.nan
                    row[f"precision_{k}"] = 100 * float(e[rj].mean()) if rj.any() else np.nan
                T["pre_sweep"].append(row)
        # ---- candidate cutoffs for the ICA-window reference
        now_pct = 100 * float(rejected(R, "now", now_z(R))[single].mean())
        win_pct = 100 * float((sg[f"score__{EQUAL_COST_BASE}"] > Z_ANCHOR).mean())
        fine = np.round(np.arange(0.0, 12.0001, 0.01), 2)
        for W in [PRE] + DERIVED:
            sco = sg[f"score__{W}"].to_numpy()
            pct = np.array([100 * np.mean(sco > z) for z in fine])
            e45 = sg.ev_n45.to_numpy(dtype=bool)
            prec = np.array([100 * np.mean(e45[sco > z]) if (sco > z).any() else np.nan for z in fine])
            low = np.where(np.nan_to_num(prec, nan=100.0) < 90)[0]      # above this z the precision stays >= 90 %
            z90 = float(fine[0]) if not len(low) else (float(fine[low[-1] + 1]) if low[-1] + 1 < len(fine) else np.nan)
            cs = sco[sg.clean.to_numpy(dtype=bool)]
            T["pre_cutoffs"].append(dict(
                subject=s, design=W, now_pct_singles=now_pct,
                z_same_cost_as_now=float(fine[np.argmax(pct <= now_pct)]),
                window_20_60_z2_5_pct_singles=win_pct,
                z_same_cost_as_window_20_60_z2_5=float(fine[np.argmax(pct <= win_pct)]),
                **{f"z_clean_p{q:g}": float(np.percentile(cs, q)) for q in FALSE_ALARM_PCT},
                z_precision_n45_90=z90,
                clean_median=float(np.median(cs))))
        # ---- equal cost: each design's cutoff set to reject the same % of single pulses
        for q in MATCHED_PCT:
            for W in DESIGNS + DERIVED:
                sco = sg[f"score__{W}"].to_numpy()
                c = float(np.quantile(sco, 1 - q / 100))
                rj = sco > c
                row = dict(subject=s, pct_target=q, window=W, cutoff=c, n_rejected=int(rj.sum()),
                           pct_rejected=100 * float(rj.mean()),
                           pct_clean_rejected=100 * float(rj[sg.clean.to_numpy(dtype=bool)].mean()))
                for k in EVIDENCE:
                    e = sg[f"ev_{k}"].to_numpy(dtype=bool)
                    row[f"precision_{k}"] = 100 * float(e[rj].mean()) if rj.any() else np.nan
                    row[f"hit_{k}"] = 100 * float(rj[e].mean()) if e.any() else np.nan
                T["matched"].append(row)
        # bootstrap of the equal-cost precision: the session's single pulses resampled (the same draws for every
        # design, so the designs are compared on the same pulses), each design's cutoff set anew in every draw
        rng_m = np.random.default_rng(2)
        idx = rng_m.integers(0, len(sg), size=(N_BOOT, len(sg)))
        E = {k: sg[f"ev_{k}"].to_numpy(dtype=bool)[idx] for k in EVIDENCE}
        for W in DESIGNS + DERIVED:
            Sb = sg[f"score__{W}"].to_numpy()[idx]
            for q in MATCHED_PCT:
                rj = Sb > np.quantile(Sb, 1 - q / 100, axis=1)[:, None]
                nr = np.maximum(rj.sum(axis=1), 1)
                for k in EVIDENCE:
                    mboot.setdefault((W, q, k), []).append(100 * (rj & E[k]).sum(axis=1) / nr)
        T["marginal"] += marginal_rows(s, sg)
        # ---- what the pulses of each band of the pre-reference score carry in the N45 window, before ICA and after
        #      the decider's cleaning (each pulse minus the typical pulses of its block)
        band_data[s] = pre_band_data(R)
        T["pre_bands"] += pre_band_rows(s, band_data[s], np.random.default_rng(3))
        # ---- the pulses whose decision changes: each window vs 15-100 ms at z 2.5; the ICA-window reference at its
        #      anchor vs the 20-60 ms window reference at z 2.5
        pairs = [(W, BASE_WINDOW, Z_ANCHOR, Z_ANCHOR) for W in NEW_WINDOWS] + [(PRE, PRE_GATE, PRE_Z_ANCHOR, Z_ANCHOR)]
        for W, B, zw, zb in pairs:
            b = (w[f"score__{B}"] > zb).to_numpy()
            n = (w[f"score__{W}"] > zw).to_numpy()
            for name, m in (("both", b & n), (f"only {B}", b & ~n), ("only new", ~b & n)):
                m = m & single
                row = dict(subject=s, window=W, against=B, z_new=zw, z_against=zb, set=name, n=int(m.sum()),
                           n_open_loop=int((m & groups["open_loop"]).sum()),
                           n_intervention=int((m & w.block.str.startswith("iv").to_numpy()).sum()),
                           n_clean=int((m & clean).sum()))
                for k in EVIDENCE:
                    row[f"n_{k}"] = int((m & ev[k]).sum())
                row["pulses"] = " ".join(f"{otr.BLOCK_SHORT.get(r.block, r.block)}#{int(r.trial_in_stage)}"
                                         for r in w[m].itertuples())
                T["disagreement"].append(row)
        # ---- calibrations
        rr_now = res["fit"]["rr_mm_current"]
        vm = res["meta"]["variants"]["current"]
        T["calibration"].append(dict(subject=s, window="now", z=now_z(R),
                                     **{f"dropped_{k}": len(vm["dropped"][k]) for k in ("ocular", "qc", "post")},
                                     trials_for_prime=vm["n_kept"], dropped_ocular_trials=" ".join(map(str, vm["dropped"]["ocular"])),
                                     dipole_position=vm["position_index"], dipole_shift_mm_vs_now=0.0,
                                     post_mad_mean=vm["post_mad_mean_std"][0], post_mad_sd=vm["post_mad_mean_std"][1]))
        for W in DESIGNS:
            for z in ZGRID[W]:
                cm = calibration_meta(R, W, z)
                rr = R["pre_fit"][f"rr_mm_{pre_name(z)}"] if W == PRE else R["fit"][f"rr_mm_{wz(W, z)}"]
                T["calibration"].append(dict(
                    subject=s, window=W, z=z, **{f"dropped_{k}": len(cm["dropped"][k]) for k in ("ocular", "qc", "post")},
                    trials_for_prime=cm["n_kept"], dropped_ocular_trials=" ".join(map(str, cm["dropped"]["ocular"])),
                    dipole_position=cm["position_index"], dipole_shift_mm_vs_now=float(np.linalg.norm(rr - rr_now)),
                    sound_filter_rel_max_diff_vs_now=cm["sound_filter_rel_max_diff_vs_current"],
                    post_mad_mean=cm["post_mad_mean_std"][0], post_mad_sd=cm["post_mad_mean_std"][1]))
    A = {k: pd.DataFrame(v) for k, v in T.items()}
    if D:
        allsg = pd.concat([R["w"][R["w"].single] for R in D.values()], ignore_index=True)
        A["marginal"] = pd.concat([A["marginal"], pd.DataFrame(marginal_rows("pooled", allsg))], ignore_index=True)
        A["decisions"] = pd.concat([A["decisions"], pooled_decisions(A["decisions"], draws)], ignore_index=True)
        ps = A["pre_sweep"]
        A["pre_sweep"] = pd.concat([ps, ps.groupby(["design", "z"], sort=False).mean(numeric_only=True).reset_index()
                                    .assign(subject="pooled")], ignore_index=True)
        pc = A["pre_cutoffs"]
        A["pre_cutoffs"] = pd.concat([pc, pc.groupby("design", sort=False).mean(numeric_only=True).reset_index()
                                      .assign(subject="mean of the sessions")], ignore_index=True)
        if len({tuple(D[s]["pre_meta"]["ch_names"]) for s in band_data}) == 1:     # pulses pooled: the same channels
            bd = {k: np.concatenate([band_data[s][k] for s in band_data]) for k in band_data[next(iter(band_data))]}
            A["pre_bands"] = pd.concat([A["pre_bands"], pd.DataFrame(pre_band_rows("pooled", bd,
                                                                                   np.random.default_rng(3)))],
                                       ignore_index=True)
        mt = A["matched"]
        A["matched"] = pd.concat([mt, mt.groupby(["pct_target", "window"], sort=False).mean(numeric_only=True)
                                  .reset_index().assign(subject="pooled")], ignore_index=True)
        # mean of the sessions with bootstrap CIs, and each design against the 20-60 ms window reference on the
        # same draws
        mp = A["matched"][A["matched"].subject == "pooled"].set_index(["pct_target", "window"])
        rows = []
        for q in MATCHED_PCT:
            for W in DESIGNS + DERIVED:
                r = dict(pct_target=q, window=W)
                for k in EVIDENCE:
                    pooled = np.mean(mboot[(W, q, k)], axis=0)
                    base = np.mean(mboot[(EQUAL_COST_BASE, q, k)], axis=0)
                    r[f"precision_{k}"] = float(mp.loc[(q, W), f"precision_{k}"])
                    r[f"precision_{k}_ci_lo"], r[f"precision_{k}_ci_hi"] = (float(x) for x in
                                                                            np.percentile(pooled, [2.5, 97.5]))
                    if W != EQUAL_COST_BASE:
                        dk = f"diff_vs_{EQUAL_COST_BASE}_{k}"
                        r[dk] = r[f"precision_{k}"] - float(mp.loc[(q, EQUAL_COST_BASE), f"precision_{k}"])
                        r[f"{dk}_ci_lo"], r[f"{dk}_ci_hi"] = (float(x) for x in
                                                              np.percentile(pooled - base, [2.5, 97.5]))
                r["pct_clean_rejected"] = float(mp.loc[(q, W), "pct_clean_rejected"])
                rows.append(r)
        A["matched_pooled"] = pd.DataFrame(rows)
        dg = A["disagreement"]
        A["disagreement"] = pd.concat([dg, dg.drop(columns="pulses").groupby(["window", "against", "z_new", "z_against",
                                                                              "set"], sort=False)
                                       .sum(numeric_only=True).reset_index().assign(subject="pooled")], ignore_index=True)
    return A


TYPICAL_MAX = 1.0                                   # typical pulses: no ocular evidence and a pre-reference score <= this
PRE_BANDS = [1.0, 2.0, 2.5, 3.0, 3.5, 5.0]          # bands of the pre-reference score above the typical pulses


def typical_mask(R):
    w = R["w"]
    return (w.single & w.clean & (w[f"score__{PRE}"] <= TYPICAL_MAX)).to_numpy(dtype=bool)


def typical_by_block(R, x, n_min=5):
    """Per pulse, the mean of x over the typical pulses of its own block (of the session where a block has fewer
    than n_min): differences to it do not mix in the change of the N45 between blocks."""
    typ = typical_mask(R)
    blocks = R["w"].block.to_numpy()
    out = np.empty_like(x)
    base = x[typ].mean(axis=0)
    for b in np.unique(blocks):
        m = blocks == b
        t = typ & m
        out[m] = x[t].mean(axis=0) if t.sum() >= n_min else base
    return out


def n45_maps(R):
    """Per pulse, the N45-window (38-50 ms) scalp maps of the pass with the reference before the pulse: the gate's
    epoch before ICA, what the eye ICs removed, and the cleaned epoch (the decider's processing)."""
    a = R["pre_arr"]
    return dict(pre=otr.load_arrays(a, "map_n45_pre").astype(float),
                rem=otr.load_arrays(a, "map_n45_removed").astype(float),
                fin=otr.load_arrays(a, "dip_pre").astype(float).mean(axis=2))


def pre_band_data(R):
    """The typical pulses and the single pulses of each band of the pre-reference score (PRE_BANDS), each minus the
    typical pulses of its block: the Fp1/Fpz/Fp2 mean over 38-50 ms before ICA and after the decider's cleaning, and
    the cleaned N45-window map."""
    w = R["w"]
    single = w.single.to_numpy(dtype=bool)
    sco = w[f"score__{PRE}"].to_numpy()
    typ = typical_mask(R)
    band = np.full(len(w), -1)
    edges = PRE_BANDS + [np.inf]
    for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        band[single & (sco > lo) & (sco <= hi)] = b
    keep = typ | (band >= 0)
    fb = w.veog_mean_n45.to_numpy(dtype=float)
    fa = w[f"final_veog_n45__{PRE}"].to_numpy(dtype=float)
    fin = n45_maps(R)["fin"]
    return dict(band=band[keep], typical=typ[keep], ev=w.ev_n45.to_numpy(dtype=bool)[keep],
                fp_before=(fb - typical_by_block(R, fb))[keep], fp_after=(fa - typical_by_block(R, fa))[keep],
                res=(fin - typical_by_block(R, fin))[keep])


def pre_band_rows(s, d, rng, n_draw=200):
    """Per band: the frontal offset before ICA and what is left of it after cleaning (mean, mean |.|, SE), and the
    RMS over channels of the band's mean cleaned map minus typical, against the RMS expected from as many typical
    pulses (median of n_draw draws with replacement): a ratio near 1 = nothing left beyond noise."""
    edges = PRE_BANDS + [np.inf]
    ti = np.where(d["typical"])[0]
    rows = []
    for b in range(-1, len(edges) - 1):
        m = d["typical"] if b < 0 else d["band"] == b
        n = int(m.sum())
        if not n:
            continue
        fb, fa = d["fp_before"][m], d["fp_after"][m]
        row = dict(subject=s, band="typical" if b < 0 else f"({edges[b]:g}, {edges[b + 1]:g}]",
                   z_from=-np.inf if b < 0 else edges[b], z_to=TYPICAL_MAX if b < 0 else edges[b + 1], n_single=n,
                   pct_n45=100 * float(d["ev"][m].mean()),
                   fp_before_mean_uV=float(fb.mean()), fp_before_abs_uV=float(np.abs(fb).mean()),
                   fp_after_mean_uV=float(fa.mean()), fp_after_se_uV=float(fa.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
                   fp_after_abs_uV=float(np.abs(fa).mean()))
        if b >= 0:
            rms = float(np.sqrt(np.mean(d["res"][m].mean(axis=0) ** 2)))
            floor = float(np.median([np.sqrt(np.mean(d["res"][rng.choice(ti, n)].mean(axis=0) ** 2))
                                     for _ in range(n_draw)]))
            row.update(residual_rms_uV=rms, residual_rms_noise_uV=floor, residual_rms_ratio=rms / floor)
        rows.append(row)
    return rows


def marginal_rows(s, sg):
    """% of single pulses with each kind of evidence (and clean) in bands of each design's score."""
    rows = []
    for W in DESIGNS + DERIVED:
        edges = [-np.inf] + (MARGINAL_EDGES if W in WINDOWS else PRE_MARGINAL_EDGES) + [np.inf]
        sco = sg[f"score__{W}"].to_numpy()
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (sco > lo) & (sco <= hi)
            n = int(m.sum())
            row = dict(subject=s, window=W, z_from=lo, z_to=hi, n_single=n,
                       pct_clean=100 * float(sg.clean.to_numpy(dtype=bool)[m].mean()) if n else np.nan)
            for k in EVIDENCE:
                kk = int(sg[f"ev_{k}"].to_numpy(dtype=bool)[m].sum())
                row[f"pct_{k}"] = 100 * kk / n if n else np.nan
                row[f"pct_{k}_ci_lo"], row[f"pct_{k}_ci_hi"] = otr.wilson(kk, n)
            rows.append(row)
    return rows


def pooled_decisions(dec, draws):
    """Mean over the sessions (rates, changes), sums (counts), minimum (fewest kept; calibration trials: also the
    minimum); the CI of the mean change from the sessions' bootstrap draws."""
    out = []
    for (W, z), g in dec.groupby(["window", "z"], sort=False):
        r = dict(subject="pooled", window=W, z=z)
        for c in dec.columns:
            if c in ("subject", "window", "z", "fewest_kept_block") or c.endswith(("_ci_lo", "_ci_hi")):
                continue
            if c.startswith("n_") or c == "calibration_dropped_ocular":
                r[c] = g[c].sum()
            elif c == "fewest_kept_open_loop_block":
                r[c] = g[c].min()
            else:
                r[c] = g[c].mean()
            if c == "calibration_trials_for_prime":
                r["calibration_trials_for_prime_min"] = g[c].min()
        for proc in ("fixed", "own"):
            dr = draws.get((W, z, proc))
            if dr and len(dr) == len(g):
                lo, hi = np.percentile(np.mean(dr, axis=0), [2.5, 97.5])
                r[f"evoked_change_mean_{proc}_ci_lo"], r[f"evoked_change_mean_{proc}_ci_hi"] = float(lo), float(hi)
        out.append(r)
    return pd.DataFrame(out)


def write_tables(D, A):
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    for name, df in A.items():
        df.to_csv(OUT / "tables" / f"{name}.csv", index=False)
    for s, R in D.items():
        R["w"].to_csv(OUT / "tables" / f"pulses_{s}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 80, "display.max_colwidth", 120,
                           "display.float_format", "{:.3g}".format):
        ch = A["checks"]
        bad = ch[~ch.ok.astype(bool)]
        print(f"\nChecks: {len(ch) - len(bad)}/{len(ch)} pass" + ("" if not len(bad) else ":"))
        for r in bad.itertuples():
            print(f"  FAIL {r.part} {r.subject} {r.check}: {r.detail}")
        print("\nOcular evidence (frontal channels; % of single pulses):")
        print(A["evidence"].drop(columns=["label", "columns"]).to_string(index=False))
        cols = ["pct_singles_rejected", "pct_triplets_rejected", "pct_clean_rejected", "hit_n45", "precision_n45",
                "hit_pre", "precision_pre", "n_kept_strong_n45", "calibration_trials_for_prime",
                "calibration_trials_for_prime_min", "fewest_kept_open_loop_block", "evoked_change_mean_fixed_pct",
                "evoked_change_mean_own_pct"]
        dec = A["decisions"]
        print("\nDecisions (pooled: mean over sessions; counts summed; fewest = minimum):")
        print(dec[dec.subject == "pooled"].set_index(["window", "z"])[[c for c in cols if c in dec]].to_string())
        for s in D:
            print(f"\n{s}:")
            print(dec[(dec.subject == s) & dec.window.isin([PRE, "now"])].set_index(["window", "z"])[
                [c for c in cols if c in dec and not c.endswith("_min")]].to_string())
        print("\nCutoffs for the ICA-window reference (per session; 'mean of the sessions' last):")
        print(A["pre_cutoffs"].to_string(index=False))
        mp = A["matched_pooled"]
        print("\nEqual cost (each design's cutoff rejects q % of single pulses): % of the rejected with the evidence, "
              f"mean of the sessions [95 % CI]; difference to {EQUAL_COST_BASE} ms (window reference), same draws")
        for k in EVIDENCE:
            print(f"  {EVIDENCE[k]['label']}:")
            for q in MATCHED_PCT:
                g = mp[mp.pct_target == q].set_index("window")
                cells = []
                for W in DESIGNS + DERIVED:
                    c = f"{short(W)} {g.loc[W, f'precision_{k}']:.0f}"
                    if W != EQUAL_COST_BASE:
                        dk = f"diff_vs_{EQUAL_COST_BASE}_{k}"
                        c += f" ({g.loc[W, dk]:+.0f} [{g.loc[W, dk + '_ci_lo']:+.0f}, {g.loc[W, dk + '_ci_hi']:+.0f}])"
                    else:
                        c += f" [{g.loc[W, f'precision_{k}_ci_lo']:.0f}, {g.loc[W, f'precision_{k}_ci_hi']:.0f}]"
                    cells.append(c)
                print(f"    {q:>2} %: " + " · ".join(cells))
        dg = A["disagreement"]
        print("\nPulses whose decision changes (single pulses, all sessions):")
        print(dg[dg.subject == "pooled"].drop(columns=["subject", "pulses"], errors="ignore").to_string(index=False))
        mg = A["marginal"]
        print("\n% of single pulses with a frontal offset 38-50 ms per band of the score (all sessions):")
        for W in [PRE, PRE_GATE, "pre30-60"]:
            g = mg[(mg.subject == "pooled") & (mg.window == W)]
            print(f"  {W}: " + " · ".join(f"({r.z_from:g}, {r.z_to:g}] {r.pct_n45:.0f} % of {r.n_single}"
                                         for r in g.itertuples()))
        print("\nBands of the pre-reference score: Fp mean over 38-50 ms before ICA and after cleaning, each pulse minus "
              "the typical pulses of its block (µV); RMS of the band's mean cleaned map against noise:")
        print(A["pre_bands"].drop(columns=["z_from", "z_to"]).to_string(index=False))
    print(f"\ntables written to {OUT / 'tables'}")


# %% 4 · Figures ------------------------------------------------------------------------------------------------------
INK, INK_2, INK_MUTED, GRID_C, BASE_C = otr.INK, otr.INK_2, otr.INK_MUTED, otr.GRID_C, otr.BASE_C
WASH, BAND_C, HEAD_C, HEOG_C = otr.WASH, otr.BAND_C, otr.HEAD_C, otr.HEOG_C
EV_C = "#0b0b0b"                                     # pulses with evidence (scatter): ink, the others light grey
NOEV_C = "#c3c2b7"
PRE_C = "#6250d6"                                    # the ICA-window reference: violet, dashed, star (told apart from
PRE_MARKER = "*"                                     #   the blue 30-100 ms line by its style and marker)
SESSION_MARKER = otr.SESSION_MARKER
POOLED_C = otr.POOLED_C
CUT_LEVELS = [2.5, 3.0, 3.5]                         # the ICA-window reference's candidate cutoffs drawn in the figures
CUT_C = {2.5: "#86b6ef", 3.0: "#2a78d6", 3.5: "#0d366b"}   # sequential blue ramp (validated in the first review)


def design_style(W):
    """(colour, marker, line style) of a design."""
    if W == PRE or W in DERIVED:
        return PRE_C, PRE_MARKER, "--"
    if W == "now":
        return HEAD_C, "*", "--"
    return WINDOW_COLOR[W], WINDOW_MARKER[W], "-"


def save(fig, name):
    path = OUT / "figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  wrote {path.relative_to(OUT)}")


def window_handles(with_now=True, now_label=None, with_pre=False):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=WINDOW_COLOR[W], marker=WINDOW_MARKER[W], ms=6, lw=1.8,
                label=wl(W) + (" (the decider's window)" if W == BASE_WINDOW else "")) for W in WINDOWS]
    if with_pre:
        h.append(Line2D([], [], color=PRE_C, marker=PRE_MARKER, ms=9, lw=1.8, ls="--",
                        label="20–60 ms, reference before the pulse (ICA window)"))
    if with_now:
        h.append(Line2D([], [], color=HEAD_C, ls="--", lw=1.3,
                        label=now_label or "now: 15–100 ms, reference the whole epoch, z 3"))
    return h


PRE_IN_W_FIGURES = 3.5                               # the window figures show the reference before the pulse up to here


def _lines_vs_z(ax, dec, col, subject="pooled", now=True, own=False, pre=True):
    d = dec[dec.subject == subject]
    for W in WINDOWS:
        g = d[d.window == W].sort_values("z")
        ax.plot(g.z, g[col], color=WINDOW_COLOR[W], marker=WINDOW_MARKER[W], ms=5.5, lw=1.7, zorder=3)
    if pre:
        g = d[(d.window == PRE) & (d.z <= PRE_IN_W_FIGURES + 1e-9)].sort_values("z")
        ax.plot(g.z, g[col], color=PRE_C, marker=PRE_MARKER, ms=8.5, lw=1.7, ls="--", zorder=3)
    if own and "evoked_change_mean_own_pct" in d:
        for k, W in enumerate(list(WINDOWS) + ([PRE] if pre else [])):
            g = d[(d.window == W) & ((d.z - ANCHOR[W]).abs() < 1e-9)]
            if len(g):
                color, marker, _ = design_style(W)
                ax.plot([ANCHOR[W] + (0.06 if W == PRE else 0.05 + 0.025 * k)], g.evoked_change_mean_own_pct,
                        ls="none", marker=marker, ms=8.5 if W == PRE else 6, mfc="white", mec=color, mew=1.4, zorder=4)
    if now:
        n = d[d.window == "now"]
        if len(n) and np.isfinite(n[col].iloc[0]):
            ax.axhline(float(n[col].iloc[0]), color=HEAD_C, ls="--", lw=1.2, zorder=2)
    ticks = Z_GRID + ([PRE_IN_W_FIGURES] if pre else [])
    ax.set_xticks(ticks, [f"{z:g}" for z in ticks])
    ax.set_xlim(Z_GRID[0] - 0.12, ticks[-1] + 0.2)


def fig_reference(D, A):
    """w1 · How each window sets the bar: |s(t)| of the eye ICs in the calibration block and, per window, the level
    the median |s| over the window must exceed at z 2.5."""
    plt = otr.setup_mpl()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    subs = [s for s in SUBJECTS if s in D and D[s]["res"]["sources"] is not None]
    if not subs:
        return
    n_ic = max(len(D[s]["eye_ics"]) for s in subs)
    fig, axes = plt.subplots(len(subs), n_ic, figsize=(5.0 * n_ic + 0.4, 2.9 * len(subs) + 1.6), squeeze=False,
                             layout="constrained")
    for i, s in enumerate(subs):
        R = D[s]
        src, meta, m1 = R["res"]["sources"], R["meta"], R["res"]["meta"]
        t = 1e3 * np.asarray(src["times"], dtype=float)
        S = np.abs(np.asarray(src["src"], dtype=float))
        eye_src = [int(k) for k in src["eye_ics"]]
        aw = m1["artifact_window_1_s"]
        for j in range(n_ic):
            ax = axes[i, j]
            if j >= len(R["eye_ics"]):
                ax.axis("off")
                continue
            ic = R["eye_ics"][j]
            jj = eye_src.index(ic)
            lo, med, hi = np.percentile(S[:, jj], [10, 50, 90], axis=0)
            ax.axvspan(1e3 * aw[0], 1e3 * aw[1], color=WASH, lw=0, zorder=0)
            ax.fill_between(t, lo, hi, color=BAND_C, lw=0, zorder=1)
            ax.plot(t, med, color=INK, lw=1.3, zorder=3)
            top = float(np.max(hi[t >= 15]))
            sds = []
            for W in WINDOWS:
                st = meta["calibrations"][wz(W, Z_ANCHOR)]["eye_stats"][str(ic)]
                ws = meta["window_samples"][W]
                y = st["mean"] + Z_ANCHOR * st["std"]
                ax.hlines(y, ws["first_ms"], ws["last_ms"], color=WINDOW_COLOR[W], lw=2.4, zorder=4)
                ax.plot([ws["first_ms"]], [y], marker=WINDOW_MARKER[W], color=WINDOW_COLOR[W], ms=5.5, zorder=5)
                top = max(top, y)
                sds.append(f"{W.replace('-', '–')} {st['std']:.2f}")
            stc = m1["variants"]["current"]["eye_stats"][str(ic)]
            y_now = stc["mean"] + now_z(R) * stc["std"]
            ax.hlines(y_now, otr.GATE_WINDOW_MS[0], otr.GATE_WINDOW_MS[1], color=HEAD_C, lw=1.3, ls="--", zorder=4)
            stp = calibration_meta(R, PRE, PRE_Z_ANCHOR)["eye_stats"][str(ic)]
            y_pre = stp["mean"] + PRE_Z_ANCHOR * stp["std"]
            wsp = meta["window_samples"][PRE_GATE]
            ax.hlines(y_pre, wsp["first_ms"], wsp["last_ms"], color=PRE_C, lw=2.0, ls=(0, (3, 1.5)), zorder=4)
            ax.plot([wsp["last_ms"]], [y_pre], marker=PRE_MARKER, color=PRE_C, ms=8, zorder=5)
            sds.append(f"before the pulse {stp['std']:.2f}")
            top = max(top, y_now, y_pre)
            ax.set_ylim(0, 1.1 * top)
            ax.set_xlim(t[0], t[-1])
            ax.axvline(0, color=BASE_C, lw=0.8)
            ax.set_title(f"{s} · eye IC {ic} · SD of |s| over the window:\n" + " | ".join(sds[:3]) + "\n"
                         + " | ".join(sds[3:]), fontsize=7.6)
            if j == 0:
                ax.set_ylabel("|s| (IC units)")
            if i == len(subs) - 1:
                ax.set_xlabel("ms from the pulse")
    handles = [Line2D([], [], color=INK, lw=1.3, label="|s(t)| of the 125 calibration trials: median"),
               Patch(color=BAND_C, label="10–90 %"), Patch(color=WASH, label="blank (interpolated)"),
               *[Line2D([], [], color=WINDOW_COLOR[W], lw=2.4, marker=WINDOW_MARKER[W], ms=5.5,
                        label=f"{wl(W)}: mean + {Z_ANCHOR:g} SD of |s| over the window") for W in WINDOWS],
               Line2D([], [], color=PRE_C, lw=2.0, ls=(0, (3, 1.5)), marker=PRE_MARKER, ms=8,
                      label=f"20–60 ms, reference before the pulse: mean + {PRE_Z_ANCHOR:g} SD of |s| over the ICA "
                            f"window (−1100 to −5 ms)"),
               Line2D([], [], color=HEAD_C, lw=1.3, ls="--", label="now: mean + 3 SD of |s| over −30 to 100 ms")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8)
    fig.suptitle(f"Each window sets its own bar: an eye IC rejects a pulse when the median |s| over the window "
                 f"exceeds the bar (drawn over the window's span; windows at z {Z_ANCHOR:g}, the reference before the "
                 f"pulse at z {PRE_Z_ANCHOR:g})", fontsize=10.5)
    save(fig, "w1_reference_windows.png")


SUMMARY_PANELS = [
    ("pct_singles_rejected", "% of single pulses", "Single pulses rejected"),
    ("hit_n45", "% of those pulses", "Pulses with a frontal offset\n38–50 ms that are rejected"),
    ("precision_n45", "% of rejected pulses", "Rejected pulses with a frontal\noffset 38–50 ms"),
    ("precision_pre", "% of rejected pulses", "Rejected pulses with eye activity\nbefore the pulse (−210 to −10 ms)"),
    ("calibration_trials_for_prime_min", "calibration trials (of 125)", "Calibration trials left for PRIME\n(fewest "
                                                                         "of the four sessions)"),
    ("evoked_change_mean_fixed_pct", "% change from baseline", "Evoked N45 change, mean of t0–t60\n(fixed processing;"
                                                               " hollow: own decider)")]


def fig_summary(D, A):
    """w2 · The choice in one figure (mean of the sessions): top, against the cutoff; bottom left, at equal cost."""
    plt = otr.setup_mpl()
    dec, mp = A["decisions"], A["matched_pooled"]
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.8), layout="constrained")
    for ax, (col, ylab, title) in zip(list(axes[0]) + list(axes[1, 2:]), SUMMARY_PANELS):
        _lines_vs_z(ax, dec, col, own=col == "evoked_change_mean_fixed_pct")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("cutoff z (each design's own reference)", fontsize=8)
        if col.startswith(("hit", "precision")):
            ax.set_ylim(0, 100)
        if col.startswith("evoked"):
            ax.axhline(0, color=BASE_C, lw=0.9)
        if col.startswith("calibration"):
            ax.set_ylim(95, 125)
    for ax, k in zip(axes[1, :2], ("n45", "pre")):
        for W in DESIGNS:
            g = mp[mp.window == W].sort_values("pct_target")
            color, marker, ls = design_style(W)
            if W == EQUAL_COST_BASE:
                ax.fill_between(g.pct_target, g[f"precision_{k}_ci_lo"], g[f"precision_{k}_ci_hi"],
                                color=color, alpha=0.15, lw=0, zorder=1)
            ax.plot(g.pct_target, g[f"precision_{k}"], color=color, marker=marker, ms=8 if W == PRE else 5.5, lw=1.7,
                    ls=ls, zorder=3)
        ax.set_ylim(0, 100)
        ax.set_xticks(MATCHED_PCT)
        ax.set_xlabel("% of single pulses rejected (each design's cutoff set to it)", fontsize=8)
        ax.set_ylabel("% of rejected pulses")
        lab = EVIDENCE[k]["label"].replace("frontal offset", "a frontal\noffset").replace("eye activity before",
                                                                                           "eye\nactivity before")
        ax.set_title(f"Equal cost: rejected pulses with {lab}\n(band: 95 % CI of 20–60 ms)", fontsize=9.5)
    fig.legend(handles=window_handles(with_pre=True), loc="outside lower center", ncol=4, fontsize=8.5)
    ev = A["evidence"]
    pv = ev.groupby("evidence").pct_singles.mean()
    fig.suptitle("Which window and cutoff for the N45 (38–50 ms)? Mean of pilot003–006. Frontal offset: Fp mean or F7 − F8 "
                 f"over 38–50 ms (before ICA, baseline −15 to −5 ms) more than {EVIDENCE_K:g} robust SD from the "
                 f"session's median ({pv.get('n45', np.nan):.0f} % of single pulses; eye activity before the pulse: "
                 f"{pv.get('pre', np.nan):.0f} %)", fontsize=10.5)
    save(fig, "w2_summary.png")


def fig_sessions(D, A):
    """w3 · The same per session."""
    plt = otr.setup_mpl()
    dec = A["decisions"]
    subs = [s for s in SUBJECTS if s in D]
    panels = [SUMMARY_PANELS[i] for i in (0, 1, 2, 3)] + [("calibration_trials_for_prime", "calibration trials",
                                                            "Calibration trials left for PRIME")]
    fig, axes = plt.subplots(len(subs), len(panels), figsize=(3.6 * len(panels), 2.6 * len(subs) + 1.3), squeeze=False,
                             layout="constrained")
    for i, s in enumerate(subs):
        for j, (col, ylab, title) in enumerate(panels):
            ax = axes[i, j]
            _lines_vs_z(ax, dec, col, subject=s)
            if col.startswith(("hit", "precision")):
                ax.set_ylim(0, 100)
            if col.startswith("calibration"):              # whole trials, and no zoom onto a one-trial difference
                v = dec.loc[dec.subject == s, col]
                ax.set_ylim(np.floor(v.min()) - 3, np.ceil(v.max()) + 3)
                ax.yaxis.get_major_locator().set_params(integer=True)
            if i == 0:
                ax.set_title(title.replace(" (−210 to −10 ms)", ""), fontsize=9)
            ax.set_ylabel(f"{s}\n{ylab}" if j == 0 else ylab, fontsize=8.5 if j == 0 else 8)
            if i == len(subs) - 1:
                ax.set_xlabel("cutoff z", fontsize=8)
    fig.legend(handles=window_handles(with_pre=True), loc="outside lower center", ncol=4, fontsize=8.5)
    fig.suptitle("Per session: cost and catch of each window and cutoff (the reference before the pulse up to z "
                 f"{PRE_IN_W_FIGURES:g}; its full range in p3)", fontsize=11)
    save(fig, "w3_per_session.png")


def cost_curve(w, W, k):
    sg = w[w.single]
    sco, e = sg[f"score__{W}"].to_numpy(), sg[f"ev_{k}"].to_numpy(dtype=bool)
    x = np.array([100 * np.mean(sco > c) for c in SWEEP])
    y = np.array([100 * np.mean(sco[e] > c) for c in SWEEP]) if e.any() else np.full(len(SWEEP), np.nan)
    return x, y


def fig_cost_curves(D, A):
    """w4 · Equal-cost comparison: share of the pulses with evidence that are rejected, against the share of all single
    pulses rejected, as the cutoff moves (markers: z 2, 2.5, 3)."""
    plt = otr.setup_mpl()
    from matplotlib.lines import Line2D
    subs = [s for s in SUBJECTS if s in D]
    cols = subs + ["pooled"]
    dec = A["decisions"]
    xmax = 10.0
    fig, axes = plt.subplots(len(EVIDENCE), len(cols), figsize=(3.5 * len(cols), 3.2 * len(EVIDENCE) + 1.2),
                             squeeze=False, layout="constrained")
    for i, (k, spec) in enumerate(EVIDENCE.items()):
        label = spec["label"]
        for j, s in enumerate(cols):
            ax = axes[i, j]
            ax.plot([0, xmax], [0, xmax], ls=":", color=INK_MUTED, lw=1.0, zorder=1)
            for W in DESIGNS:
                color, marker, ls = design_style(W)
                marks = (2.0, 2.5, 3.0) if W in WINDOWS else tuple(CUT_LEVELS)
                idx = {z: int(np.argmin(np.abs(SWEEP - z))) for z in marks}
                if s == "pooled":
                    xy = [cost_curve(D[t]["w"], W, k) for t in subs]
                    x, y = np.mean([a for a, _ in xy], axis=0), np.mean([b for _, b in xy], axis=0)
                else:
                    x, y = cost_curve(D[s]["w"], W, k)
                m = x <= xmax + 1
                ax.plot(x[m], y[m], color=color, lw=1.6, ls=ls, zorder=3)
                for z, ii in idx.items():
                    filled = abs(z - ANCHOR[W]) < 1e-9
                    ax.plot([x[ii]], [y[ii]], marker=marker, ms=(9 if W == PRE else 6.5) if filled else
                            (8 if W == PRE else 5), color=color, mfc=color if filled else "white", mew=1.3,
                            ls="none", zorder=4)
            n = dec[(dec.subject == s) & (dec.window == "now")]
            if len(n):
                ax.plot([n.pct_singles_rejected.iloc[0]], [n[f"hit_{k}"].iloc[0]], marker="*", ms=11, color=HEAD_C,
                        mfc="white", mew=1.3, ls="none", zorder=5)
            ax.set_xlim(0, xmax)
            ax.set_ylim(0, 100)
            if s == "pooled":
                tl = "mean of the sessions"
            else:
                tl = f"{s} · {int(D[s]['w'].loc[D[s]['w'].single, f'ev_{k}'].sum())} pulses with it"
            ax.set_title(tl, fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f"% of pulses with {label}\nthat are rejected", fontsize=8.5)
            if i == len(EVIDENCE) - 1:
                ax.set_xlabel("% of single pulses rejected", fontsize=8.5)
    handles = window_handles(with_now=False, with_pre=True) + [
        Line2D([], [], color=INK_2, marker="o", ms=6.5, ls="none",
               label=f"filled: z {Z_ANCHOR:g} (window references), {PRE_Z_ANCHOR:g} (before the pulse)"),
        Line2D([], [], color=INK_2, marker="o", ms=5, mfc="white", ls="none", label="hollow: z 2 and 3 / 2.5 and 3.5"),
        Line2D([], [], color=HEAD_C, marker="*", ms=11, mfc="white", ls="none", label="now (decider as it runs)"),
        Line2D([], [], color=INK_MUTED, ls=":", label="rejecting pulses at random")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=8.5)
    fig.suptitle("Equal cost: as the cutoff moves, how many of the pulses with ocular evidence does each design "
                 "reject for a given share of single pulses rejected? (higher = catches more at the same cost)",
                 fontsize=10.5)
    save(fig, "w4_cost_curves.png")


def fig_scores(D, A):
    """w5 · Score of each new window against the 15-100 ms score (single pulses), coloured by frontal evidence."""
    plt = otr.setup_mpl()
    from matplotlib.lines import Line2D
    subs = [s for s in SUBJECTS if s in D]
    lo, hi = -1.5, 8.0
    fig, axes = plt.subplots(len(subs), len(NEW_WINDOWS), figsize=(3.3 * len(NEW_WINDOWS) + 0.4, 3.2 * len(subs) + 1.1),
                             squeeze=False, layout="constrained")
    for i, s in enumerate(subs):
        sg = D[s]["w"][D[s]["w"].single]
        e = sg.ev_n45.to_numpy(dtype=bool)
        x = np.clip(sg[f"score__{BASE_WINDOW}"].to_numpy(), lo, hi)
        for j, W in enumerate(NEW_WINDOWS):
            ax = axes[i, j]
            y = np.clip(sg[f"score__{W}"].to_numpy(), lo, hi)
            ax.plot([lo, hi], [lo, hi], color=GRID_C, lw=1.0, zorder=1)
            ax.scatter(x[~e], y[~e], s=8, color=NOEV_C, lw=0, zorder=2)
            ax.scatter(x[e], y[e], s=12, color=EV_C, lw=0, zorder=3)
            ax.axvline(Z_ANCHOR, color=WINDOW_COLOR[BASE_WINDOW], lw=1.0, ls="--", zorder=2)
            ax.axhline(Z_ANCHOR, color=WINDOW_COLOR[W], lw=1.2, ls="--", zorder=2)
            xb, yb = x > Z_ANCHOR, y > Z_ANCHOR
            for m, tx, ty, ha, va, name in ((~xb & yb, lo + 0.2, hi - 0.2, "left", "top", f"only {W}"),
                                            (xb & ~yb, hi - 0.2, lo + 0.2, "right", "bottom", "only 15–100"),
                                            (xb & yb, hi - 0.2, hi - 0.2, "right", "top", "both")):
                ax.text(tx, ty, f"{name}: {int(m.sum())}\n({int((m & e).sum())} with an offset)", ha=ha, va=va,
                        fontsize=7.2, color=INK_2)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_title(f"{s} · {wl(W)} vs 15–100 ms", fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f"score, {wl(W)}" if len(NEW_WINDOWS) == 1 else "score, window of the column")
            if i == len(subs) - 1:
                ax.set_xlabel("score, 15–100 ms")
    handles = [Line2D([], [], color=EV_C, marker="o", ms=4.5, ls="none", label="frontal offset 38–50 ms"),
               Line2D([], [], color=NOEV_C, marker="o", ms=4.5, ls="none", label="no offset"),
               Line2D([], [], color=INK_2, ls="--", lw=1.0, label=f"cutoff z {Z_ANCHOR:g}")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8.5)
    fig.suptitle("Single pulses: each window's score against the 15–100 ms score (values beyond the axes at the edge)",
                 fontsize=10.5)
    save(fig, "w5_scores_vs_15-100.png")


def fig_cards(R, A):
    """w6 · Pulses whose decision at z 2.5 depends on the window: frontal traces (top) and the deciding eye IC's |s|
    with both windows' bars (bottom)."""
    plt = otr.setup_mpl()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    res, w, meta, s = R["res"], R["w"], R["meta"], R["subject"]
    sg = w[w.single]
    base = sg[f"score__{BASE_WINDOW}"] > Z_ANCHOR
    rows = []
    for W in NEW_WINDOWS:
        new = sg[f"score__{W}"] > Z_ANCHOR
        for who, other, m in ((W, BASE_WINDOW, new & ~base), (BASE_WINDOW, W, base & ~new)):
            d_all = sg[m].sort_values(f"score__{who}")
            if not len(d_all):
                continue
            d = d_all.iloc[np.round(np.linspace(0, len(d_all) - 1, min(CARDS_PER_ROW, len(d_all)))).astype(int)]
            rows.append((who, other, W, d.drop_duplicates("stem"), d_all))
    if not rows:
        return
    stems = list(dict.fromkeys(st for *_, d, _ in rows for st in d.stem))
    pos = {st: i for i, st in enumerate(stems)}
    tv, th = otr.eog_traces(res, stems, "veog"), otr.eog_traces(res, stems, "heog")
    src = otr.load_arrays(res, "src", stems).astype(float)
    t_ep = np.asarray(res["meta"]["times_epoch_ms"])
    t_pre, b_pre, t_post, b_post = otr.calm_band(res, "veog")
    eye = R["eye_ics"]
    nrow = len(rows)
    H = 2.75 * nrow + 1.4
    fig = plt.figure(figsize=(3.3 * CARDS_PER_ROW + 2.2, H))
    outer = fig.add_gridspec(nrow, CARDS_PER_ROW + 1, width_ratios=[0.62] + [1] * CARDS_PER_ROW, hspace=0.62,
                             wspace=0.28, **otr.page(H, 1.0, 0.8))
    for r, (who, other, W, d, d_all) in enumerate(rows):
        axl = fig.add_subplot(outer[r, 0])
        axl.axis("off")
        k45 = int(d_all.ev_n45.sum())
        kpre = int(d_all.ev_pre.sum())
        axl.text(0.98, 0.5, f"rejected only by\n{wl(who)}\n(kept by {wl(other)})\n\n{len(d_all)} single pulses:\n"
                 f"{k45} with a frontal\noffset 38–50 ms,\n{kpre} with eye activity\nbefore the pulse",
                 ha="right", va="center", fontsize=7.8, transform=axl.transAxes)
        for c, (_, rr) in enumerate(d.iterrows()):      # a Series: column names such as score__15-100 kept
            n = pos[rr["stem"]]
            cell = outer[r, c + 1].subgridspec(2, 1, hspace=0.1)
            axt = fig.add_subplot(cell[0])
            axt.axvspan(1e3 * N45_S[0], 1e3 * N45_S[1], color=WASH, lw=0, zorder=0)
            axt.fill_between(t_pre, b_pre[0], b_pre[1], color=BAND_C, lw=0, zorder=1)
            axt.fill_between(t_post, b_post[0], b_post[1], color=BAND_C, lw=0, zorder=1)
            for (tp, vp, tq, vq), color in ((th, HEOG_C), (tv, INK)):
                axt.plot(tp, vp[n], color=color, lw=0.7, alpha=0.55, zorder=2)
                axt.plot(tq, vq[n], color=color, lw=1.0, zorder=3)
            axt.axvline(0, color=BASE_C, lw=0.8)
            axt.set_xlim(-210, 100)
            axt.tick_params(labelbottom=False, labelsize=6.5)
            axt.set_ylabel("µV", fontsize=6.5)
            sign = (["offset 38–50 ms"] if bool(rr["ev_n45"]) else []) + \
                   (["eye activity before"] if bool(rr["ev_pre"]) else [])
            axt.set_title(f"{otr.label_trial(rr)} · {otr.CAT_SHORT[rr['category']]} · "
                          f"{BASE_WINDOW.replace('-', '–')}: {rr[f'score__{BASE_WINDOW}']:.2f} · "
                          f"{W.replace('-', '–')}: {rr[f'score__{W}']:.2f}\n"
                          + ("; ".join(sign) if sign else "no frontal sign"), fontsize=6.8)
            axz = fig.add_subplot(cell[1], sharex=axt)
            rd = rr
            ic = int(rd[f"argmax__{who}"])
            j = eye.index(ic)
            a = np.abs(src[n][j])
            axz.plot(t_ep, a, color=INK_2, lw=0.9, zorder=2)
            top = float(np.max(a[t_ep >= 15]))
            for WW in (BASE_WINDOW, W):
                st = meta["calibrations"][wz(WW, Z_ANCHOR)]["eye_stats"][str(ic)]
                ws = meta["window_samples"][WW]
                level = st["mean"] + Z_ANCHOR * st["std"]
                med = st["mean"] + rd[f"zeye{j + 1}__{WW}"] * st["std"]
                axz.hlines(level, ws["first_ms"], ws["last_ms"], color=WINDOW_COLOR[WW], lw=1.1, ls="--", zorder=3)
                axz.hlines(med, ws["first_ms"], ws["last_ms"], color=WINDOW_COLOR[WW], lw=2.6, zorder=4)
                top = max(top, level, med)
            axz.set_ylim(0, 1.15 * top)
            axz.axvline(0, color=BASE_C, lw=0.8)
            axz.set_ylabel(f"IC {ic} |s|", fontsize=6.5)
            axz.tick_params(labelsize=6.5)
            if r == nrow - 1:
                axz.set_xlabel("ms from the pulse", fontsize=7)
    handles = [Line2D([], [], color=INK, lw=1.1, label="Fp1/Fpz/Fp2 mean (VEOG)"),
               Line2D([], [], color=HEOG_C, lw=1.1, label="F7 − F8 (HEOG)"),
               Patch(color=BAND_C, label="VEOG of the calm pulses, 5–95 %"),
               Patch(color=WASH, label="38–50 ms (N45 window; frontal offset measured)"),
               Line2D([], [], color=INK_2, lw=1.4, label="bars: median |s| over the window (thick), bar to pass (dashed)"),
               *[Line2D([], [], color=WINDOW_COLOR[W], lw=2.4, label=wl(W)) for W in WINDOWS]]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{s} · single pulses whose decision at z {Z_ANCHOR:g} depends on the window (up to {CARDS_PER_ROW} "
                 f"per row, spread over the rejecting window's score) · top: frontal channels before ICA, faint = raw "
                 f"pre window · bottom: the eye IC that decides", y=1 - 0.25 / H, fontsize=9.5)
    save(fig, f"w6_cards_{s}.png")


def fig_outcome(D, A):
    """w7 · The outcome: evoked N45 change per evaluation block (own decider at z 2.5) and the mean change against the
    cutoff (fixed processing)."""
    plt = otr.setup_mpl()
    from matplotlib.lines import Line2D
    ev, dec = A["evoked"], A["decisions"]
    subs = [s for s in SUBJECTS if s in D]
    eval_b = list(otr.EVAL_BLOCKS)
    fig, axes = plt.subplots(len(subs), 2, figsize=(14, 2.7 * len(subs) + 1.4), squeeze=False, layout="constrained",
                             gridspec_kw=dict(width_ratios=[1.6, 1]))
    order = list(WINDOWS) + [PRE, "now"]
    for i, s in enumerate(subs):
        ax = axes[i, 0]
        e = ev[(ev.subject == s) & (ev.processing == "own") & ev.block.isin(eval_b)]
        for k, W in enumerate(order):
            sel = (e.window == W) if W == "now" else (e.window == W) & ((e.z - ANCHOR[W]).abs() < 1e-9)
            g = e[sel].set_index("block").reindex(eval_b)
            x = np.arange(len(eval_b)) + (k - (len(order) - 1) / 2) * 0.12
            color, marker, _ = design_style(W)
            ax.errorbar(x, g.change_pct, yerr=[g.change_pct - g.change_ci_lo, g.change_ci_hi - g.change_pct],
                        fmt=marker, color=color, ms=8 if marker == "*" else 5,
                        mfc="white" if W == "now" else color, capsize=2, lw=1.0, elinewidth=1.0)
        ax.axhline(0, color=BASE_C, lw=0.9)
        ax.set_xticks(range(len(eval_b)), eval_b)
        ax.set_ylabel("% change from baseline")
        ax.set_title(f"{s} · evoked N45 change per block, each design's own decider (windows z {Z_ANCHOR:g}, "
                     f"pre-stimulus reference z {PRE_Z_ANCHOR:g}; 95 % CI)", fontsize=8.5)
        ax = axes[i, 1]
        _lines_vs_z(ax, dec, "evoked_change_mean_fixed_pct", subject=s, own=True)
        ax.axhline(0, color=BASE_C, lw=0.9)
        ax.set_ylabel("% change, mean t0–t60")
        ax.set_title(f"{s} · mean change against the cutoff (fixed processing)", fontsize=8.5)
        if i == len(subs) - 1:
            ax.set_xlabel("cutoff z")
    handles = window_handles(now_label="now (decider as it runs): dashed, star", with_pre=True) + [
        Line2D([], [], color=INK_2, marker="o", ms=6, mfc="white", ls="none", label="hollow: own decider")]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, fontsize=8.5)
    fig.suptitle("Outcome: does the window change the evoked N45 change from baseline? (fixed processing: the decider's "
                 "cleaning and dipole now, only the kept pulses differ)", fontsize=10.5)
    save(fig, "w7_outcome.png")


# ---- the reference before the pulse (ICA window) on 20-60 ms: which cutoff? -------------------------------------
PRE_BAND_C =["#9ec5f4", "#6fa6ea", "#3987e5", "#256abf", "#184f95", "#0d366b"]    # higher score: darker
LADDER_TARGETS = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 6.0]
PRE_ZLIM = (1.5, 6.0)                                # x range of the panels against the cutoff
CLEAN_C = "#a3a29b"                                  # histograms: no ocular evidence (grey), eye activity before the
PRE_ONLY_C = "#b9b1ee"                               #   pulse only (light violet), a frontal offset 38-50 ms (ink)


def band_rule(lo, hi):
    """Which candidate cutoffs reject the pulses scoring lo < score <= hi."""
    rej = [z for z in CUT_LEVELS if lo >= z - 1e-9]
    kept = [z for z in CUT_LEVELS if hi <= z + 1e-9]
    if not kept:
        return "rejected at every candidate cutoff"
    if not rej:
        return "kept at every candidate cutoff"
    return f"rejected at z {', '.join(f'{z:g}' for z in rej)}, kept at {', '.join(f'{z:g}' for z in kept)}"


def fig_pre_reference(D, A):
    """p1 · The reference before the pulse, per eye IC: its scalp map; |s(t)| over the ICA window, where mean and SD
    are taken (lines: mean + z SD at the candidate cutoffs); |s(t)| of the calibration trials after the pulse with the
    bars the median |s| over 20-60 ms must exceed (and the 20-60 ms window reference's bar at z 2.5)."""
    plt = otr.setup_mpl()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    rows = [(s, ic) for s in SUBJECTS if s in D and D[s]["res"]["sources"] is not None for ic in D[s]["eye_ics"]]
    if not rows:
        return
    H = 1.8 * len(rows) + 1.9
    fig = plt.figure(figsize=(17, H))
    gs = fig.add_gridspec(len(rows), 3, width_ratios=[0.42, 2.45, 1.75], hspace=0.95, wspace=0.1, left=0.02,
                          right=0.99, **otr.page(H, 0.8, 1.2))
    for r, (s, ic) in enumerate(rows):
        R = D[s]
        w = R["w"]
        j = R["eye_ics"].index(ic)
        ci = R["res"]["cal"]
        comp = np.asarray(ci["components"], dtype=float)[:, ic]
        ax = fig.add_subplot(gs[r, 0])
        otr.draw_topo(ax, comp, otr.topo_info(list(ci["ch_names"])), float(np.max(np.abs(comp))))
        ax.set_title(f"{s}\neye IC {ic}", fontsize=8, loc="center")
        st = calibration_meta(R, PRE, PRE_Z_ANCHOR)["eye_stats"][str(ic)]
        mean, sd = st["mean"], st["std"]
        ps, src = R["pre_sources"], R["res"]["sources"]
        tp, tq = 1e3 * np.asarray(ps["times"], dtype=float), 1e3 * np.asarray(src["times"], dtype=float)
        Sp = np.abs(np.asarray(ps["src"], dtype=float)[:, [int(k) for k in ps["eye_ics"]].index(ic)])
        Sq = np.abs(np.asarray(src["src"], dtype=float)[:, [int(k) for k in src["eye_ics"]].index(ic)])
        stw = R["meta"]["calibrations"][wz(PRE_GATE, Z_ANCHOR)]["eye_stats"][str(ic)]
        lvl_w = stw["mean"] + Z_ANCHOR * stw["std"]
        ws = R["meta"]["window_samples"][PRE_GATE]
        bp, bq = np.percentile(Sp, [10, 50, 90], axis=0), np.percentile(Sq, [10, 50, 90], axis=0)
        top = 1.2 * max(mean + max(CUT_LEVELS) * sd, float(np.max(bp[2])), float(np.max(bq[2][tq >= 30])))
        axp = fig.add_subplot(gs[r, 1])
        axq = fig.add_subplot(gs[r, 2], sharey=axp)
        aw = R["res"]["meta"]["artifact_window_1_s"]
        axq.axvspan(1e3 * aw[0], 1e3 * aw[1], color=WASH, lw=0, zorder=0)
        for ax, t, (lo, med, hi) in ((axp, tp, bp), (axq, tq, bq)):
            ax.fill_between(t, lo, hi, color=BAND_C, lw=0, zorder=1)
            ax.plot(t, med, color=INK, lw=0.8 if ax is axp else 1.2, zorder=3)
            ax.set_xlim(t[0], t[-1])
        axp.axhline(mean, color=INK_2, lw=1.0, ls=":", zorder=2)
        for z in CUT_LEVELS:
            axp.axhline(mean + z * sd, color=CUT_C[z], lw=1.3, zorder=4)
            axq.hlines(mean + z * sd, ws["first_ms"], ws["last_ms"], color=CUT_C[z], lw=2.6, zorder=4)
        y_w = min(lvl_w, 0.96 * top)                  # above the axis (the TMS decay in pilot006): at the top edge
        axq.hlines(y_w, ws["first_ms"], ws["last_ms"], color=WINDOW_COLOR[PRE_GATE], lw=1.6, ls="--", zorder=5)
        if lvl_w > 0.96 * top:
            axq.text(ws["last_ms"] + 1.5, y_w, f"↑ {lvl_w:.2f}", color=WINDOW_COLOR[PRE_GATE], fontsize=7.5,
                     va="center")
        axq.axvline(0, color=BASE_C, lw=0.8)
        axp.set_ylim(0, top)
        above = 100 * float(np.mean(Sp > mean + PRE_Z_ANCHOR * sd))
        med_clean = float(np.median(w.loc[w.clean, f"med_abs_s{j + 1}__{PRE_GATE}"]))
        axp.set_title(f"before the pulse: ICA window, 1–100 Hz, {Sp.shape[0]} calibration trials · |s| mean {mean:.2f}, "
                      f"SD {sd:.2f} · {above:.1f} % of its samples above mean + {PRE_Z_ANCHOR:g} SD", fontsize=7.8)
        axq.set_title(f"after the pulse: calibration trials · bars over 20–60 ms: "
                      f"{' / '.join(f'{mean + z * sd:.2f}' for z in CUT_LEVELS)} (z {' / '.join(f'{z:g}' for z in CUT_LEVELS)})"
                      f"\n20–60 ms window reference at z {Z_ANCHOR:g}: {lvl_w:.2f} · clean pulses' median |s| "
                      f"{med_clean:.2f} (score {(med_clean - mean) / sd:+.2f})", fontsize=7.8)
        axp.set_ylabel("|s| (IC units)", fontsize=8)
        axq.tick_params(labelleft=False)
        if r == len(rows) - 1:
            axp.set_xlabel("ms from the pulse")
            axq.set_xlabel("ms from the pulse")
    handles = [Line2D([], [], color=INK, lw=1.2, label="|s(t)|, median over the calibration trials"),
               Patch(color=BAND_C, label="10–90 %"),
               Line2D([], [], color=INK_2, lw=1.0, ls=":", label="mean of |s| over the ICA window"),
               *[Line2D([], [], color=CUT_C[z], lw=2.2, label=f"mean + {z:g} SD (cutoff z {z:g})") for z in CUT_LEVELS],
               Line2D([], [], color=WINDOW_COLOR[PRE_GATE], lw=1.6, ls="--",
                      label=f"20–60 ms window reference: mean + {Z_ANCHOR:g} SD of |s| over 20–60 ms"),
               Patch(color=WASH, label="blank (interpolated)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("The reference before the pulse: mean and SD of each eye IC's |s| over every sample of the ICA "
                 "window (left); a pulse is rejected when, for any eye IC, its median |s| over 20–60 ms after the pulse "
                 "exceeds mean + z SD (right; the same y scale)", y=1 - 0.25 / H, fontsize=10)
    save(fig, "p1_pre_reference.png")


def fig_pre_scores(D, A):
    """p2 · The score under the reference before the pulse, single pulses without a frontal offset in the N45 window
    (top: no evidence grey, eye activity before the pulse only light violet) and with one (bottom, downwards), with
    the candidate cutoffs and what each rejects."""
    plt = otr.setup_mpl()
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    subs = [s for s in SUBJECTS if s in D]
    cols = subs + ["all"]
    bw = 0.25
    bins = np.arange(-1.0, 10.0001, bw)
    x0 = bins[:-1]
    fig, axes = plt.subplots(2, len(cols), figsize=(3.9 * len(cols), 7.0), sharex=True, squeeze=False,
                             layout="constrained", gridspec_kw=dict(height_ratios=[1.15, 1]))
    for j, s in enumerate(cols):
        sg = (pd.concat([D[t]["w"][D[t]["w"].single] for t in subs], ignore_index=True) if s == "all"
              else D[s]["w"][D[s]["w"].single])
        raw = sg[f"score__{PRE}"].to_numpy()
        sco = np.clip(raw, bins[0] + 1e-6, bins[-1] - 1e-6)
        e45 = sg.ev_n45.to_numpy(dtype=bool)
        epre = sg.ev_pre.to_numpy(dtype=bool) & ~e45
        cln = sg.clean.to_numpy(dtype=bool)
        hc, hp, he = (np.histogram(sco[m], bins)[0] for m in (cln, epre, e45))
        ax, axb = axes[0, j], axes[1, j]
        ax.bar(x0, hc, width=bw, align="edge", color=CLEAN_C, lw=0, zorder=2)
        ax.bar(x0, hp, bottom=hc, width=bw, align="edge", color=PRE_ONLY_C, lw=0, zorder=2)
        ax.set_yscale("log")
        ax.set_ylim(0.7, 2.5 * max(1, (hc + hp).max()))
        axb.bar(x0, he, width=bw, align="edge", color=EV_C, lw=0, zorder=2)
        axb.set_ylim(max(5.0, 1.2 * he.max()), 0)
        for z in CUT_LEVELS:
            for a in (ax, axb):
                a.axvline(z, color=CUT_C[z], lw=1.3, ls="--", zorder=3)
        for k, z in enumerate(CUT_LEVELS):
            ax.text(0.98, 0.96 - 0.075 * k, f"z {z:g}: clean {100 * np.mean(raw[cln] > z):.1f} %, offset "
                    f"{100 * np.mean(raw[e45] > z):.0f} %", transform=ax.transAxes, ha="right", va="top",
                    fontsize=7.6, color=CUT_C[z], zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.6))
        tl = "all sessions (pulses pooled)" if s == "all" else s
        ax.set_title(f"{tl}\n{len(sg)} single pulses: {int(cln.sum())} clean, {int(epre.sum())} with eye activity "
                     f"before only", fontsize=8.2)
        axb.text(0.98, 0.04, f"{int(e45.sum())} with a frontal offset 38–50 ms", transform=axb.transAxes,
                 ha="right", va="bottom", fontsize=8, color=INK_2)
        if j == 0:
            ax.set_ylabel("single pulses without an offset\n(log scale)")
            axb.set_ylabel("single pulses with an offset")
        axb.set_xlabel("score, 20–60 ms against the\nreference before the pulse", fontsize=8.5)
        axb.set_xlim(bins[0], bins[-1])
    handles = [Patch(color=CLEAN_C, label="no ocular evidence (clean)"),
               Patch(color=PRE_ONLY_C, label="eye activity before the pulse only"),
               Patch(color=EV_C, label="frontal offset 38–50 ms (Fp mean or F7 − F8, before ICA)"),
               *[Line2D([], [], color=CUT_C[z], lw=1.3, ls="--", label=f"cutoff z {z:g}") for z in CUT_LEVELS]]
    fig.legend(handles=handles, loc="outside lower center", ncol=6, fontsize=8.5)
    fig.suptitle("Scores under the reference before the pulse (bins of 0.25; scores beyond −1 / 10 in the edge bins); "
                 "text: % of the clean and of the offset pulses each cutoff rejects", fontsize=10.5)
    save(fig, "p2_pre_scores.png")


def fig_pre_cutoff(D, A):
    """p3 · Choosing the cutoff for the reference before the pulse: per session (thin) and the mean of the sessions
    (thick) against the cutoff, with the decider now and the 20-60 ms window reference at z 2.5 for comparison."""
    plt = otr.setup_mpl()
    from matplotlib.lines import Line2D
    subs = [s for s in SUBJECTS if s in D]
    dec, mg = A["decisions"], A["marginal"]
    sw = A["pre_sweep"][A["pre_sweep"].design == PRE]
    pooled = dec[dec.subject == "pooled"]
    now = pooled[pooled.window == "now"].iloc[0]
    win = pooled[(pooled.window == PRE_GATE) & ((pooled.z - Z_ANCHOR).abs() < 1e-9)].iloc[0]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.0), layout="constrained")

    def refs(ax, col):
        ax.axhline(now[col], color=HEAD_C, ls="--", lw=1.2, zorder=2)
        ax.axhline(win[col], color=WINDOW_COLOR[PRE_GATE], ls=":", lw=1.6, zorder=2)

    def cuts(ax):
        for z in CUT_LEVELS:
            ax.axvline(z, color=CUT_C[z], lw=1.1, ls="--", zorder=1)

    def sweep_panel(ax, col, ylab, title, ylim=None):
        for k, s in enumerate(subs):
            g = sw[sw.subject == s].sort_values("z")
            m = (g.z >= PRE_ZLIM[0] - 1e-9) & (g.z <= PRE_ZLIM[1] + 1e-9)
            g = g[m]
            mk = np.where((2 * g.z.to_numpy()) % 1 == 0)[0]
            ax.plot(g.z, g[col], color=INK_2, lw=0.8, marker=SESSION_MARKER.get(s, "o"), ms=4, mfc="white",
                    markevery=list(mk), zorder=3)
        g = sw[sw.subject == "pooled"].sort_values("z")
        g = g[(g.z >= PRE_ZLIM[0] - 1e-9) & (g.z <= PRE_ZLIM[1] + 1e-9)]
        ax.plot(g.z, g[col], color=PRE_C, lw=2.4, zorder=4)
        for z in CUT_LEVELS:
            v = g.loc[(g.z - z).abs() < 1e-9, col]
            if len(v):
                ax.plot([z], [float(v.iloc[0])], marker=PRE_MARKER, ms=11, color=PRE_C, ls="none", zorder=5)
                ax.annotate(f"{float(v.iloc[0]):.1f}", (z, float(v.iloc[0])), textcoords="offset points",
                            xytext=(5, 5), fontsize=7.5, color=PRE_C)
        refs(ax, col)
        cuts(ax)
        ax.set_xlim(*PRE_ZLIM)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlabel("cutoff z (reference before the pulse)", fontsize=8)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9.5)

    # (a) how often a pulse in a band of the score shows a frontal offset in the N45 window
    ax = axes[0, 0]
    edges = PRE_MARGINAL_EDGES
    labels = [f"≤ {edges[0]:g}"] + [f"{a:g}–{b:g}" for a, b in zip(edges[:-1], edges[1:])] + [f"> {edges[-1]:g}"]
    xs = np.arange(len(labels))
    g = mg[(mg.subject == "pooled") & (mg.window == PRE)].reset_index(drop=True)
    ax.fill_between(xs, g.pct_n45_ci_lo, g.pct_n45_ci_hi, color=PRE_C, alpha=0.15, lw=0, zorder=1)
    ax.plot(xs, g.pct_n45, color=PRE_C, lw=2.2, marker=PRE_MARKER, ms=9, zorder=4)
    for k, s in enumerate(subs):
        gs_ = mg[(mg.subject == s) & (mg.window == PRE)].reset_index(drop=True)
        ok = gs_.n_single >= 3
        ax.plot((xs + (k - (len(subs) - 1) / 2) * 0.08)[ok], gs_.pct_n45[ok], color=INK_2, lw=0, marker=SESSION_MARKER.get(s, "o"),
                ms=4.5, mfc="white", zorder=3)
    for x, n in zip(xs, g.n_single):
        ax.text(x, 101, f"{int(n)}", ha="center", va="bottom", fontsize=6.5, color=INK_2)
    for z in CUT_LEVELS:
        ax.axvline(edges.index(z) + 0.5, color=CUT_C[z], lw=1.1, ls="--", zorder=1)
    ax.set_xticks(xs, labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylim(0, 108)
    ax.set_xlabel("band of the score (numbers: single pulses, all sessions)", fontsize=8)
    ax.set_ylabel("% with a frontal offset 38–50 ms")
    ax.set_title("How often a pulse in a band of the score\nhas a frontal offset in the N45 window", fontsize=9.5)
    sweep_panel(axes[0, 1], "pct_singles_rejected", "% of single pulses", "Single pulses rejected", (0, 12))
    sweep_panel(axes[0, 2], "pct_clean_rejected", "% of the clean pulses",
                "False alarms: clean pulses rejected\n(no ocular evidence)", (0, 6))
    sweep_panel(axes[0, 3], "precision_n45", "% of the rejected", "Rejected pulses with a frontal\noffset 38–50 ms",
                (0, 105))
    sweep_panel(axes[1, 0], "hit_n45", "% of those pulses", "Pulses with a frontal offset 38–50 ms\nthat are rejected",
                (0, 100))
    # (f) calibration trials left for PRIME, (g) the outcome: from the calibrations and passes at PRE_Z_GRID
    d = dec[dec.window == PRE]
    for ax, col, ylab, title in ((axes[1, 1], "calibration_trials_for_prime", "calibration trials (of 125)",
                                  "Calibration trials left for PRIME\n(dashed: fewest of the sessions)"),
                                 (axes[1, 2], "evoked_change_mean_fixed_pct", "% change from baseline",
                                  "Evoked N45 change, mean of t0–t60\n(fixed processing; hollow: own decider)")):
        for s in subs:
            g = d[d.subject == s].sort_values("z")
            ax.plot(g.z, g[col], color=INK_2, lw=0.8, marker=SESSION_MARKER.get(s, "o"), ms=4, mfc="white", zorder=3)
        g = d[d.subject == "pooled"].sort_values("z")
        ax.plot(g.z, g[col], color=PRE_C, lw=2.4, marker=PRE_MARKER, ms=9, zorder=4)
        if col == "calibration_trials_for_prime":
            ax.plot(g.z, g.calibration_trials_for_prime_min, color=PRE_C, lw=1.4, ls="--", zorder=4)
            ax.set_ylim(95, 126)
        else:
            a = g[(g.z - PRE_Z_ANCHOR).abs() < 1e-9]
            ax.plot([PRE_Z_ANCHOR + 0.08], a.evoked_change_mean_own_pct, marker=PRE_MARKER, ms=10, mfc="white",
                    mec=PRE_C, mew=1.4, ls="none", zorder=5)
            ax.axhline(0, color=BASE_C, lw=0.9)
        refs(ax, col)
        cuts(ax)
        ax.set_xlim(PRE_Z_GRID[0] - 0.15, PRE_Z_GRID[-1] + 0.15)
        ax.set_xticks(PRE_Z_GRID, [f"{z:g}" for z in PRE_Z_GRID])
        ax.set_xlabel("cutoff z (reference before the pulse)", fontsize=8)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9.5)
    # (h) the same cost: catch against cost for the two references on 20-60 ms and the decider now
    ax = axes[1, 3]
    xmax = 10.0
    ax.plot([0, xmax], [0, xmax], ls=":", color=INK_MUTED, lw=1.0, zorder=1)
    for W, marks in ((PRE, CUT_LEVELS), (PRE_GATE, (2.0, 2.5, 3.0))):
        color, marker, ls = design_style(W)
        xy = [cost_curve(D[t]["w"], W, "n45") for t in subs]
        x, y = np.mean([a for a, _ in xy], axis=0), np.mean([b for _, b in xy], axis=0)
        m = x <= xmax + 1
        ax.plot(x[m], y[m], color=color, lw=1.8, ls=ls, zorder=3)
        for z in marks:
            ii = int(np.argmin(np.abs(SWEEP - z)))
            ax.plot([x[ii]], [y[ii]], marker=marker, ms=9 if W == PRE else 6.5, color=color, mfc="white", mew=1.3,
                    ls="none", zorder=4)
            ax.annotate(f"{z:g}", (x[ii], y[ii]), textcoords="offset points", fontsize=7.5, color=color,
                        xytext=(-6, 7) if W == PRE else (7, -11), ha="right" if W == PRE else "left")
    ax.plot([now.pct_singles_rejected], [now.hit_n45], marker="*", ms=12, color=HEAD_C, mfc="white", mew=1.3,
            ls="none", zorder=5)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 100)
    ax.set_xlabel("% of single pulses rejected (mean of the sessions)", fontsize=8)
    ax.set_ylabel("% of the offset pulses rejected")
    ax.set_title("At the same cost: the two references on\n20–60 ms (markers: cutoffs) and the decider now",
                 fontsize=9.5)
    handles = ([Line2D([], [], color=INK_2, lw=0.8, marker=SESSION_MARKER.get(s, "o"), ms=4.5, mfc="white", label=s)
                for s in subs]
               + [Line2D([], [], color=PRE_C, lw=2.4, marker=PRE_MARKER, ms=9,
                         label="mean of the sessions (band: 95 % CI, all pulses)"),
                  Line2D([], [], color=HEAD_C, ls="--", lw=1.2, marker="*", ms=10, mfc="white",
                         label="now: 15–100 ms, reference the whole epoch, z 3"),
                  Line2D([], [], color=WINDOW_COLOR[PRE_GATE], ls=":", lw=1.6,
                         label=f"20–60 ms, window reference, z {Z_ANCHOR:g}"),
                  *[Line2D([], [], color=CUT_C[z], lw=1.1, ls="--", label=f"cutoff z {z:g}") for z in CUT_LEVELS]])
    fig.legend(handles=handles, loc="outside lower center", ncol=6, fontsize=8.5)
    fig.suptitle("The reference before the pulse (ICA window) on 20–60 ms: what each cutoff costs and catches "
                 "(thick: mean of pilot003–006; thin: sessions)", fontsize=11)
    save(fig, "p3_pre_cutoff.png")


def fig_pre_maps(D, A, subs, tag):
    """p4 · Scalp maps in the N45 window (38-50 ms) by band of the score under the reference before the pulse: the
    epoch before ICA and after cleaning minus the typical pulses (no ocular evidence, score <= TYPICAL_MAX), and what
    the eye ICs remove; right, the frontal channels before ICA. tag: a session, or 'all' (bands pooled over the
    sessions, each pulse minus its own session's typical pulses)."""
    plt = otr.setup_mpl()
    from matplotlib.patches import Patch
    names = [list(D[s]["pre_meta"]["ch_names"]) for s in subs]
    if any(n != names[0] for n in names):
        print(f"  p4 {tag}: the sessions have other channels; skipped")
        return
    info = otr.topo_info(names[0])
    edges = PRE_BANDS + [np.inf]
    nb = len(edges) - 1
    keys = ("pre", "rem", "fin", "veog", "heog")
    acc = {k: [[] for _ in range(nb + 1)] for k in keys}      # index 0: the typical pulses, 1..nb: the bands
    n_off = np.zeros(nb + 1, int)
    t_ep = None
    for s in subs:
        R = D[s]
        w = R["w"]
        single = w.single.to_numpy(dtype=bool)
        sco = w[f"score__{PRE}"].to_numpy()
        mp = n45_maps(R)
        tr = {k: otr.load_arrays(R["res"], k, list(w.stem)).astype(float) for k in ("veog", "heog")}
        t_ep = np.asarray(R["res"]["meta"]["times_epoch_ms"])
        typ = typical_mask(R)
        base = {k: typical_by_block(R, mp[k]) for k in ("pre", "fin")}
        masks = [typ] + [single & (sco > a) & (sco <= b) for a, b in zip(edges[:-1], edges[1:])]
        for b, m in enumerate(masks):
            for k in ("pre", "fin"):
                acc[k][b].append(mp[k][m] if b == 0 else mp[k][m] - base[k][m])
            acc["rem"][b].append(mp["rem"][m])
            for k in ("veog", "heog"):
                acc[k][b].append(tr[k][m])
            n_off[b] += int((m & w.ev_n45.to_numpy(dtype=bool)).sum())
    agg = [{k: np.concatenate(acc[k][b]) for k in keys} for b in range(nb + 1)]
    n = [len(a["pre"]) for a in agg]
    mean = [{k: a[k].mean(axis=0) if len(a[k]) else None for k in keys} for a in agg]
    cols = [("pre", "before ICA\nminus typical"), ("rem", "removed by\nthe eye ICs"), ("fin", "after cleaning\nminus typical")]
    typ_title = {"pre": "before ICA", "rem": "removed by the eye ICs", "fin": "after cleaning"}
    vmax = {}
    for k, _ in cols:                                 # one scale per column, set by the bands up to the highest
        vals = [np.abs(mean[b][k]) for b in range(1, nb + 1)      # candidate cutoff (the higher bands exceed it)
                if mean[b][k] is not None and edges[b] <= max(CUT_LEVELS) + 1e-9]
        vmax[k] = float(np.percentile(np.concatenate(vals), 99)) if vals else 1.0
    H = 1.5 * (nb + 1) + 2.0
    fig = plt.figure(figsize=(17, H))
    gs = fig.add_gridspec(nb + 2, 6, width_ratios=[1.45, 1, 1, 1, 0.22, 3.4], height_ratios=[1] * (nb + 1) + [0.12],
                          hspace=0.45, wspace=0.08, left=0.01, right=0.99, **otr.page(H, 1.05, 0.55))
    ims = {}
    for b in range(nb + 1):
        axl = fig.add_subplot(gs[b, 0])
        axl.axis("off")
        if b == 0:
            txt = (f"typical pulses\nno ocular evidence,\nscore ≤ {TYPICAL_MAX:g}; n = {n[0]}\n(maps: own scale, "
                   f"not minus typical)")
            color = INK_MUTED
        else:
            lo, hi = edges[b - 1], edges[b]
            lab = f"{lo:g} < score ≤ {hi:g}" if np.isfinite(hi) else f"score > {lo:g}"
            pct = 100 * n_off[b] / n[b] if n[b] else np.nan
            txt = f"{lab}\nn = {n[b]}, {pct:.0f} % with an offset\n{band_rule(lo, hi)}"
            color = PRE_BAND_C[b - 1]
        axl.text(0.9, 0.5, txt, ha="right", va="center", fontsize=7.6, transform=axl.transAxes)
        axl.plot([0.96], [0.5], marker="s", ms=9, color=color, transform=axl.transAxes, clip_on=False)
        for c, (k, title) in enumerate(cols):
            ax = fig.add_subplot(gs[b, c + 1])
            m = mean[b][k]
            if m is None:
                ax.axis("off")
                continue
            if b == 0:
                v = float(np.percentile(np.abs(m), 99)) or 1.0
                otr.draw_topo(ax, m, info, v)
                ax.set_title(f"{typ_title[k]} (±{v:.1f} µV)", fontsize=7.2, loc="center")
            else:
                ims[k] = otr.draw_topo(ax, m, info, vmax[k])
                if b == 1:
                    ax.set_title(title, fontsize=8, loc="center")
    for c, (k, _) in enumerate(cols):
        if k in ims:
            cb = fig.colorbar(ims[k], cax=fig.add_subplot(gs[nb + 1, c + 1]), orientation="horizontal")
            cb.ax.tick_params(labelsize=7)
            cb.set_label("µV, 38–50 ms (bands)", fontsize=7)
    inner = gs[:nb + 1, 5].subgridspec(2, 1, hspace=0.32)
    for r, (k, title) in enumerate((("veog", "Fp1/Fpz/Fp2 mean before ICA (VEOG), mean of the pulses"),
                                    ("heog", "F7 − F8 before ICA (HEOG), mean of the pulses"))):
        ax = fig.add_subplot(inner[r])
        ax.axvspan(*(1e3 * np.asarray(WINDOWS[PRE_GATE])), color=WASH, lw=0, zorder=0)
        for x in (1e3 * N45_S[0], 1e3 * N45_S[1]):
            ax.axvline(x, color=INK_MUTED, lw=0.8, ls=":", zorder=1)
        for b in range(nb + 1):
            if mean[b][k] is not None:
                ax.plot(t_ep, mean[b][k], color=INK_MUTED if b == 0 else PRE_BAND_C[b - 1], lw=1.6 if b else 1.2,
                        zorder=3)
        ax.axvline(0, color=BASE_C, lw=0.8)
        ax.set_xlim(t_ep[0], t_ep[-1])
        ok = t_ep >= 1e3 * WINDOWS[PRE_GATE][0]
        vals = np.concatenate([mean[b][k][ok] for b in range(nb + 1) if mean[b][k] is not None])
        pad = 0.08 * (vals.max() - vals.min())
        ax.set_ylim(vals.min() - pad, vals.max() + pad)
        ax.set_ylabel("µV")
        ax.set_title(title + " (y range from 20 ms on)", fontsize=8.5)
        if r == 1:
            ax.set_xlabel("ms from the pulse")
    fig.legend(handles=[Patch(color=WASH, label="20–60 ms (the gate's samples); dotted: 38–50 ms (N45)")],
               loc="lower right", fontsize=8, bbox_to_anchor=(0.99, 0.0))
    who = "all sessions (pulses pooled)" if tag == "all" else tag
    fig.suptitle(f"{who} · single pulses by band of the score under the reference before the pulse · N45-window maps "
                 f"(38–50 ms), each pulse minus the typical pulses of its block; one colour scale per column, set by the "
                 f"bands up to {max(CUT_LEVELS):g} (higher bands exceed it)", y=1 - 0.25 / H, fontsize=9.5)
    save(fig, f"p4_pre_maps_{tag}.png")


def ladder_picks(sco, single):
    """Single pulses nearest to LADDER_TARGETS (each at most once), in the order of the score."""
    idx = np.where(single)[0]
    picks = []
    for t in LADDER_TARGETS:
        tol = 0.12 if t < 4 else 0.5
        for i in idx[np.argsort(np.abs(sco[idx] - t))][:6]:
            if i not in picks and abs(sco[i] - t) <= tol:
                picks.append(int(i))
                break
    return sorted(picks, key=lambda i: sco[i])


def fig_pre_ladder(R, A):
    """p5 · Single pulses ordered by the score under the reference before the pulse, around the candidate cutoffs:
    frontal channels before ICA (top) and the N45-window maps before ICA and after cleaning, minus the typical
    pulses."""
    plt = otr.setup_mpl()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    w, s = R["w"], R["subject"]
    single = w.single.to_numpy(dtype=bool)
    sco = w[f"score__{PRE}"].to_numpy()
    picks = ladder_picks(sco, single)
    if not picks:
        return
    mp = n45_maps(R)
    maps = {k: mp[k][picks] - typical_by_block(R, mp[k])[picks] for k in ("pre", "fin")}
    stems = list(w.stem.iloc[picks])
    tv = otr.load_arrays(R["res"], "veog", stems).astype(float)
    th = otr.load_arrays(R["res"], "heog", stems).astype(float)
    t_ep = np.asarray(R["res"]["meta"]["times_epoch_ms"])
    info = otr.topo_info(list(R["pre_meta"]["ch_names"]))
    n = len(picks)
    H = 6.4
    fig = plt.figure(figsize=(1.3 * n + 2.6, H))
    gs = fig.add_gridspec(3, n + 2, width_ratios=[1.0] + [1] * n + [0.13], height_ratios=[1.0, 1.15, 1.15],
                          hspace=0.3, wspace=0.12, left=0.01, right=0.96, **otr.page(H, 1.1, 0.75))
    ok = t_ep >= 15
    yl = np.percentile(np.concatenate([tv[:, ok].ravel(), th[:, ok].ravel()]), [0.5, 99.5])
    yl = (min(yl[0], -10.0), max(yl[1], 10.0))
    vmax = {k: float(np.percentile(np.abs(maps[k]), 98)) or 1.0 for k in maps}
    axes_top, ims = [], {}
    for c, i in enumerate(picks):
        rr = w.iloc[i]
        ax = fig.add_subplot(gs[0, c + 1])
        ax.axvspan(*(1e3 * np.asarray(WINDOWS[PRE_GATE])), color=WASH, lw=0, zorder=0)
        ax.plot(t_ep, th[c], color=HEOG_C, lw=0.9, zorder=2)
        ax.plot(t_ep, tv[c], color=INK, lw=1.0, zorder=3)
        ax.axvline(0, color=BASE_C, lw=0.7)
        ax.set_xlim(t_ep[0], t_ep[-1])
        ax.set_ylim(*yl)
        ax.tick_params(labelsize=6, labelleft=c == 0)
        sign = (["offset"] if bool(rr["ev_n45"]) else []) + (["before"] if bool(rr["ev_pre"]) else [])
        ax.set_title(f"{otr.label_trial(rr)} · {otr.CAT_SHORT[rr['category']]}\n{sco[i]:.2f} (window "
                     f"{rr[f'score__{PRE_GATE}']:.2f})\n{', '.join(sign) if sign else '—'}", fontsize=6.8, loc="center")
        axes_top.append(ax)
        for r, k in ((1, "pre"), (2, "fin")):
            axm = fig.add_subplot(gs[r, c + 1])
            ims[k] = otr.draw_topo(axm, maps[k][c], info, vmax[k])
    for r, (k, lab) in enumerate((("", "frontal channels\nbefore ICA (µV)"), ("pre", "38–50 ms before ICA\nminus typical"),
                                  ("fin", "38–50 ms after cleaning\nminus typical"))):
        axl = fig.add_subplot(gs[r, 0])
        axl.axis("off")
        axl.text(0.95, 0.5, lab, ha="right", va="center", fontsize=8, transform=axl.transAxes)
        if k:
            cb = fig.colorbar(ims[k], cax=fig.add_subplot(gs[r, n + 1]))
            cb.ax.tick_params(labelsize=6.5)
            cb.set_label("µV", fontsize=7)
    fig.canvas.draw()
    boxes = [a.get_position() for a in axes_top]
    for z in CUT_LEVELS:                              # the cutoffs between the columns they separate
        gaps = [c for c in range(n - 1) if sco[picks[c]] <= z < sco[picks[c + 1]]]
        if not gaps:
            continue
        c = gaps[0]
        x = 0.5 * (boxes[c].x1 + boxes[c + 1].x0)
        fig.add_artist(Line2D([x, x], [0.75 / H * 0.6, boxes[c].y1 + 0.02], color=CUT_C[z], lw=1.6, ls="--",
                              transform=fig.transFigure))
        fig.text(x, boxes[c].y1 + 0.025, f"z {z:g}", color=CUT_C[z], ha="center", va="bottom", fontsize=8.5,
                 fontweight="bold")
    handles = [Line2D([], [], color=INK, lw=1.0, label="Fp1/Fpz/Fp2 mean"), Line2D([], [], color=HEOG_C, lw=1.0, label="F7 − F8"),
               Patch(color=WASH, label="20–60 ms (the gate's samples)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{s} · single pulses ordered by the score under the reference before the pulse (titles: pulse, score, "
                 f"the 20–60 ms window reference's score, evidence) · maps minus the typical pulses of the same block "
                 f"(no evidence, score ≤ {TYPICAL_MAX:g})", y=1 - 0.2 / H, fontsize=9.5)
    save(fig, f"p5_pre_ladder_{s}.png")


def fig_pre_vs_window(D, A):
    """p6 · Single pulses: the score under the reference before the pulse against the 20-60 ms window reference's
    score (the same median |s| per eye IC, scaled by two references)."""
    plt = otr.setup_mpl()
    from matplotlib.lines import Line2D
    subs = [s for s in SUBJECTS if s in D]
    lo, hi = -1.5, 10.0
    fig, axes = plt.subplots(1, len(subs), figsize=(4.1 * len(subs), 4.6), squeeze=False, layout="constrained")
    for j, s in enumerate(subs):
        sg = D[s]["w"][D[s]["w"].single]
        xr, yr = sg[f"score__{PRE_GATE}"].to_numpy(), sg[f"score__{PRE}"].to_numpy()
        x, y = np.clip(xr, lo, hi), np.clip(yr, lo, hi)
        e = sg.ev_n45.to_numpy(dtype=bool)
        ax = axes[0, j]
        ax.plot([lo, hi], [lo, hi], color=GRID_C, lw=1.0, zorder=1)
        ax.scatter(x[~e], y[~e], s=8, color=NOEV_C, lw=0, zorder=2)
        ax.scatter(x[e], y[e], s=12, color=EV_C, lw=0, zorder=3)
        ax.axvline(Z_ANCHOR, color=WINDOW_COLOR[PRE_GATE], lw=1.2, ls="--", zorder=2)
        for z in CUT_LEVELS:
            ax.axhline(z, color=CUT_C[z], lw=1.1, ls="--", zorder=2)
        rho = pd.Series(xr).corr(pd.Series(yr), method="spearman")
        a, b = xr > Z_ANCHOR, yr > PRE_Z_ANCHOR
        ax.text(0.03, 0.97, f"Spearman {rho:.2f}\nwindow {Z_ANCHOR:g} only: {int((a & ~b).sum())} "
                f"({int((a & ~b & e).sum())} offset)\nbefore {PRE_Z_ANCHOR:g} only: {int((b & ~a).sum())} "
                f"({int((b & ~a & e).sum())} offset)\nboth: {int((a & b).sum())} ({int((a & b & e).sum())} offset)",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.4, color=INK_2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(s, fontsize=9)
        ax.set_xlabel("score, 20–60 ms window reference")
        if j == 0:
            ax.set_ylabel("score, reference before the pulse")
    handles = [Line2D([], [], color=EV_C, marker="o", ms=4.5, ls="none", label="frontal offset 38–50 ms"),
               Line2D([], [], color=NOEV_C, marker="o", ms=4.5, ls="none", label="no offset"),
               Line2D([], [], color=WINDOW_COLOR[PRE_GATE], ls="--", lw=1.2, label=f"window reference, z {Z_ANCHOR:g}"),
               *[Line2D([], [], color=CUT_C[z], ls="--", lw=1.1, label=f"before the pulse, z {z:g}") for z in CUT_LEVELS]]
    fig.legend(handles=handles, loc="outside lower center", ncol=6, fontsize=8.5)
    fig.suptitle("The same 20–60 ms samples scored against two references: each eye IC's median |s| maps linearly "
                 "from one score to the other (one line per deciding IC); beyond −1.5 / 10 at the edge", fontsize=10.5)
    save(fig, "p6_pre_vs_window.png")


FIGURES = ["reference", "summary", "sessions", "cost", "scores", "cards", "outcome",
           "pre_reference", "pre_scores", "pre_cutoff", "pre_maps", "pre_ladder", "pre_vs_window"]


def make_figures(D, A, which=None):
    which = set(which or FIGURES)
    for name, fn in (("reference", fig_reference), ("summary", fig_summary), ("sessions", fig_sessions),
                     ("cost", fig_cost_curves), ("scores", fig_scores), ("outcome", fig_outcome),
                     ("pre_reference", fig_pre_reference), ("pre_scores", fig_pre_scores),
                     ("pre_cutoff", fig_pre_cutoff), ("pre_vs_window", fig_pre_vs_window)):
        if name in which:
            fn(D, A)
    if "cards" in which:
        for R in D.values():
            fig_cards(R, A)
    if "pre_maps" in which:
        for s in D:
            fig_pre_maps(D, A, [s], s)
        fig_pre_maps(D, A, list(D), "all")
    if "pre_ladder" in which:
        for R in D.values():
            fig_pre_ladder(R, A)


# %% 9 · Main ------------------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract", action="store_true", help="child mode: extract one session")
    ap.add_argument("--part", choices=["windows", "pre"], default="windows",
                    help="child: the window references or the pre-stimulus (ICA window) reference")
    ap.add_argument("--subject")
    ap.add_argument("--budget", type=float, default=None, help="child: stop after about this many seconds (resumable)")
    ap.add_argument("--plots-only", action="store_true", help="tables and figures from what is extracted")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--figures", nargs="*", default=None, help=f"subset of {FIGURES}")
    ap.add_argument("--no-tables", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args, _ = ap.parse_known_args(argv)
    if args.extract:
        if not args.subject:
            ap.error("--extract needs --subject")
        fn = extract_pre if args.part == "pre" else extract_windows
        sys.exit(0 if fn(args.subject, budget=args.budget) else 3)
    subjects = args.subjects or SUBJECTS
    if not args.plots_only:
        for s in subjects:
            if not otr.is_complete("variants", s):       # the first review's replay comes first
                otr.run_child("variants", s)
            if not is_complete(s):
                run_child(s, part="windows")
            if not is_complete_pre(s):
                run_child(s, part="pre")
    D = load_all(subjects)
    if not D:
        print("nothing extracted yet")
        return
    A = analyze(D)
    if not args.no_tables:
        write_tables(D, A)
    if not args.no_figures:
        make_figures(D, A, args.figures)


if __name__ == "__main__":
    main()
