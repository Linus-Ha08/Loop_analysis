#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ocular_window_review.py - the ocular gate on shorter windows: which window (15-100, 30-100, 30-60, 20-60, 20-70 ms)
and which cutoff (z = 2.0, 2.25, 2.5, 2.75, 3.0)?

Follow-up of ocular_threshold_review.py (same folder). It imports that script's machinery (git export, the override of
_detect_ocular_trials, the replay recorder, the N45 fit) and builds on its results in Loop/offline_ocular_window/replay
(the decider as it runs, 'current', and the 15-100 ms window): run ocular_threshold_review.py first.

The window of the gate is two things here:
  - the samples the score is the median of (the decider's trial_reject_opts.ocular.post_timerange, now [15 ms, end]),
  - the samples the reference distribution of |s| (mean, SD) is taken from (detect_ocular_trials_ref, reference "gate").
A pulse is rejected when, for any eye IC, the median |s| over the window exceeds mean + z x SD of the calibration
block's |s| over the same window.

Why other windows: in pilot006 an eye IC carries the decay of the TMS artifact right after the blank (median |s| 8.8 at
15-30 ms, 0.8 later). Over 15-100 ms it triples that IC's reference SD (4.46 vs 1.55 over 30-100 ms), so the IC can
hardly reject a pulse. Windows starting at 20 or 30 ms leave (part of) the transient out; 30-60 and 20-60 ms are the
neighbourhood of the N45 (dipole window 38-50 ms).

How it runs (nothing in prime/ is written)
------------------------------------------
Child process per session, importing the same decider export as the first review (HEAD, 007b932):
  1. Calibrations: for every window W and cutoff z in Z_GRID, calibrate() on the session's 125 calibration trials with
     post_timerange = W, reference = W and z_threshold = z - the ocular drops, the trials left for PRIME, SOUND /
     SSP-SIR, the post-MAD statistics and the dipole fit, each as the decider computes them (calibrations/*.npy).
  2. Pulses: for every window except 15-100 ms one full variant at z = Z_ANCHOR (2.5): every pulse through that
     calibration's preprocess_post (decision, cleaned epoch, N45, post-MAD z) with the recorder on its ICA calls.
     15-100 ms at 2.5 is the first review's window_z2.5 variant, and 'current' is the decider as it runs.
     A window added to WINDOWS later gets its calibrations and its own pass over the pulses; its rows and arrays are
     appended to the block files (checked: the same epoch, frontal measures and scalp maps as the earlier pass).
A window's score (max over the eye ICs of the median z over W) depends on the window's reference, not on the cutoff,
so the ocular decision at any cutoff is score > z. Exact: checked against preprocess_post's own decisions at z 2.5 for
every window, and for 15-100 ms at z 2, 3 and 3.5 (the first review's variants).

Analysis: what each window and cutoff costs, and how well it catches ocular pulses
----------------------------------------------------------------------------------
- Decisions: ocular(W, z) = score_W > z; kept(W, z) = not ocular and the post-MAD gates of 'current' passed (the same
  MAD gate everywhere, so only the ocular gate differs). Calibration numbers come from the (W, z) calibrations.
  'now' is the decider as it runs (15-100 ms, reference the whole epoch, z 3).
- Ocular evidence, the same for every window and independent of the gate (no ICA): Fp1/Fpz/Fp2 mean (VEOG) or F7 - F8
  (HEOG) peak-to-peak above median + EVIDENCE_K x robust SD of the session's single pulses, over 30-60 ms ('n45': a
  frontal deflection around the N45), 30-100 ms ('late') and in the raw pre window -210 to -10 ms ('pre': eye
  activity before the pulse). Spans starting before 30 ms are not used: the decay of the TMS artifact dominates their
  frontal peak-to-peak in some sessions.
- Per window and cutoff: % rejected (singles, triplets, intervention singles), hit rate (% of the pulses with evidence
  that are rejected), precision (% of the rejected with evidence), kept pulses with strong evidence, calibration
  trials for PRIME, fewest pulses kept in an open-loop block, and the outcome: evoked N45 change from baseline to
  t0-t60 with one fixed processing ('current') and each (W, z)'s kept pulses; at z 2.5 also each window's own decider
  (its own gates, cleaning and dipole), with bootstrap CIs.
- Equal cost: each window's cutoff set so that it rejects the same q % of single pulses; precision per evidence, mean
  of the sessions with bootstrap CIs, and each window minus 15-100 ms on the same draws. Cost curves over all cutoffs.
- The pulses whose decision at z 2.5 changes with the window, with their evidence (table and example cards).

Run
---
    python ocular_window_review.py                    # extract what is missing (about 9 min per session: 25
                                                      # calibrations, a pass over the 1425 pulses per new window),
                                                      # then tables and figures (about 1.5 min)
    python ocular_window_review.py --plots-only
    python ocular_window_review.py --plots-only --figures summary cards
    python ocular_window_review.py --extract --subject pilot006 [--budget 150]     # one child run, resumable

Outputs (Loop/offline_ocular_window/windows/)
    sub-<ID>/calibrations/<W>_z<z>.npy   bundle, dipole fitting info and override record of every (W, z) calibration
    sub-<ID>/scores.csv                  one row per pulse x new window (z 2.5 variant): score per eye IC, decision,
                                         N45, post-MAD z; frontal measures over every window
    sub-<ID>/blocks/<block>.npz          each new window's dipole-window epochs; scalp maps over every window
    sub-<ID>/meta.json, fitting.npz
    tables/      decisions (per window and cutoff, and 'now'; 'pooled' = mean of the sessions), matched and
                 matched_pooled (equal cost, with CIs), disagreement (z 2.5), marginal (evidence per band of the score),
                 evoked (per block), calibration, reference (eye-IC statistics per window), evidence, checks,
                 pulses_<ID> (one row per pulse, every window)
    figures/     w1 reference per window · w2 summary · w3 per session · w4 cost curves · w5 scores against
                 15-100 ms · w6 cards per session · w7 outcome
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

# ocular evidence, the same for every window and independent of the gate: Fp1/Fpz/Fp2 mean (VEOG) or F7 - F8 (HEOG)
# peak-to-peak above median + EVIDENCE_K x robust SD of the session's single pulses (STRONG_K: strong evidence)
EVIDENCE_K = 3.0
STRONG_K = 6.0
EVIDENCE = {"n45": ("30-60", "frontal deflection 30–60 ms"),           # around the N45, after the TMS decay
            "late": ("30-100", "frontal deflection 30–100 ms"),
            "pre": (None, "eye activity before the pulse")}             # raw pre window, -210 to -10 ms
# (15-100 and 20-70 ms are not used as evidence: the decay of the TMS artifact before 30 ms dominates their frontal
#  peak-to-peak in some sessions, e.g. F7 - F8 median 185 µV over 15-100 ms vs 31 µV over 30-100 ms in pilot006)
MATCHED_PCT = [2, 3, 4, 5, 6, 8]                    # equal-cost comparison: each window's cutoff set to reject q %
MARGINAL_EDGES = [1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5]
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


def sub_dir(subject):
    return OUT / f"sub-{subject}"


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


def run_child(subject, budget=None):
    cmd = [sys.executable, str(SCRIPT), "--extract", "--subject", subject]
    if budget:
        cmd += ["--budget", str(budget)]
    for _ in range(50):
        r = subprocess.run(cmd)
        if r.returncode == 0:
            return True
        if r.returncode != 3:
            raise RuntimeError(f"extraction failed for windows / {subject} (exit code {r.returncode})")
    raise RuntimeError(f"extraction of windows / {subject} did not finish")


def is_complete(subject):
    d = sub_dir(subject)
    f = d / "meta.json"
    if not (f.is_file() and (d / "scores.csv").is_file()):
        return False
    m = json.loads(f.read_text(encoding="utf-8"))
    return bool(m.get("complete", False)) and set(WINDOWS) <= set(m.get("windows", {})) and \
        all(abs(z - z1) < 1e-9 for z, z1 in zip(Z_GRID, m.get("z_grid", []))) and len(m.get("z_grid", [])) == len(Z_GRID)


# %% 2 · Load the extracted results --------------------------------------------------------------------------------

def wl(window):
    return window.replace("-", "–") + " ms"


def load_session(s):
    """One row per pulse (the first review's wide table) plus every window's score, decision at z 2.5, N45 and the
    frontal measures; the ocular evidence flags; the calibrations (meta) and dipole fits of every (window, cutoff)."""
    res = otr.load_subject(s)
    d = sub_dir(s)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    sc = pd.read_csv(d / "scores.csv", keep_default_na=False, na_values=[""], low_memory=False)
    sc["reason"] = sc["reason"].fillna("").astype(str)
    eye = [int(i) for i in meta["eye_ics"]]
    if eye != [int(i) for i in res["meta"]["eye_ics"]]:
        raise RuntimeError(f"{s}: other eye ICs than in the first review")
    w = res["w"].set_index("stem")
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
    for W in WINDOWS:
        w[f"kept_own__{W}"] = w[f"reason__{W}"].eq("")
    w = w.reset_index()
    single = w.single.to_numpy(dtype=bool)
    lines = {}
    for k, (span, _) in EVIDENCE.items():
        v, h = (w[f"veog_ptp__{span}"], w[f"heog_ptp__{span}"]) if span else (w.veog_pre_ptp, w.heog_pre_ptp)
        L = {}
        for tag, kk in (("", EVIDENCE_K), ("_strong", STRONG_K)):
            lv = float(v[single].median() + kk * otr.robust_sd(v[single]))
            lh = float(h[single].median() + kk * otr.robust_sd(h[single]))
            w[f"ev_{k}{tag}"] = ((v > lv) | (h > lh)).to_numpy()
            L[f"veog_line{tag}_uV"], L[f"heog_line{tag}_uV"] = lv, lh
        L["pct_singles"] = 100 * float(w.loc[single, f"ev_{k}"].mean())
        lines[k] = L
    S = w.loc[single, [f"score__{W}" for W in WINDOWS]]
    w["calm_all"] = False                                # single pulses at or below the median of every window
    w.loc[single, "calm_all"] = (S <= S.median()).all(axis=1).to_numpy()
    for k in EVIDENCE:
        lines[k]["pct_calm_pulses"] = 100 * float(w.loc[w.calm_all, f"ev_{k}"].mean())
    return dict(subject=s, res=res, meta=meta, w=w, dir=d, eye_ics=eye, evidence=lines,
                fit=dict(np.load(d / "fitting.npz")), arr=dict(dir=d, w=w))


def load_all(subjects):
    D = {}
    for s in subjects:
        if otr.is_complete("variants", s) and is_complete(s):
            D[s] = load_session(s)
        else:
            print(f"not extracted yet: {s}")
    return D


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

def analyze(D):
    T = {k: [] for k in ("checks", "evidence", "reference", "decisions", "matched", "marginal", "disagreement", "evoked",
                         "calibration")}
    draws = {}                                           # (window, z, processing) -> per session: mean-change draws
    mboot = {}                                           # (window, q, evidence) -> per session: equal-cost precision draws
    for s, R in D.items():
        w, meta, res = R["w"], R["meta"], R["res"]
        single = w.single.to_numpy(dtype=bool)
        rng = np.random.default_rng(1)
        # ---- checks: extraction (this review, the first review) and analysis
        for part, m in (("windows", meta), ("first review: replay", res["meta"]), ("first review: online", res["online_meta"])):
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
        for W in NEW_WINDOWS:
            a = (w[f"score__{W}"] > Z_ANCHOR).to_numpy() == w[f"reason__{W}"].eq("ocular").to_numpy()
            check(f"{W} ms: score > {Z_ANCHOR:g} == preprocess_post's own ocular decision", a.all(),
                  f"{int(a.sum())}/{len(a)} agree")
            arrays, kind, (L, ir) = proc_spec(R, W)
            rd = np.abs(otr.n45_fit(otr.load_arrays(arrays, kind), L, ir) / w[f"amp__{W}"].to_numpy() - 1)
            check(f"{W} ms: n45_fit() == the decider's fit_trial (float32 epochs), max rel. diff",
                  np.nanmax(rd) < 1e-4, f"{np.nanmax(rd):.1e}")
        dd = max(float(np.max(np.abs(w[f"veog_ptp__{BASE_WINDOW}"] - w.veog_gate_ptp))),
                 float(np.max(np.abs(w[f"heog_ptp__{BASE_WINDOW}"] - w.heog_gate_ptp))))
        check("frontal peak-to-peak over 15-100 ms == the first review's (the same epoch before ICA)", dd < 1e-6,
              f"max |diff| {dd:.1e} µV")
        # ---- the evidence
        for k, L in R["evidence"].items():
            T["evidence"].append(dict(subject=s, evidence=k, label=EVIDENCE[k][1], span_ms=EVIDENCE[k][0] or "-210 to -10",
                                      **L))
        # ---- reference statistics per window and eye IC, and the eye ICs' |s| over time in the calibration block
        prof = {}
        src = res["sources"]
        if src is not None:
            t = 1e3 * np.asarray(src["times"], dtype=float)
            Sabs = np.abs(np.asarray(src["src"], dtype=float))
            for j, ic in enumerate(int(i) for i in src["eye_ics"]):
                prof[ic] = {f"median_abs_s_{a}_{b}ms": float(np.median(Sabs[:, j][:, (t >= a) & (t < b)]))
                            for a, b in PROFILE_SPANS}
        for ic in R["eye_ics"]:
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
        # ---- decisions: cost, catch, outcome per window and cutoff (and the decider now)
        ev = {k: w[f"ev_{k}"].to_numpy(dtype=bool) & single for k in EVIDENCE}
        strong = {k: w[f"ev_{k}_strong"].to_numpy(dtype=bool) & single for k in EVIDENCE}
        mad_ok = w["mad_ok__current"].to_numpy(dtype=bool)
        groups = {"singles": single, "open_loop": (w.category == "open_loop").to_numpy(),
                  "triplets": (w.category == "prime_triplet").to_numpy(),
                  "intervention_singles": single & w.block.str.startswith("iv").to_numpy()}
        blocks_ol = [b for b in otr.OPEN_LOOP_BLOCKS if b in set(w.block)]
        for W, z in [(W, z) for W in WINDOWS for z in Z_GRID] + [("now", now_z(R))]:
            rej = rejected(R, W, z)
            kept = ~rej & mad_ok
            cm = res["meta"]["variants"]["current"] if W == "now" else meta["calibrations"][wz(W, z)]
            row = dict(subject=s, window=W, z=z)
            for g, m in groups.items():
                row[f"pct_{g}_rejected"] = 100 * float(rej[m].mean())
            row["n_singles_rejected"] = int((rej & single).sum())
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
            boot = N_BOOT if (W == "now" or abs(z - Z_ANCHOR) < 1e-9) else 0
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
        # ---- equal cost: each window's cutoff set to reject the same % of single pulses
        sg = w[w.single]
        for q in MATCHED_PCT:
            for W in WINDOWS:
                sco = sg[f"score__{W}"].to_numpy()
                c = float(np.quantile(sco, 1 - q / 100))
                rj = sco > c
                row = dict(subject=s, pct_target=q, window=W, cutoff=c, n_rejected=int(rj.sum()),
                           pct_rejected=100 * float(rj.mean()))
                for k in EVIDENCE:
                    e = sg[f"ev_{k}"].to_numpy(dtype=bool)
                    row[f"precision_{k}"] = 100 * float(e[rj].mean()) if rj.any() else np.nan
                    row[f"hit_{k}"] = 100 * float(rj[e].mean()) if e.any() else np.nan
                T["matched"].append(row)
        # bootstrap of the equal-cost precision: the session's single pulses resampled (the same draws for every
        # window, so the windows are compared on the same pulses), each window's cutoff set anew in every draw
        rng_m = np.random.default_rng(2)
        idx = rng_m.integers(0, len(sg), size=(N_BOOT, len(sg)))
        E = {k: sg[f"ev_{k}"].to_numpy(dtype=bool)[idx] for k in EVIDENCE}
        for W in WINDOWS:
            Sb = sg[f"score__{W}"].to_numpy()[idx]
            for q in MATCHED_PCT:
                rj = Sb > np.quantile(Sb, 1 - q / 100, axis=1)[:, None]
                nr = np.maximum(rj.sum(axis=1), 1)
                for k in EVIDENCE:
                    mboot.setdefault((W, q, k), []).append(100 * (rj & E[k]).sum(axis=1) / nr)
        T["marginal"] += marginal_rows(s, sg)
        # ---- the pulses that change their decision at z 2.5 when the window changes
        b = (w[f"score__{BASE_WINDOW}"] > Z_ANCHOR).to_numpy()
        for W in NEW_WINDOWS:
            n = (w[f"score__{W}"] > Z_ANCHOR).to_numpy()
            for name, m in (("both", b & n), (f"only {BASE_WINDOW}", b & ~n), ("only new", ~b & n)):
                m = m & single
                row = dict(subject=s, window=W, set=name, n=int(m.sum()),
                           n_open_loop=int((m & groups["open_loop"]).sum()),
                           n_intervention=int((m & w.block.str.startswith("iv").to_numpy()).sum()))
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
        for W in WINDOWS:
            for z in Z_GRID:
                cm = meta["calibrations"][wz(W, z)]
                T["calibration"].append(dict(
                    subject=s, window=W, z=z, **{f"dropped_{k}": len(cm["dropped"][k]) for k in ("ocular", "qc", "post")},
                    trials_for_prime=cm["n_kept"], dropped_ocular_trials=" ".join(map(str, cm["dropped"]["ocular"])),
                    dipole_position=cm["position_index"],
                    dipole_shift_mm_vs_now=float(np.linalg.norm(R["fit"][f"rr_mm_{wz(W, z)}"] - rr_now)),
                    sound_filter_rel_max_diff_vs_now=cm["sound_filter_rel_max_diff_vs_current"],
                    post_mad_mean=cm["post_mad_mean_std"][0], post_mad_sd=cm["post_mad_mean_std"][1]))
    A = {k: pd.DataFrame(v) for k, v in T.items()}
    if D:
        allsg = pd.concat([R["w"][R["w"].single] for R in D.values()], ignore_index=True)
        A["marginal"] = pd.concat([A["marginal"], pd.DataFrame(marginal_rows("pooled", allsg))], ignore_index=True)
        A["decisions"] = pd.concat([A["decisions"], pooled_decisions(A["decisions"], draws)], ignore_index=True)
        mt = A["matched"]
        A["matched"] = pd.concat([mt, mt.groupby(["pct_target", "window"], sort=False).mean(numeric_only=True)
                                  .reset_index().assign(subject="pooled")], ignore_index=True)
        # mean of the sessions with bootstrap CIs, and each window against 15-100 ms on the same draws
        mp = A["matched"][A["matched"].subject == "pooled"].set_index(["pct_target", "window"])
        rows = []
        for q in MATCHED_PCT:
            for W in WINDOWS:
                r = dict(pct_target=q, window=W)
                for k in EVIDENCE:
                    pooled = np.mean(mboot[(W, q, k)], axis=0)
                    base = np.mean(mboot[(BASE_WINDOW, q, k)], axis=0)
                    r[f"precision_{k}"] = float(mp.loc[(q, W), f"precision_{k}"])
                    r[f"precision_{k}_ci_lo"], r[f"precision_{k}_ci_hi"] = (float(x) for x in
                                                                            np.percentile(pooled, [2.5, 97.5]))
                    if W != BASE_WINDOW:
                        r[f"diff_vs_{BASE_WINDOW}_{k}"] = r[f"precision_{k}"] - float(mp.loc[(q, BASE_WINDOW),
                                                                                             f"precision_{k}"])
                        r[f"diff_vs_{BASE_WINDOW}_{k}_ci_lo"], r[f"diff_vs_{BASE_WINDOW}_{k}_ci_hi"] = (
                            float(x) for x in np.percentile(pooled - base, [2.5, 97.5]))
                rows.append(r)
        A["matched_pooled"] = pd.DataFrame(rows)
        dg = A["disagreement"]
        A["disagreement"] = pd.concat([dg, dg.drop(columns="pulses").groupby(["window", "set"], sort=False).sum(
            numeric_only=True).reset_index().assign(subject="pooled")], ignore_index=True)
    return A


def marginal_rows(s, sg):
    """% of single pulses with each kind of evidence in bands of each window's score."""
    rows, edges = [], [-np.inf] + MARGINAL_EDGES + [np.inf]
    for W in WINDOWS:
        sco = sg[f"score__{W}"].to_numpy()
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (sco > lo) & (sco <= hi)
            n = int(m.sum())
            row = dict(subject=s, window=W, z_from=lo, z_to=hi, n_single=n)
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
        print("\nOcular evidence (frontal peak-to-peak lines, % of single pulses, % of the calm pulses):")
        print(A["evidence"].drop(columns=["label"]).to_string(index=False))
        cols = ["pct_singles_rejected", "pct_triplets_rejected", "hit_n45", "precision_n45", "hit_late", "precision_late",
                "hit_pre", "precision_pre", "n_kept_strong_n45", "calibration_trials_for_prime",
                "calibration_trials_for_prime_min", "fewest_kept_open_loop_block", "evoked_change_mean_fixed_pct",
                "evoked_change_mean_own_pct"]
        dec = A["decisions"]
        print("\nDecisions (pooled: mean over sessions; counts summed; fewest = minimum):")
        print(dec[dec.subject == "pooled"].set_index(["window", "z"])[[c for c in cols if c in dec]].to_string())
        for s in D:
            print(f"\n{s}:")
            print(dec[dec.subject == s].set_index(["window", "z"])[[c for c in cols if c in dec and
                                                                     not c.endswith("_min")]].to_string())
        mp = A["matched_pooled"]
        print("\nEqual cost (each window's cutoff rejects q % of single pulses): % of the rejected with the evidence, "
              "mean of the sessions [95 % CI]; difference to 15-100 ms on the same bootstrap draws")
        for k in EVIDENCE:
            print(f"  {EVIDENCE[k][1]}:")
            for q in MATCHED_PCT:
                g = mp[mp.pct_target == q].set_index("window")
                cells = []
                for W in WINDOWS:
                    c = f"{W} {g.loc[W, f'precision_{k}']:.0f} [{g.loc[W, f'precision_{k}_ci_lo']:.0f}, " \
                        f"{g.loc[W, f'precision_{k}_ci_hi']:.0f}]"
                    if W != BASE_WINDOW:
                        dk = f"diff_vs_{BASE_WINDOW}_{k}"
                        c += f" ({g.loc[W, dk]:+.0f} [{g.loc[W, dk + '_ci_lo']:+.0f}, {g.loc[W, dk + '_ci_hi']:+.0f}])"
                    cells.append(c)
                print(f"    {q:>2} %: " + " · ".join(cells))
        dg = A["disagreement"]
        print(f"\nPulses whose decision at z {Z_ANCHOR:g} changes with the window (single pulses, all sessions):")
        print(dg[dg.subject == "pooled"].drop(columns=["subject", "pulses"], errors="ignore").to_string(index=False))
        mg = A["marginal"]
        print("\n% of single pulses with a frontal deflection 30-60 ms per band of the score (all sessions):")
        print(mg[mg.subject == "pooled"].pivot_table(index=["z_from", "z_to"], columns="window", values="pct_n45",
                                                     sort=False).round(1).to_string())
    print(f"\ntables written to {OUT / 'tables'}")


# %% 4 · Figures ------------------------------------------------------------------------------------------------------
INK, INK_2, INK_MUTED, GRID_C, BASE_C = otr.INK, otr.INK_2, otr.INK_MUTED, otr.GRID_C, otr.BASE_C
WASH, BAND_C, HEAD_C, HEOG_C = otr.WASH, otr.BAND_C, otr.HEAD_C, otr.HEOG_C
EV_C = "#0b0b0b"                                     # pulses with evidence (scatter): ink, the others light grey
NOEV_C = "#c3c2b7"


def save(fig, name):
    path = OUT / "figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  wrote {path.relative_to(OUT)}")


def window_handles(with_now=True, now_label=None):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=WINDOW_COLOR[W], marker=WINDOW_MARKER[W], ms=6, lw=1.8,
                label=wl(W) + (" (the decider's window)" if W == BASE_WINDOW else "")) for W in WINDOWS]
    if with_now:
        h.append(Line2D([], [], color=HEAD_C, ls="--", lw=1.3,
                        label=now_label or "now: 15–100 ms, reference the whole epoch, z 3"))
    return h


def _lines_vs_z(ax, dec, col, subject="pooled", now=True, own=False):
    d = dec[dec.subject == subject]
    for W in WINDOWS:
        g = d[d.window == W].sort_values("z")
        ax.plot(g.z, g[col], color=WINDOW_COLOR[W], marker=WINDOW_MARKER[W], ms=5.5, lw=1.7, zorder=3)
    if own and "evoked_change_mean_own_pct" in d:
        for k, W in enumerate(WINDOWS):
            g = d[(d.window == W) & ((d.z - Z_ANCHOR).abs() < 1e-9)]
            if len(g):
                ax.plot([Z_ANCHOR + 0.05 + 0.025 * k], g.evoked_change_mean_own_pct, ls="none",
                        marker=WINDOW_MARKER[W], ms=6, mfc="white", mec=WINDOW_COLOR[W], mew=1.4, zorder=4)
    if now:
        n = d[d.window == "now"]
        if len(n) and np.isfinite(n[col].iloc[0]):
            ax.axhline(float(n[col].iloc[0]), color=HEAD_C, ls="--", lw=1.2, zorder=2)
    ax.set_xticks(Z_GRID, [f"{z:g}" for z in Z_GRID])
    ax.set_xlim(Z_GRID[0] - 0.12, Z_GRID[-1] + 0.2)


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
            top = max(top, y_now)
            ax.set_ylim(0, 1.1 * top)
            ax.set_xlim(t[0], t[-1])
            ax.axvline(0, color=BASE_C, lw=0.8)
            ax.set_title(f"{s} · eye IC {ic}\nSD of |s| over the window: " + " | ".join(sds), fontsize=7.6)
            if j == 0:
                ax.set_ylabel("|s| (IC units)")
            if i == len(subs) - 1:
                ax.set_xlabel("ms from the pulse")
    handles = [Line2D([], [], color=INK, lw=1.3, label="|s(t)| of the 125 calibration trials: median"),
               Patch(color=BAND_C, label="10–90 %"), Patch(color=WASH, label="blank (interpolated)"),
               *[Line2D([], [], color=WINDOW_COLOR[W], lw=2.4, marker=WINDOW_MARKER[W], ms=5.5,
                        label=f"{wl(W)}: mean + {Z_ANCHOR:g} SD of |s| over the window") for W in WINDOWS],
               Line2D([], [], color=HEAD_C, lw=1.3, ls="--", label="now: mean + 3 SD of |s| over −30 to 100 ms")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8)
    fig.suptitle(f"Each window sets its own bar: an eye IC rejects a pulse when the median |s| over the window "
                 f"exceeds the bar (drawn over the window's span, z {Z_ANCHOR:g})", fontsize=10.5)
    save(fig, "w1_reference_windows.png")


SUMMARY_PANELS = [
    ("pct_singles_rejected", "% of single pulses", "Single pulses rejected"),
    ("hit_n45", "% of those pulses", "Pulses with a frontal deflection\n30–60 ms that are rejected"),
    ("precision_n45", "% of rejected pulses", "Rejected pulses with a frontal\ndeflection 30–60 ms"),
    ("precision_pre", "% of rejected pulses", "Rejected pulses with eye activity\nbefore the pulse (−210 to −10 ms)"),
    ("calibration_trials_for_prime_min", "calibration trials (of 125)", "Calibration trials left for PRIME\n(fewest "
                                                                         "of the four sessions)"),
    ("evoked_change_mean_fixed_pct", "% change from baseline", "Evoked N45 change, mean of t0–t60\n(fixed processing;"
                                                               " hollow: own decider at 2.5)")]


def fig_summary(D, A):
    """w2 · The choice in one figure (mean of the sessions): top, against the cutoff; bottom left, at equal cost."""
    plt = otr.setup_mpl()
    dec, mp = A["decisions"], A["matched_pooled"]
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.8), layout="constrained")
    for ax, (col, ylab, title) in zip(list(axes[0]) + list(axes[1, 2:]), SUMMARY_PANELS):
        _lines_vs_z(ax, dec, col, own=col == "evoked_change_mean_fixed_pct")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("cutoff z (each window's own reference)", fontsize=8)
        if col.startswith(("hit", "precision")):
            ax.set_ylim(0, 100)
        if col.startswith("evoked"):
            ax.axhline(0, color=BASE_C, lw=0.9)
        if col.startswith("calibration"):
            ax.set_ylim(95, 125)
    for ax, k in zip(axes[1, :2], ("n45", "pre")):
        for W in WINDOWS:
            g = mp[mp.window == W].sort_values("pct_target")
            if W == BASE_WINDOW:
                ax.fill_between(g.pct_target, g[f"precision_{k}_ci_lo"], g[f"precision_{k}_ci_hi"],
                                color=WINDOW_COLOR[W], alpha=0.13, lw=0, zorder=1)
            ax.plot(g.pct_target, g[f"precision_{k}"], color=WINDOW_COLOR[W], marker=WINDOW_MARKER[W], ms=5.5, lw=1.7,
                    zorder=3)
        ax.set_ylim(0, 100)
        ax.set_xticks(MATCHED_PCT)
        ax.set_xlabel("% of single pulses rejected (each window's cutoff set to it)", fontsize=8)
        ax.set_ylabel("% of rejected pulses")
        ax.set_title("Equal cost: rejected pulses with " + EVIDENCE[k][1].replace("frontal deflection", "a frontal\n"
                     "deflection").replace("eye activity before", "eye\nactivity before")
                     + "\n(band: 95 % CI of 15–100 ms)", fontsize=9.5)
    fig.legend(handles=window_handles(), loc="outside lower center", ncol=6, fontsize=8.5)
    ev = A["evidence"]
    pv = ev.groupby("evidence").pct_singles.mean()
    fig.suptitle("Which window and cutoff? Mean of pilot003–006. Frontal deflection: Fp mean or F7 − F8 peak-to-peak "
                 f"above median + {EVIDENCE_K:g} robust SD of the session's single pulses (30–60 ms: "
                 f"{pv.get('n45', np.nan):.0f} % of single pulses; before the pulse: {pv.get('pre', np.nan):.0f} %)",
                 fontsize=10.5)
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
    fig.legend(handles=window_handles(), loc="outside lower center", ncol=6, fontsize=8.5)
    fig.suptitle("Per session: cost and catch of each window and cutoff", fontsize=11)
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
    idx = {z: int(np.argmin(np.abs(SWEEP - z))) for z in (2.0, 2.5, 3.0)}
    for i, (k, (span, label)) in enumerate(EVIDENCE.items()):
        for j, s in enumerate(cols):
            ax = axes[i, j]
            ax.plot([0, xmax], [0, xmax], ls=":", color=INK_MUTED, lw=1.0, zorder=1)
            for W in WINDOWS:
                if s == "pooled":
                    xy = [cost_curve(D[t]["w"], W, k) for t in subs]
                    x, y = np.mean([a for a, _ in xy], axis=0), np.mean([b for _, b in xy], axis=0)
                else:
                    x, y = cost_curve(D[s]["w"], W, k)
                m = x <= xmax + 1
                ax.plot(x[m], y[m], color=WINDOW_COLOR[W], lw=1.6, zorder=3)
                for z, ii in idx.items():
                    ax.plot([x[ii]], [y[ii]], marker=WINDOW_MARKER[W], ms=6.5 if z == Z_ANCHOR else 5,
                            color=WINDOW_COLOR[W], mfc=WINDOW_COLOR[W] if z == Z_ANCHOR else "white", mew=1.3,
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
    handles = window_handles(with_now=False) + [
        Line2D([], [], color=INK_2, marker="o", ms=6.5, ls="none", label=f"filled: z {Z_ANCHOR:g}"),
        Line2D([], [], color=INK_2, marker="o", ms=5, mfc="white", ls="none", label="hollow: z 2 and 3"),
        Line2D([], [], color=HEAD_C, marker="*", ms=11, mfc="white", ls="none", label="now (decider as it runs)"),
        Line2D([], [], color=INK_MUTED, ls=":", label="rejecting pulses at random")]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, fontsize=8.5)
    fig.suptitle("Equal cost: as the cutoff moves, how many of the pulses with ocular evidence does each window "
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
                ax.text(tx, ty, f"{name}: {int(m.sum())}\n({int((m & e).sum())} with deflection)", ha=ha, va=va,
                        fontsize=7.2, color=INK_2)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_title(f"{s} · {wl(W)} vs 15–100 ms", fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f"score, {wl(W)}" if len(NEW_WINDOWS) == 1 else "score, window of the column")
            if i == len(subs) - 1:
                ax.set_xlabel("score, 15–100 ms")
    handles = [Line2D([], [], color=EV_C, marker="o", ms=4.5, ls="none", label="frontal deflection 30–60 ms"),
               Line2D([], [], color=NOEV_C, marker="o", ms=4.5, ls="none", label="no deflection"),
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
                 f"{k45} with a frontal\ndeflection 30–60 ms,\n{kpre} with eye activity\nbefore the pulse",
                 ha="right", va="center", fontsize=7.8, transform=axl.transAxes)
        for c, (_, rr) in enumerate(d.iterrows()):      # a Series: column names such as score__15-100 kept
            n = pos[rr["stem"]]
            cell = outer[r, c + 1].subgridspec(2, 1, hspace=0.1)
            axt = fig.add_subplot(cell[0])
            axt.axvspan(30, 60, color=WASH, lw=0, zorder=0)
            axt.fill_between(t_pre, b_pre[0], b_pre[1], color=BAND_C, lw=0, zorder=1)
            axt.fill_between(t_post, b_post[0], b_post[1], color=BAND_C, lw=0, zorder=1)
            for (tp, vp, tq, vq), color in ((th, HEOG_C), (tv, INK)):
                axt.plot(tp, vp[n], color=color, lw=0.7, alpha=0.55, zorder=2)
                axt.plot(tq, vq[n], color=color, lw=1.0, zorder=3)
            axt.axvline(0, color=BASE_C, lw=0.8)
            axt.set_xlim(-210, 100)
            axt.tick_params(labelbottom=False, labelsize=6.5)
            axt.set_ylabel("µV", fontsize=6.5)
            spans = [lab for k, lab in (("n45", "30–60"), ("late", "30–100")) if bool(rr[f"ev_{k}"])]
            sign = (["deflection " + ", ".join(spans) + " ms"] if spans else []) + \
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
               Patch(color=WASH, label="30–60 ms (frontal deflection measured)"),
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
    order = list(WINDOWS) + ["now"]
    for i, s in enumerate(subs):
        ax = axes[i, 0]
        e = ev[(ev.subject == s) & (ev.processing == "own") & ev.block.isin(eval_b)]
        for k, W in enumerate(order):
            g = e[(e.window == W) & ((W == "now") | ((e.z - Z_ANCHOR).abs() < 1e-9))].set_index("block").reindex(eval_b)
            x = np.arange(len(eval_b)) + (k - (len(order) - 1) / 2) * 0.13
            ax.errorbar(x, g.change_pct, yerr=[g.change_pct - g.change_ci_lo, g.change_ci_hi - g.change_pct],
                        fmt=WINDOW_MARKER.get(W, "*"), color=WINDOW_COLOR.get(W, HEAD_C), ms=5 if W != "now" else 8,
                        mfc="white" if W == "now" else WINDOW_COLOR.get(W), capsize=2, lw=1.0, elinewidth=1.0)
        ax.axhline(0, color=BASE_C, lw=0.9)
        ax.set_xticks(range(len(eval_b)), eval_b)
        ax.set_ylabel("% change from baseline")
        ax.set_title(f"{s} · evoked N45 change per block, each window's own decider at z {Z_ANCHOR:g} (95 % CI)",
                     fontsize=8.5)
        ax = axes[i, 1]
        _lines_vs_z(ax, dec, "evoked_change_mean_fixed_pct", subject=s, own=True)
        ax.axhline(0, color=BASE_C, lw=0.9)
        ax.set_ylabel("% change, mean t0–t60")
        ax.set_title(f"{s} · mean change against the cutoff (fixed processing)", fontsize=8.5)
        if i == len(subs) - 1:
            ax.set_xlabel("cutoff z")
    handles = window_handles(now_label="now (decider as it runs): dashed, star") + [
        Line2D([], [], color=INK_2, marker="o", ms=6, mfc="white", ls="none", label="hollow: own decider at 2.5")]
    fig.legend(handles=handles, loc="outside lower center", ncol=7, fontsize=8.5)
    fig.suptitle("Outcome: does the window change the evoked N45 change from baseline? (fixed processing: the decider's "
                 "cleaning and dipole now, only the kept pulses differ)", fontsize=10.5)
    save(fig, "w7_outcome.png")


FIGURES = ["reference", "summary", "sessions", "cost", "scores", "cards", "outcome"]


def make_figures(D, A, which=None):
    which = set(which or FIGURES)
    for name, fn in (("reference", fig_reference), ("summary", fig_summary), ("sessions", fig_sessions),
                     ("cost", fig_cost_curves), ("scores", fig_scores), ("outcome", fig_outcome)):
        if name in which:
            fn(D, A)
    if "cards" in which:
        for R in D.values():
            fig_cards(R, A)


# %% 9 · Main ------------------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract", action="store_true", help="child mode: extract one session")
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
        sys.exit(0 if extract_windows(args.subject, budget=args.budget) else 3)
    subjects = args.subjects or SUBJECTS
    if not args.plots_only:
        for s in subjects:
            if not otr.is_complete("variants", s):       # the first review's replay comes first
                otr.run_child("variants", s)
            if not is_complete(s):
                run_child(s)
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
