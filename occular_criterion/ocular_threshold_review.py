#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ocular_threshold_review.py - the ocular gate with its reference distribution taken over the gate window (15-100 ms):
how does that criterion behave, and which cutoff (z = 2, 2.5, 3 or 3.5) is best?

The gate (Preprocessor.preprocess_post in prime/decider), per ICLabel "eye blink" IC of the calibration's ICA:

    raw post window [-30, 100] ms
      -> baseline, first artifact blank [-2, 14] ms, 500 Hz low-pass, bad-channel interpolation, average reference
      -> s(t) = ica.get_sources(epoch)
      -> z(t) = (|s(t)| - mean) / SD            mean, SD: the REFERENCE DISTRIBUTION of |s|, from the calibration block
      -> score = median of z(t) over the gate window [15, 100] ms
      -> reject if the score of ANY eye IC > z_threshold          ("ocular z" = max over the eye ICs of the score)

z is an increasing straight-line function of |s|, so the median of z(t) is (median |s(t)| - mean) / SD: a pulse is
rejected when the median |s| over 15-100 ms of any eye IC exceeds   mean + z_threshold x SD   of the reference.

Where the reference distribution comes from (_detect_ocular_trials, at calibration, 125 calibration trials):
    decider now      every sample of every calibration trial's post epoch, -30 to 100 ms: the pre-pulse part,
                     the blanked artifact (|s| near 0) and 15-100 ms
    this review      only the samples of the gate window, 15-100 ms: the same samples the score is the median of

How the change is made - nothing in prime/ is written
-----------------------------------------------------
The variants run in a child process that imports the decider from a verified git export of HEAD (Loop/prime_<sha7>,
the export / verify routine of offline_fwdmodel.ipynb). In that process only, two things are overridden:
  1. the reference window: the decider's module-level _detect_ocular_trials is swapped for detect_ocular_trials_ref()
     below - a copy of the decider's function in which three lines (marked CHANGED) take the mean and SD of |s| over
     the reference window. On every session the copy is first run with the whole epoch as reference and must return
     exactly what the decider's own function returns (bad trials, mean, SD, sample indices); the swap must be called
     exactly once per calibration, else the run stops.
  2. the cutoff: trial_reject_opts.ocular.z_threshold of the Preprocessor's config, which calibrate() and
     preprocess_post() both read.
preprocess_post() itself is untouched: it reads the mean and SD from the calibration bundle.

Variants (VARIANTS), each re-calibrated on the session's own 125 calibration trials
    current          the decider at HEAD as it is (whole-epoch reference, its own cutoff: 3.0 since 007b932)
    window_z2.0 ...  window_z3.5: reference = gate window, z_threshold = 2, 2.5, 3, 3.5
The cutoff also decides which calibration trials are dropped as ocular, and so SOUND, SSP-SIR, the dipole fit, the
post-MAD statistics and the trials PRIME is trained on - hence one calibration per cutoff. The ICA, the eye ICs and the
bad channels are fitted before the gate, so they are the same in every variant (checked), and so the window ocular z
of a pulse is the same for all four cutoffs (checked): the cutoffs differ only in where they cut.
Every pulse of the session (baseline, calibration, intervention, evaluation) goes through every variant's own
preprocess_post(); a recorder on its two ICA calls keeps the eye-IC sources, so the scores are the decider's arithmetic.

Validation: "online" re-runs the code and bundle that ran each session (SESSION_CODE) and must reproduce the logged
decisions (intervention post_reject_reason, ocular decision, tep_amplitude_raw; calibration ocular drops). Cross-checks
against the earlier replays (CROSS_CHECKS): window_z2.5 must equal the window-only patch of offline_ocular_mad pulse by
pulse, and the current decider's ocular z must equal offline_ocular/current (to floating-point precision). The parent's
dipole fit for averages, n45_fit(), must reproduce the decider's fit_trial on every stored pulse. tables/checks.csv.

How to judge (section 7)
------------------------
The four cutoffs split the single pulses into bands of window ocular z: <= 2 (kept by every cutoff), 2-2.5, 2.5-3,
3-3.5 (decided by the choice), > 3.5 (rejected by every cutoff). What the pulses at stake are: frontal evidence (Fp VEOG
or F7-F8 HEOG peak-to-peak in 15-100 ms before ICA above the 97.5th percentile of the calm half of the session's
pulses; as a curve over z, and per band), their single-trial N45, and the evoked N45 of each band's average against
equally large averages of calm pulses - these use one fixed processing, the decider as it runs (AMP_VARIANT). What each
cutoff costs: single pulses and triplets rejected, intervention singles that lose their finetuning label, calibration
trials left for PRIME, the fewest pulses kept in an open-loop block. And the outcome: the evoked N45 change from
baseline to t0..t60, once with each cutoff's own calibration and once with the fixed processing (only the kept pulses
differ). figures/3_decision_summary.png and tables/decision_table.csv put this side by side.

Figures: 1_reference_window (the criterion in IC units: both reference distributions, the cutoffs, the pulses),
2_scores, 3_decision_summary, 4_band_cards_<ID> (example pulses per band), 5_ladder_<ID> (scalp maps along z),
6_amplitude_vs_z, 7_outcome_blocks, 8_window_vs_current (the two scales, equivalent cutoffs), 9_eye_ics,
10_cleaning_<ID> (what ICA removes / leaves per band), 11_timeline.
Tables: decision_table, frontal_by_z, bands, evoked, rejections, blocks, calibration, eye_ics, equivalent, checks,
pulses_<ID> (one row per pulse, every variant's decision side by side).

Run
---
    python ocular_threshold_review.py                       # extract what is missing, then tables and figures
    python ocular_threshold_review.py --plots-only          # tables and figures from what is extracted
    python ocular_threshold_review.py --extract --part variants --subject pilot006 [--budget 150]
    python ocular_threshold_review.py --extract --part online --subject pilot006 [--budget 150]
                                                             # one child run; resumable block by block

Outputs (Loop/offline_ocular_window/)
    replay/sub-<ID>/scores.csv              one row per pulse x variant: ocular z per eye IC, decision, N45, post MAD z,
                                            frontal (EOG) measures
    replay/sub-<ID>/blocks/<block>.npz      per-pulse arrays: gate-window maps, Fp / F7-F8 traces, eye-IC sources,
                                            cleaned N45 maps (ARRAY_VARIANT), each variant's dipole-window epoch
    replay/sub-<ID>/calibration_<variant>.npy   each variant's bundle and dipole fitting info (+ gate override record)
    replay/sub-<ID>/calibration_info.npz, calibration_sources.npz (the calibration's eye-IC sources), fitting.npz,
                    meta.json (settings, per-variant calibration summary, checks)
    online/sub-<ID>/                        the validation run
    tables/, figures/                       section 7 and 8
Runtime: the variants take about 0.25 s per pulse (5 variants), i.e. about 6 min per session; tables and figures
about 1.5 min for four sessions.

The previous version of this script (online vs current decider, cutoffs 1-3, outputs in Loop/offline_ocular) is
old/ocular_threshold_review_pre-window-ref_2026-10-23.py.
"""

# %% 0 · Parameters ---------------------------------------------------------------------------------------------
import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import types
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SUBJECTS = ["pilot003", "pilot004", "pilot005", "pilot006"]
SESSION_ID = "ses-prime"
RUN_IDS = {}                       # e.g. {"pilot005": "105"}; missing: the one run folder with prime/trials_intervention.csv

# The question: the reference distribution of the ocular gate, and the cutoff
REF_WINDOW = "gate"                # "gate": the samples of the gate window (post_timerange: 15 ms to the end of the post
                                   # window, 100 ms); (start_s, end_s): another window containing it; None: whole epoch
Z_GRID = [2.0, 2.5, 3.0, 3.5]      # candidate cutoffs; each one gets its own re-calibration
CURRENT_COMMIT = "HEAD"            # the decider the variants are built on (HEAD = 007b932 "ocular z = 3; max_wait = 0.50")


def variant_name(z):
    return f"window_z{float(z):.1f}"


WINDOW_VARIANTS = [variant_name(z) for z in Z_GRID]
VARIANTS = {"current": dict(ref_window=None, z=None),          # z None: the commit's own z_threshold
            **{variant_name(z): dict(ref_window=REF_WINDOW, z=float(z)) for z in Z_GRID}}
ARRAY_VARIANT = variant_name(Z_GRID[len(Z_GRID) // 2])         # extraction: whose cleaned-epoch maps are stored per pulse
AMP_VARIANT = "current"            # analysis: the fixed processing (cleaning, dipole) for the single-trial N45 and the
                                   # band comparisons - the decider as it runs; every variant also has its own (outcome)

# Validation: the code that ran each session, with the decider settings that differed from the commit
VALIDATE_ONLINE = True
F4FD395 = "f4fd3954c4f5b1e31a5be06c82e39d9beb175154"         # "ocular thresh z 2 --> 2.5", 17 Sep 2026
H007B932 = "007b932d0a4ed4bdb4643fdb2f95441ff0eb6a66"        # "ocular z = 3 ; max_wait = 0.50", 15 Oct 2026
SESSION_CODE = {
    "pilot003": dict(commit=F4FD395, overrides={"ocular_z_threshold": 2.0}),   # ran before f4fd395
    "pilot004": dict(commit=F4FD395, overrides={"ocular_z_threshold": 2.0}),
    "pilot005": dict(commit=F4FD395, overrides={}),
    "pilot006": dict(commit=H007B932, overrides={}),     # calibrated 15 Oct 2026 09:46, after 007b932 (08:57)
}
# Earlier replays the variants must agree with (folder under Loop, variant, compared columns); skipped when absent
CROSS_CHECKS = {
    "earlier window-only replay (a1d8593 + ocular_gate_window_only.patch, z 2.5)":
        ("offline_ocular_mad/window", variant_name(2.5), ("z_max", "reason", "amp")),
    "earlier current-decider replay (a1d8593, whole epoch)": ("offline_ocular/current", "current", ("z_max",)),
}
GIT_EXE = None                     # None: git on PATH or the usual Git for Windows locations

# Analysis and figures
CALM_QUANTILE = 0.5                # calm reference pulses: single pulses with window ocular z <= the session's median
EVIDENCE_PCT = 97.5                # frontal evidence: Fp VEOG or F7-F8 HEOG peak-to-peak in the gate window (before ICA)
STRONG_PCT = 99.5                  #   above this percentile of the calm pulses (each; together <= 5 % / 1 % of calm pulses)
N_BOOT = 1000
MIN_BAND_N = 8                     # fewer pulses: no estimate for a band
FRONTAL_BIN_EDGES = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.5]
                                   # bins of the frontal-evidence curve over window ocular z (outer bins open)
CARDS_PER_BAND = 5
CARD_BANDS = [(1.5, 2.0)] + list(zip(Z_GRID[:-1], Z_GRID[1:])) + [(Z_GRID[-1], Z_GRID[-1] + 1.5)]
LADDER_LEVELS = [-0.5, 0.0, 0.5, 1.0, 1.5, 1.75, 2.25, 2.75, 3.25, 3.75, 5.0, 7.0]
LADDER_PER_LEVEL = 6
GATE_WINDOW_MS = (15, 100)         # the gate's window ([15 ms, end of the post window])
N45_WINDOW_MS = (38, 50)           # dipole window of the decider
VEOG_CHANNELS = ["Fp1", "Fpz", "Fp2"]
HEOG_PAIR = ("F7", "F8")
ROI_CHANNELS = ["C1", "C3", "FC1", "FC3", "CP1", "CP3"]   # left M1 (baseline_candidates)
OPEN_LOOP_BLOCKS = ["baseline", "t0", "t15", "t30", "t60"]
EVAL_BLOCKS = ["t0", "t15", "t30", "t60"]
ROBUST_K = 3.0                     # robust outlier line: median + ROBUST_K x 1.4826 x MAD of the single-pulse ocular z
SWEEP = np.round(np.arange(-1.0, 8.0001, 0.05), 2)       # cutoff grid of the sweeps
AMP_BIN_EDGES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0]

LOOP_ROOT = None                   # None: LOOP_ROOT env var, D:\Linus\Loop, or a folder above this script with BIDS + prime
OUT_ROOT = None                    # None: <Loop>/offline_ocular_window

# %% 1 · Paths, sessions -------------------------------------------------------------------------------------------
BLOCK_ORDER = ["baseline", "calibration", "iv1", "iv2", "iv3", "iv4", "t0", "t15", "t30", "t60"]
EVAL_LABELS = {1: "t0", 2: "t15", 3: "t30", 4: "t60"}
RAW_SFREQ_EXPECTED = 5000.0
STORE_SFREQ = 1000.0               # traces are stored at 1 kHz (scores and maps are computed at the native rate)
MANIFEST_NAME = ".prime_export.json"
ICLABEL_CLASSES = ["brain", "muscle artifact", "eye blink", "heart beat", "line noise", "channel noise", "other"]


def _script_path():
    try:
        return Path(__file__).resolve()
    except NameError:              # interactive window without __file__
        return None


def find_loop_root():
    if LOOP_ROOT:
        return Path(LOOP_ROOT)
    env = os.environ.get("LOOP_ROOT")
    for c in ([env] if env else []) + [r"D:\Linus\Loop"]:
        if c and (Path(c) / "BIDS").is_dir() and (Path(c) / "prime").is_dir():
            return Path(c)
    for start in [p for p in (_script_path(), Path.cwd()) if p is not None]:
        for d in (start if start.is_dir() else start.parent, *start.parents):
            if (d / "BIDS").is_dir() and (d / "prime").is_dir():
                return d
    raise FileNotFoundError("Loop folder not found: set LOOP_ROOT (parameter or environment variable)")


def first_existing(paths, what):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    raise FileNotFoundError(f"{what} not found; looked for:\n  " + "\n  ".join(str(p) for p in paths))


ROOT = find_loop_root()
PRIME_DIR = ROOT / "prime"
PRIME_REPO = next((d for d in (PRIME_DIR / "decider", PRIME_DIR)
                   if (d / ".git").exists() or (d / "2_intervention.py").is_file()), PRIME_DIR / "decider")
FORWARD_PATH = first_existing([d / "offline_data" / "fsaverage" / "fsaverage-fwd.fif" for d in (PRIME_REPO, PRIME_DIR, ROOT)],
                              "the decider's forward model fsaverage-fwd.fif")
OUT = Path(OUT_ROOT) if OUT_ROOT else ROOT / "offline_ocular_window"
if OUT.resolve().is_relative_to(PRIME_DIR.resolve()):
    raise ValueError(f"OUT_ROOT inside prime/: {OUT}")
SCRIPT = _script_path() or ROOT / "scripts_analysis" / "ocular_threshold_review.py"


def session_paths(subject):
    ses = ROOT / "BIDS" / f"sub-{subject}" / SESSION_ID
    run = RUN_IDS.get(subject)
    if run is None:
        runs = sorted({p.parent.parent.name for p in ses.glob("*/prime/trials_intervention.csv")})
        if len(runs) != 1:
            raise FileNotFoundError(f"expected one run folder with prime/trials_intervention.csv in {ses}, found {runs}")
        run = runs[0]
    online_dir = ses / str(run) / "prime"
    bundle = next((online_dir / n for n in ("calibration_bundle.npy", "calibration_bunble.npy")
                   if (online_dir / n).is_file()), None)
    if bundle is None:
        raise FileNotFoundError(f"no calibration bundle in {online_dir}")
    return dict(subject=subject, run=str(run), online_dir=online_dir, bundle=bundle)


def part_dir(part, subject):
    return OUT / ("replay" if part == "variants" else part) / f"sub-{subject}"


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, float):
        return None if not np.isfinite(x) else x
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    return x


# %% 2 · Code exports (git) ------------------------------------------------------------------------------------------
# As offline_fwdmodel.ipynb: the commit is copied out of the repository with git ls-tree / cat-file (no checkout, the
# repository and its working tree are only read), every file is checked against its blob id, and a manifest is kept.

def find_git(explicit=None):
    candidates = [explicit, os.environ.get("GIT_EXE"), shutil.which("git"),
                  r"C:\Program Files\Git\cmd\git.exe", r"C:\Program Files\Git\bin\git.exe",
                  str(Path.home() / "AppData" / "Local" / "Programs" / "Git" / "cmd" / "git.exe")]
    return next((str(c) for c in candidates if c and Path(c).is_file()), None)


def run_git(git, repo, *args, stdin=None):
    r = subprocess.run([git, "-C", str(repo), *args], input=stdin, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:2])} failed: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout


def blob_sha1(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def commit_files(git, repo, commit):
    files = {}
    for rec in run_git(git, repo, "ls-tree", "-r", "-z", "--full-tree", commit).split(b"\0"):
        if not rec:
            continue
        meta, path = rec.split(b"\t", 1)
        mode, kind, sha = meta.decode().split()
        if kind != "blob":
            continue
        if mode == "120000":
            raise RuntimeError(f"{path.decode()} is a symbolic link in {commit}; not supported")
        files[path.decode("utf-8")] = sha
    return files


def read_blobs(git, repo, shas):
    data = run_git(git, repo, "cat-file", "--batch", stdin="".join(s + "\n" for s in shas).encode())
    out, pos = {}, 0
    for sha in shas:
        nl = data.index(b"\n", pos)
        header = data[pos:nl].decode().split()
        if len(header) != 3 or header[0] != sha or header[1] != "blob":
            raise RuntimeError(f"unexpected git cat-file output for {sha}: {header}")
        size = int(header[2])
        out[sha] = data[nl + 1:nl + 1 + size]
        pos = nl + 1 + size + 1
    return out


def export_commit(git, repo, commit, out_dir):
    out_dir = Path(out_dir)
    full = run_git(git, repo, "rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()
    date, subject = run_git(git, repo, "log", "-1", "--format=%ad%n%s", "--date=iso", full).decode().strip().split("\n", 1)
    files = commit_files(git, repo, full)
    blobs = read_blobs(git, repo, sorted(set(files.values())))
    tmp = out_dir.with_name(out_dir.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    for path, sha in files.items():
        if blob_sha1(blobs[sha]) != sha:
            raise RuntimeError(f"{path}: content does not match its blob id {sha}")
        target = tmp / Path(*path.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blobs[sha])
    manifest = {"commit": full, "commit_date": date, "commit_subject": subject, "repository": str(repo),
                "files": files, "exported": datetime.now().isoformat(timespec="seconds"),
                "git": run_git(git, repo, "--version").decode().strip()}
    (tmp / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    tmp.rename(out_dir)
    return manifest


def verify_export(out_dir, commit):
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    problems = []
    if not manifest["commit"].startswith(commit.lower()):
        problems.append(f"the export is of {manifest['commit'][:10]}, not {commit[:10]}")
    for path, sha in manifest["files"].items():
        f = out_dir / Path(*path.split("/"))
        if not f.is_file():
            problems.append(f"missing: {path}")
        elif blob_sha1(f.read_bytes()) != sha:
            problems.append(f"changed since the export: {path}")
    known = {Path(*p.split("/")) for p in manifest["files"]}
    for f in out_dir.rglob("*.py"):
        rel = f.relative_to(out_dir)
        if rel not in known and "__pycache__" not in rel.parts:
            problems.append(f"not in the commit: {rel.as_posix()}")
    return manifest, problems


def resolve_commit(ref):
    """Full SHA of a commit of prime/decider (HEAD allowed). Without git, an existing export of a SHA is accepted."""
    git = find_git(GIT_EXE)
    if git and (PRIME_REPO / ".git").exists():
        return run_git(git, PRIME_REPO, "rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()
    if ref.upper() == "HEAD":
        raise RuntimeError("resolving HEAD needs git and the repository prime/decider/.git")
    d = ROOT / f"prime_{ref[:7]}"
    if (d / MANIFEST_NAME).is_file():
        return json.loads((d / MANIFEST_NAME).read_text(encoding="utf-8"))["commit"]
    raise RuntimeError(f"cannot resolve {ref}: no git and no export {d}")


def ensure_export(commit):
    out_dir = ROOT / f"prime_{commit[:7]}"
    if not (out_dir / MANIFEST_NAME).is_file():
        if out_dir.exists():
            raise RuntimeError(f"{out_dir} exists but has no {MANIFEST_NAME}: not an export; move it away")
        git = find_git(GIT_EXE)
        if not git or not (PRIME_REPO / ".git").exists():
            raise RuntimeError(f"exporting {commit[:10]} needs git and {PRIME_REPO / '.git'}")
        m = export_commit(git, PRIME_REPO, commit, out_dir)
        print(f"exported {m['commit'][:10]} ({len(m['files'])} files) to {out_dir}")
    manifest, problems = verify_export(out_dir, commit)
    if problems:
        raise RuntimeError(f"{out_dir} fails verification: " + "; ".join(problems[:8]) + ". Delete it to export afresh.")
    return out_dir, manifest


def working_tree_changes():
    """Files of prime/decider that differ from HEAD (for the note of the variants)."""
    git = find_git(GIT_EXE)
    if not git or not (PRIME_REPO / ".git").exists():
        return None
    out = run_git(git, PRIME_REPO, "status", "--porcelain", "--untracked-files=no").decode().strip()
    return [l[3:] for l in out.splitlines() if l.strip()]


# %% 3 · The override: reference window of the ocular gate, and the cutoff ------------------------------------------

def reference_samples(times, time_mask, ref_window):
    """Sample indices of the reference distribution. It must contain the gate window (the score's samples)."""
    times = np.asarray(times)
    if ref_window is None:
        ref = np.arange(len(times))
    elif isinstance(ref_window, str):
        if ref_window != "gate":
            raise ValueError(f"REF_WINDOW {ref_window!r}: use 'gate', None or (start_s, end_s)")
        ref = np.asarray(time_mask)
    else:
        lo, hi = ref_window
        lo, hi = (-np.inf if lo is None else float(lo)), (np.inf if hi is None else float(hi))
        ref = np.where((times >= lo) & (times <= hi))[0]
    if not np.isin(time_mask, ref).all():
        raise ValueError("the reference window must contain the gate window (the samples the score is taken over)")
    return ref


def detect_ocular_trials_ref(ica, epochs, tmin, tmax, ocular_component_indices, z_threshold, ref_window=None):
    """The decider's _detect_ocular_trials (prime_core/preprocessing/preprocessor.py; the same in f4fd395 ... 007b932)
    with one change: the reference distribution of |s| - the mean and SD of the z-score - is taken over `ref_window`
    of every calibration trial instead of over every sample of the post epoch. Changed lines are marked CHANGED.

    ref_window  None: every sample - the decider's function (run_variant checks it returns the decider's exact result)
                "gate": the samples of the gate window [tmin, tmax], the samples the score is the median of
                (start_s, end_s): any window containing the gate window
    """
    from scipy.stats import zscore                     # the decider's module imports the same function
    sources = ica.get_sources(epochs)
    source_data = sources.get_data(copy=True)

    if tmax is None:
        tmax = np.inf
    if tmin is None:
        tmin = -np.inf

    time_mask = np.where((sources.times >= tmin) & (sources.times <= tmax))[0]
    ref_mask = reference_samples(sources.times, time_mask, ref_window)            # CHANGED: new, the reference samples
    gate_in_ref = np.searchsorted(ref_mask, time_mask)                              # CHANGED: new, the gate within them
    bad_trials = np.array([])
    thresholds = {}

    for component_idx in ocular_component_indices:
        abs_time_course = np.abs(source_data[:, component_idx, ref_mask])           # CHANGED: was [:, component_idx, :]
        n_trials, n_times = abs_time_course.shape
        flat_time_course = abs_time_course.ravel()
        z_scored = zscore(flat_time_course).reshape(n_trials, n_times)

        assert z_scored.shape == (n_trials, n_times)

        median_z = np.median(z_scored[:, gate_in_ref], axis=1)                      # CHANGED: was [:, time_mask]
        bad_now = np.where(median_z > z_threshold)[0]
        thresholds[component_idx] = {
            'std': np.std(flat_time_course),
            'mean': np.mean(flat_time_course),
            'time_indices_of_interest': time_mask,
        }
        bad_trials = np.union1d(bad_trials, bad_now)

    return list(bad_trials.astype(int)), thresholds


def same_gate_output(a, b):
    (bad_a, thr_a), (bad_b, thr_b) = a, b
    same = (list(map(int, bad_a)) == list(map(int, bad_b)) and sorted(thr_a) == sorted(thr_b) and all(
        thr_a[k]["mean"] == thr_b[k]["mean"] and thr_a[k]["std"] == thr_b[k]["std"]
        and np.array_equal(thr_a[k]["time_indices_of_interest"], thr_b[k]["time_indices_of_interest"]) for k in thr_a))
    return dict(ok=bool(same), detail="identical (bad trials, mean, SD, gate samples)" if same else
                f"differs: bad trials {list(map(int, bad_a))} vs {list(map(int, bad_b))}")


@contextlib.contextmanager
def ocular_reference(P, ref_window, info, capture=None):
    """Inside the block, the decider's calibration builds the ocular gate's reference over `ref_window`.

    ref_window None: the decider's own function runs (only counted). Otherwise the copy runs, after a check on this
    very data that the copy with the whole epoch as reference returns what the decider's function returns.
    capture: a dict that receives the eye-IC sources of the calibration trials (for the reference-window figure)."""
    orig = P.pp_module._detect_ocular_trials
    info["calls"] = 0

    def gate(ica, epochs, tmin, tmax, ocular_component_indices, z_threshold):
        info["calls"] += 1
        decider = orig(ica, epochs, tmin, tmax, ocular_component_indices, z_threshold)
        src = ica.get_sources(epochs)
        times, data = np.asarray(src.times, dtype=float), src.get_data(copy=True)
        tmax_ = np.inf if tmax is None else tmax
        tmin_ = -np.inf if tmin is None else tmin
        gate_idx = np.where((times >= tmin_) & (times <= tmax_))[0]
        ref_idx = reference_samples(times, gate_idx, ref_window)
        if ref_window is None:
            out = decider
        else:
            copy = detect_ocular_trials_ref(ica, epochs, tmin, tmax, ocular_component_indices, z_threshold, None)
            info["copy_equals_decider"] = same_gate_output(decider, copy)
            if not info["copy_equals_decider"]["ok"]:
                raise RuntimeError("detect_ocular_trials_ref() with the whole epoch differs from the decider's "
                                   "_detect_ocular_trials: the copy is out of date - compare it with the decider's code")
            out = detect_ocular_trials_ref(ica, epochs, tmin, tmax, ocular_component_indices, z_threshold, ref_window)
        # the reference statistics, recomputed here from the sources: mean and SD of |s| over the reference samples
        diffs = []
        for ic, st in out[1].items():
            a = np.abs(data[:, int(ic), ref_idx])
            diffs += [abs(np.mean(a) - st["mean"]) / st["mean"], abs(np.std(a) - st["std"]) / st["std"]]
        info.update(ref_stats_rel_diff=float(max(diffs)) if diffs else np.nan,
                    gate_ms=[1e3 * times[gate_idx[0]], 1e3 * times[gate_idx[-1]]], n_gate=int(len(gate_idx)),
                    ref_ms=[1e3 * times[ref_idx[0]], 1e3 * times[ref_idx[-1]]], n_ref=int(len(ref_idx)),
                    n_trials=int(data.shape[0]), z_threshold=float(z_threshold))
        if capture is not None and "src" not in capture:
            eye = sorted(int(i) for i in ocular_component_indices)
            capture.update(src=data[:, eye, :].astype(np.float32), times=times, gate_idx=gate_idx, eye_ics=np.asarray(eye))
        return out

    P.pp_module._detect_ocular_trials = gate
    try:
        yield info
    finally:
        P.pp_module._detect_ocular_trials = orig


def set_z(pp, z):
    """The cutoff: calibrate() reads pp._opts (a copy made in __init__), preprocess_post() reads pp._cfg."""
    pp._cfg.trial_reject_opts.ocular.z_threshold = float(z)
    pp._opts = pp._cfg.to_dicts()
    return pp


def apply_overrides(pp, overrides):
    if "ocular_z_threshold" in overrides:
        set_z(pp, overrides["ocular_z_threshold"])
    return pp


# %% 4 · Extraction (child process: one part x one subject) ---------------------------------------------------------

def import_prime(code_dir):
    """Import prime_core from an export; refuse a prime_core that is already loaded from elsewhere."""
    code_dir = Path(code_dir).resolve()
    loaded = [m for m in sys.modules if m == "prime_core" or m.startswith("prime_core.")]
    for m in loaded:
        f = getattr(sys.modules[m], "__file__", None)
        if f and not Path(f).resolve().is_relative_to(code_dir):
            raise RuntimeError(f"{m} is already imported from {f}: run the extraction in a fresh process")
    sys.dont_write_bytecode = True                       # keep the export a clean copy of the commit
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    if not hasattr(np, "trapz"):                         # NumPy >= 2.4 renamed it; calibrate() still calls np.trapz
        np.trapz = np.trapezoid
    warnings.filterwarnings("ignore")
    import mne
    mne.set_log_level("ERROR")
    import prime_core.preprocessing.preprocessor as preprocessor_module
    from prime_core.preprocessing.preprocessor import Preprocessor
    from prime_core.preprocessing.dipole_fitter import DipoleFitter
    from prime_core.preprocessing.utils.resampling import resample_buffer_polyphase
    from prime_core import prime_config
    outside = [m for m, mod in sys.modules.items() if (m == "prime_core" or m.startswith("prime_core."))
               and getattr(mod, "__file__", None) and not Path(mod.__file__).resolve().is_relative_to(code_dir)]
    if outside:
        raise RuntimeError(f"modules imported from outside the export: {outside}")
    if not callable(getattr(preprocessor_module, "_detect_ocular_trials", None)):
        raise RuntimeError("this decider has no _detect_ocular_trials: the reference-window override does not apply")
    return types.SimpleNamespace(mne=mne, pp_module=preprocessor_module, Preprocessor=Preprocessor,
                                 DipoleFitter=DipoleFitter, resample=resample_buffer_polyphase, cfg=prime_config)


def post_rate(pp, P):
    return float(getattr(pp, "_post_sfreq", P.cfg.get_processed_sfreq()))


def make_fitter(P, rate):
    try:
        return P.DipoleFitter(FORWARD_PATH, sfreq=rate)
    except TypeError:                                    # commits before the 5 kHz post window: 1 kHz, no sfreq
        return P.DipoleFitter(FORWARD_PATH)


def trial_table(sp, online_bundle):
    """Every pulse of the session in recording order, with what the online decider logged."""
    od, rows = sp["online_dir"], []
    if (od / "trials_baseline.csv").is_file():
        d = pd.read_csv(od / "trials_baseline.csv")
        for r in d.itertuples():
            rows.append(dict(block="baseline", stem=f"baseline_{int(r.trial_in_stage):04d}",
                             trial_in_stage=int(r.trial_in_stage), category="open_loop", condition="baseline"))
    t = pd.read_csv(od / "trials_intervention.csv")
    for c in ("is_forced", "postprocessing_failed", "preprocessing_failed"):
        if c in t:
            t[c] = t[c].astype(str).str.lower().eq("true")
    dropped = {int(i): k for k, idx in online_bundle["trials_dropped"].items() for i in np.asarray(idx).ravel()}
    cal_reason = {"ocular": "ocular", "qc": "pre_mad", "post": "post_mad"}
    for r in t.itertuples():
        stem = f"{r.stage}_{int(r.trial_in_stage):04d}"
        if r.stage == "calibration":
            rows.append(dict(block="calibration", stem=stem, trial_in_stage=int(r.trial_in_stage), category="open_loop",
                             condition="calibration",
                             online_reason=cal_reason.get(dropped.get(int(r.trial_in_stage)), "")))
        elif str(r.stage).startswith("intervention_block_"):
            reason = getattr(r, "post_reject_reason", np.nan)
            rows.append(dict(block=f"iv{int(str(r.stage).rsplit('_', 1)[1])}", stem=stem,
                             trial_in_stage=int(r.trial_in_stage), category=r.condition, condition=r.condition,
                             is_forced=bool(r.is_forced), online_failed=bool(r.postprocessing_failed),
                             online_reason=reason if isinstance(reason, str) else "",
                             online_amp=float(r.tep_amplitude_raw) if pd.notna(r.tep_amplitude_raw) else np.nan,
                             prediction_probability=float(r.prediction_probability)
                             if pd.notna(r.prediction_probability) else np.nan))
    for f in sorted(od.glob("trials_evaluation_*.csv"), key=lambda p: int(p.stem.rsplit("_", 1)[1])):
        d = pd.read_csv(f)
        run = int(d.run_index.iloc[0])
        for r in d.itertuples():
            rows.append(dict(block=EVAL_LABELS.get(run, f"evaluation_{run}"),
                             stem=f"evaluation_{run}_{int(r.trial_in_stage):04d}",
                             trial_in_stage=int(r.trial_in_stage), category="open_loop", condition="evaluation"))
    df = pd.DataFrame(rows)
    rank = {b: i for i, b in enumerate(BLOCK_ORDER)}
    df["block_rank"] = df.block.map(lambda b: rank.get(b, len(rank)))
    df = df.sort_values(["block_rank", "trial_in_stage"], kind="stable").reset_index(drop=True)
    df["order"] = np.arange(len(df))
    df["single"] = df.category != "prime_triplet"
    return df


class IcaRecorder:
    """Records the inputs and outputs of the two ICA calls inside preprocess_post(); the computation is unchanged.

    Instance attributes shadow the ICA methods, so only this ICA object (one bundle's) is affected."""

    def __init__(self, ica):
        self.ica, self._get, self._apply = ica, ica.get_sources, ica.apply
        ica.get_sources, ica.apply = self.get_sources, self.apply
        self.reset()

    def reset(self):
        self.pre = self.src = self.removed = self.times = None

    def get_sources(self, inst, *args, **kwargs):
        out = self._get(inst, *args, **kwargs)
        self.pre = inst.get_data(copy=True)[0]          # the epoch the gate looks at (before ICA cleaning)
        self.times = np.asarray(inst.times, dtype=float).copy()
        self.src = out.get_data(copy=True)[0]            # all IC sources
        return out

    def apply(self, inst, *args, **kwargs):
        before = inst.get_data(copy=True)[0]
        out = self._apply(inst, *args, **kwargs)
        self.removed = before - inst.get_data(copy=True)[0]    # what the eye ICs took out
        return out

    def remove(self):
        for name in ("get_sources", "apply"):
            self.ica.__dict__.pop(name, None)


@contextlib.contextmanager
def capture_get_ica(P, store):
    """ICLabel output of the calibration: labels, the full 7-class probabilities, the IC topographies."""
    orig = P.pp_module.get_ica

    def get_ica(epochs_or_raw, *args, **kwargs):
        ica, excluded, labels = orig(epochs_or_raw, *args, **kwargs)
        store["labels"] = [str(l) for l in labels["labels"]]
        store["proba_pred"] = np.asarray(labels["y_pred_proba"], dtype=float)
        try:
            from mne_icalabel.iclabel import iclabel_label_components
            store["proba_all"] = np.asarray(iclabel_label_components(epochs_or_raw, ica, inplace=False), dtype=float)
        except Exception as err:                           # older mne-icalabel: predicted-label probability only
            store["proba_all_error"] = repr(err)
        try:
            import mne_icalabel
            store["icalabel_version"] = mne_icalabel.__version__
        except Exception:
            pass
        store["excluded"] = {k: [int(i) for i in v] for k, v in excluded.items()}
        store["components"] = np.asarray(ica.get_components(), dtype=float)
        store["n_components"] = int(ica.n_components_)
        return ica, excluded, labels

    P.pp_module.get_ica = get_ica
    try:
        yield store
    finally:
        P.pp_module.get_ica = orig


def add_calibration_trials(P, pp, sp, trials):
    """The session's 125 _trial_raw buffers into a Preprocessor, as run_calibration() of the decider."""
    od = sp["online_dir"]
    raw_sfreq = P.cfg.get_raw_sfreq()
    cal_tmin = P.cfg.get_calibration_time_range()[0]
    tic = time.perf_counter()
    for stem in trials.loc[trials.block == "calibration", "stem"]:
        x = np.load(od / f"{stem}_trial_raw.npy")
        pp.add_trial(x, cal_tmin + np.arange(len(x)) / raw_sfreq)
    return time.perf_counter() - tic


def compare_bundles(rep, online):
    """Does a re-calibration reproduce another bundle (ICA, eye ICs, gate statistics, dropped trials)?"""
    oc_r, oc_o = rep["ocular_threshold_post"], online["ocular_threshold_post"]
    rows = {
        "bad channels": (sorted(map(str, rep["bad_channels"])) == sorted(map(str, online["bad_channels"])),
                         f"{sorted(map(str, rep['bad_channels']))} vs {sorted(map(str, online['bad_channels']))}"),
        "eye ICs": (sorted(map(int, oc_r)) == sorted(map(int, oc_o)), f"{sorted(map(int, oc_r))} vs {sorted(map(int, oc_o))}"),
    }
    if rows["eye ICs"][0]:
        d = max(max(abs(oc_r[k]["mean"] - oc_o[k]["mean"]), abs(oc_r[k]["std"] - oc_o[k]["std"])) for k in oc_o)
        rows["eye-IC |s| mean/std, max |diff|"] = (d < 1e-8, f"{d:.1e}")
    for k in ("ocular", "qc", "post"):
        a, b = sorted(map(int, rep["trials_dropped"][k])), sorted(map(int, online["trials_dropped"][k]))
        rows[f"dropped ({k})"] = (a == b, f"{len(a)} vs {len(b)}" + ("" if a == b else f": {a} vs {b}"))
    ur, uo = rep["ica"].unmixing_matrix_, online["ica"].unmixing_matrix_
    same_shape = ur.shape == uo.shape
    rows["ICA unmixing, max |diff|"] = (same_shape and float(np.max(np.abs(ur - uo))) < 1e-6,
                                        f"{float(np.max(np.abs(ur - uo))):.1e}" if same_shape else f"{ur.shape} vs {uo.shape}")
    return {k: dict(ok=bool(v[0]), detail=v[1]) for k, v in rows.items()}


def gate_value(stats, eye_ics, src):
    """The gate as preprocess_post computes it: per eye IC the median z over the gate samples; and their max."""
    z = [float(np.median((np.abs(src[ic, stats[ic]["time_indices_of_interest"]]) - stats[ic]["mean"]) / stats[ic]["std"]))
         for ic in eye_ics]
    return z, max(z)


def run_blocks(blocks, trials, out, one_trial, t_start, budget, say, label):
    """Run block by block; a block counts as done when its csv exists (written last). False if the budget ran out."""
    per_trial_s = None
    for block in blocks:
        f_csv, f_npz = out / "blocks" / f"{block}.csv", out / "blocks" / f"{block}.npz"
        if f_csv.is_file():                              # written after the npz
            continue
        rows = trials[trials.block == block]
        if budget is not None and per_trial_s is not None and \
                time.time() - t_start + per_trial_s * len(rows) * 1.2 > budget:
            break
        tic = time.time()
        recs, arrs = [], {}
        for r in rows.itertuples():
            rec, arrays = one_trial(r.stem)
            recs += rec
            for k, v in arrays.items():
                arrs.setdefault(k, []).append(np.asarray(v, dtype=np.float32))
        per_trial_s = max(per_trial_s or 0.0, (time.time() - tic) / max(len(rows), 1))
        res = rows.merge(pd.DataFrame(recs), on="stem", how="left")
        if arrs:
            np.savez(f_npz, stems=np.asarray(rows.stem, dtype=str), **{k: np.stack(v) for k, v in arrs.items()})
        res.to_csv(f_csv, index=False)
        say(f"  {block:12s} {len(rows):4d} pulses  {time.time() - tic:5.1f} s  {label(res)}")
        if budget is not None and time.time() - t_start > budget:
            break
    return all((out / "blocks" / f"{b}.csv").is_file() for b in blocks)


def internal_checks(scores, tag=""):
    c = {}
    agree = (scores.rejected_ocular == scores.reason.fillna("").eq("ocular")).sum()
    c[f"{tag}replay: ocular z > cutoff vs preprocess_post's own reason"] = dict(
        ok=int(agree) == len(scores), detail=f"{int(agree)}/{len(scores)} agree")
    s = scores.same_as_returned.map({"True": True, "False": False, True: True, False: False}).dropna().astype(bool)
    c[f"{tag}replay: dipole window of last_post_full == preprocess_post output"] = dict(
        ok=bool(s.all()), detail=f"{int(s.sum())}/{len(s)} accepted pulses identical")
    return c


# ---------------------------------------------------------------------- part "online": the validation run

def extract_online(subject, budget=None, verbose=True):
    """The code and bundle that ran the session reproduce its logged decisions (calibration + intervention)."""
    t_start = time.time()
    say = print if verbose else (lambda *a, **k: None)
    sp = session_paths(subject)
    od = sp["online_dir"]
    sc = SESSION_CODE[subject]
    commit, overrides = resolve_commit(sc["commit"]), dict(sc.get("overrides", {}))
    code_dir, manifest = ensure_export(commit)
    P = import_prime(code_dir)
    out = part_dir("online", subject)
    (out / "blocks").mkdir(parents=True, exist_ok=True)
    say(f"[online / {subject}] code {commit[:10]} ({manifest['commit_subject']}); overrides {overrides or 'none'}")
    online_bundle = np.load(sp["bundle"], allow_pickle=True).item()
    trials = trial_table(sp, online_bundle)
    raw_sfreq = P.cfg.get_raw_sfreq()
    post_i_tmin = P.cfg.get_post_initial_time_range()[0]
    dip_tmin, dip_tmax = P.cfg.get_dipole_time_range()
    post_tmin = P.cfg.get_post_time_range()[0]

    rep_file = out / "calibration_replay.npy"
    if rep_file.is_file():
        rep = np.load(rep_file, allow_pickle=True).item()
    else:
        say("  replaying the online calibration ...")
        acc = apply_overrides(P.Preprocessor(FORWARD_PATH), overrides)
        t_add = add_calibration_trials(P, acc, sp, trials)
        log, tic = io.StringIO(), time.perf_counter()
        with contextlib.redirect_stdout(log):
            acc.calibrate()
        rep = dict(check=compare_bundles(acc.calibration_params, online_bundle), log=log.getvalue(),
                   seconds_add_trial=t_add, seconds_calibrate=time.perf_counter() - tic)
        np.save(rep_file, rep, allow_pickle=True)

    pp = apply_overrides(P.Preprocessor.from_bundle(online_bundle, FORWARD_PATH), overrides)
    pp.keep_full_epochs = True
    rate = post_rate(pp, P)
    fitter = make_fitter(P, rate)
    fitter._fitting_info = np.load(od / "dipole_fitting_info.npy", allow_pickle=True).item()
    thr = float(pp._cfg.trial_reject_opts.ocular.z_threshold)
    stats = online_bundle["ocular_threshold_post"]
    eye_ics = sorted(int(k) for k in stats)
    d0 = int(round((dip_tmin - post_tmin) * rate))
    dip_idx = slice(d0, int(round((dip_tmax - post_tmin) * rate)) + 1)
    rec = IcaRecorder(online_bundle["ica"])

    def one_trial(stem):
        x = np.load(od / f"{stem}_post_raw.npy")
        rec.reset()
        ret = pp.preprocess_post(x, post_i_tmin + np.arange(len(x)) / raw_sfreq)
        if rec.pre is None:
            raise RuntimeError("the recorder did not see the ICA calls: is this the decider's preprocess_post?")
        final = np.asarray(pp.last_post_full)
        dip = np.ascontiguousarray(final[:, dip_idx])
        z_eye, z_max = gate_value(stats, eye_ics, rec.src)
        return [dict(stem=stem, z_max=z_max, rejected_ocular=bool(z_max > thr), reason=pp.last_post_reject_reason or "",
                     amp=float(fitter.fit_trial(dip)),
                     same_as_returned=None if ret is None else bool(np.array_equal(np.asarray(ret), dip)))], {}

    blocks = [b for b in ("calibration", "iv1", "iv2", "iv3", "iv4") if b in set(trials.block)]
    complete = run_blocks(blocks, trials, out, one_trial, t_start, budget, say,
                          lambda r: f"(ocular z > {thr:g}: {int(r.rejected_ocular.sum())})")
    rec.remove()
    checks = {"calibration replay vs the online bundle": rep["check"]}
    if complete:
        scores = pd.concat([pd.read_csv(out / "blocks" / f"{b}.csv", keep_default_na=False, na_values=[""])
                            for b in blocks], ignore_index=True)
        for c in ("reason", "online_reason"):
            scores[c] = scores[c].fillna("").astype(str)
        scores.insert(0, "subject", subject)
        scores.to_csv(out / "scores.csv", index=False)
        checks.update(internal_checks(scores))
        iv = scores[scores.block.str.startswith("iv")]
        a = iv.reason.to_numpy() == iv.online_reason.to_numpy()
        checks["intervention: replayed reject reason == online post_reject_reason"] = dict(
            ok=bool(a.all()), detail=f"{int(a.sum())}/{len(iv)} agree")
        acc_ = iv[(iv.reason == "") & iv.online_amp.notna()]
        if len(acc_):
            rd = np.abs(acc_.amp - acc_.online_amp) / np.abs(acc_.online_amp)
            checks["intervention: N45 amplitude vs online tep_amplitude_raw, max rel. diff"] = dict(
                ok=bool(rd.max() < 1e-3), detail=f"{rd.max():.1e} over {len(acc_)} pulses")
        cal = scores[scores.block == "calibration"]
        a = cal.rejected_ocular.to_numpy() == (cal.online_reason == "ocular").to_numpy()
        checks["calibration: ocular z > cutoff vs the online bundle's ocular drops"] = dict(
            ok=bool(a.all()), detail=f"{int(a.sum())}/{len(cal)} agree")
    meta = dict(part="online", subject=subject, run=sp["run"], complete=complete, commit=commit,
                commit_subject=manifest["commit_subject"], overrides=overrides, z_threshold=thr, eye_ics=eye_ics,
                checks=checks, written=datetime.now().isoformat(timespec="seconds"))
    (out / "meta.json").write_text(json.dumps(to_jsonable(meta), indent=1), encoding="utf-8")
    report(f"online / {subject}", complete, checks, t_start, say)
    return complete


def report(tag, complete, checks, t_start, say):
    if complete:
        bad = [f"{k}: {v['detail']}" for k, v in flatten_checks(checks) if not v["ok"]]
        say(f"[{tag}] done in {time.time() - t_start:.0f} s; checks: " + ("all pass" if not bad else "; ".join(bad)))
    else:
        say(f"[{tag}] stopped after {time.time() - t_start:.0f} s (budget); run again to continue")


def flatten_checks(checks, prefix=""):
    for k, v in checks.items():
        if isinstance(v, dict) and "ok" in v:
            yield prefix + k, v
        elif isinstance(v, dict):
            yield from flatten_checks(v, prefix + k + ": ")


# ---------------------------------------------------------------------- part "variants": all variants of one session

def calibrate_variants(P, sp, trials, out, say):
    """One re-calibration per variant on the session's 125 calibration trials (cached as calibration_<variant>.npy)."""
    files = {v: out / f"calibration_{v}.npy" for v in VARIANTS}
    missing = [v for v, f in files.items() if not f.is_file()]
    if not missing:
        return files
    changed = working_tree_changes()
    if changed:
        say(f"  NOTE: prime/decider has uncommitted changes ({', '.join(changed[:6])}); the variants use the commit")
    acc = P.Preprocessor(FORWARD_PATH)
    t_add = add_calibration_trials(P, acc, sp, trials)
    default_z = float(acc._cfg.trial_reject_opts.ocular.z_threshold)
    iclabel = {} if not (out / "iclabel.npy").is_file() else None
    sources = {} if not (out / "calibration_sources.npz").is_file() else None
    for v in missing:
        spec = VARIANTS[v]
        z = default_z if spec["z"] is None else spec["z"]
        set_z(acc, z)
        info, log, tic = {}, io.StringIO(), time.perf_counter()
        with contextlib.redirect_stdout(log), ocular_reference(P, spec["ref_window"], info, capture=sources), \
                (capture_get_ica(P, iclabel) if iclabel is not None else contextlib.nullcontext()):
            _, dipole_buffers = acc.calibrate()
        t_cal = time.perf_counter() - tic
        if info["calls"] != 1:
            raise RuntimeError(f"{v}: calibrate() called _detect_ocular_trials {info['calls']} times, expected once: "
                               f"the override did not take effect as intended")
        fitter = make_fitter(P, post_rate(acc, P))
        with contextlib.redirect_stdout(io.StringIO()):
            fitter.calibrate(dipole_buffers)
        bundle = acc.calibration_params
        np.save(files[v], dict(bundle=bundle, fitting_info=fitter.fitting_info, gate=info, log=log.getvalue(),
                               spec=dict(spec, z_used=z), z_threshold=z, seconds_add_trial=t_add, seconds_calibrate=t_cal),
                allow_pickle=True)
        if iclabel:
            np.save(out / "iclabel.npy", iclabel, allow_pickle=True)
            iclabel = None
        if sources and "src" in sources:
            np.savez(out / "calibration_sources.npz", **sources)
            sources = None
        dr = bundle["trials_dropped"]
        say(f"  calibrated {v:12s} (z {z:g}, reference {info['ref_ms'][0]:.1f}-{info['ref_ms'][1]:.1f} ms, "
            f"{info['n_ref']} samples): dropped ocular {len(dr['ocular'])}, qc {len(dr['qc'])}, post {len(dr['post'])}; "
            f"kept {len(bundle['trials_kept'])}  ({t_cal:.0f} s)")
    return files


def extract_variants(subject, budget=None, verbose=True):
    """All variants x every pulse of one session. Writes each block as soon as it is done; True when complete."""
    t_start = time.time()
    say = print if verbose else (lambda *a, **k: None)
    sp = session_paths(subject)
    od = sp["online_dir"]
    commit = resolve_commit(CURRENT_COMMIT)
    code_dir, manifest = ensure_export(commit)
    P = import_prime(code_dir)
    mne = P.mne
    from scipy.stats import median_abs_deviation, zscore
    out = part_dir("variants", subject)
    (out / "blocks").mkdir(parents=True, exist_ok=True)
    say(f"[variants / {subject}] decider {commit[:10]} ({manifest['commit_subject']}) from {code_dir.name}; "
        f"variants {', '.join(VARIANTS)}")
    online_bundle = np.load(sp["bundle"], allow_pickle=True).item()
    trials = trial_table(sp, online_bundle)
    raw_sfreq = P.cfg.get_raw_sfreq()
    if abs(raw_sfreq - RAW_SFREQ_EXPECTED) > 1e-6:
        raise ValueError(f"raw_sfreq {raw_sfreq} in the commit's prime.yaml, expected {RAW_SFREQ_EXPECTED}")
    post_i_tmin = P.cfg.get_post_initial_time_range()[0]
    post_tmin = P.cfg.get_post_time_range()[0]
    dip_tmin, dip_tmax = P.cfg.get_dipole_time_range()

    cal_files = calibrate_variants(P, sp, trials, out, say)
    skip_blocks = budget is not None and time.time() - t_start > 0.5 * budget

    runs = {}
    for v in VARIANTS:
        c = np.load(cal_files[v], allow_pickle=True).item()
        pp = set_z(P.Preprocessor.from_bundle(c["bundle"], FORWARD_PATH), c["z_threshold"])
        pp.keep_full_epochs = True                       # as Decider.__init__
        fitter = make_fitter(P, post_rate(pp, P))
        fitter._fitting_info = c["fitting_info"]         # DipoleFitter has no public restore method
        stats = c["bundle"]["ocular_threshold_post"]
        runs[v] = types.SimpleNamespace(pp=pp, fitter=fitter, cal=c, stats=stats, thr=float(c["z_threshold"]),
                                        eye_ics=sorted(int(k) for k in stats), gts=c["bundle"]["good_trial_stats_post"],
                                        rec=None)
    R0 = runs[ARRAY_VARIANT]
    pp0 = R0.pp
    rate = post_rate(pp0, P)
    eye_ics = R0.eye_ics
    glob_thr = [float(v) for v in pp0._cfg.trial_reject_opts.post.global_zscore_threshold]
    loc_thr = float(pp0._cfg.trial_reject_opts.post.local_zscore_threshold)
    reject_range = [float(v) for v in pp0._cfg.reject_range]
    info_post = getattr(pp0, "_info_post", None) or pp0._info_processed
    ch_names = list(pp0._info["ch_names"])
    step = int(round(rate / STORE_SFREQ))
    d0 = int(round((dip_tmin - post_tmin) * rate))
    dip_idx = slice(d0, int(round((dip_tmax - post_tmin) * rate)) + 1)
    gate_idx = np.asarray(R0.stats[eye_ics[0]]["time_indices_of_interest"])
    veog_idx = [ch_names.index(c) for c in VEOG_CHANNELS]
    heog_idx = [ch_names.index(c) for c in HEOG_PAIR]
    roi_idx = [ch_names.index(c) for c in ROI_CHANNELS]
    bad_set = set(map(str, R0.cal["bundle"]["bad_channels"]))
    good_idx = [i for i, c in enumerate(ch_names) if c not in bad_set]
    proc_names = list(np.load(od / "proc_info.npy", allow_pickle=True).item()["ch_names"])
    if proc_names != ch_names:
        raise ValueError("channel order of the session (proc_info.npy) differs from the forward model's")
    for v, R in runs.items():                            # same ICA, eye ICs and gate samples in every variant
        if R.eye_ics != eye_ics or not np.array_equal(R.stats[eye_ics[0]]["time_indices_of_interest"], gate_idx):
            raise RuntimeError(f"{v}: other eye ICs or gate samples than {ARRAY_VARIANT}")

    def one_trial(stem):
        x = np.load(od / f"{stem}_post_raw.npy")
        tt = post_i_tmin + np.arange(len(x)) / raw_sfreq
        recs, arrays, first, fin = [], {}, None, None
        for v, R in runs.items():
            R.rec.reset()
            tic = time.perf_counter()
            ret = R.pp.preprocess_post(x, tt)
            t_ms = (time.perf_counter() - tic) * 1e3
            if R.rec.pre is None or R.rec.removed is None:
                raise RuntimeError("the recorder did not see the ICA calls: is this the decider's preprocess_post?")
            final = np.asarray(R.pp.last_post_full)
            z_eye, z_max = gate_value(R.stats, R.eye_ics, R.rec.src)
            dip = np.ascontiguousarray(final[:, dip_idx])
            ep = mne.EpochsArray(final[np.newaxis], info_post, tmin=post_tmin, verbose=False)
            rd = ep.crop(reject_range[0], reject_range[1]).get_data(copy=True)
            row = dict(stem=stem, variant=v, z_threshold=R.thr, z_max=z_max,
                       z_argmax_ic=R.eye_ics[int(np.argmax(z_eye))], rejected_ocular=bool(z_max > R.thr),
                       reason=R.pp.last_post_reject_reason or "", amp=float(R.fitter.fit_trial(dip)),
                       z_mad_global=float((median_abs_deviation(rd, axis=(1, 2))[0] - R.gts["mads_mean"])
                                          / R.gts["mads_std"]),
                       z_mad_local_max=float(np.max(np.abs(zscore(median_abs_deviation(rd, axis=2), axis=1)))),
                       final_veog_n45=float(final[veog_idx][:, dip_idx].mean()),
                       final_roi_n45=float(final[roi_idx][:, dip_idx].mean()),
                       same_as_returned=None if ret is None else bool(np.array_equal(np.asarray(ret), dip)), t_ms=t_ms)
            for j, z in enumerate(z_eye):
                row[f"z_eye{j + 1}"] = z
            if first is None:
                first = R.rec
            # every variant's gate must see the same epoch, sources and ICA removal (ICA fitted before the gate)
            row["same_inputs"] = R.rec is first or bool(
                np.array_equal(R.rec.pre, first.pre) and np.array_equal(R.rec.src, first.src)
                and np.array_equal(R.rec.removed, first.removed))
            recs.append(row)
            arrays[f"dip_{v}"] = dip
            if v == ARRAY_VARIANT:
                fin = final
        # what the gate sees, the same in every variant: frontal (EOG-like) measures, independent of the ICA
        pre, removed, src = first.pre, first.removed, first.src
        veog = pre[veog_idx].mean(axis=0)
        heog = pre[heog_idx[0]] - pre[heog_idx[1]]
        pre_raw = np.load(od / f"{stem}_pre_raw.npy")
        pre_win = np.asarray(P.resample(pre_raw, raw_sfreq, STORE_SFREQ), dtype=float).T    # (n_ch, 201), raw
        pw = pre_win - pre_win[good_idx].mean(axis=0, keepdims=True)
        veog_pre, heog_pre = pw[veog_idx].mean(axis=0), pw[heog_idx[0]] - pw[heog_idx[1]]
        common = dict(veog_gate_mean=float(veog[gate_idx].mean()), veog_gate_ptp=float(np.ptp(veog[gate_idx])),
                      heog_gate_mean=float(heog[gate_idx].mean()), heog_gate_ptp=float(np.ptp(heog[gate_idx])),
                      gfp_gate=float(pre[:, gate_idx].std(axis=0).mean()),
                      removed_gate_rms=float(np.sqrt(np.mean(removed[:, gate_idx] ** 2))),
                      veog_pre_ptp=float(np.ptp(veog_pre)), heog_pre_ptp=float(np.ptp(heog_pre)),
                      veog_pre_mean_abs=float(np.mean(np.abs(veog_pre - veog_pre.mean()))))
        for r in recs:
            r.update(common)
        n45 = fin[:, dip_idx]
        arrays.update(gate_mean=pre[:, gate_idx].mean(axis=1), gate_ms=np.mean(pre[:, gate_idx] ** 2, axis=1),
                      removed_ms=np.mean(removed[:, gate_idx] ** 2, axis=1), veog=veog[::step], heog=heog[::step],
                      src=src[eye_ics][:, ::step], veog_pre=veog_pre, heog_pre=heog_pre,
                      n45_mean=n45.mean(axis=1), n45_ms=np.mean(n45 ** 2, axis=1),
                      roi_final=fin[roi_idx].mean(axis=0)[::step], veog_final=fin[veog_idx].mean(axis=0)[::step])
        return recs, arrays

    blocks = [b for b in BLOCK_ORDER if b in set(trials.block)] + sorted(set(trials.block) - set(BLOCK_ORDER))
    complete = False
    if not skip_blocks:
        for R in runs.values():
            R.rec = IcaRecorder(R.cal["bundle"]["ica"])
        complete = run_blocks(blocks, trials, out, one_trial, t_start, budget, say,
                              lambda r: "ocular: " + " ".join(
                                  f"{v.replace('window_', '')} {int(((r.variant == v) & (r.reason == 'ocular')).sum())}"
                                  for v in runs))
        for R in runs.values():
            R.rec.remove()

    # ---- calibration info, fitting, meta (every run)
    ica0 = R0.cal["bundle"]["ica"]
    ic = np.load(out / "iclabel.npy", allow_pickle=True).item() if (out / "iclabel.npy").is_file() else {}
    cal_arrays = dict(ch_names=np.asarray(ch_names), eye_ics=np.asarray(eye_ics),
                      components=np.asarray(ica0.get_components(), dtype=float),
                      iclabel_labels=np.asarray(ic.get("labels", []), dtype=str),
                      iclabel_proba_all=np.asarray(ic.get("proba_all", np.zeros((0, 0)))),
                      iclabel_proba_pred=np.asarray(ic.get("proba_pred", [])),
                      bad_channels=np.asarray(sorted(bad_set), dtype=str))
    fit_arrays = {}
    fwd = mne.read_forward_solution(str(FORWARD_PATH), verbose=False)
    L = fwd["sol"]["data"] - np.mean(fwd["sol"]["data"], axis=0)        # as dipole_fitter: average-reference leadfield
    variant_meta, checks = {}, {}
    for v, R in runs.items():
        b, fi = R.cal["bundle"], R.cal["fitting_info"]
        for k in ("ocular", "qc", "post"):
            cal_arrays[f"dropped_{k}_{v}"] = np.asarray(b["trials_dropped"][k], dtype=int).ravel()
        cal_arrays[f"eye_mean_{v}"] = np.asarray([R.stats[i]["mean"] for i in eye_ics])
        cal_arrays[f"eye_std_{v}"] = np.asarray([R.stats[i]["std"] for i in eye_ics])
        p = int(fi["position_index"])
        fit_arrays[f"L_pos_{v}"] = L[:, 3 * p:3 * p + 3]
        fit_arrays[f"index_range_{v}"] = np.asarray(fi["index_range"], dtype=int)
        fit_arrays[f"rr_mm_{v}"] = 1e3 * np.asarray(fwd["source_rr"][p], dtype=float)     # dipole position, head coords
        g = R.cal["gate"]
        variant_meta[v] = dict(z_threshold=R.thr, ref_window=VARIANTS[v]["ref_window"], ref_window_ms=g.get("ref_ms"),
                               n_ref_samples=g.get("n_ref"), gate_window_ms=g.get("gate_ms"), n_gate_samples=g.get("n_gate"),
                               eye_stats={str(i): dict(mean=float(R.stats[i]["mean"]), std=float(R.stats[i]["std"]))
                                          for i in eye_ics},
                               dropped={k: list(map(int, np.asarray(b["trials_dropped"][k]).ravel()))
                                        for k in ("ocular", "qc", "post")},
                               n_kept=int(np.asarray(b["trials_kept"]).size), position_index=p,
                               index_range=list(map(int, fi["index_range"])),
                               post_mad_mean_std=[float(R.gts["mads_mean"]), float(R.gts["mads_std"])],
                               seconds_calibrate=float(R.cal["seconds_calibrate"]))
        if VARIANTS[v]["ref_window"] is not None:
            checks[f"{v}: copy of _detect_ocular_trials with the whole epoch == the decider's function"] = \
                g["copy_equals_decider"]
            checks[f"{v}: reference mean / SD == mean / SD of |s| over the reference samples"] = dict(
                ok=bool(g["ref_stats_rel_diff"] < 1e-12), detail=f"max rel. diff {g['ref_stats_rel_diff']:.1e}")
        checks[f"{v}: the calibration called the gate function once"] = dict(ok=g["calls"] == 1,
                                                                               detail=f"{g['calls']} call(s)")
    for v, R in runs.items():                            # the ICA is fitted before the gate: identical everywhere
        if v == ARRAY_VARIANT:
            continue
        cmp = compare_bundles(R.cal["bundle"], R0.cal["bundle"])
        checks[f"{v} vs {ARRAY_VARIANT}: same ICA, eye ICs, bad channels"] = {
            k: cmp[k] for k in ("bad channels", "eye ICs", "ICA unmixing, max |diff|")}
        if v in WINDOW_VARIANTS:
            same = all(R.stats[i]["mean"] == R0.stats[i]["mean"] and R.stats[i]["std"] == R0.stats[i]["std"]
                       for i in eye_ics)
            checks[f"{v} vs {ARRAY_VARIANT}: same reference mean / SD"] = dict(ok=bool(same), detail="identical" if same
                                                                                else "differ")
    if subject in SESSION_CODE and resolve_commit(SESSION_CODE[subject]["commit"]) == commit and \
            not SESSION_CODE[subject].get("overrides"):
        checks["current re-calibration vs the online bundle (same code)"] = compare_bundles(runs["current"].cal["bundle"],
                                                                                            online_bundle)
    np.savez(out / "calibration_info.npz", **cal_arrays)
    np.savez(out / "fitting.npz", **fit_arrays)

    if complete:
        scores = pd.concat([pd.read_csv(out / "blocks" / f"{b}.csv", keep_default_na=False, na_values=[""])
                            for b in blocks], ignore_index=True)
        for c in ("reason", "online_reason"):
            scores[c] = scores[c].fillna("").astype(str) if c in scores else ""
        scores.insert(0, "subject", subject)
        scores.to_csv(out / "scores.csv", index=False)
        for v, R in runs.items():
            s_v = scores[scores.variant == v]
            checks.update(internal_checks(s_v, tag=f"{v}: "))
            cal = s_v[s_v.block == "calibration"]
            drop = set(map(int, np.asarray(R.cal["bundle"]["trials_dropped"]["ocular"]).ravel()))
            mism = int((cal.rejected_ocular != cal.trial_in_stage.isin(drop)).sum())
            checks[f"{v}: calibration pulses, ocular z > cutoff vs its re-calibration's ocular drops"] = dict(
                ok=mism == 0, detail=f"{len(cal) - mism}/{len(cal)} agree")
        wz = scores[scores.variant.isin(WINDOW_VARIANTS)].pivot(index="stem", columns="variant", values="z_max")
        dz = float(np.nanmax(wz.max(axis=1) - wz.min(axis=1))) if wz.shape[1] > 1 else 0.0
        checks["window variants: the same ocular z for every pulse"] = dict(ok=dz == 0.0,
                                                                          detail=f"max spread {dz:.1e}")
        si = scores.same_inputs.map({"True": True, "False": False, True: True, False: False}).astype(bool)
        checks["every variant's gate saw the same epoch, eye-IC sources and ICA removal"] = dict(
            ok=bool(si.all()), detail=f"{int(si.sum())}/{len(si)} pulse x variant rows identical to the first variant")
        checks.update(cross_checks(scores, subject))
    post_i_tmax = P.cfg.get_post_initial_time_range()[1]
    post_tmax = P.cfg.get_post_time_range()[1]
    qc_tmin, qc_tmax = P.cfg.get_qc_time_range()
    t_epoch = post_i_tmin + np.arange(int(round((post_i_tmax - post_i_tmin) * rate)) + 1) / rate
    t_final = post_tmin + np.arange(int(round((post_tmax - post_tmin) * rate)) + 1) / rate
    t_pre = qc_tmin + np.arange(int(round((qc_tmax - qc_tmin) * STORE_SFREQ)) + 1) / STORE_SFREQ
    t_dip = dip_tmin + np.arange(dip_idx.stop - dip_idx.start) / rate
    meta = dict(part="variants", subject=subject, run=sp["run"], complete=complete, commit=commit,
                commit_subject=manifest["commit_subject"], commit_date=manifest["commit_date"], code_dir=str(code_dir),
                ref_window=REF_WINDOW, z_grid=Z_GRID, array_variant=ARRAY_VARIANT, variants=variant_meta,
                eye_ics=eye_ics, post_sfreq=rate, store_sfreq=STORE_SFREQ, bad_channels=sorted(bad_set),
                n_ica_components=int(ica0.n_components_), post_mad_thresholds=dict(global_=glob_thr, local=loc_thr),
                reject_range_s=reject_range, baseline_s=[float(v) for v in pp0._cfg.baseline],
                artifact_window_1_s=[float(v) for v in pp0._cfg.artifact_window_1], ch_names=ch_names,
                times_epoch_ms=list(np.round(1e3 * t_epoch[::step], 3)),       # gate epoch traces (-30..100 ms)
                times_final_ms=list(np.round(1e3 * t_final[::step], 3)),       # cleaned epoch traces (1..100 ms)
                times_pre_win_ms=list(np.round(1e3 * t_pre, 3)),               # raw pre window (-210..-10 ms)
                times_dip_ms=list(np.round(1e3 * t_dip, 3)),                   # dipole window at the post rate
                gate_window_ms=[float(1e3 * t_epoch[gate_idx[0]]), float(1e3 * t_epoch[gate_idx[-1]])],
                forward=str(FORWARD_PATH), forward_sha256=hashlib.sha256(FORWARD_PATH.read_bytes()).hexdigest()[:16],
                versions=dict(python=platform.python_version(), numpy=np.__version__, mne=mne.__version__,
                              pandas=pd.__version__, scipy=__import__("scipy").__version__),
                checks=checks, written=datetime.now().isoformat(timespec="seconds"))
    (out / "meta.json").write_text(json.dumps(to_jsonable(meta), indent=1), encoding="utf-8")
    report(f"variants / {subject}", complete, checks, t_start, say)
    return complete


def cross_checks(scores, subject):
    """The variants against the earlier replays of the same pulses (CROSS_CHECKS)."""
    out = {}
    for name, (rel, v, cols) in CROSS_CHECKS.items():
        f = ROOT / rel / f"sub-{subject}" / "scores.csv"
        if v not in VARIANTS or not f.is_file() or (v in WINDOW_VARIANTS and REF_WINDOW != "gate"):
            continue
        old = pd.read_csv(f, keep_default_na=False, na_values=[""])
        old["reason"] = old["reason"].fillna("").astype(str) if "reason" in old else ""
        new = scores[scores.variant == v]
        m = new.merge(old[["stem", *cols]], on="stem", suffixes=("", "_old"))
        out[f"{name}: pulses matched"] = dict(ok=len(m) == len(new) == len(old), detail=f"{len(m)} of {len(new)} / {len(old)}")
        # floating-point agreement: another machine or BLAS thread count changes the last digits of the ICA fit,
        # and SOUND's iterations carry that on (the online check allows 1e-3 on the amplitude, too)
        if "z_max" in cols:
            dz = float(np.max(np.abs(m.z_max - m.z_max_old)))
            out[f"{name}: ocular z, max |diff|"] = dict(ok=dz < 1e-6, detail=f"{dz:.1e}")
        if "reason" in cols:
            a = m.reason == m.reason_old
            out[f"{name}: reject reason"] = dict(ok=bool(a.all()), detail=f"{int(a.sum())}/{len(m)} agree")
        if "amp" in cols:
            rd = float(np.max(np.abs(m.amp - m.amp_old) / np.abs(m.amp_old)))
            out[f"{name}: N45 amplitude, max rel. diff"] = dict(ok=rd < 1e-3, detail=f"{rd:.1e}")
    return out


def run_child(part, subject, budget=None):
    """Run one extraction in a fresh interpreter until it reports complete (no budget: one run does everything)."""
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


def is_complete(part, subject):
    d = part_dir(part, subject)
    f = d / "meta.json"
    return f.is_file() and json.loads(f.read_text(encoding="utf-8")).get("complete", False) and (d / "scores.csv").is_file()


# %% 5 · Load the extracted results ---------------------------------------------------------------------------------
PER_VARIANT = ["z_max", "z_argmax_ic", "reason", "amp", "z_mad_global", "z_mad_local_max", "final_roi_n45",
               "final_veog_n45", "mad_ok", "kept"]
COMMON = ["block", "trial_in_stage", "category", "condition", "order", "single", "is_forced", "online_reason",
          "online_amp", "prediction_probability", "veog_gate_mean", "veog_gate_ptp", "heog_gate_mean", "heog_gate_ptp",
          "gfp_gate", "removed_gate_rms", "veog_pre_ptp", "heog_pre_ptp", "veog_pre_mean_abs"]


def load_subject(s):
    d = part_dir("variants", s)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    df = pd.read_csv(d / "scores.csv", keep_default_na=False, na_values=[""], low_memory=False)
    for c in ("reason", "online_reason"):
        df[c] = df[c].fillna("").astype(str) if c in df else ""
    df["is_forced"] = df.is_forced.astype(str).str.lower().eq("true") if "is_forced" in df else False
    df["single"] = df.category != "prime_triplet"
    g, l = meta["post_mad_thresholds"]["global_"], meta["post_mad_thresholds"]["local"]
    lo, hi = (-np.inf if g[0] is None else g[0]), (np.inf if g[1] is None else g[1])
    df["mad_ok"] = (df.z_mad_global >= lo) & (df.z_mad_global <= hi) & (df.z_mad_local_max <= l)
    df["kept"] = df.reason == ""
    res = dict(meta=meta, df=df, dir=d, subject=s, cal=dict(np.load(d / "calibration_info.npz", allow_pickle=True)),
               fit=dict(np.load(d / "fitting.npz")), thr={v: m["z_threshold"] for v, m in meta["variants"].items()})
    f = d / "calibration_sources.npz"
    res["sources"] = dict(np.load(f)) if f.is_file() else None
    om = part_dir("online", s) / "meta.json"
    res["online_meta"] = json.loads(om.read_text(encoding="utf-8")) if om.is_file() else None
    res["variants"] = [v for v in VARIANTS if v in set(df.variant)]
    res["w"] = wide(res)
    return res


def wide(res):
    """One row per pulse: the measures that do not depend on the variant, and <column>__<variant> for those that do.
    z_win: the window ocular z (the same in every window variant); z_cur: the current decider's (whole epoch).
    band: 0 = z_win <= Z_GRID[0], 1 = (Z_GRID[0], Z_GRID[1]], ..., len(Z_GRID) = above the last cutoff.
    calm: single pulses with z_win at or below the session's median. frontal: Fp VEOG or F7-F8 HEOG peak-to-peak in the
    gate window (before ICA) above the EVIDENCE_PCT percentile of the calm pulses; frontal_pre: the same in the raw
    pre-stimulus window (-210..-10 ms: free of the TMS artifact, blind to blinks the pulse sets off). rel_amp: the
    AMP_VARIANT N45 amplitude / the median of the calm open-loop pulses passing its MAD gates."""
    df, variants = res["df"], res["variants"]
    base = df[df.variant == variants[0]].set_index("stem")
    w = base[[c for c in COMMON if c in base]].copy()
    for v in variants:
        d = df[df.variant == v].set_index("stem")
        for c in PER_VARIANT + [c for c in d if c.startswith("z_eye")]:
            if c in d:
                w[f"{c}__{v}"] = d[c]
    wv = [v for v in WINDOW_VARIANTS if v in variants]
    w["z_win"] = w[f"z_max__{wv[0]}"]
    w["z_cur"] = w["z_max__current"] if "current" in variants else np.nan
    w["band"] = np.digitize(w.z_win, Z_GRID, right=True)
    sg = w[w.single]
    q = float(sg.z_win.quantile(CALM_QUANTILE))
    w["calm"] = w.single & (w.z_win <= q)
    calm = w[w.calm]
    ev = dict(calm_z_max=q, n_calm=int(len(calm)))
    for k, col in (("veog", "veog_gate_ptp"), ("heog", "heog_gate_ptp")):
        ev[k] = float(np.percentile(calm[col], EVIDENCE_PCT))
        ev[f"{k}_strong"] = float(np.percentile(calm[col], STRONG_PCT))
    w["frontal"] = (w.veog_gate_ptp > ev["veog"]) | (w.heog_gate_ptp > ev["heog"])
    w["frontal_strong"] = (w.veog_gate_ptp > ev["veog_strong"]) | (w.heog_gate_ptp > ev["heog_strong"])
    ev["veog_pre"] = float(np.percentile(calm.veog_pre_ptp, EVIDENCE_PCT))
    ev["heog_pre"] = float(np.percentile(calm.heog_pre_ptp, EVIDENCE_PCT))
    w["frontal_pre"] = (w.veog_pre_ptp > ev["veog_pre"]) | (w.heog_pre_ptp > ev["heog_pre"])
    ev["calm_rate"] = float(w.frontal[w.calm].mean())
    ev["calm_rate_strong"] = float(w.frontal_strong[w.calm].mean())
    ev["calm_rate_pre"] = float(w.frontal_pre[w.calm].mean())
    res["evidence"] = ev
    A, M = f"amp__{AMP_VARIANT}", f"mad_ok__{AMP_VARIANT}"
    w["amp_ok"] = w.single & w[M].astype(bool)
    w["rel_amp"] = w[A] / w.loc[w.calm & (w.category == "open_loop") & w.amp_ok, A].median()
    return w.reset_index()


def load_all(subjects):
    D = {}
    for s in subjects:
        if is_complete("variants", s):
            D[s] = load_subject(s)
        else:
            print(f"not extracted yet: variants / {s}")
    return D


def load_arrays(res, kind, stems=None):
    """Arrays of one kind for the pulses in `stems` (default: all, in the order of res['w']); each kind is read from
    the block files once per subject and kept."""
    cache = res.setdefault("_arrays", {})
    if "_pos" not in res:
        res["_pos"] = {s: i for i, s in enumerate(res["w"].stem)}
    pos = res["_pos"]
    if kind not in cache:
        out = None
        for f in sorted((res["dir"] / "blocks").glob("*.npz")):
            with np.load(f) as z:
                st, a = [str(s) for s in z["stems"]], z[kind]
            if out is None:
                out = np.full((len(pos),) + a.shape[1:], np.nan, dtype=np.float32)
            out[[pos[s] for s in st]] = a
        cache[kind] = out
    return cache[kind] if stems is None else cache[kind][[pos[s] for s in stems]]


def n45_fit(epochs, L_pos, index_range):
    """The decider's single-trial dipole amplitude (DipoleFitter.fit_trial, free orientation) at the calibrated
    position: per sample of index_range, Q = pinv(L) y and the R² of L Q; the amplitude |Q| at the best R².
    epochs (n_ch, n_t) or (n, n_ch, n_t). Checked against fit_trial on every stored pulse (tables/checks.csv)."""
    Y = np.asarray(epochs, dtype=float)
    one = Y.ndim == 2
    Y = Y[None] if one else Y
    a, b = int(index_range[0]), int(index_range[1])
    Y = Y[:, :, a:b]
    L = np.asarray(L_pos, dtype=float)
    Q = np.einsum("kc,nct->nkt", np.linalg.pinv(L), Y)
    res = Y - np.einsum("ck,nkt->nct", L, Q)
    ss_res = (res ** 2).sum(axis=1)
    ss_tot = ((Y - Y.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    best = np.argmax(1.0 - ss_res / ss_tot, axis=1)
    amp = np.linalg.norm(Q[np.arange(len(Q)), :, best], axis=1)
    return float(amp[0]) if one else amp


def fit_of(res, v):
    return res["fit"][f"L_pos_{v}"], res["fit"][f"index_range_{v}"]


def evoked_n45(res, v, stems, n_boot=0, rng=None):
    """N45 of the average of `stems` (variant v's cleaned epochs and dipole). With n_boot: bootstrap amplitudes, the
    pulses resampled with replacement (drawn as multinomial counts, so the resampled means are one matrix product)."""
    stems = list(stems)
    if len(stems) < 2:
        return np.nan, None
    L, (a, b) = fit_of(res, v)
    X = load_arrays(res, f"dip_{v}", stems)[:, :, a:b].astype(float)        # the samples the fit looks at
    if not np.all(np.isfinite(X)):
        raise ValueError(f"missing dipole-window epochs for {v}")
    amp = n45_fit(X.mean(axis=0), L, (0, b - a))
    if not n_boot:
        return amp, None
    n = len(X)
    C = (rng or np.random.default_rng(0)).multinomial(n, np.full(n, 1.0 / n), size=n_boot)
    return amp, n45_fit((C @ X.reshape(n, -1) / n).reshape(n_boot, X.shape[1], b - a), L, (0, b - a))


def subset_means(pool, k, n_draw, rng):
    """Means of n_draw random subsets of k rows of pool (without replacement), as one matrix product."""
    pick = np.argpartition(rng.random((n_draw, len(pool))), k, axis=1)[:, :k]
    S = np.zeros((n_draw, len(pool)))
    np.put_along_axis(S, pick, 1.0, axis=1)
    return (S @ pool.reshape(len(pool), -1) / k).reshape((n_draw,) + pool.shape[1:])


# %% 6 · Figures: style and helpers ------------------------------------------------------------------------------------
INK, INK_2, INK_MUTED, GRID_C, BASE_C = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
WASH = "#f3f2ee"                   # gate / N45 window shading
BAND_C = "#d9d7cf"                 # range of the calm pulses
REJ_C = "#b02a2a"                  # text colour of "rejected" labels (always with the word, never colour alone)
CAT_ORDER = ["open_loop", "predetermined_single", "prime_single_pulse", "prime_triplet"]
CAT_LABEL = {"open_loop": "open-loop single (baseline, calibration, evaluation)",
             "predetermined_single": "predetermined single (intervention)",
             "prime_single_pulse": "PRIME single (intervention)", "prime_triplet": "PRIME triplet (intervention)"}
CAT_SHORT = {"open_loop": "open-loop", "predetermined_single": "predet.", "prime_single_pulse": "PRIME",
             "prime_triplet": "triplet"}
BLOCK_SHORT = {"baseline": "base", "calibration": "cal"}
# Validated with the dataviz validator (light surface): categories as PRIME_online_performance.ipynb, eye ICs the first
# three categorical slots (all pairs pass), cutoffs and z bands one-hue ordinal ramps (stricter / higher z = darker).
# Four sessions do not fit a categorical palette that passes all pairs, so sessions are told apart by marker + legend.
CAT_COLOR = {"open_loop": "#1baf7a", "predetermined_single": "#5b8ff9", "prime_single_pulse": "#e8684a",
             "prime_triplet": "#6250d6"}
CAT_MARKER = {"open_loop": "o", "predetermined_single": "s", "prime_single_pulse": "D", "prime_triplet": "^"}
IC_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
_RAMP = ["#0d366b", "#184f95", "#1c5cab", "#2a78d6", "#3987e5", "#6da7ec", "#86b6ef"]
CUT_COLOR = {z: _RAMP[int(round(i * (len(_RAMP) - 1) / max(len(Z_GRID) - 1, 1)))] for i, z in enumerate(sorted(Z_GRID))}
HEAD_C = "#595959"                 # the current decider (dashed)
HEOG_C = "#6250d6"                 # F7 - F8 trace (violet: apart from the eye-IC colours of the panel below)
BAND_COLORS = ["#86b6ef", "#5598e7", "#256abf", "#184f95", "#0d366b"]     # ordinal ramp, validated (5 steps)
POOLED_C = "#2a78d6"
SESSION_MARKER = {"pilot003": "o", "pilot004": "s", "pilot005": "D", "pilot006": "^", "pilot007": "v", "pilot008": "P"}
BAND_LABELS = ([f"≤ {Z_GRID[0]:g}"] + [f"{a:g}–{b:g}" for a, b in zip(Z_GRID[:-1], Z_GRID[1:])]
               + [f"> {Z_GRID[-1]:g}"])


def band_meaning(b):
    if b == 0:
        return "kept by every cutoff"
    if b == len(Z_GRID):
        return "rejected by every cutoff"
    return f"rejected at z ≤ {Z_GRID[b - 1]:g}, kept at z ≥ {Z_GRID[b]:g}"


def setup_mpl():
    import matplotlib
    if "ipykernel" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 130, "savefig.bbox": "tight", "font.size": 9,
        "axes.titlesize": 9.5, "axes.labelsize": 9, "axes.edgecolor": BASE_C, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID_C, "grid.linewidth": 0.6,
        "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round", "legend.frameon": False, "legend.fontsize": 8, "figure.facecolor": "white",
        "axes.titlecolor": INK, "text.color": INK, "axes.titlelocation": "left",
    })
    return plt


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "blue_gray_red", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f4a9a8", "#e34948", "#8a1c1b"])


def sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "blue_seq", ["#f6f8fb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


_INFO_CACHE = {}


def topo_info(ch_names):
    import mne
    key = tuple(ch_names)
    if key not in _INFO_CACHE:
        info = mne.create_info(list(ch_names), STORE_SFREQ, "eeg")
        info.set_montage(mne.channels.make_standard_montage("standard_1005"))
        _INFO_CACHE[key] = info
    return _INFO_CACHE[key]


def draw_topo(ax, values, info, vmax, cmap=None, vmin=None):
    import mne
    im, _ = mne.viz.plot_topomap(np.asarray(values, dtype=float), info, axes=ax, show=False,
                                 cmap=cmap or diverging_cmap(), vlim=(-vmax if vmin is None else vmin, vmax),
                                 contours=0, sensors=True, res=48, extrapolate="head", outlines="head")
    return im


def page(H, top_in=0.95, bottom_in=0.4):
    """Gridspec margins in inches (figures with many topomaps are laid out by hand)."""
    return dict(top=1 - top_in / H, bottom=bottom_in / H)


def gate_mask_ms(times_ms):
    t = np.asarray(times_ms)
    return (t >= GATE_WINDOW_MS[0]) & (t <= GATE_WINDOW_MS[1])


def robust_sd(x):
    x = np.asarray(x, dtype=float)
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def robust_line(z):
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    return float(np.median(z) + ROBUST_K * robust_sd(z))


def pct_above(z, c):
    z = np.asarray(z, dtype=float)
    return 100.0 * np.mean(z > c) if len(z) else np.nan


def wilson(k, n, zc=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + zc ** 2 / n
    c = (p + zc ** 2 / (2 * n)) / d
    h = zc * np.sqrt(p * (1 - p) / n + zc ** 2 / (4 * n ** 2)) / d
    return 100 * (c - h), 100 * (c + h)


def label_trial(r):
    return f"{BLOCK_SHORT.get(r.block, r.block)} #{int(r.trial_in_stage)}"


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  wrote {path.relative_to(OUT)}")


def eog_traces(res, stems, which="veog"):
    """Fp mean (veog) or F7-F8 (heog): the raw pre window (-210..-10 ms, average reference over the good channels,
    shifted to meet the gate epoch on -30..-15 ms) and the gate epoch itself (-30..100 ms, before ICA)."""
    meta = res["meta"]
    t_post, t_pre = np.asarray(meta["times_epoch_ms"]), np.asarray(meta["times_pre_win_ms"])
    v_post = load_arrays(res, which, stems).astype(float)
    v_pre = load_arrays(res, f"{which}_pre", stems).astype(float)
    ov_post, ov_pre = (t_post >= -30) & (t_post <= -15), (t_pre >= -30) & (t_pre <= -15)
    v_pre = v_pre - v_pre[:, ov_pre].mean(axis=1, keepdims=True) + v_post[:, ov_post].mean(axis=1, keepdims=True)
    return t_pre, v_pre, t_post, v_post


def calm_band(res, which="veog"):
    """5-95 % range of the traces of the calm single pulses (the reference band in the trace panels)."""
    key = f"band_{which}"
    if key not in res:
        w = res["w"]
        t_pre, v_pre, t_post, v_post = eog_traces(res, w.stem[w.calm], which)
        res[key] = (t_pre, np.percentile(v_pre, [5, 95], axis=0), t_post, np.percentile(v_post, [5, 95], axis=0))
    return res[key]


def draw_eog(ax, res, stem_traces, band=True, lw=1.0):
    """stem_traces: {"veog": (v_pre, v_post), "heog": (...)} of one or more pulses."""
    t_pre, b_pre, t_post, b_post = calm_band(res, "veog")
    ax.axvspan(*GATE_WINDOW_MS, color=WASH, lw=0, zorder=0)
    if band:
        ax.fill_between(t_pre, b_pre[0], b_pre[1], color=BAND_C, lw=0, zorder=1)
        ax.fill_between(t_post, b_post[0], b_post[1], color=BAND_C, lw=0, zorder=1)
    for which, color, ls in (("heog", HEOG_C, "-"), ("veog", INK, "-")):
        if which not in stem_traces:
            continue
        vp, vq = stem_traces[which]
        for a, b in zip(np.atleast_2d(vp), np.atleast_2d(vq)):
            ax.plot(t_pre, a, color=color, lw=lw * 0.8, alpha=0.55, ls=ls, zorder=2)
            ax.plot(t_post, b, color=color, lw=lw, alpha=0.95, ls=ls, zorder=3)
    ax.axvline(0, color=BASE_C, lw=0.8, zorder=1)
    ax.set_xlim(-210, 100)


def cut_line_handles(plt, with_head=True, head_z=3.0):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=CUT_COLOR[z], lw=1.6, label=f"window reference, cutoff z {z:g}") for z in Z_GRID]
    if with_head:
        h.append(Line2D([], [], color=HEAD_C, lw=1.4, ls="--", label=f"current decider (whole epoch), z {head_z:g}"))
    return h


def x_axis_cutoffs(ax, head=True):
    """x positions: the cutoffs, then the current decider ("now") after a gap."""
    x_head = Z_GRID[-1] + 0.8
    ticks = list(Z_GRID) + ([x_head] if head else [])
    ax.set_xticks(ticks, [f"{z:g}" for z in Z_GRID] + (["now"] if head else []))
    if head:
        ax.axvline((Z_GRID[-1] + x_head) / 2, color=GRID_C, lw=1.0)
    ax.set_xlim(Z_GRID[0] - 0.3, x_head + 0.3 if head else Z_GRID[-1] + 0.3)
    return x_head


# %% 7 · Analysis: tables ----------------------------------------------------------------------------------------------
GROUPS = [("open-loop singles", lambda w: w.category == "open_loop"),
          ("predetermined singles", lambda w: w.category == "predetermined_single"),
          ("PRIME singles", lambda w: w.category == "prime_single_pulse"),
          ("PRIME triplets", lambda w: w.category == "prime_triplet"),
          ("intervention singles", lambda w: w.single & w.block.str.startswith("iv")),
          ("all single pulses", lambda w: w.single)]


def analyze(D):
    """Every table of section 7 as DataFrames (written by write_tables, drawn by the figures)."""
    rng = np.random.default_rng(0)
    A = {k: [] for k in ("checks", "rejections", "blocks", "calibration", "eye_ics", "frontal_by_z", "bands", "evoked",
                         "equivalent")}
    for s, res in D.items():
        w, meta = res["w"], res["meta"]
        for name, v in flatten_checks(meta["checks"]):
            A["checks"].append(dict(part="variants", subject=s, check=name, ok=v["ok"], detail=v["detail"]))
        if res["online_meta"]:
            for name, v in flatten_checks(res["online_meta"]["checks"]):
                A["checks"].append(dict(part="online", subject=s, check=name, ok=v["ok"], detail=v["detail"]))
        # the parent's N45 fit (for averages) == the decider's fit_trial on every stored single pulse
        for v in res["variants"]:
            X = load_arrays(res, f"dip_{v}")
            rd = np.abs(n45_fit(X, *fit_of(res, v)) / w[f"amp__{v}"].to_numpy() - 1)
            A["checks"].append(dict(part="analysis", subject=s, check=f"{v}: n45_fit() == the decider's fit_trial "
                                    "(float32 epochs), max rel. diff", ok=bool(np.nanmax(rd) < 1e-4),
                                    detail=f"{np.nanmax(rd):.1e}; {int((rd < 1e-6).sum())}/{len(rd)} below 1e-6"))
        # --- rejections per group and block
        for v in res["variants"]:
            reason = w[f"reason__{v}"]
            for g, sel in GROUPS:
                m = sel(w).to_numpy()
                A["rejections"].append(dict(subject=s, variant=v, z_threshold=res["thr"][v], group=g, n=int(m.sum()),
                                            pct_ocular=100 * float((reason[m] == "ocular").mean()),
                                            pct_any_gate=100 * float((reason[m] != "").mean()),
                                            n_kept=int((reason[m] == "").sum())))
            for b in dict.fromkeys(w.block):
                m = (w.single & (w.block == b)).to_numpy()
                A["blocks"].append(dict(subject=s, variant=v, z_threshold=res["thr"][v], block=b, n_single=int(m.sum()),
                                        pct_ocular=100 * float((reason[m] == "ocular").mean()),
                                        n_kept=int((reason[m] == "").sum())))
        # --- calibration per variant, against the fixed processing (AMP_VARIANT)
        bundles = {v: np.load(res["dir"] / f"calibration_{v}.npy", allow_pickle=True).item()["bundle"]
                   for v in res["variants"]}
        ref_b, rr_ref = bundles[AMP_VARIANT], res["fit"].get(f"rr_mm_{AMP_VARIANT}")
        for v in res["variants"]:
            vm, b = meta["variants"][v], bundles[v]
            rr = res["fit"].get(f"rr_mm_{v}")
            both = (w[f"kept__{v}"] & w[f"kept__{AMP_VARIANT}"]).to_numpy()
            ad = np.abs(w[f"amp__{v}"][both] / w[f"amp__{AMP_VARIANT}"][both] - 1)
            A["calibration"].append(dict(
                subject=s, variant=v, z_threshold=vm["z_threshold"],
                reference_ms=f"{vm['ref_window_ms'][0]:.1f}–{vm['ref_window_ms'][1]:.1f}", n_reference_samples=vm["n_ref_samples"],
                **{f"dropped_{k}": len(vm["dropped"][k]) for k in ("ocular", "qc", "post")},
                n_kept_for_prime=vm["n_kept"], dropped_ocular_trials=" ".join(map(str, vm["dropped"]["ocular"])),
                **{f"IC{ic}_mean_sd": f"{vm['eye_stats'][str(ic)]['mean']:.3f} / {vm['eye_stats'][str(ic)]['std']:.3f}"
                   for ic in meta["eye_ics"]},
                post_mad_mean=vm["post_mad_mean_std"][0], post_mad_sd=vm["post_mad_mean_std"][1],
                dipole_position=vm["position_index"], dipole_index_range=f"{vm['index_range'][0]}–{vm['index_range'][1]}",
                dipole_xyz_mm=" ".join(f"{x:.0f}" for x in rr) if rr is not None else "",
                dipole_shift_mm_vs_ref=float(np.linalg.norm(rr - rr_ref)) if rr is not None and rr_ref is not None else np.nan,
                sound_filter_rel_max_diff_vs_ref=float(np.max(np.abs(b["sound_filter"] - ref_b["sound_filter"]))
                                                       / np.max(np.abs(ref_b["sound_filter"]))),
                sspsir_kernel_rel_max_diff_vs_ref=float(np.max(np.abs(b["sspsir_filter_kernel"] - ref_b["sspsir_filter_kernel"]))
                                                        / np.max(np.abs(ref_b["sspsir_filter_kernel"]))),
                single_n45_median_abs_rel_diff_vs_ref=float(np.median(ad)) if len(ad) else np.nan,
                single_n45_p95_abs_rel_diff_vs_ref=float(np.percentile(ad, 95)) if len(ad) else np.nan))
        wv = [v for v in WINDOW_VARIANTS if v in res["variants"]][0]
        lab, P = res["cal"].get("iclabel_labels"), res["cal"].get("iclabel_proba_all")
        for j, ic in enumerate(meta["eye_ics"]):
            A["eye_ics"].append(dict(subject=s, ic=ic, p_eye=float(P[ic, ICLABEL_CLASSES.index("eye blink")])
                                     if P is not None and np.size(P) else np.nan,
                                     label=str(lab[ic]) if lab is not None and len(lab) > ic else "",
                                     whole_epoch_mean=meta["variants"]["current"]["eye_stats"][str(ic)]["mean"],
                                     whole_epoch_sd=meta["variants"]["current"]["eye_stats"][str(ic)]["std"],
                                     window_mean=meta["variants"][wv]["eye_stats"][str(ic)]["mean"],
                                     window_sd=meta["variants"][wv]["eye_stats"][str(ic)]["std"],
                                     pct_single_argmax=100 * float((w[f"z_argmax_ic__{wv}"][w.single] == ic).mean())))
        # --- frontal evidence over window ocular z, and the bands between the cutoffs (variant-independent)
        sg = w[w.single]
        A["frontal_by_z"] += frontal_curve(s, sg)
        calm_ok = sg[sg.calm & sg.amp_ok]
        L, (ia, ib) = fit_of(res, AMP_VARIANT)
        pool = load_arrays(res, f"dip_{AMP_VARIANT}", list(calm_ok.stem))[:, :, ia:ib].astype(float)
        for bnd in range(len(Z_GRID) + 1):
            d = sg[sg.band == bnd]
            row = band_row(s, bnd, d, rng)
            dd = d[d.amp_ok]
            if len(dd) >= MIN_BAND_N and len(pool) > len(dd):
                xb = load_arrays(res, f"dip_{AMP_VARIANT}", list(dd.stem))[:, :, ia:ib].astype(float)
                a_band = n45_fit(xb.mean(axis=0), L, (0, ib - ia))
                null = n45_fit(subset_means(pool, len(dd), N_BOOT, rng), L, (0, ib - ia))
                row.update(evoked_n45_band=a_band, evoked_n45_matched_calm_median=float(np.median(null)),
                           evoked_ratio_to_matched_calm=a_band / float(np.median(null)),
                           evoked_ratio_null_lo=float(np.percentile(null, 2.5) / np.median(null)),
                           evoked_ratio_null_hi=float(np.percentile(null, 97.5) / np.median(null)),
                           evoked_percentile_in_matched_calm=100 * float(np.mean(null < a_band)))
            A["bands"].append(row)
        # --- evoked N45 per block of each variant's kept pulses: with the variant's own cleaning and dipole (what that
        #     cutoff would give, calibration included) and with the fixed processing (only the trial selection differs)
        blocks = [b for b in BLOCK_ORDER if b in set(w.block)]
        for v in res["variants"]:
            base, rng_v = {}, np.random.default_rng(1)
            for b in blocks:
                d = w[w.single & (w.block == b) & w[f"kept__{v}"]]
                row = dict(subject=s, variant=v, z_threshold=res["thr"][v], block=b, n_kept_single=len(d),
                           single_n45_mean=float(d[f"amp__{v}"].mean()) if len(d) else np.nan)
                for tag, pv in (("own", v), ("fixed", AMP_VARIANT)):
                    amp, boots = evoked_n45(res, pv, d.stem, n_boot=N_BOOT if b in OPEN_LOOP_BLOCKS else 0, rng=rng_v)
                    row[f"evoked_n45_{tag}"] = amp
                    if b == "baseline":
                        base[tag] = (amp, boots)
                    elif b in OPEN_LOOP_BLOCKS and tag in base and base[tag][0]:
                        row[f"change_{tag}_pct"] = 100 * (amp / base[tag][0] - 1)
                        if boots is not None and base[tag][1] is not None:
                            ch = 100 * (boots / base[tag][1] - 1)
                            row[f"change_{tag}_ci_lo"], row[f"change_{tag}_ci_hi"] = (float(x) for x in
                                                                                    np.percentile(ch, [2.5, 97.5]))
                A["evoked"].append(row)
        # --- equivalent cutoffs between the window and the current scale (same % of single pulses rejected)
        zs_w, zs_c = sg.z_win.to_numpy(), sg.z_cur.to_numpy()
        for c in Z_GRID:
            q = np.mean(zs_w > c)
            A["equivalent"].append(dict(subject=s, scale="window", cutoff=c, pct_single_above=100 * q,
                                        same_pct_on_other_scale=float(np.quantile(zs_c, 1 - q)) if 0 < q < 1 else np.nan))
        c_now = res["thr"].get("current", np.nan)
        q = np.mean(zs_c > c_now)
        A["equivalent"].append(dict(subject=s, scale="current (whole epoch)", cutoff=c_now, pct_single_above=100 * q,
                                    same_pct_on_other_scale=float(np.quantile(zs_w, 1 - q)) if 0 < q < 1 else np.nan))
    # pooled (all sessions' single pulses together, each flagged against its own session's calm pulses)
    if D:
        allw = pd.concat([D[s]["w"].assign(subject=s) for s in D], ignore_index=True)
        sg = allw[allw.single]
        A["frontal_by_z"] += frontal_curve("pooled", sg)
        for bnd in range(len(Z_GRID) + 1):
            A["bands"].append(band_row("pooled", bnd, sg[sg.band == bnd], rng))
    A = {k: pd.DataFrame(v) for k, v in A.items()}
    A["decision_table"] = decision_table(D, A)
    return A


def frontal_curve(s, sg):
    """% of single pulses with frontal evidence per bin of window ocular z (FRONTAL_BIN_EDGES; open ends)."""
    e = FRONTAL_BIN_EDGES
    idx = np.digitize(sg.z_win, e)
    rows = []
    for b in range(len(e) + 1):
        d = sg[idx == b]
        if not len(d):
            continue
        k, n = int(d.frontal.sum()), len(d)
        lo, hi = wilson(k, n)
        rows.append(dict(subject=s, z_from=-np.inf if b == 0 else e[b - 1], z_to=np.inf if b == len(e) else e[b],
                         z_median=float(d.z_win.median()), n_single=n, pct_frontal=100 * k / n, ci_lo=lo, ci_hi=hi,
                         pct_frontal_pre=100 * float(d.frontal_pre.mean())))
    return rows


def band_row(s, bnd, d, rng):
    dd = d[d.amp_ok]
    k, n = int(d.frontal.sum()), len(d)
    lo, hi = wilson(k, n)
    row = dict(subject=s, band=bnd, band_label=BAND_LABELS[bnd], meaning=band_meaning(bnd), n_single=n,
               n_open_loop=int((d.category == "open_loop").sum()), n_intervention=int(d.block.str.startswith("iv").sum()),
               pct_frontal=100 * k / n if n else np.nan, pct_frontal_ci_lo=lo, pct_frontal_ci_hi=hi,
               pct_frontal_strong=100 * float(d.frontal_strong.mean()) if n else np.nan,
               pct_frontal_pre=100 * float(d.frontal_pre.mean()) if n else np.nan,
               veog_ptp_median_uV=float(d.veog_gate_ptp.median()) if n else np.nan,
               heog_ptp_median_uV=float(d.heog_gate_ptp.median()) if n else np.nan,
               n_n45=len(dd))
    if len(dd) >= MIN_BAND_N:
        x = dd.rel_amp.to_numpy()
        boots = np.median(x[rng.integers(0, len(x), size=(N_BOOT, len(x)))], axis=1)
        row.update(rel_n45_median=float(np.median(x)), rel_n45_ci_lo=float(np.percentile(boots, 2.5)),
                   rel_n45_ci_hi=float(np.percentile(boots, 97.5)), rel_n45_robust_sd=robust_sd(x))
    return row


def decision_table(D, A):
    """Per session and cutoff: what it costs and what it lets through. 'pooled': mean over sessions for rates and
    changes, sum for counts (n_ ..., calibration_dropped_ocular), minimum for the fewest-trials columns."""
    rows = []
    rej, blk, ev = A["rejections"], A["blocks"], A["evoked"]
    for s, res in D.items():
        w = res["w"]
        sg = w[w.single]
        for v in res["variants"]:
            r = rej[(rej.subject == s) & (rej.variant == v)].set_index("group")
            b = blk[(blk.subject == s) & (blk.variant == v) & blk.block.isin(OPEN_LOOP_BLOCKS)]
            e = ev[(ev.subject == s) & (ev.variant == v)].set_index("block")
            ev_blocks = [b_ for b_ in EVAL_BLOCKS if b_ in e.index]
            vm = res["meta"]["variants"][v]
            rej_oc = sg[sg[f"reason__{v}"] == "ocular"]
            kept = sg[sg[f"kept__{v}"]]
            rows.append(dict(
                subject=s, variant=v, cutoff=res["thr"][v], reference="whole epoch" if v == "current" else "15–100 ms",
                pct_singles_rejected_ocular=r.loc["all single pulses", "pct_ocular"],
                pct_open_loop_rejected_ocular=r.loc["open-loop singles", "pct_ocular"],
                pct_triplets_rejected_ocular=r.loc["PRIME triplets", "pct_ocular"],
                pct_intervention_singles_without_label=r.loc["intervention singles", "pct_any_gate"],
                calibration_dropped_ocular=len(vm["dropped"]["ocular"]), calibration_trials_for_prime=vm["n_kept"],
                fewest_kept_open_loop_block=int(b.n_kept.min()) if len(b) else np.nan,
                fewest_kept_block=str(b.loc[b.n_kept.idxmin(), "block"]) if len(b) else "",
                pct_rejected_with_frontal_evidence=100 * float(rej_oc.frontal.mean()) if len(rej_oc) else np.nan,
                pct_kept_with_frontal_evidence=100 * float(kept.frontal.mean()),
                pct_calm_with_frontal_evidence=100 * res["evidence"]["calm_rate"],
                n_kept_with_strong_evidence=int(kept.frontal_strong.sum()),
                kept_rel_n45_median=float(kept.rel_amp[kept.amp_ok].median()),
                evoked_change_t0_pct=e.loc["t0", "change_own_pct"] if "t0" in e.index else np.nan,
                evoked_change_mean_t0_t60_pct=float(e.loc[ev_blocks, "change_own_pct"].mean()),
                evoked_change_mean_t0_t60_fixed_pct=float(e.loc[ev_blocks, "change_fixed_pct"].mean())))
    df = pd.DataFrame(rows)
    if len(df):
        pooled = []
        for v, g in df.groupby("variant", sort=False):
            r = dict(subject="pooled", variant=v, cutoff=g.cutoff.iloc[0] if g.cutoff.nunique() == 1 else np.nan,
                     reference=g.reference.iloc[0])
            for c in df.columns[4:]:
                if c == "fewest_kept_block":
                    continue
                if c.startswith("n_") or c.startswith("calibration_dropped"):
                    r[c] = g[c].sum()
                elif c.startswith("fewest") or c == "calibration_trials_for_prime":
                    r[c] = g[c].min()
                else:
                    r[c] = g[c].mean()
            pooled.append(r)
        df = pd.concat([df, pd.DataFrame(pooled)], ignore_index=True)
    return df

def write_tables(D, A):
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    for name, df in A.items():
        df.to_csv(OUT / "tables" / f"{name}.csv", index=False)
    for s, res in D.items():
        res["w"].to_csv(OUT / "tables" / f"pulses_{s}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 60, "display.max_colwidth", 120,
                           "display.float_format", "{:.3g}".format):
        ch = A["checks"]
        bad = ch[~ch.ok.astype(bool)]
        print(f"\nChecks: {len(ch) - len(bad)}/{len(ch)} pass" + ("" if not len(bad) else ":"))
        for r in bad.itertuples():
            print(f"  FAIL {r.part} {r.subject} {r.check}: {r.detail}")
        dec = A["decision_table"]
        cols = [c for c in dec.columns if c not in ("subject", "variant", "reference", "fewest_kept_block")]
        print("\nDecision table (pooled: mean over sessions, counts summed, fewest = minimum):")
        print(dec[dec.subject == "pooled"].set_index("variant")[cols].T.to_string())
        for s in D:
            print(f"\n{s}:")
            print(dec[dec.subject == s].set_index("variant")[cols].T.to_string())
        print("\nBands of window ocular z (single pulses):")
        print(A["bands"].drop(columns=["meaning"]).to_string(index=False))
        print("\nEvoked N45 per open-loop block (change vs baseline, %):")
        e = A["evoked"]
        for tag, what in (("own", "each cutoff with its own calibration"), ("fixed", f"{AMP_VARIANT} processing")):
            print(f"  {what}:")
            print(e[e.block.isin(EVAL_BLOCKS)].pivot_table(index=["subject", "block"], columns="variant",
                                                           values=f"change_{tag}_pct", sort=False).round(1).to_string())
        print("\nEquivalent cutoffs (same % of single pulses):")
        print(A["equivalent"].to_string(index=False))
    print(f"\ntables written to {OUT / 'tables'}")


# %% 8 · Figures --------------------------------------------------------------------------------------------------------

def fig_reference(D):
    """1 · How the criterion works with the window reference."""
    plt = setup_mpl()
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    subs = [s for s in SUBJECTS if s in D and D[s]["sources"] is not None]
    if not subs:
        return
    wv = WINDOW_VARIANTS[0]
    n_ic = max(len(D[s]["sources"]["eye_ics"]) for s in subs)
    fig = plt.figure(figsize=(3.9 * n_ic + 5.4, 3.0 * len(subs) + 1.9), layout="constrained")
    gs = fig.add_gridspec(len(subs), n_ic + 1, width_ratios=[1.35] + [1] * n_ic)
    z_now = None
    for i, s in enumerate(subs):
        res = D[s]
        src, meta, w = res["sources"], res["meta"], res["w"]
        t_ms = 1e3 * np.asarray(src["times"])
        S = np.abs(np.asarray(src["src"], dtype=float))           # (calibration trials, eye ICs, samples)
        gate = np.asarray(src["gate_idx"])
        eye = [int(k) for k in src["eye_ics"]]
        z_now = res["thr"]["current"]
        ax = fig.add_subplot(gs[i, 0])
        aw = meta["artifact_window_1_s"]
        ax.axvspan(t_ms[gate[0]], t_ms[gate[-1]], color=WASH, lw=0, zorder=0)
        ax.axvspan(1e3 * aw[0], 1e3 * aw[1], color=BAND_C, lw=0, zorder=0)
        top = 0.0
        for j, ic in enumerate(eye):
            p90 = np.percentile(S[:, j], 90, axis=0)
            ax.plot(t_ms, np.median(S[:, j], axis=0), color=IC_COLORS[j % 3], lw=1.6, label=f"IC {ic}: median")
            ax.plot(t_ms, p90, color=IC_COLORS[j % 3], lw=0.9, ls=":", label=f"IC {ic}: 90th percentile")
            top = max(top, float(p90.max()))
        ax.set_ylim(0, top * 1.28)
        ax.text(0.5 * (1e3 * aw[0] + 1e3 * aw[1]), top * 1.2, "blank", ha="center", va="top", fontsize=7.5, color=INK_2)
        ax.text(0.5 * (t_ms[gate[0]] + t_ms[gate[-1]]), top * 1.2, "15–100 ms: the score's samples\n= the reference here",
                ha="center", va="top", fontsize=7.5, color=INK_2)
        ax.text(t_ms[0] + 1, top * 1.2, "reference now:\n−30 to 100 ms", ha="left", va="top", fontsize=7.5, color=INK_2)
        ax.axvline(0, color=BASE_C, lw=0.8)
        ax.set_xlim(t_ms[0], t_ms[-1])
        ax.set_ylabel("|s| (IC units)")
        ax.set_xlabel("ms from the pulse")
        ax.legend(fontsize=6.8, loc="center left", ncol=1)
        ax.set_title(f"{s} · |s(t)| of the eye ICs over the {S.shape[0]} calibration trials", fontsize=8.5)
        for j, ic in enumerate(eye):
            ax = fig.add_subplot(gs[i, j + 1])
            stc, stw = meta["variants"]["current"]["eye_stats"][str(ic)], meta["variants"][wv]["eye_stats"][str(ic)]
            whole, win = S[:, j].ravel(), S[:, j][:, gate].ravel()
            zc = w[f"z_eye{j + 1}__current"][w.single].to_numpy()
            zw = w[f"z_eye{j + 1}__{wv}"][w.single].to_numpy()
            med = stw["mean"] + zw * stw["std"]                     # median |s| over 15-100 ms of each single pulse
            cut_w = {z: stw["mean"] + z * stw["std"] for z in Z_GRID}
            cut_c = stc["mean"] + z_now * stc["std"]
            xmax = 1.12 * max(np.percentile(win, 99), max(cut_w.values()), cut_c)
            bins = np.linspace(0, xmax, 70)
            ax.hist(np.clip(med, 0, xmax), bins, density=True, color=BAND_C, lw=0)
            ax.hist(np.clip(whole, 0, xmax), bins, density=True, histtype="step", color=INK_MUTED, lw=1.2)
            ax.hist(np.clip(win, 0, xmax), bins, density=True, histtype="step", color=INK, lw=1.4)
            ax.set_yscale("log")
            for z, x in cut_w.items():
                ax.axvline(x, color=CUT_COLOR[z], lw=1.5)
            ax.axvline(cut_c, color=HEAD_C, lw=1.3, ls="--")
            ax.set_xlim(0, xmax)
            pct = " · ".join(f"{z:g}: {pct_above(zw, z):.1f}" for z in Z_GRID)
            ax.set_title(f"IC {ic} · reference now mean {stc['mean']:.2f}, SD {stc['std']:.2f} → here {stw['mean']:.2f}, "
                         f"{stw['std']:.2f}\n% of single pulses above, z {pct} · now {pct_above(zc, z_now):.1f}",
                         fontsize=7.6)
            ax.set_xlabel("|s| (values beyond the axis in the last bin)", fontsize=8)
            if j == 0:
                ax.set_ylabel("density (log scale)")
    handles = [Patch(color=BAND_C, label="single pulses: median |s| over 15–100 ms (what the gate compares)"),
               Line2D([], [], color=INK_MUTED, lw=1.2, label="reference now: calibration samples −30 to 100 ms"),
               Line2D([], [], color=INK, lw=1.4, label="reference here: calibration samples 15–100 ms"),
               *cut_line_handles(plt, head_z=z_now)]
    fig.legend(handles=handles, loc="outside lower center", ncol=4)
    fig.suptitle("The criterion with the window reference: an eye IC rejects a pulse when its median |s| over 15–100 ms "
                 "exceeds mean + z × SD of the reference (vertical lines: z = "
                 + ", ".join(f"{z:g}" for z in Z_GRID) + "; dashed: the decider now)", fontsize=10.5)
    save(fig, OUT / "figures" / "1_reference_window.png")


def fig_scores(D):
    """2 · Window ocular z of every pulse, per condition, with the four cutoffs."""
    plt = setup_mpl()
    from matplotlib.lines import Line2D
    subs = [s for s in SUBJECTS if s in D]
    fig, axes = plt.subplots(len(subs), 2, figsize=(13.5, 2.7 * len(subs) + 1.3), squeeze=False, layout="constrained",
                             gridspec_kw=dict(width_ratios=[2.3, 1]))
    edges = np.arange(-1.5, 8.0001, 0.25)
    for i, s in enumerate(subs):
        w = D[s]["w"]
        for j, (cats, title) in enumerate(((CAT_ORDER[:3], "single pulses"), (CAT_ORDER[3:], "PRIME triplets"))):
            ax = axes[i, j]
            sub = w[w.category.isin(cats)]
            bottom = np.zeros(len(edges) - 1)
            for c in cats:
                z = np.clip(sub.z_win[sub.category == c].to_numpy(), edges[0], edges[-1] - 1e-9)
                h, _ = np.histogram(z, edges)
                ax.bar(edges[:-1], h, width=np.diff(edges), bottom=bottom, align="edge", color=CAT_COLOR[c],
                       edgecolor="white", linewidth=0.6, label=CAT_LABEL[c] if i == 0 else None)
                bottom += h
            z_all = sub.z_win.to_numpy()
            for z in Z_GRID:
                ax.axvline(z, color=CUT_COLOR[z], lw=1.4)
            txt = f"n = {len(z_all)} · % above " + " · ".join(f"{z:g}: {pct_above(z_all, z):.1f}" for z in Z_GRID)
            if j == 0:
                rl = robust_line(z_all)
                ax.axvline(rl, color=INK_2, ls=":", lw=1.3)
                txt += f"\nmedian + {ROBUST_K:g}·robust SD = {rl:.2f} (dotted)"
            n_over = int((z_all > edges[-1]).sum())
            if n_over:
                txt += f"\n{n_over} above {edges[-1]:g} (max {z_all.max():.1f}), counted in the last bin"
            ax.text(0.99, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=7.8, color=INK_2)
            ax.set_xlim(edges[0], edges[-1])
            ax.set_xlabel("window ocular z")
            if j == 0:
                ax.set_ylabel("pulses")
            ax.set_title(f"{s} · {title}" + (f" · eye ICs {D[s]['meta']['eye_ics']}" if j == 0 else ""))
    h, l = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[0, 1].get_legend_handles_labels()
    cut = [Line2D([], [], color=CUT_COLOR[z], lw=1.4, label=f"cutoff {z:g}") for z in Z_GRID]
    fig.legend(h + h2 + cut, l + l2 + [c.get_label() for c in cut], loc="outside lower center", ncol=4)
    fig.suptitle("Window ocular z of every pulse (max over the eye ICs of the median z over 15–100 ms, reference "
                 "15–100 ms of the calibration block)", fontsize=11)
    save(fig, OUT / "figures" / "2_scores.png")


def _sessions_vs_cutoff(ax, dec, col, subs, pooled=True):
    x_head = x_axis_cutoffs(ax)
    for s in subs:
        d = dec[dec.subject == s].set_index("variant")
        mk = SESSION_MARKER.get(s, "o")
        ax.plot(Z_GRID, [d.loc[variant_name(z), col] for z in Z_GRID], color=INK_2, lw=0.9, marker=mk, ms=5,
                mfc="white", mew=1.1)
        if "current" in d.index:
            ax.plot([x_head], [d.loc["current", col]], color=INK_2, marker=mk, ms=5, mfc="white", mew=1.1, ls="none")
    if pooled:
        p = dec[dec.subject == "pooled"].set_index("variant")
        ax.plot(Z_GRID, [p.loc[variant_name(z), col] for z in Z_GRID], color=POOLED_C, lw=2.4, marker="o", ms=6,
                zorder=5)
        if "current" in p.index:
            ax.plot([x_head], [p.loc["current", col]], color=POOLED_C, marker="o", ms=6, ls="none", zorder=5)
    ax.set_xlabel("cutoff z, window reference (now: the decider as it runs)", fontsize=8)
    return x_head


def _sessions_by_band(ax, bands, col, subs, lo=None, hi=None):
    xb = np.arange(len(BAND_LABELS))
    for b in range(1, len(Z_GRID)):
        ax.axvspan(b - 0.45, b + 0.45, color=WASH, lw=0, zorder=0)
    for k, s in enumerate(subs):
        b = bands[bands.subject == s].set_index("band").reindex(xb)
        ax.plot(xb + (k - (len(subs) - 1) / 2) * 0.09, b[col], color=INK_2, marker=SESSION_MARKER.get(s, "o"), ms=5,
                mfc="white", mew=1.1, ls="none", zorder=3)
    p = bands[bands.subject == "pooled"].set_index("band").reindex(xb)
    if lo and hi and lo in p:
        ax.errorbar(xb, p[col], yerr=[p[col] - p[lo], p[hi] - p[col]], color=POOLED_C, lw=2.0, marker="o", ms=6,
                    capsize=3, zorder=4)
    ax.set_xticks(xb, [f"{lab}\nn {int(n)}" for lab, n in zip(BAND_LABELS, p.n_single.fillna(0))], fontsize=7.8)
    ax.set_xlabel("band of window ocular z (shaded: decided by the cutoff)", fontsize=8)
    return xb


def fig_decision(D, A):
    """3 · The judgement in one figure: what each cutoff costs (top) and what the pulses at stake look like (bottom)."""
    plt = setup_mpl()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    dec, bands = A["decision_table"], A["bands"]
    subs = [s for s in SUBJECTS if s in D]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.6), layout="constrained")
    top = [("pct_singles_rejected_ocular", "% of single pulses", "Single pulses rejected by the ocular gate", True),
           ("pct_intervention_singles_without_label", "% of intervention singles",
            "Intervention singles without a TEP label\n(any post gate: no PRIME finetuning)", True),
           ("calibration_trials_for_prime", "calibration trials (of 125)",
            "Calibration trials PRIME is trained on\n(after the ocular, pre- and post-MAD gates)", False),
           ("fewest_kept_open_loop_block", "single pulses (of 100)",
            "Fewest pulses kept in an open-loop block\n(baseline, t0–t60; all post gates)", False)]
    for ax, (col, ylab, title, pooled) in zip(axes[0], top):
        _sessions_vs_cutoff(ax, dec, col, subs, pooled=pooled)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9)
    # (e) how ocular the pulses are along window ocular z
    ax = axes[1, 0]
    fz = A["frontal_by_z"]
    lo_x, hi_x = FRONTAL_BIN_EDGES[0] - 0.3, FRONTAL_BIN_EDGES[-1] + 0.3
    x_of = lambda d: np.clip(d.z_median, lo_x, hi_x)
    for z in Z_GRID:
        ax.axvline(z, color=CUT_COLOR[z], lw=1.1)
    for s in subs:
        d = fz[(fz.subject == s) & (fz.n_single >= MIN_BAND_N)]
        ax.plot(x_of(d), d.pct_frontal, color=INK_MUTED, lw=0.8, marker=SESSION_MARKER.get(s, "o"), ms=3.5,
                mfc="white", mew=0.9)
    d = fz[(fz.subject == "pooled") & (fz.n_single >= MIN_BAND_N)]
    ax.fill_between(x_of(d), d.ci_lo, d.ci_hi, color=POOLED_C, alpha=0.18, lw=0)
    ax.plot(x_of(d), d.pct_frontal, color=POOLED_C, lw=2.2, marker="o", ms=4)
    calm = 100 * np.mean([D[s]["evidence"]["calm_rate"] for s in subs])
    ax.axhline(calm, color=INK_MUTED, ls="--", lw=1.0)
    ax.set_ylim(0, 100)
    ax.set_xlim(lo_x - 0.1, hi_x + 0.1)
    ax.set_xlabel(f"window ocular z (bin median; ≥ {MIN_BAND_N} pulses per bin)", fontsize=8)
    ax.set_ylabel("% of single pulses with frontal evidence")
    ax.set_title(f"How ocular are the pulses at each z?\nFp or F7−F8 peak-to-peak, 15–100 ms > calm {EVIDENCE_PCT:g}th pct",
                 fontsize=9)
    # (f) single-trial N45 per band
    ax = axes[1, 1]
    _sessions_by_band(ax, bands, "rel_n45_median", subs, "rel_n45_ci_lo", "rel_n45_ci_hi")
    ax.axhline(1.0, color=BASE_C, lw=0.9)
    ax.set_ylabel("single-trial N45 ÷ calm open-loop median")
    ax.set_title(f"Single-trial N45 per band (median, 95 % CI)\n{AMP_VARIANT} decider's cleaning and dipole", fontsize=9)
    # (g) evoked N45 of each band against equally large calm averages
    ax = axes[1, 2]
    xb = np.arange(len(BAND_LABELS))
    for b in range(1, len(Z_GRID)):
        ax.axvspan(b - 0.45, b + 0.45, color=WASH, lw=0, zorder=0)
    for k, s in enumerate(subs):
        bb = bands[bands.subject == s].set_index("band").reindex(xb)
        if "evoked_ratio_to_matched_calm" not in bb:
            continue
        x = xb + (k - (len(subs) - 1) / 2) * 0.12
        ax.vlines(x, bb.evoked_ratio_null_lo, bb.evoked_ratio_null_hi, color=BASE_C, lw=3.0, zorder=2)
        ax.plot(x, bb.evoked_ratio_to_matched_calm, color=INK_2, marker=SESSION_MARKER.get(s, "o"), ms=5.5, mfc="white",
                mew=1.2, ls="none", zorder=3)
    ax.axhline(1.0, color=BASE_C, lw=0.9)
    pb = bands[bands.subject == "pooled"].set_index("band").reindex(xb)
    ax.set_xticks(xb, [f"{lab}\nn {int(n)}" for lab, n in zip(BAND_LABELS, pb.n_single.fillna(0))], fontsize=7.8)
    ax.set_xlabel(f"band of window ocular z (bands with < {MIN_BAND_N} pulses left out)", fontsize=8)
    ax.set_ylabel("evoked N45 ÷ median of matched calm averages")
    ax.set_title(f"Evoked N45 of each band's average vs {N_BOOT}\naverages of as many calm pulses (grey: 95 % range)",
                 fontsize=9)
    # (h) the outcome
    ax = axes[1, 3]
    x_head = _sessions_vs_cutoff(ax, dec, "evoked_change_mean_t0_t60_pct", subs, pooled=True)
    p = dec[dec.subject == "pooled"].set_index("variant")
    ax.plot(Z_GRID, [p.loc[variant_name(z), "evoked_change_mean_t0_t60_fixed_pct"] for z in Z_GRID], color=POOLED_C,
            lw=1.6, ls="--", marker="o", ms=4, mfc="white")
    if "current" in p.index:
        ax.plot([x_head], [p.loc["current", "evoked_change_mean_t0_t60_fixed_pct"]], color=POOLED_C, marker="o", ms=4,
                mfc="white", ls="none")
    ax.axhline(0, color=BASE_C, lw=0.9)
    ax.set_ylabel("% change from baseline")
    ax.set_title("Outcome: evoked N45 change from baseline\n(mean of t0–t60; each cutoff its own calibration)", fontsize=9)
    handles = ([Line2D([], [], color=INK_2, marker=SESSION_MARKER.get(s, "o"), ms=5, mfc="white", lw=0.9, label=s)
                for s in subs]
               + [Line2D([], [], color=POOLED_C, lw=2.4, marker="o", ms=6, label="mean of the sessions / all pulses (95 % CI)"),
                  Line2D([], [], color=POOLED_C, lw=1.6, ls="--", marker="o", ms=4, mfc="white",
                         label=f"outcome with the {AMP_VARIANT} decider's processing for every cutoff"),
                  Line2D([], [], color=INK_MUTED, ls="--", lw=1.0, label="frontal evidence in the calm half of the pulses"),
                  Patch(color=WASH, label="bands decided by the choice of cutoff")]
               + [Line2D([], [], color=CUT_COLOR[z], lw=1.2, label=f"cutoff {z:g}") for z in Z_GRID])
    fig.legend(handles=handles, loc="outside lower center", ncol=6, fontsize=8)
    fig.suptitle("Which cutoff with the window reference? Top: what each cutoff costs · bottom: the pulses whose "
                 "decision depends on the cutoff, and the outcome", fontsize=11.5)
    save(fig, OUT / "figures" / "3_decision_summary.png")


def fig_band_cards(res):
    """4 · Example pulses from each band (evenly spread over the band's z), what the gate saw."""
    plt = setup_mpl()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    w, meta, s = res["w"], res["meta"], res["subject"]
    sg = w[w.single]
    rows = []
    for lo, hi in CARD_BANDS:
        d_all = sg[(sg.z_win > lo) & (sg.z_win <= hi)].sort_values("z_win")
        d = d_all.iloc[np.round(np.linspace(0, len(d_all) - 1, CARDS_PER_BAND)).astype(int)] \
            if len(d_all) > CARDS_PER_BAND else d_all
        rows.append((lo, hi, d.drop_duplicates("stem"), d_all))
    stems = [st for *_, d, _ in rows for st in d.stem]
    if not stems:
        return
    gm, src = load_arrays(res, "gate_mean", stems), load_arrays(res, "src", stems)
    tv, th = eog_traces(res, stems, "veog"), eog_traces(res, stems, "heog")
    pos = {st: i for i, st in enumerate(stems)}
    info = topo_info(meta["ch_names"])
    t_ep = np.asarray(meta["times_epoch_ms"])
    tg = gate_mask_ms(t_ep)
    wv, eye = WINDOW_VARIANTS[0], meta["eye_ics"]
    means = np.array([meta["variants"][wv]["eye_stats"][str(i)]["mean"] for i in eye])
    stds = np.array([meta["variants"][wv]["eye_stats"][str(i)]["std"] for i in eye])
    ncol, nrow = CARDS_PER_BAND, len(rows)
    H = 2.6 * nrow + 1.5
    fig = plt.figure(figsize=(3.5 * ncol + 1.9, H))
    outer = fig.add_gridspec(nrow, ncol + 1, width_ratios=[0.5] + [1] * ncol, hspace=0.66, wspace=0.24,
                             **page(H, 1.25, 0.75))
    for r, (lo, hi, d, d_all) in enumerate(rows):
        axl = fig.add_subplot(outer[r, 0])
        axl.axis("off")
        k = int(d_all.frontal.sum())
        b = int(np.digitize((lo + hi) / 2, Z_GRID, right=True))
        axl.text(0.98, 0.5, f"{lo:g} < z ≤ {hi:g}\n{len(d_all)} single pulses\n{k} ({100 * k / max(len(d_all), 1):.0f} %) "
                 f"frontal\n\n{band_meaning(b) if lo >= Z_GRID[0] else 'top of what every cutoff keeps'}",
                 ha="right", va="center", fontsize=7.8, transform=axl.transAxes, wrap=True)
        for c, rr in enumerate(d.itertuples()):
            n = pos[rr.stem]
            cell = outer[r, c + 1].subgridspec(2, 2, width_ratios=[1.0, 1.75], hspace=0.12, wspace=0.45)
            ax = fig.add_subplot(cell[:, 0])
            vmax = float(np.max(np.abs(gm[n]))) or 1.0
            draw_topo(ax, gm[n], info, vmax)
            ax.set_xlabel(f"±{vmax:.0f} µV", fontsize=6.5, color=INK_2, labelpad=1)
            axt = fig.add_subplot(cell[0, 1])
            draw_eog(axt, res, {"veog": (tv[1][n], tv[3][n]), "heog": (th[1][n], th[3][n])})
            axt.tick_params(labelbottom=False, labelsize=6.5)
            axt.set_ylabel("µV", fontsize=6.5)
            axt.set_title(f"{label_trial(rr)} · {CAT_SHORT[rr.category]} · z {rr.z_win:.2f}"
                          + (" · frontal" if rr.frontal else ""), fontsize=7.2, x=-0.62)
            axz = fig.add_subplot(cell[1, 1], sharex=axt)
            axz.axvspan(*GATE_WINDOW_MS, color=WASH, lw=0, zorder=0)
            for j in range(len(eye)):
                z = (np.abs(src[n][j]) - means[j]) / stds[j]
                axz.plot(t_ep, z, color=IC_COLORS[j % 3], lw=0.9)
                axz.hlines(np.median(z[tg]), *GATE_WINDOW_MS, color=IC_COLORS[j % 3], lw=2.2)
            for zc in (lo, hi):
                if zc in CUT_COLOR:
                    axz.hlines(zc, t_ep[0], t_ep[-1], color=CUT_COLOR[zc], lw=0.9)
            axz.axvline(0, color=BASE_C, lw=0.8)
            axz.set_ylabel("eye-IC z", fontsize=6.5)
            axz.tick_params(labelsize=6.5)
            if r == nrow - 1:
                axz.set_xlabel("ms from the pulse", fontsize=7)
    handles = [Patch(color=BAND_C, label="Fp mean: 5–95 % of the calm pulses"),
               Line2D([], [], color=INK, lw=1.2, label="Fp1/Fpz/Fp2 mean (VEOG)"),
               Line2D([], [], color=HEOG_C, lw=1.2, label="F7 − F8 (HEOG)"),
               *[Line2D([], [], color=IC_COLORS[j % 3], lw=1.6, label=f"IC {ic}: z(t), bar = median 15–100 ms")
                 for j, ic in enumerate(eye)],
               *[Line2D([], [], color=CUT_COLOR[z], lw=1.2, label=f"cutoff {z:g}") for z in Z_GRID]]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{s} · single pulses from each band of window ocular z ({CARDS_PER_BAND} spread evenly over the band)"
                 f"\nmap: 15–100 ms mean before ICA · top: VEOG / HEOG, faint = raw pre window · bottom: eye-IC z(t) on "
                 f"the window reference (1 kHz; the title gives the exact score)", y=1 - 0.22 / H, fontsize=9.5)
    save(fig, OUT / "figures" / f"4_band_cards_{s}.png")


def fig_ladder(res):
    """5 · Scalp maps of the epoch the gate sees, at increasing window ocular z, with the four cutoffs."""
    plt = setup_mpl()
    w, s, meta = res["w"], res["subject"], res["meta"]
    singles = w[w.single]
    rows, used = [], set()
    edges = [-np.inf] + list(Z_GRID) + [np.inf]
    for lv in LADDER_LEVELS:
        b = int(np.digitize(lv, Z_GRID, right=True))
        cand = singles[(singles.z_win > edges[b]) & (singles.z_win <= edges[b + 1])]
        d = cand.assign(dist=(cand.z_win - lv).abs()).sort_values("dist")
        pick = d[~d.stem.isin(used)].head(LADDER_PER_LEVEL)
        if not len(pick) or pick.dist.min() > 0.5:
            continue
        used.update(pick.stem)
        rows.append((lv, pick.sort_values("z_win")))
    if not rows:
        return
    stems = [st for _, p in rows for st in p.stem]
    gm = load_arrays(res, "gate_mean", stems)
    tv = eog_traces(res, stems, "veog")
    pos = {st: i for i, st in enumerate(stems)}
    info = topo_info(meta["ch_names"])
    k = LADDER_PER_LEVEL
    H = 1.62 * len(rows) + 1.3
    fig = plt.figure(figsize=(1.5 * k + 5.0, H))
    gs = fig.add_gridspec(len(rows), k + 4, width_ratios=[0.6] + [1] * k + [0.07, 0.62, 3.0], hspace=0.5,
                          wspace=0.08, **page(H, 1.05, 0.45))
    row_axes = []
    for i, (lv, pick) in enumerate(rows):
        maps = [gm[pos[st]] for st in pick.stem]
        vmax = float(np.percentile(np.abs(np.concatenate(maps)), 99)) or 1.0
        axl = fig.add_subplot(gs[i, 0])
        axl.axis("off")
        axl.text(0.95, 0.5, f"z ≈ {lv:g}", ha="right", va="center", fontsize=10, color=INK, transform=axl.transAxes)
        axs = []
        for j, (r, m) in enumerate(zip(pick.itertuples(), maps)):
            ax = fig.add_subplot(gs[i, j + 1])
            im = draw_topo(ax, m, info, vmax)
            ax.set_title(f"z {r.z_win:.2f}{' · F' if r.frontal else ''}\n{label_trial(r)} · {CAT_SHORT[r.category]}",
                         fontsize=6.8, loc="center")
            axs.append(ax)
        row_axes.append(axs)
        cb = fig.colorbar(im, cax=fig.add_subplot(gs[i, k + 1]))
        cb.ax.tick_params(labelsize=6.5)
        cb.set_label("µV", fontsize=7)
        axt = fig.add_subplot(gs[i, k + 3])
        idx = [pos[st] for st in pick.stem]
        draw_eog(axt, res, {"veog": (tv[1][idx], tv[3][idx])}, lw=0.9)
        axt.tick_params(labelsize=7)
        if i == 0:
            axt.set_title(f"{', '.join(VEOG_CHANNELS)} mean of these {k} pulses, µV (faint: raw pre window)\nband: 5–95 % "
                          f"of the calm pulses; shaded: gate window", fontsize=7.5)
        if i < len(rows) - 1:
            axt.tick_params(labelbottom=False)
        else:
            axt.set_xlabel("ms from the pulse", fontsize=8)
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    renderer = fig.canvas.get_renderer()
    for z in Z_GRID:
        for i in range(len(rows) - 1):
            if rows[i][0] <= z < rows[i + 1][0]:
                lo = row_axes[i][0].get_window_extent(renderer).transformed(inv).y0
                hi = row_axes[i + 1][0].title.get_window_extent(renderer).transformed(inv).y1
                y = (lo + hi) / 2
                fig.add_artist(plt.Line2D([0.02, 0.985], [y, y], transform=fig.transFigure, color=CUT_COLOR[z], lw=1.4))
                fig.text(0.02, y + 0.002, f"cutoff {z:g}", color=CUT_COLOR[z], va="bottom", fontsize=8.5)
                break
    fig.suptitle(f"{s} · single pulses at increasing window ocular z (F = frontal evidence): scalp map of the epoch the "
                 f"gate sees (before ICA), mean {GATE_WINDOW_MS[0]}–{GATE_WINDOW_MS[1]} ms, the {k} pulses nearest each "
                 f"level within its band; a pulse below a cutoff line is kept by that cutoff", y=1 - 0.2 / H,
                 fontsize=9.5)
    save(fig, OUT / "figures" / f"5_ladder_{s}.png")


def binned_stats(z, v, edges, fn, n_boot=500, min_n=8, seed=0):
    z, v = np.asarray(z, dtype=float), np.asarray(v, dtype=float)
    rng = np.random.default_rng(seed)
    idx = np.digitize(z, edges)
    out = []
    for b in range(len(edges) + 1):
        m = idx == b
        if m.sum() < min_n:
            continue
        x = v[m]
        boots = [fn(rng.choice(x, len(x))) for _ in range(n_boot)]
        out.append(dict(lo=-np.inf if b == 0 else edges[b - 1], hi=np.inf if b == len(edges) else edges[b],
                        z_med=float(np.median(z[m])), n=int(m.sum()), est=float(fn(x)),
                        ci_lo=float(np.percentile(boots, 2.5)), ci_hi=float(np.percentile(boots, 97.5))))
    return pd.DataFrame(out)


def amplitude_sweep(d):
    """Kept = window ocular z <= cutoff (MAD gates passed): mean and SEM of the relative N45, as ratios to keeping
    every pulse (no ocular gate)."""
    z, a = d.z_win.to_numpy(), d.rel_amp.to_numpy()
    ref_mean, ref_sem = np.mean(a), np.std(a, ddof=1) / np.sqrt(len(a))
    rows = []
    for c in SWEEP:
        k = a[z <= c]
        ok = len(k) > 2
        rows.append(dict(cutoff=c, n=len(k), kept_pct=100 * len(k) / len(a),
                         mean_ratio=np.mean(k) / ref_mean if ok else np.nan,
                         sem_ratio=(np.std(k, ddof=1) / np.sqrt(len(k))) / ref_sem if ok else np.nan))
    return pd.DataFrame(rows)


def _cut_lines(ax):
    for z in Z_GRID:
        ax.axvline(z, color=CUT_COLOR[z], lw=1.1)


def _sweep_limits(ax, sw, x0=0.5, x1=6.0):
    """Axis limits from the part of the sweep that is drawn (very low cutoffs keep few pulses and explode the SEM)."""
    v = sw[(sw.cutoff >= x0) & (sw.cutoff <= x1)]
    lo = float(np.nanmin([v.mean_ratio.min(), v.sem_ratio.min(), 0.95]))
    hi = float(np.nanmax([v.mean_ratio.max(), v.sem_ratio.max(), 1.05]))
    ax.set_xlim(x0, x1)
    ax.set_ylim(lo - 0.02, hi + 0.02)


def fig_amplitude(D):
    """6 · Does ocular activity leak into the single-trial N45? (variant-independent z, AMP_VARIANT N45)"""
    plt = setup_mpl()
    from scipy.stats import spearmanr
    subs = [s for s in SUBJECTS if s in D]
    fig = plt.figure(figsize=(15, 2.9 * (1 + len(subs)) + 1.2), layout="constrained")
    gs = fig.add_gridspec(1 + len(subs), 3)
    Dd = pd.concat([D[s]["w"][D[s]["w"].amp_ok].assign(subject=s) for s in subs], ignore_index=True)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, j]) for j in range(3))
    for name, m, color, ls in (("open-loop singles", (Dd.category == "open_loop").to_numpy(), CAT_COLOR["open_loop"], "-"),
                               ("all single pulses", np.ones(len(Dd), bool), INK, "--")):
        bs = binned_stats(Dd.z_win[m], Dd.rel_amp[m], AMP_BIN_EDGES, np.median)
        ax_a.errorbar(bs.z_med, bs.est, yerr=[bs.est - bs.ci_lo, bs.ci_hi - bs.est], color=color, ls=ls, lw=1.6,
                      marker="o", ms=4, capsize=2, label=f"{name}: median per bin, 95 % CI")
        if name.startswith("open"):
            for r in bs.itertuples():
                ax_a.text(r.z_med, r.ci_hi + 0.01, f"{r.n}", fontsize=6.3, color=INK_MUTED, ha="center", va="bottom")
        sd = binned_stats(Dd.z_win[m], Dd.rel_amp[m], AMP_BIN_EDGES, robust_sd)
        ax_b.errorbar(sd.z_med, sd.est, yerr=[sd.est - sd.ci_lo, sd.ci_hi - sd.est], color=color, ls=ls, lw=1.6,
                      marker="o", ms=4, capsize=2, label=name)
    for ax in (ax_a, ax_b):
        _cut_lines(ax)
        ax.set_xlim(-1.0, 7.0)
        ax.set_xlabel("window ocular z (bin median)")
    ax_a.axhline(1.0, color=BASE_C, lw=0.9)
    ax_a.set_ylabel("relative N45 amplitude")
    ax_a.set_title(f"{len(subs)} sessions pooled · median N45 per z bin (÷ each session's calm open-loop median;\n"
                   f"numbers: pulses per bin; vertical lines: the cutoffs)", fontsize=8.5)
    ax_a.legend(loc="upper left", fontsize=7)
    ax_b.set_ylabel("robust SD (relative units)")
    ax_b.set_title("trial-to-trial spread: robust SD per z bin", fontsize=8.5)
    ax_b.legend(loc="upper left", fontsize=7)
    sw = amplitude_sweep(Dd)
    ax_c.plot(sw.cutoff, sw.mean_ratio, color="#eb6834", lw=1.8, label="mean of the kept pulses")
    ax_c.plot(sw.cutoff, sw.sem_ratio, color="#2a78d6", lw=1.8, label="SEM of that mean")
    ax_c.axhline(1.0, color=BASE_C, lw=0.9)
    _cut_lines(ax_c)
    _sweep_limits(ax_c, sw)
    ax_c.set_xlabel("cutoff (kept: window ocular z ≤ cutoff)")
    ax_c.set_ylabel("ratio to no ocular gate")
    kept = ", ".join(f"{z:g}: {sw.loc[(sw.cutoff - z).abs().idxmin(), 'kept_pct']:.0f} %" for z in Z_GRID)
    ax_c.set_title(f"pooled · cutoff sweep · pulses kept at {kept}", fontsize=8.5)
    ax_c.legend(loc="lower right", fontsize=7.5)
    for i, s in enumerate(subs, start=1):
        d = D[s]["w"][D[s]["w"].amp_ok]
        ax = fig.add_subplot(gs[i, 0:2])
        for c in CAT_ORDER[:3]:
            m = (d.category == c).to_numpy()
            ax.scatter(np.minimum(d.z_win[m], 8), np.minimum(d.rel_amp[m], 3), s=8, color=CAT_COLOR[c],
                       marker=CAT_MARKER[c], alpha=0.35, lw=0, label=CAT_SHORT[c])
        bs = binned_stats(d.z_win, d.rel_amp, AMP_BIN_EDGES, np.median)
        ax.errorbar(bs.z_med, bs.est, yerr=[bs.est - bs.ci_lo, bs.ci_hi - bs.est], color=INK, lw=1.6, marker="o", ms=4,
                    capsize=2, label="median per bin, 95 % CI")
        _cut_lines(ax)
        ax.axhline(1.0, color=BASE_C, lw=0.9)
        ax.set_xlim(-1.0, 8.2)
        ax.set_ylim(0, 3.05)
        kept_ol = d[(d.category == "open_loop") & (d.z_win <= Z_GRID[-1])]
        rho = spearmanr(kept_ol.z_win, kept_ol.rel_amp)
        ax.set_title(f"{s} · single pulses passing the MAD gates (amplitude > 3 or z > 8 drawn at the edge) · ρ(N45, z) "
                     f"among open-loop pulses with z ≤ {Z_GRID[-1]:g}: {rho.statistic:+.2f} (p {rho.pvalue:.1g})",
                     fontsize=8.5)
        ax.set_ylabel("relative N45 amplitude")
        if i == 1:
            ax.legend(loc="upper right", fontsize=7, ncol=4)
        if i == len(subs):
            ax.set_xlabel("window ocular z")
        sw = amplitude_sweep(d)
        ax = fig.add_subplot(gs[i, 2])
        ax.plot(sw.cutoff, sw.mean_ratio, color="#eb6834", lw=1.8)
        ax.plot(sw.cutoff, sw.sem_ratio, color="#2a78d6", lw=1.8)
        ax.axhline(1.0, color=BASE_C, lw=0.9)
        _cut_lines(ax)
        _sweep_limits(ax, sw)
        ax.set_ylabel("ratio to no ocular gate")
        ax.set_title(f"{s} · mean (orange) and SEM (blue) of the kept pulses", fontsize=8.5)
        if i == len(subs):
            ax.set_xlabel("cutoff (kept: window ocular z ≤ cutoff)")
    fig.suptitle(f"Does ocular activity leak into the N45? Single-trial dipole amplitude ({AMP_VARIANT} decider: fit_trial after "
                 f"ICA, SOUND, SSP-SIR; MAD gates applied) against window ocular z", fontsize=10.5)
    save(fig, OUT / "figures" / "6_amplitude_vs_z.png")


def variant_style(v, thr):
    if v == "current":
        return dict(color=HEAD_C, ls="--", label=f"current decider (whole epoch, z {thr:g})")
    z = VARIANTS[v]["z"]
    return dict(color=CUT_COLOR[z], ls="-", label=f"window reference, z {z:g}")


def fig_outcome(D, A):
    """7 · Does the cutoff change the outcome? Evoked N45 per open-loop block of each cutoff's kept pulses."""
    plt = setup_mpl()
    ev = A["evoked"]
    subs = [s for s in SUBJECTS if s in D]
    fig, axes = plt.subplots(3, len(subs), figsize=(4.5 * len(subs), 10.6), squeeze=False, layout="constrained")
    blocks = OPEN_LOOP_BLOCKS
    x = np.arange(len(blocks))
    variants = WINDOW_VARIANTS + ["current"]
    for i, s in enumerate(subs):
        e = ev[ev.subject == s]
        ref = e[e.variant == AMP_VARIANT].set_index("block").reindex(blocks)
        for r, (tag, what) in enumerate((("own", "each cutoff with its own calibration"),
                                         ("fixed", f"{AMP_VARIANT} decider's cleaning and dipole for every cutoff"))):
            ax = axes[r, i]
            ax.fill_between(x[1:], ref[f"change_{tag}_ci_lo"].iloc[1:], ref[f"change_{tag}_ci_hi"].iloc[1:], color=BAND_C,
                            lw=0, label=f"95 % bootstrap CI ({AMP_VARIANT} decider)")
            for v in variants:
                d = e[e.variant == v].set_index("block").reindex(blocks)
                y = d[f"change_{tag}_pct"].to_numpy(dtype=float)
                y[0] = 0.0
                ax.plot(x, y, marker="o", ms=4, lw=1.5, **variant_style(v, D[s]["thr"][v]))
            ax.axhline(0, color=BASE_C, lw=0.9)
            ax.set_xticks(x, blocks)
            ax.set_title(f"{s} · evoked N45 change from baseline\n{what}", fontsize=8.5)
            if i == 0:
                ax.set_ylabel("% change of the evoked N45")
        ax = axes[2, i]
        wbar = 0.8 / len(variants)
        lo_all = []
        for k, v in enumerate(variants):
            d = e[e.variant == v].set_index("block").reindex(blocks)
            st = variant_style(v, D[s]["thr"][v])
            ax.bar(x + (k - (len(variants) - 1) / 2) * wbar, d.n_kept_single, width=wbar * 0.92, color=st["color"],
                   edgecolor="white", lw=0.5, hatch="///" if v == "current" else None, label=st["label"])
            lo_all.append(d.n_kept_single.min())
        ax.set_ylim(max(0, min(lo_all) - 12), 101)
        ax.set_xticks(x, blocks)
        ax.set_title(f"{s} · single pulses kept per block (of 100; all post gates)", fontsize=8.5)
        if i == 0:
            ax.set_ylabel("pulses kept")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=6, fontsize=8)
    fig.suptitle("Does the cutoff move the outcome? Evoked N45 = dipole fit on the block average of the kept single "
                 "pulses; the change from baseline per evaluation block, and the pulses each block keeps", fontsize=11)
    save(fig, OUT / "figures" / "7_outcome_blocks.png")


def fig_window_vs_current(D, A):
    """8 · Same pulses, two references: window vs whole epoch (the decider now)."""
    plt = setup_mpl()
    from scipy.stats import spearmanr
    subs = [s for s in SUBJECTS if s in D]
    fig, axes = plt.subplots(len(subs), 2, figsize=(12.5, 3.7 * len(subs) + 1.0), squeeze=False, layout="constrained",
                             gridspec_kw=dict(width_ratios=[1, 1.3]))
    eq = A["equivalent"]
    for i, s in enumerate(subs):
        w = D[s]["w"]
        z_now = D[s]["thr"]["current"]
        ax = axes[i, 0]
        lim = [min(w.z_cur.min(), w.z_win.min()) - 0.2, 8.2]
        for c in CAT_ORDER:
            m = w.category == c
            ax.scatter(np.minimum(w.z_cur[m], 8), np.minimum(w.z_win[m], 8), s=7, color=CAT_COLOR[c],
                       marker=CAT_MARKER[c], alpha=0.45, lw=0, label=CAT_SHORT[c] if i == 0 else None)
        ax.plot(lim, lim, color=BASE_C, lw=1.0)
        ax.axvline(z_now, color=HEAD_C, ls="--", lw=1.0)
        for z in Z_GRID:
            ax.axhline(z, color=CUT_COLOR[z], lw=0.9)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ok = np.isfinite(w.z_cur) & np.isfinite(w.z_win)
        slope, icpt = np.polyfit(w.z_cur[ok], w.z_win[ok], 1)
        ax.set_title(f"{s} · Spearman ρ {spearmanr(w.z_cur, w.z_win).statistic:.4f} · window ≈ {slope:.2f} × now "
                     f"{icpt:+.2f}", fontsize=8.5)
        ax.set_xlabel("ocular z, decider now (whole-epoch reference; > 8 at 8)")
        ax.set_ylabel("ocular z, window reference")
        ax = axes[i, 1]
        sg = w[w.single]
        ax.plot(SWEEP, [pct_above(sg.z_win, x) for x in SWEEP], color=POOLED_C, lw=1.8, label="window reference")
        ax.plot(SWEEP, [pct_above(sg.z_cur, x) for x in SWEEP], color=HEAD_C, lw=1.6, ls="--",
                label="whole-epoch reference (decider now)")
        for z in Z_GRID:
            ax.plot([z], [pct_above(sg.z_win, z)], marker="o", color=CUT_COLOR[z], ms=6, ls="none")
        ax.plot([z_now], [pct_above(sg.z_cur, z_now)], marker="D", color=HEAD_C, ms=6, ls="none")
        e = eq[eq.subject == s]
        txt = "\n".join([f"window {r.cutoff:g} ≙ now {r.same_pct_on_other_scale:.2f}  ({r.pct_single_above:.1f} %)"
                         for r in e[e.scale == "window"].itertuples()] +
                        [f"now {r.cutoff:g} ≙ window {r.same_pct_on_other_scale:.2f}  ({r.pct_single_above:.1f} %)"
                         for r in e[e.scale != "window"].itertuples()])
        ax.text(0.98, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK_2, family="monospace")
        ax.set_xlim(0, 6)
        ax.set_ylim(0, None)
        ax.set_xlabel("cutoff on ocular z")
        ax.set_ylabel("% of single pulses above")
        ax.set_title(f"{s} · rejection curves (dots: the four window cutoffs; diamond: the decider now)", fontsize=8.5)
        if i == 0:
            ax.legend(fontsize=7.5, loc="center right")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=4)
    fig.suptitle("Same pulses, two references: the window reference against the whole-epoch reference the decider uses "
                 "now (equivalent cutoffs: the same % of single pulses)", fontsize=11)
    save(fig, OUT / "figures" / "8_window_vs_current.png")


def fig_eye_ics(D):
    """9 · The ICs the gate scores, and each one's window score per single pulse."""
    plt = setup_mpl()
    subs = [s for s in SUBJECTS if s in D]
    wv = WINDOW_VARIANTS[0]
    n_ic = max(len(D[s]["meta"]["eye_ics"]) for s in subs)
    H = 2.55 * len(subs) + 0.9
    fig = plt.figure(figsize=(2.3 * n_ic + 5.6, H))
    gs = fig.add_gridspec(len(subs), n_ic + 1, width_ratios=[1] * n_ic + [2.5], hspace=0.7, wspace=0.28,
                          **page(H, 0.85, 0.45))
    for i, s in enumerate(subs):
        res = D[s]
        ci, meta, w = res["cal"], res["meta"], res["w"]
        info = topo_info(list(ci["ch_names"]))
        P = ci.get("iclabel_proba_all")
        comp = ci["components"]
        for j, ic in enumerate(meta["eye_ics"]):
            ax = fig.add_subplot(gs[i, j])
            draw_topo(ax, comp[:, ic], info, np.max(np.abs(comp[:, ic])))
            p_eye = (f"P(eye) {P[ic, ICLABEL_CLASSES.index('eye blink')]:.2f}" if P is not None and np.size(P)
                     else "P(eye) n/a")
            stc, stw = meta["variants"]["current"]["eye_stats"][str(ic)], meta["variants"][wv]["eye_stats"][str(ic)]
            ax.set_title(f"{s} · IC {ic} · {p_eye}\n|s| now {stc['mean']:.2f} ± {stc['std']:.2f}\nwindow "
                         f"{stw['mean']:.2f} ± {stw['std']:.2f}", fontsize=7.5, loc="center")
        ax = fig.add_subplot(gs[i, n_ic])
        sg = w[w.single]
        rng = np.random.default_rng(0)
        for j, ic in enumerate(meta["eye_ics"]):
            z = sg[f"z_eye{j + 1}__{wv}"].to_numpy()
            ax.scatter(j + rng.uniform(-0.28, 0.28, len(z)), np.minimum(z, 10), s=5, color=IC_COLORS[j % 3],
                       alpha=0.35, lw=0)
            ax.text(j, 10.2, " · ".join(f"{pct_above(z, c):.1f}" for c in Z_GRID), ha="center", va="bottom",
                    fontsize=7, color=INK_2)
        for z in Z_GRID:
            ax.axhline(z, color=CUT_COLOR[z], lw=1.0)
        ax.set_ylim(-1.5, 11.4)
        ax.set_xticks(range(len(meta["eye_ics"])), [f"IC {ic}" for ic in meta["eye_ics"]])
        ax.set_ylabel("median z, 15–100 ms")
        ax.set_title("each eye IC's window score, single pulses (> 10 at 10); numbers: % above "
                     + " · ".join(f"{z:g}" for z in Z_GRID), fontsize=8)
    fig.suptitle("The ICs the gate scores (ICLabel 'eye blink'), with their |s| mean ± SD under both references",
                 y=1 - 0.25 / H, fontsize=11)
    save(fig, OUT / "figures" / "9_eye_ics.png")


def fig_cleaning(res):
    """10 · What the eye ICs remove and what is left after cleaning, by band of window ocular z."""
    plt = setup_mpl()
    w, s, meta = res["w"], res["subject"], res["meta"]
    d = w[w.single].copy()
    calm_z = res["evidence"]["calm_z_max"]
    edges = [calm_z] + list(Z_GRID)
    labels = [f"calm (z ≤ {calm_z:.2f})", f"{calm_z:.2f} < z ≤ {Z_GRID[0]:g}"] + \
             [f"{a:g} < z ≤ {b:g}" for a, b in zip(Z_GRID[:-1], Z_GRID[1:])] + [f"z > {Z_GRID[-1]:g}"]
    colors = [INK_MUTED] + BAND_COLORS
    d["grp"] = np.digitize(d.z_win, edges, right=True)
    stems = list(d.stem)
    arr = {k: load_arrays(res, k, stems).astype(float) for k in
           ("gate_ms", "removed_ms", "n45_mean", "n45_ms", "veog", "roi_final", "veog_final")}
    info = topo_info(meta["ch_names"])
    t_ep, t_fin = np.asarray(meta["times_epoch_ms"]), np.asarray(meta["times_final_ms"])
    agg = {}
    for b in range(len(labels)):
        m = (d.grp == b).to_numpy()
        if m.sum() < 3:
            continue
        agg[b] = dict(n=int(m.sum()), pre_rms=np.sqrt(arr["gate_ms"][m].mean(axis=0)),
                      rem_rms=np.sqrt(arr["removed_ms"][m].mean(axis=0)), fin_mean=arr["n45_mean"][m].mean(axis=0),
                      fin_rms=np.sqrt(arr["n45_ms"][m].mean(axis=0)), pre_veog_sd=arr["veog"][m].std(axis=0),
                      fin_roi_mean=arr["roi_final"][m].mean(axis=0), fin_veog_sd=arr["veog_final"][m].std(axis=0))
    if not agg:
        return
    cols = [("pre_rms", "before ICA\nRMS, 15–100 ms", True), ("rem_rms", "removed by eye ICs\nRMS, 15–100 ms", True),
            ("fin_mean", "after cleaning\nmean, 38–50 ms", False), ("fin_rms", "after cleaning\nRMS, 38–50 ms", True)]
    vmax = {key: (float(np.percentile(np.concatenate([agg[b][key] for b in agg]) if seq else
                                      np.abs(np.concatenate([agg[b][key] for b in agg])), 99)) or 1.0)
            for key, _, seq in cols}
    nb = len(labels)
    H = 1.6 * nb + 2.0
    fig = plt.figure(figsize=(15.5, H))
    gs = fig.add_gridspec(nb + 1, 7, width_ratios=[1.0, 1, 1, 1, 1, 0.3, 3.6], height_ratios=[1] * nb + [0.12],
                          hspace=0.35, wspace=0.1, **page(H, 1.0, 0.45))
    ims = {}
    for b in range(nb):
        axl = fig.add_subplot(gs[b, 0])
        axl.axis("off")
        axl.text(0.86, 0.5, f"{labels[b]}\nn = {agg[b]['n'] if b in agg else 0}", ha="right", va="center", fontsize=8.2,
                 transform=axl.transAxes)
        axl.plot([0.95], [0.5], marker="s", ms=9, color=colors[b], transform=axl.transAxes, clip_on=False)
        for j, (key, title, seq) in enumerate(cols):
            ax = fig.add_subplot(gs[b, j + 1])
            if b in agg:
                ims[key] = draw_topo(ax, agg[b][key], info, vmax[key], cmap=sequential_cmap() if seq else None,
                                     vmin=0.0 if seq else None)
            else:
                ax.axis("off")
            if b == 0:
                ax.set_title(title, fontsize=8, loc="center")
    for j, (key, _, _) in enumerate(cols):
        if key in ims:
            cb = fig.colorbar(ims[key], cax=fig.add_subplot(gs[nb, j + 1]), orientation="horizontal")
            cb.ax.tick_params(labelsize=7)
            cb.set_label("µV", fontsize=7)
    inner = gs[:nb, 6].subgridspec(3, 1, hspace=0.5)
    traces = [("pre_veog_sd", t_ep, "Fp1/Fpz/Fp2 mean before ICA: SD over pulses (log scale)", GATE_WINDOW_MS, True),
              ("fin_roi_mean", t_fin, f"left-M1 ROI ({', '.join(ROI_CHANNELS)}) after cleaning: mean", N45_WINDOW_MS, False),
              ("fin_veog_sd", t_fin, "Fp1/Fpz/Fp2 mean after cleaning: SD over pulses", N45_WINDOW_MS, False)]
    for k, (key, t, title, win, logy) in enumerate(traces):
        ax = fig.add_subplot(inner[k])
        ax.axvspan(*win, color=WASH, lw=0, zorder=0)
        for b in sorted(agg):
            ax.plot(t, agg[b][key], color=colors[b], lw=1.5, label=f"{labels[b]} ({agg[b]['n']})")
        ax.axvline(0, color=BASE_C, lw=0.8)
        ax.set_xlim(t[0], 100)
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(1, None)
        ax.set_ylabel("µV")
        ax.set_title(title, fontsize=8.5)
        if k == 0:
            ax.legend(fontsize=6.8, ncol=2, loc="upper left")
        if k == 2:
            ax.set_xlabel("ms from the pulse")
    fig.suptitle(f"{s} · single pulses grouped by window ocular z · cleaned epochs of {meta.get('array_variant', ARRAY_VARIANT)} · maps share one "
                 f"colour scale per column", y=1 - 0.25 / H, fontsize=10)
    save(fig, OUT / "figures" / f"10_cleaning_{s}.png")


def fig_timeline(D):
    """11 · Window ocular z across the session, and per block."""
    plt = setup_mpl()
    subs = [s for s in SUBJECTS if s in D]
    fig = plt.figure(figsize=(16, 2.8 * len(subs) + 1.0), layout="constrained")
    gs = fig.add_gridspec(len(subs), 2, width_ratios=[3.1, 1])
    for i, s in enumerate(subs):
        w = D[s]["w"]
        ax = fig.add_subplot(gs[i, 0])
        ymax = min(10, np.nanmax(w.z_win) + 0.5)
        blocks = list(dict.fromkeys(w.block))
        for k, b in enumerate(blocks):
            g = w[w.block == b]
            x0, x1 = g.order.min() - 0.5, g.order.max() + 0.5
            if k % 2:
                ax.axvspan(x0, x1, color="#f6f5f2", lw=0, zorder=0)
            ax.text((x0 + x1) / 2, ymax, b, ha="center", va="bottom", fontsize=7.5, color=INK_2)
        for c in CAT_ORDER:
            m = w.category == c
            ax.scatter(w.order[m], np.minimum(w.z_win[m], ymax), s=7, color=CAT_COLOR[c], marker=CAT_MARKER[c],
                       alpha=0.6, lw=0, label=CAT_SHORT[c] if i == 0 else None)
        for z in Z_GRID:
            ax.axhline(z, color=CUT_COLOR[z], lw=1.0)
        ax.set_ylim(min(-1.5, np.nanmin(w.z_win) - 0.2), ymax)
        ax.set_xlim(-5, w.order.max() + 5)
        ax.set_ylabel("window ocular z")
        ax.set_title(f"{s} · values above {ymax:.1f} drawn at {ymax:.1f}", fontsize=8.5, pad=14)
        if i == len(subs) - 1:
            ax.set_xlabel("pulse, in recording order")
        if i == 0:
            ax.legend(loc="upper left", ncol=4, fontsize=7.5, bbox_to_anchor=(0, 1.32))
        ax2 = fig.add_subplot(gs[i, 1])
        sg = w[w.single]
        for z in Z_GRID:
            y = [pct_above(sg.z_win[sg.block == b], z) for b in blocks]
            ax2.plot(range(len(blocks)), y, color=CUT_COLOR[z], marker="o", ms=4, lw=1.5, label=f"z > {z:g}")
        ax2.set_xticks(range(len(blocks)), [BLOCK_SHORT.get(b, b) for b in blocks], rotation=60, fontsize=7.5)
        ax2.set_ylim(0, None)
        ax2.set_ylabel("% of single pulses")
        ax2.set_title("above each cutoff, per block", fontsize=8.5)
        if i == 0:
            ax2.legend(fontsize=7, ncol=2, loc="best")
    fig.suptitle("Window ocular z across the session (horizontal lines: the four cutoffs)", fontsize=11)
    save(fig, OUT / "figures" / "11_timeline.png")


FIGURES = ["reference", "scores", "decision", "cards", "ladder", "amplitude", "outcome", "compare", "eye_ics",
           "cleaning", "timeline"]


def make_figures(D, A, which=None):
    which = set(which or FIGURES)
    for name, fn in (("reference", fig_reference), ("scores", fig_scores), ("amplitude", fig_amplitude),
                     ("eye_ics", fig_eye_ics), ("timeline", fig_timeline)):
        if name in which:
            fn(D)
    for name, fn in (("decision", fig_decision), ("outcome", fig_outcome), ("compare", fig_window_vs_current)):
        if name in which:
            fn(D, A)
    for s, res in D.items():
        for name, fn in (("cards", fig_band_cards), ("ladder", fig_ladder), ("cleaning", fig_cleaning)):
            if name in which:
                fn(res)

# %% 9 · Main ------------------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract", action="store_true", help="child mode: extract one part x one subject")
    ap.add_argument("--part", choices=["online", "variants"])
    ap.add_argument("--subject")
    ap.add_argument("--budget", type=float, default=None, help="child: stop after about this many seconds (resumable)")
    ap.add_argument("--plots-only", action="store_true", help="tables and figures from what is extracted")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--figures", nargs="*", default=None, help=f"subset of {FIGURES}")
    ap.add_argument("--no-tables", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-validation", action="store_true", help="skip the online validation runs")
    args, _ = ap.parse_known_args(argv)

    if args.extract:
        if not (args.part and args.subject):
            ap.error("--extract needs --part and --subject")
        fn = extract_online if args.part == "online" else extract_variants
        sys.exit(0 if fn(args.subject, budget=args.budget) else 3)

    subjects = args.subjects or SUBJECTS
    if not args.plots_only:
        for s in subjects:
            if VALIDATE_ONLINE and not args.no_validation and s in SESSION_CODE and not is_complete("online", s):
                run_child("online", s)
            if not is_complete("variants", s):
                run_child("variants", s)
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
