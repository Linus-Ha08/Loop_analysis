#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ocular_threshold_review.py - where should the cutoff of the ocular gate sit?

The online decider rejects a pulse when the eye-blink ICs are too active after the pulse. This script re-runs
exactly that part of the online pipeline for every pulse of a session, keeps the score (not only the yes/no
decision) and plots the scores, the trials around the cutoff and what the cutoff does to the N45.

The online gate (Preprocessor.preprocess_post in prime/decider):

    raw post window [-30, 100] ms
      -> baseline, first artifact blank, [500 Hz low-pass,] bad-channel interpolation, average reference
      -> sources of the ICLabel "eye blink" ICs:  s(t) = ica.get_sources(epoch)
      -> z(t) = (|s(t)| - mean) / std   (mean and std of |s| over every sample of every calibration trial)
      -> score = median of z(t) over [15, 100] ms;  reject if the score of ANY eye IC > z_threshold

So the decision statistic is z_max = max over the eye ICs of that median ("ocular z" below).

Two setups (SETUPS)
-------------------
online   The code that ran the session (prime commit in SESSION_CODE, plus the settings that differed, e.g. the
         ocular threshold 2.0 of pilot003/004) with the session's own calibration bundle. Reproduces the online
         decisions; the script checks this against trials_intervention.csv and the bundle's calibration drops.
current  The decider as it is now (CURRENT_COMMIT, default HEAD of prime/decider), re-calibrated on the session's
         own 125 calibration trials as a new session would be. A cutoff chosen today acts on this scale, which
         is not the scale of the pilots (5 kHz post window, 500 Hz low-pass, baseline [-15, -5] ms, blank [-2, 14]).

How it runs
-----------
Nothing is written into prime/. Each setup's code is exported from git into Loop/prime_<sha7> (the export /
verify routine of offline_fwdmodel.ipynb) and imported from there in a child process, because two prime_core
versions cannot live in one interpreter. The child calls the decider's own Preprocessor.preprocess_post() on
every saved _post_raw buffer with a recorder on the two ICA calls inside it (get_sources, apply), so the scores
come from the decider's arithmetic, not from a re-implementation. It also fits the N45 dipole (fit_trial) on
every trial, including the rejected ones.

    python ocular_threshold_review.py                     # extract what is missing, then tables and figures
    python ocular_threshold_review.py --plots-only        # tables and figures from what is already extracted
    python ocular_threshold_review.py --extract --setup online --subject pilot005 [--budget 150]
                                                           # one child run; resumes block by block

Outputs (Loop/offline_ocular/)
    <setup>/sub-<ID>/scores.csv            one row per pulse: ocular z per eye IC, z_max, replayed and online
                                           rejection reason, N45 amplitude, post MAD z, frontal (EOG) measures
    <setup>/sub-<ID>/blocks/<block>.npz    per-pulse arrays at 1 kHz: pre-ICA epoch, ICA-removed part, eye-IC
                                           sources, final epoch, pre-stimulus window
    <setup>/sub-<ID>/calibration_info.npz  eye ICs, their topographies, ICLabel probabilities, gate statistics
    <setup>/sub-<ID>/meta.json             code, settings, rates, checks
    figures/, tables/                      everything below section 6
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

SUBJECTS = ["pilot003", "pilot004", "pilot005"]
SESSION_ID = "ses-prime"
RUN_IDS = {}                       # e.g. {"pilot005": "105"}; missing: the one run folder with prime/trials_intervention.csv

SETUPS = ["online", "current"]

# The prime commit that ran each session, and the decider settings the session ran with where they differ from it.
F4FD395 = "f4fd3954c4f5b1e31a5be06c82e39d9beb175154"     # "ocular thresh z 2 --> 2.5", 17 Sep 2026
SESSION_CODE = {
    "pilot003": dict(commit=F4FD395, overrides={"ocular_z_threshold": 2.0}),   # ran before f4fd395
    "pilot004": dict(commit=F4FD395, overrides={"ocular_z_threshold": 2.0}),
    "pilot005": dict(commit=F4FD395, overrides={}),
}
CURRENT_COMMIT = "HEAD"            # "current" setup: a commit (or HEAD) of prime/decider
CURRENT_OVERRIDES = {}             # e.g. {"ocular_z_threshold": 2.0}: only changes which pulses count as rejected

REPLAY_ONLINE_CALIBRATION = True   # online setup: also replay calibrate() for the ICLabel probabilities and a check
                                   # that the replay reproduces the online bundle (scores always use the online bundle)
GIT_EXE = None                     # None: git on PATH or the usual Git for Windows locations

# Figures and tables
CANDIDATE_CUTOFFS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
LADDER_LEVELS = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0]   # ocular-z levels of the topoplot ladder
LADDER_PER_LEVEL = 6               # pulses per level (the nearest ones)
CARDS_PER_SIDE = 8                 # pulses just below / just above the cutoff in the trial-card figure
Z_BIN_EDGES = [1.0, 1.75, 2.5, 4.0]   # 5 bins for the bin-average figure
GATE_WINDOW_MS = (15, 100)         # the gate's window ([15 ms, end of the post window])
N45_WINDOW_MS = (38, 50)           # dipole window of the decider
VEOG_CHANNELS = ["Fp1", "Fpz", "Fp2"]
HEOG_PAIR = ("F7", "F8")
ROI_CHANNELS = ["C1", "C3", "FC1", "FC3", "CP1", "CP3"]   # left M1 (baseline_candidates)
ROBUST_K = 3.0                     # robust outlier line: median + ROBUST_K x 1.4826 x MAD of the single-pulse z_max
SWEEP = np.round(np.arange(0.0, 8.0001, 0.05), 2)        # cutoff grid of the sweeps

LOOP_ROOT = None                   # None: LOOP_ROOT env var, D:\Linus\Loop, or a folder above this script with BIDS + prime
OUT_ROOT = None                    # None: <Loop>/offline_ocular

# %% 1 · Paths, sessions -------------------------------------------------------------------------------------------
BLOCK_ORDER = ["baseline", "calibration", "iv1", "iv2", "iv3", "iv4", "t0", "t15", "t30", "t60"]
EVAL_LABELS = {1: "t0", 2: "t15", 3: "t30", 4: "t60"}
RAW_SFREQ_EXPECTED = 5000.0
STORE_SFREQ = 1000.0               # arrays are stored at 1 kHz (the scores are computed at the native rate)
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
OUT = Path(OUT_ROOT) if OUT_ROOT else ROOT / "offline_ocular"
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


def setup_dir(setup, subject):
    return OUT / setup / f"sub-{subject}"


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
    """Files of prime/decider that differ from HEAD (for the warning of the current setup)."""
    git = find_git(GIT_EXE)
    if not git or not (PRIME_REPO / ".git").exists():
        return None
    out = run_git(git, PRIME_REPO, "status", "--porcelain", "--untracked-files=no").decode().strip()
    return [l[3:] for l in out.splitlines() if l.strip()]


def setup_code(setup, subject):
    if setup == "online":
        sc = SESSION_CODE[subject]
        return resolve_commit(sc["commit"]), dict(sc.get("overrides", {}))
    if setup == "current":
        return resolve_commit(CURRENT_COMMIT), dict(CURRENT_OVERRIDES)
    raise ValueError(f"unknown setup {setup!r}")


# %% 3 · Extraction (child process: one setup x one subject) -------------------------------------------------------

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
    return types.SimpleNamespace(mne=mne, pp_module=preprocessor_module, Preprocessor=Preprocessor,
                                 DipoleFitter=DipoleFitter, resample=resample_buffer_polyphase, cfg=prime_config)


def apply_overrides(pp, overrides):
    if "ocular_z_threshold" in overrides:
        pp._cfg.trial_reject_opts.ocular.z_threshold = float(overrides["ocular_z_threshold"])
    pp._opts = pp._cfg.to_dicts()                        # calibrate() reads this copy, made in __init__
    return pp


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

    Instance attributes shadow the ICA methods, so only this ICA object (the bundle's) is affected."""

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


def replay_calibration(P, sp, trials, overrides, fit_dipole):
    """run_calibration() of the decider on the session's 125 _trial_raw buffers (add_trial x 125 -> calibrate())."""
    od = sp["online_dir"]
    raw_sfreq = P.cfg.get_raw_sfreq()
    cal_tmin = P.cfg.get_calibration_time_range()[0]
    pp = apply_overrides(P.Preprocessor(FORWARD_PATH), overrides)
    pp.keep_full_epochs = True
    tic = time.perf_counter()
    for stem in trials.loc[trials.block == "calibration", "stem"]:
        x = np.load(od / f"{stem}_trial_raw.npy")
        pp.add_trial(x, cal_tmin + np.arange(len(x)) / raw_sfreq)
    t_add = time.perf_counter() - tic
    store, log = {}, io.StringIO()
    tic = time.perf_counter()
    with contextlib.redirect_stdout(log), capture_get_ica(P, store):
        _, dipole_buffers = pp.calibrate()
    t_cal = time.perf_counter() - tic
    res = dict(bundle=pp.calibration_params, iclabel=store, log=log.getvalue(), seconds_add_trial=t_add,
               seconds_calibrate=t_cal)
    if fit_dipole:
        fitter = make_fitter(P, post_rate(pp, P))
        with contextlib.redirect_stdout(io.StringIO()):
            fitter.calibrate(dipole_buffers)
        res["fitting_info"] = fitter.fitting_info
    return res


def compare_bundles(rep, online):
    """Does the replayed calibration reproduce the online bundle?"""
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


def extract(setup, subject, budget=None, verbose=True):
    """One setup x one subject. Writes each block as soon as it is done; returns True when everything is there."""
    t_start = time.time()
    say = print if verbose else (lambda *a, **k: None)
    sp = session_paths(subject)
    od = sp["online_dir"]
    commit, overrides = setup_code(setup, subject)
    code_dir, manifest = ensure_export(commit)
    P = import_prime(code_dir)
    mne = P.mne
    out = setup_dir(setup, subject)
    (out / "blocks").mkdir(parents=True, exist_ok=True)
    say(f"[{setup} / {subject}] code {commit[:10]} ({manifest['commit_subject']}) from {code_dir.name}; "
        f"overrides {overrides or 'none'}")

    online_bundle = np.load(sp["bundle"], allow_pickle=True).item()
    trials = trial_table(sp, online_bundle)
    raw_sfreq = P.cfg.get_raw_sfreq()
    if abs(raw_sfreq - RAW_SFREQ_EXPECTED) > 1e-6:
        raise ValueError(f"raw_sfreq {raw_sfreq} in the commit's prime.yaml, expected {RAW_SFREQ_EXPECTED}")
    post_i_tmin = P.cfg.get_post_initial_time_range()[0]
    post_tmin = P.cfg.get_post_time_range()[0]
    dip_tmin, dip_tmax = P.cfg.get_dipole_time_range()
    checks, calinfo = {}, {}

    # ------------------------------------------------------------------ calibration of the setup
    if setup == "online":
        bundle = online_bundle
        fitting_info = np.load(od / "dipole_fitting_info.npy", allow_pickle=True).item()
        calinfo["source"] = f"online bundle {sp['bundle'].name}"
        if REPLAY_ONLINE_CALIBRATION:
            rep_file = out / "calibration_replay.npy"
            if rep_file.is_file():
                rep = np.load(rep_file, allow_pickle=True).item()
            else:
                say("  replaying the online calibration (ICLabel probabilities, reproduction check) ...")
                r = replay_calibration(P, sp, trials, overrides, fit_dipole=False)
                rep = dict(iclabel=r["iclabel"], check=compare_bundles(r["bundle"], online_bundle), log=r["log"],
                           seconds_calibrate=r["seconds_calibrate"], seconds_add_trial=r["seconds_add_trial"])
                np.save(rep_file, rep, allow_pickle=True)
            checks["calibration replay vs online bundle"] = rep["check"]
            calinfo["iclabel"] = rep["iclabel"]
            if not all(v["ok"] for v in rep["check"].values()):
                say("  NOTE: the replayed calibration differs from the online bundle: " +
                    "; ".join(f"{k}: {v['detail']}" for k, v in rep["check"].items() if not v["ok"]))
    else:
        cal_file = out / "calibration_current.npy"
        if cal_file.is_file():
            cal = np.load(cal_file, allow_pickle=True).item()
        else:
            changed = working_tree_changes()
            if changed:
                say(f"  NOTE: prime/decider has uncommitted changes ({', '.join(changed[:6])}); the current setup "
                    f"uses commit {commit[:10]}, not the working tree")
            say("  re-calibrating on the session's 125 calibration trials with the current code ...")
            r = replay_calibration(P, sp, trials, overrides, fit_dipole=True)
            cal = dict(bundle=r["bundle"], fitting_info=r["fitting_info"], iclabel=r["iclabel"], log=r["log"],
                       seconds_calibrate=r["seconds_calibrate"], seconds_add_trial=r["seconds_add_trial"])
            np.save(cal_file, cal, allow_pickle=True)       # saved before the recorder touches the ICA object
            say(f"  calibration: {r['seconds_add_trial']:.0f} s add_trial + {r['seconds_calibrate']:.0f} s calibrate()")
        bundle, fitting_info = cal["bundle"], cal["fitting_info"]
        calinfo["source"] = "re-calibrated on the session's calibration trials"
        calinfo["iclabel"] = cal["iclabel"]

    pp = apply_overrides(P.Preprocessor.from_bundle(bundle, FORWARD_PATH), overrides)
    pp.keep_full_epochs = True                           # as Decider.__init__
    rate = post_rate(pp, P)
    fitter = make_fitter(P, rate)
    fitter._fitting_info = fitting_info                  # DipoleFitter has no public restore method
    thr = float(pp._cfg.trial_reject_opts.ocular.z_threshold)
    stats = bundle["ocular_threshold_post"]
    eye_ics = sorted(int(k) for k in stats)
    gts = bundle["good_trial_stats_post"]
    glob_thr = [float(v) for v in pp._cfg.trial_reject_opts.post.global_zscore_threshold]
    loc_thr = float(pp._cfg.trial_reject_opts.post.local_zscore_threshold)
    reject_range = [float(v) for v in pp._cfg.reject_range]
    info_post = getattr(pp, "_info_post", None) or pp._info_processed
    ch_names = list(pp._info["ch_names"])
    step = int(round(rate / STORE_SFREQ))
    d0 = int(round((dip_tmin - post_tmin) * rate))
    dip_idx = slice(d0, int(round((dip_tmax - post_tmin) * rate)) + 1)
    gate_idx = np.asarray(stats[eye_ics[0]]["time_indices_of_interest"])
    veog_idx = [ch_names.index(c) for c in VEOG_CHANNELS]
    heog_idx = [ch_names.index(c) for c in HEOG_PAIR]
    roi_idx = [ch_names.index(c) for c in ROI_CHANNELS]
    good_idx = [i for i, c in enumerate(ch_names) if c not in set(map(str, bundle["bad_channels"]))]
    proc_names = list(np.load(od / "proc_info.npy", allow_pickle=True).item()["ch_names"])
    if proc_names != ch_names:
        raise ValueError("channel order of the session (proc_info.npy) differs from the forward model's")

    rec = IcaRecorder(bundle["ica"])

    def one_trial(stem):
        x = np.load(od / f"{stem}_post_raw.npy")
        rec.reset()
        tic = time.perf_counter()
        ret = pp.preprocess_post(x, post_i_tmin + np.arange(len(x)) / raw_sfreq)
        t_ms = (time.perf_counter() - tic) * 1e3
        if rec.pre is None or rec.removed is None:
            raise RuntimeError("the recorder did not see the ICA calls: is this the decider's preprocess_post?")
        reason = pp.last_post_reject_reason or ""
        final = np.asarray(pp.last_post_full)
        # the gate, exactly as preprocess_post computes it
        z_eye = []
        for ic in eye_ics:
            st = stats[ic]
            z = (np.abs(rec.src[ic, st["time_indices_of_interest"]]) - st["mean"]) / st["std"]
            z_eye.append(float(np.median(z)))
        z_max = max(z_eye)
        # N45 dipole amplitude of every pulse, rejected or not (accepted: the same samples preprocess_post returns)
        dip = np.ascontiguousarray(final[:, dip_idx])
        same_as_returned = None if ret is None else bool(np.array_equal(np.asarray(ret), dip))
        amp = float(fitter.fit_trial(dip))
        # the post-stimulus MAD gates on the final epoch, as preprocess_post (also for ocular-rejected pulses)
        ep = mne.EpochsArray(final[np.newaxis], info_post, tmin=post_tmin, verbose=False)
        rd = ep.crop(reject_range[0], reject_range[1]).get_data(copy=True)
        from scipy.stats import median_abs_deviation, zscore
        z_glob = float((median_abs_deviation(rd, axis=(1, 2))[0] - gts["mads_mean"]) / gts["mads_std"])
        z_loc = float(np.max(np.abs(zscore(median_abs_deviation(rd, axis=2), axis=1))))
        # frontal (EOG-like) measures, independent of the ICA
        pre = rec.pre
        veog = pre[veog_idx].mean(axis=0)
        heog = pre[heog_idx[0]] - pre[heog_idx[1]]
        removed = rec.removed
        gate = slice(int(gate_idx[0]), int(gate_idx[-1]) + 1)
        pre_raw = np.load(od / f"{stem}_pre_raw.npy")
        pre_win = np.asarray(P.resample(pre_raw, raw_sfreq, STORE_SFREQ), dtype=float).T     # (n_ch, 201), raw
        pw = pre_win - pre_win[good_idx].mean(axis=0, keepdims=True)
        veog_pre = pw[veog_idx].mean(axis=0)
        row = dict(z_max=z_max, z_argmax_ic=eye_ics[int(np.argmax(z_eye))], rejected_ocular=bool(z_max > thr),
                   reason=reason, amp=amp, z_mad_global=z_glob, z_mad_local_max=z_loc,
                   veog_gate_mean=float(veog[gate].mean()), veog_gate_ptp=float(np.ptp(veog[gate])),
                   heog_gate_mean=float(heog[gate].mean()), gfp_gate=float(pre[:, gate].std(axis=0).mean()),
                   removed_gate_rms=float(np.sqrt(np.mean(removed[:, gate] ** 2))),
                   veog_pre_ptp=float(np.ptp(veog_pre)), veog_pre_mean_abs=float(np.mean(np.abs(veog_pre - veog_pre.mean()))),
                   final_veog_n45=float(final[veog_idx][:, dip_idx].mean()),
                   final_roi_n45=float(final[roi_idx][:, dip_idx].mean()),
                   same_as_returned=same_as_returned, t_ms=t_ms)
        for j, z in enumerate(z_eye):
            row[f"z_eye{j + 1}"] = z
        arrays = dict(pre_ica=pre[:, ::step], removed=removed[:, ::step], src=rec.src[eye_ics][:, ::step],
                      final=final[:, ::step], pre_win=pre_win)
        return row, arrays

    # ------------------------------------------------------------------ blocks
    blocks = [b for b in BLOCK_ORDER if b in set(trials.block)] + sorted(set(trials.block) - set(BLOCK_ORDER))
    per_trial_s = None
    # a calibration that used up half the budget: stop here, the next run starts with the blocks
    skip_blocks = budget is not None and time.time() - t_start > 0.5 * budget
    for block in ([] if skip_blocks else blocks):
        f_csv, f_npz = out / "blocks" / f"{block}.csv", out / "blocks" / f"{block}.npz"
        if f_csv.is_file() and f_npz.is_file():
            continue
        rows = trials[trials.block == block]
        if budget is not None and per_trial_s is not None and \
                time.time() - t_start + per_trial_s * len(rows) * 1.2 > budget:
            break
        tic = time.time()
        recs, arrs = [], {}
        for r in rows.itertuples():
            row, arrays = one_trial(r.stem)
            recs.append(dict(stem=r.stem, **row))
            for k, v in arrays.items():
                arrs.setdefault(k, []).append(np.asarray(v, dtype=np.float32))
        per_trial_s = max(per_trial_s or 0.0, (time.time() - tic) / max(len(rows), 1))
        res = rows.merge(pd.DataFrame(recs), on="stem", how="left")
        np.savez(f_npz, stems=np.asarray(rows.stem, dtype=str), **{k: np.stack(v) for k, v in arrs.items()})
        res.to_csv(f_csv, index=False)                    # written last: a block counts as done when its csv exists
        say(f"  {block:12s} {len(rows):4d} pulses  {time.time() - tic:5.1f} s  "
            f"(ocular z > {thr:g}: {int(res.rejected_ocular.sum())})")
        if budget is not None and time.time() - t_start > budget:
            break
    rec.remove()
    complete = all((out / "blocks" / f"{b}.csv").is_file() for b in blocks)

    # ------------------------------------------------------------------ calibration info and meta (every run)
    comp = None
    try:
        comp = np.asarray(bundle["ica"].get_components(), dtype=float)
    except Exception:
        pass
    ic = calinfo.get("iclabel", {})
    np.savez(out / "calibration_info.npz", ch_names=np.asarray(ch_names), eye_ics=np.asarray(eye_ics),
             eye_mean=np.asarray([stats[i]["mean"] for i in eye_ics]), eye_std=np.asarray([stats[i]["std"] for i in eye_ics]),
             components=comp if comp is not None else np.zeros((0, 0)),
             iclabel_labels=np.asarray(ic.get("labels", []), dtype=str),
             iclabel_proba_all=np.asarray(ic.get("proba_all", np.zeros((0, 0)))),
             iclabel_proba_pred=np.asarray(ic.get("proba_pred", [])),
             bad_channels=np.asarray(list(map(str, bundle["bad_channels"])), dtype=str),
             dropped_ocular=np.asarray(bundle["trials_dropped"]["ocular"], dtype=int),
             dropped_qc=np.asarray(bundle["trials_dropped"]["qc"], dtype=int),
             dropped_post=np.asarray(bundle["trials_dropped"]["post"], dtype=int))

    if complete:
        scores = pd.concat([pd.read_csv(out / "blocks" / f"{b}.csv") for b in blocks], ignore_index=True)
        scores.insert(0, "setup", setup)
        scores.insert(0, "subject", subject)
        scores["online_reason"] = scores["online_reason"].fillna("") if "online_reason" in scores else ""
        scores.to_csv(out / "scores.csv", index=False)
        checks.update(internal_checks(scores, thr))
        if setup == "online":
            checks.update(online_checks(scores))
        else:
            cal = scores[scores.block == "calibration"]
            drop = set(map(int, bundle["trials_dropped"]["ocular"]))
            mism = int((cal.rejected_ocular != cal.trial_in_stage.isin(drop)).sum())
            checks["calibration: ocular z > threshold vs the re-calibration's ocular drops"] = dict(
                ok=mism == 0, detail=f"{len(cal) - mism}/{len(cal)} agree")

    meta = dict(setup=setup, subject=subject, run=sp["run"], complete=complete, commit=commit,
                commit_subject=manifest["commit_subject"], commit_date=manifest["commit_date"], code_dir=str(code_dir),
                overrides=overrides, z_threshold=thr, eye_ics=eye_ics,
                eye_stats={str(i): dict(mean=float(stats[i]["mean"]), std=float(stats[i]["std"]),
                                        n_gate_samples=int(len(stats[i]["time_indices_of_interest"])))
                           for i in eye_ics},
                gate_window_s=[float(rec.times[gate_idx[0]]) if rec.times is not None else None, None],
                post_sfreq=rate, store_sfreq=STORE_SFREQ, bad_channels=list(map(str, bundle["bad_channels"])),
                n_ica_components=int(bundle["ica"].n_components_), calibration_source=calinfo.get("source"),
                post_mad_thresholds=dict(global_=glob_thr, local=loc_thr), reject_range_s=reject_range,
                baseline_s=[float(v) for v in pp._cfg.baseline],
                artifact_window_1_s=[float(v) for v in pp._cfg.artifact_window_1],
                post_lowpass=bool(bundle.get("post_lowpass", False)),
                ch_names=ch_names, times_pre_ica_ms=list(np.round(np.arange(-30, 100.5, 1.0), 3)),
                times_final_ms=list(np.round(np.arange(1, 100.5, 1.0), 3)),
                times_pre_win_ms=list(np.round(np.arange(-210, -9.5, 1.0), 3)),
                forward=str(FORWARD_PATH), forward_sha256=hashlib.sha256(FORWARD_PATH.read_bytes()).hexdigest()[:16],
                versions=dict(python=platform.python_version(), numpy=np.__version__, mne=mne.__version__,
                              pandas=pd.__version__, scipy=__import__("scipy").__version__),
                checks=checks, written=datetime.now().isoformat(timespec="seconds"))
    (out / "meta.json").write_text(json.dumps(to_jsonable(meta), indent=1), encoding="utf-8")
    if complete:
        bad = {k: v for k, v in checks.items() if isinstance(v, dict) and "ok" in v and not v["ok"]}
        say(f"[{setup} / {subject}] done in {time.time() - t_start:.0f} s; checks: "
            + ("all pass" if not bad else "; ".join(f"{k}: {v['detail']}" for k, v in bad.items())))
    else:
        say(f"[{setup} / {subject}] stopped after {time.time() - t_start:.0f} s (budget); run again to continue")
    return complete


def internal_checks(scores, thr):
    c = {}
    agree = (scores.rejected_ocular == scores.reason.fillna("").eq("ocular")).sum()
    c["replay: ocular z > threshold vs preprocess_post's own reason"] = dict(
        ok=int(agree) == len(scores), detail=f"{int(agree)}/{len(scores)} agree")
    s = scores.same_as_returned.map({"True": True, "False": False, True: True, False: False}).dropna().astype(bool)
    c["replay: dipole window of last_post_full == preprocess_post output"] = dict(
        ok=bool(s.all()), detail=f"{int(s.sum())}/{len(s)} accepted pulses identical")
    return c


def online_checks(scores):
    c = {}
    iv = scores[scores.block.str.startswith("iv")]
    if len(iv):
        a = iv.reason.fillna("").to_numpy() == iv.online_reason.fillna("").to_numpy()
        c["intervention: replayed reject reason == online post_reject_reason"] = dict(
            ok=bool(a.all()), detail=f"{int(a.sum())}/{len(iv)} agree")
        o = (iv.reason.fillna("") == "ocular").to_numpy() == (iv.online_reason.fillna("") == "ocular").to_numpy()
        c["intervention: ocular decision == online"] = dict(ok=bool(o.all()), detail=f"{int(o.sum())}/{len(iv)} agree")
        acc = iv[(iv.reason.fillna("") == "") & iv.online_amp.notna()]
        if len(acc):
            rd = np.abs(acc.amp - acc.online_amp) / np.abs(acc.online_amp)
            c["intervention: N45 amplitude vs online tep_amplitude_raw, max rel. diff"] = dict(
                ok=bool(rd.max() < 1e-3), detail=f"{rd.max():.1e} over {len(acc)} pulses")
    cal = scores[scores.block == "calibration"]
    if len(cal):
        a = cal.rejected_ocular.to_numpy() == (cal.online_reason.fillna("") == "ocular").to_numpy()
        c["calibration: ocular z > threshold vs the bundle's ocular drops"] = dict(
            ok=bool(a.all()), detail=f"{int(a.sum())}/{len(cal)} agree")
    return c


def run_child(setup, subject, budget=None):
    """Run extract() in a fresh interpreter until it reports complete (no budget: one run does everything)."""
    cmd = [sys.executable, str(SCRIPT), "--extract", "--setup", setup, "--subject", subject]
    if budget:
        cmd += ["--budget", str(budget)]
    for _ in range(50):
        r = subprocess.run(cmd)
        if r.returncode == 0:
            return True
        if r.returncode != 3:
            raise RuntimeError(f"extraction failed for {setup} / {subject} (exit code {r.returncode})")
    raise RuntimeError(f"extraction of {setup} / {subject} did not finish")


def is_complete(setup, subject):
    f = setup_dir(setup, subject) / "meta.json"
    return f.is_file() and json.loads(f.read_text(encoding="utf-8")).get("complete", False) and \
        (setup_dir(setup, subject) / "scores.csv").is_file()


# %% 4 · Load the extracted scores --------------------------------------------------------------------------------------

def load_setup(setup, subjects=None):
    R = {}
    for s in subjects or SUBJECTS:
        if not is_complete(setup, s):
            continue
        d = setup_dir(setup, s)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        df = pd.read_csv(d / "scores.csv", keep_default_na=False, na_values=[""])
        for c in ("reason", "online_reason"):
            df[c] = df[c].fillna("").astype(str) if c in df else ""
        df["single"] = df.category != "prime_triplet"
        thr = meta["z_threshold"]
        g, l = meta["post_mad_thresholds"]["global_"], meta["post_mad_thresholds"]["local"]
        g_lo = -np.inf if g[0] is None else g[0]
        g_hi = np.inf if g[1] is None else g[1]
        df["mad_ok"] = (df.z_mad_global >= g_lo) & (df.z_mad_global <= g_hi) & (df.z_mad_local_max <= l)
        ci = dict(np.load(d / "calibration_info.npz", allow_pickle=True))
        R[s] = dict(meta=meta, df=df, cal=ci, dir=d, thr=thr, setup=setup, subject=s)
    return R


def load_arrays(res, kind, stems=None):
    """Arrays of one kind for the pulses in `stems` (default: all, in scores.csv order)."""
    want = list(res["df"].stem) if stems is None else list(stems)
    pos = {s: i for i, s in enumerate(want)}
    out = None
    for f in sorted((res["dir"] / "blocks").glob("*.npz")):
        z = np.load(f)
        st = [str(s) for s in z["stems"]]
        sel = [i for i, s in enumerate(st) if s in pos]
        if not sel:
            continue
        a = z[kind][sel]
        if out is None:
            out = np.full((len(want),) + a.shape[1:], np.nan, dtype=np.float32)
        out[[pos[st[i]] for i in sel]] = a
    return out



# %% 5 · Figures: style and helpers ------------------------------------------------------------------------------------
INK, INK_2, INK_MUTED, GRID_C, BASE_C = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
WASH = "#f3f2ee"                   # gate / N45 window shading
BAND_C = "#d9d7cf"                 # range of the clean pulses
REJ_C = "#b02a2a"                  # text colour of "rejected" labels (always with the word, never colour alone)
CAT_ORDER = ["open_loop", "predetermined_single", "prime_single_pulse", "prime_triplet"]
CAT_LABEL = {"open_loop": "open-loop single (baseline, calibration, evaluation)",
             "predetermined_single": "predetermined single (intervention)",
             "prime_single_pulse": "PRIME single (intervention)", "prime_triplet": "PRIME triplet (intervention)"}
CAT_SHORT = {"open_loop": "open-loop", "predetermined_single": "predet.", "prime_single_pulse": "PRIME",
             "prime_triplet": "triplet"}
BLOCK_SHORT = {"baseline": "base", "calibration": "cal"}
# Validated with the dataviz validator (all pairs, light surface): open-loop aqua, the two condition colours of
# PRIME_online_performance.ipynb, triplets violet. Markers are the secondary encoding in scatter plots.
CAT_COLOR = {"open_loop": "#1baf7a", "predetermined_single": "#5b8ff9", "prime_single_pulse": "#e8684a",
             "prime_triplet": "#6250d6"}
CAT_MARKER = {"open_loop": "o", "predetermined_single": "s", "prime_single_pulse": "D", "prime_triplet": "^"}
SETUP_STYLE = {"online": dict(color="#595959", ls="--", label="online (session code + its bundle)"),
               "current": dict(color="#2a78d6", ls="-", label="current decider (re-calibrated)")}
IC_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
BIN_COLORS = ["#86b6ef", "#5598e7", "#256abf", "#184f95", "#0d366b"]     # ordinal ramp, validated (5 steps)


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


def label_trial(r):
    return f"{BLOCK_SHORT.get(r.block, r.block)} #{int(r.trial_in_stage)}"


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  wrote {path.relative_to(OUT)}")


def sel_nearest(df, level, k, used):
    d = df.assign(dist=(df.z_max - level).abs()).sort_values("dist")
    return d[~d.stem.isin(used)].head(k)


def veog_traces(res, stems, pre=None, pw=None):
    """Mean of the VEOG channels: raw pre window (-210..-10 ms, average reference over the good channels, shifted to
    meet the gate epoch on -30..-15 ms) and the gate epoch itself (-30..100 ms, before ICA)."""
    meta = res["meta"]
    ch = meta["ch_names"]
    veog = [ch.index(c) for c in VEOG_CHANNELS]
    good = [i for i, c in enumerate(ch) if c not in set(meta["bad_channels"])]
    t_post, t_pre = np.asarray(meta["times_pre_ica_ms"]), np.asarray(meta["times_pre_win_ms"])
    pre = load_arrays(res, "pre_ica", stems) if pre is None else pre
    pw = load_arrays(res, "pre_win", stems) if pw is None else pw
    v_post = pre[:, veog, :].mean(axis=1)
    p = pw - pw[:, good, :].mean(axis=1, keepdims=True)
    v_pre = p[:, veog, :].mean(axis=1)
    ov_post, ov_pre = (t_post >= -30) & (t_post <= -15), (t_pre >= -30) & (t_pre <= -15)
    v_pre = v_pre - v_pre[:, ov_pre].mean(axis=1, keepdims=True) + v_post[:, ov_post].mean(axis=1, keepdims=True)
    return t_pre, v_pre, t_post, v_post


def clean_band(res, z_below=0.5):
    """5-95 % range of the VEOG traces of the single pulses with ocular z < z_below (the 'clean' reference)."""
    if "band" not in res:
        df = res["df"]
        stems = df.stem[df.single & (df.z_max < z_below)]
        t_pre, v_pre, t_post, v_post = veog_traces(res, stems)
        res["band"] = (t_pre, np.percentile(v_pre, [5, 95], axis=0), t_post, np.percentile(v_post, [5, 95], axis=0),
                       len(stems))
    return res["band"]


def draw_veog(ax, res, v_pre, v_post, band=True, lw=1.1, alpha=0.9, color=INK):
    t_pre, b_pre, t_post, b_post, _ = clean_band(res)
    ax.axvspan(*GATE_WINDOW_MS, color=WASH, lw=0, zorder=0)
    if band:
        ax.fill_between(t_pre, b_pre[0], b_pre[1], color=BAND_C, lw=0, zorder=1)
        ax.fill_between(t_post, b_post[0], b_post[1], color=BAND_C, lw=0, zorder=1)
    for vp, vq in zip(np.atleast_2d(v_pre), np.atleast_2d(v_post)):
        ax.plot(t_pre, vp, color=INK_MUTED, lw=lw * 0.9, alpha=alpha, zorder=2)
        ax.plot(t_post, vq, color=color, lw=lw, alpha=alpha, zorder=3)
    ax.axvline(0, color=BASE_C, lw=0.8, zorder=1)
    ax.set_xlim(-210, 100)


# %% 6 · Figures ------------------------------------------------------------------------------------------------------

def fig_eye_ics(R, setup):
    plt = setup_mpl()
    subs = [s for s in SUBJECTS if s in R]
    n_ic = max(len(R[s]["meta"]["eye_ics"]) for s in subs)
    H = 2.55 * len(subs) + 0.9
    fig = plt.figure(figsize=(2.3 * n_ic + 5.4, H))
    gs = fig.add_gridspec(len(subs), n_ic + 1, width_ratios=[1] * n_ic + [2.5], hspace=0.65, wspace=0.28,
                          **page(H, 0.85, 0.45))
    for i, s in enumerate(subs):
        res = R[s]
        ci, meta, df = res["cal"], res["meta"], res["df"]
        info = topo_info(list(ci["ch_names"]))
        P = ci.get("iclabel_proba_all")
        for j, ic in enumerate(meta["eye_ics"]):
            ax = fig.add_subplot(gs[i, j])
            comp = ci["components"]
            if comp.size:
                draw_topo(ax, comp[:, ic], info, np.max(np.abs(comp[:, ic])))
            p_eye = (f"P(eye) {P[ic, ICLABEL_CLASSES.index('eye blink')]:.2f}" if P is not None and P.size
                     else "P(eye) n/a")
            st = meta["eye_stats"][str(ic)]
            ax.set_title(f"{s} · IC {ic}\n{p_eye} · |s| {st['mean']:.2f} ± {st['std']:.2f}", fontsize=8, loc="center")
        ax = fig.add_subplot(gs[i, n_ic])
        singles = df[df.single]
        rng = np.random.default_rng(0)
        for j, ic in enumerate(meta["eye_ics"]):
            z = singles[f"z_eye{j + 1}"].to_numpy()
            ax.scatter(j + rng.uniform(-0.28, 0.28, len(z)), np.minimum(z, 10), s=5, color=IC_COLORS[j % 3],
                       alpha=0.35, lw=0)
            ax.text(j, 10.2, f"{pct_above(z, res['thr']):.1f} % > {res['thr']:g}", ha="center", va="bottom",
                    fontsize=7.5, color=INK_2)
        ax.axhline(res["thr"], color=INK, ls="--", lw=1.0)
        ax.set_ylim(-1, 11.2)
        ax.set_xticks(range(len(meta["eye_ics"])), [f"IC {ic}" for ic in meta["eye_ics"]])
        ax.set_ylabel("median z, 15–100 ms")
        ax.set_title("score of each eye IC, single pulses (values > 10 drawn at 10)", fontsize=8.5)
    fig.suptitle(f"The ICs the gate scores (ICLabel 'eye blink') · {SETUP_STYLE[setup]['label']}",
                 y=1 - 0.25 / H, fontsize=11)
    save(fig, OUT / "figures" / f"{setup}_0_eye_ics.png")


def fig_distribution(R, setup):
    plt = setup_mpl()
    subs = [s for s in SUBJECTS if s in R]
    fig, axes = plt.subplots(len(subs), 2, figsize=(13.5, 2.75 * len(subs) + 1.1), squeeze=False,
                             layout="constrained", gridspec_kw=dict(width_ratios=[2.3, 1]))
    edges = np.arange(-1.5, 8.0001, 0.25)
    for i, s in enumerate(subs):
        df, thr = R[s]["df"], R[s]["thr"]
        for j, (cats, title) in enumerate(((CAT_ORDER[:3], "single pulses"), (CAT_ORDER[3:], "PRIME triplets"))):
            ax = axes[i, j]
            sub = df[df.category.isin(cats)]
            bottom = np.zeros(len(edges) - 1)
            for c in cats:
                z = np.clip(sub.z_max[sub.category == c].to_numpy(), edges[0], edges[-1] - 1e-9)
                h, _ = np.histogram(z, edges)
                ax.bar(edges[:-1], h, width=np.diff(edges), bottom=bottom, align="edge", color=CAT_COLOR[c],
                       edgecolor="white", linewidth=0.6, label=CAT_LABEL[c] if i == 0 else None)
                bottom += h
            z_all = sub.z_max.to_numpy()
            rl = robust_line(z_all)
            ax.axvline(thr, color=INK, ls="--", lw=1.2)
            if j == 0:
                ax.axvline(rl, color=INK_2, ls=":", lw=1.3)
            n_over = int((z_all > edges[-1]).sum())
            txt = (f"n = {len(z_all)} · cutoff {thr:g} (dashed): {pct_above(z_all, thr):.1f} % rejected"
                   + (f"\nmedian + {ROBUST_K:g}·robust SD = {rl:.2f} (dotted)" if j == 0 else "")
                   + (f"\n{n_over} above {edges[-1]:g} (max {z_all.max():.1f}), counted in the last bin" if n_over else ""))
            ax.text(0.99, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK_2)
            ax.set_xlim(edges[0], edges[-1])
            ax.set_xlabel("ocular z")
            if j == 0:
                ax.set_ylabel("pulses")
            ax.set_title(f"{s} · {title}" + (f" · eye ICs {R[s]['meta']['eye_ics']}" if j == 0 else ""))
    handles, labels = [], []
    for ax in axes[0]:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    fig.suptitle(f"Ocular z of every pulse (max over eye ICs of the median z over 15–100 ms) · "
                 f"{SETUP_STYLE[setup]['label']}", fontsize=11)
    save(fig, OUT / "figures" / f"{setup}_1_distribution.png")


def fig_rejection_curves(R, setup):
    plt = setup_mpl()
    subs = [s for s in SUBJECTS if s in R]
    fig, axes = plt.subplots(1, len(subs), figsize=(4.7 * len(subs), 3.9), squeeze=False, sharey=True,
                             layout="constrained")
    for i, s in enumerate(subs):
        df, thr, ax = R[s]["df"], R[s]["thr"], axes[0, i]
        for c in CAT_ORDER:
            z = df.z_max[df.category == c].to_numpy()
            if len(z):
                ax.plot(SWEEP, [pct_above(z, x) for x in SWEEP], color=CAT_COLOR[c], lw=1.8,
                        label=f"{CAT_SHORT[c]} (n = {len(z)})")
        ax.axvline(thr, color=INK, ls="--", lw=1.1)
        ax.set_xlim(0, 6)
        ax.set_ylim(0, None)
        ax.set_xlabel("cutoff on ocular z")
        if i == 0:
            ax.set_ylabel("% of pulses rejected by the ocular gate")
        ax.set_title(f"{s} · session cutoff {thr:g} (dashed)")
        ax.legend(loc="upper right")
    fig.suptitle(f"What each cutoff costs, per condition · {SETUP_STYLE[setup]['label']}", fontsize=11)
    save(fig, OUT / "figures" / f"{setup}_2_rejection_curves.png")


def fig_topo_ladder(res):
    plt = setup_mpl()
    df, thr, s, setup = res["df"], res["thr"], res["subject"], res["setup"]
    singles = df[df.single]
    info = topo_info(res["meta"]["ch_names"])
    tg = gate_mask_ms(res["meta"]["times_pre_ica_ms"])
    rows, used = [], set()
    for lv in LADDER_LEVELS:
        pick = sel_nearest(singles, lv, LADDER_PER_LEVEL, used)
        if not len(pick) or pick.dist.min() > 0.5:
            continue
        used.update(pick.stem)
        rows.append((lv, pick.sort_values("z_max")))
    stems = [st for _, p in rows for st in p.stem]
    pre = load_arrays(res, "pre_ica", stems)
    pw = load_arrays(res, "pre_win", stems)
    t_pre, v_pre, t_post, v_post = veog_traces(res, stems, pre, pw)
    pos = {st: i for i, st in enumerate(stems)}
    k = LADDER_PER_LEVEL
    H = 1.62 * len(rows) + 1.3
    fig = plt.figure(figsize=(1.5 * k + 5.0, H))
    gs = fig.add_gridspec(len(rows), k + 4, width_ratios=[0.6] + [1] * k + [0.07, 0.62, 3.0], hspace=0.5,
                          wspace=0.08, **page(H, 1.05, 0.45))
    row_axes = []
    for i, (lv, pick) in enumerate(rows):
        maps = [pre[pos[st]][:, tg].mean(axis=1) for st in pick.stem]
        vmax = float(np.percentile(np.abs(np.concatenate(maps)), 99)) or 1.0
        axl = fig.add_subplot(gs[i, 0])
        axl.axis("off")
        axl.text(0.95, 0.5, f"z ≈ {lv:g}", ha="right", va="center", fontsize=10, color=INK, transform=axl.transAxes)
        axs = []
        for j, (r, m) in enumerate(zip(pick.itertuples(), maps)):
            ax = fig.add_subplot(gs[i, j + 1])
            im = draw_topo(ax, m, info, vmax)
            status = "rejected" if r.z_max > thr else "kept"
            ax.set_title(f"z {r.z_max:.2f} · {status}\n{label_trial(r)} · {CAT_SHORT[r.category]}", fontsize=6.8,
                         loc="center", color=INK if status == "kept" else REJ_C)
            axs.append(ax)
        row_axes.append(axs)
        cax = fig.add_subplot(gs[i, k + 1])
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=6.5)
        cb.set_label("µV", fontsize=7)
        axt = fig.add_subplot(gs[i, k + 3])
        idx = [pos[st] for st in pick.stem]
        draw_veog(axt, res, v_pre[idx], v_post[idx], lw=0.9, alpha=0.8)
        axt.tick_params(labelsize=7)
        if i == 0:
            n_clean = clean_band(res)[4]
            axt.set_title(f"{', '.join(VEOG_CHANNELS)} mean of these {k} pulses, µV (grey: raw pre window)\n"
                          f"band: 5–95 % of the {n_clean} pulses with z < 0.5; shaded: gate window", fontsize=7.5)
        if i < len(rows) - 1:
            axt.tick_params(labelbottom=False)
        else:
            axt.set_xlabel("ms from the pulse", fontsize=8)
    # the cutoff: a line between the last level below it and the first level above it
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    renderer = fig.canvas.get_renderer()
    for i in range(len(rows) - 1):
        if rows[i][0] <= thr < rows[i + 1][0]:
            lo = row_axes[i][0].get_window_extent(renderer).transformed(inv).y0
            hi = row_axes[i + 1][0].title.get_window_extent(renderer).transformed(inv).y1
            y = (lo + hi) / 2
            fig.add_artist(plt.Line2D([0.02, 0.985], [y, y], transform=fig.transFigure, color=REJ_C, ls="--", lw=1.3))
            fig.text(0.02, y + 0.002, f"cutoff {thr:g}: pulses below the line are kept", color=REJ_C,
                     va="bottom", fontsize=8.5)
            break
    fig.suptitle(f"{s} · {SETUP_STYLE[setup]['label']}\nSingle pulses at increasing ocular z: scalp map of the epoch "
                 f"the gate sees (before ICA cleaning), mean {GATE_WINDOW_MS[0]}–{GATE_WINDOW_MS[1]} ms, the {k} pulses "
                 f"nearest each level, colour scale per row", y=1 - 0.2 / H, fontsize=9.5)
    save(fig, OUT / "figures" / f"{setup}_3_topo_ladder_{s}.png")


def fig_threshold_cards(res, cutoff=None):
    plt = setup_mpl()
    df, s, setup = res["df"], res["subject"], res["setup"]
    thr = res["thr"] if cutoff is None else cutoff
    meta = res["meta"]
    singles = df[df.single]
    below = singles[singles.z_max <= thr].sort_values("z_max", ascending=False).head(CARDS_PER_SIDE).iloc[::-1]
    above = singles[singles.z_max > thr].sort_values("z_max").head(CARDS_PER_SIDE)
    pick = pd.concat([below, above])
    stems = list(pick.stem)
    pre = load_arrays(res, "pre_ica", stems)
    src = load_arrays(res, "src", stems)
    pw = load_arrays(res, "pre_win", stems)
    t_pre, v_pre, t_post, v_post = veog_traces(res, stems, pre, pw)
    info = topo_info(meta["ch_names"])
    tg = gate_mask_ms(t_post)
    eye = meta["eye_ics"]
    means = np.array([meta["eye_stats"][str(i)]["mean"] for i in eye])
    stds = np.array([meta["eye_stats"][str(i)]["std"] for i in eye])
    ncol = 4
    nrow = int(np.ceil(len(pick) / ncol))
    H = 2.9 * nrow + 1.2
    fig = plt.figure(figsize=(4.6 * ncol, H))
    outer = fig.add_gridspec(nrow, ncol, hspace=0.62, wspace=0.16, **page(H, 1.15, 0.55))
    for n, r in enumerate(pick.itertuples()):
        cell = outer[n // ncol, n % ncol].subgridspec(2, 2, width_ratios=[1.0, 1.75], hspace=0.12, wspace=0.42)
        m = pre[n][:, tg].mean(axis=1)
        ax = fig.add_subplot(cell[:, 0])
        vmax = float(np.max(np.abs(m))) or 1.0
        draw_topo(ax, m, info, vmax)
        ax.set_xlabel(f"±{vmax:.0f} µV", fontsize=7, color=INK_2, labelpad=1)
        status = "rejected" if r.z_max > thr else "kept"
        axt = fig.add_subplot(cell[0, 1])
        draw_veog(axt, res, v_pre[n], v_post[n])
        axt.tick_params(labelbottom=False, labelsize=7)
        axt.set_ylabel("Fp µV", fontsize=7)
        axt.set_title(f"{label_trial(r)} · {CAT_SHORT[r.category]} · z {r.z_max:.2f} · {status}", fontsize=8,
                      color=INK if status == "kept" else REJ_C, x=-0.62)
        axz = fig.add_subplot(cell[1, 1], sharex=axt)
        axz.axvspan(*GATE_WINDOW_MS, color=WASH, lw=0, zorder=0)
        for j in range(len(eye)):
            z = (np.abs(src[n][j]) - means[j]) / stds[j]
            axz.plot(t_post, z, color=IC_COLORS[j % 3], lw=1.0, label=f"IC {eye[j]}")
            axz.hlines(np.median(z[tg]), GATE_WINDOW_MS[0], GATE_WINDOW_MS[1], color=IC_COLORS[j % 3], lw=2.4)
        axz.axhline(thr, color=INK, ls="--", lw=0.9)
        axz.axvline(0, color=BASE_C, lw=0.8)
        axz.set_ylabel("eye-IC z", fontsize=7)
        axz.tick_params(labelsize=7)
        if n // ncol == nrow - 1:
            axz.set_xlabel("ms from the pulse", fontsize=7.5)
        if n == 0:
            axz.legend(fontsize=6.5, loc="upper left", ncol=len(eye), handlelength=1.0)
    fig.suptitle(f"{s} · {SETUP_STYLE[setup]['label']} · the {len(below)} single pulses just below and the "
                 f"{len(above)} just above the cutoff {thr:g}\nmap: gate-window mean before ICA cleaning · top: "
                 f"mean of {', '.join(VEOG_CHANNELS)} (grey: raw pre window; band: 5–95 % of pulses with z < 0.5) · "
                 f"bottom: eye-IC z(t), thick bars = the medians the gate compares with the cutoff", y=1 - 0.25 / H,
                 fontsize=9.5)
    save(fig, OUT / "figures" / f"{setup}_4_threshold_cards_{s}.png")


AMP_BIN_EDGES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]


def rel_amplitude(df):
    """Single pulses passing the post MAD gates; N45 amplitude ÷ the session's median over the open-loop pulses with
    ocular z < 0.5 (so sessions can be pooled and 1 means 'like a clean open-loop pulse')."""
    d = df[df.single & df.mad_ok].copy()
    d["rel_amp"] = d.amp / d.amp[(d.category == "open_loop") & (d.z_max < 0.5)].median()
    return d


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
    """Kept = ocular z <= cutoff (and the MAD gates passed): mean and SEM of the relative N45 amplitude, as ratios to
    keeping every pulse (no ocular gate)."""
    z, a = d.z_max.to_numpy(), d.rel_amp.to_numpy()
    ref_mean, ref_sem = np.mean(a), np.std(a, ddof=1) / np.sqrt(len(a))
    rows = []
    for c in SWEEP:
        k = a[z <= c]
        ok = len(k) > 2
        rows.append(dict(cutoff=c, n=len(k), kept_pct=100 * len(k) / len(a),
                         mean_ratio=np.mean(k) / ref_mean if ok else np.nan,
                         sem_ratio=(np.std(k, ddof=1) / np.sqrt(len(k))) / ref_sem if ok else np.nan,
                         median_ratio=np.median(k) / np.median(a) if ok else np.nan))
    return pd.DataFrame(rows)


def _amp_panels(ax_a, ax_b, d, thr_lines, pooled):
    from scipy.stats import spearmanr
    series = [("open-loop singles", (d.category == "open_loop").to_numpy(), CAT_COLOR["open_loop"], "-"),
              ("all single pulses", np.ones(len(d), bool), INK, "--")]
    if not pooled:
        for c in CAT_ORDER[:3]:
            m = (d.category == c).to_numpy()
            ax_a.scatter(np.minimum(d.z_max[m], 8), np.minimum(d.rel_amp[m], 3), s=8, color=CAT_COLOR[c],
                         marker=CAT_MARKER[c], alpha=0.35, lw=0, zorder=1, label=CAT_SHORT[c])
    for name, m, color, ls in series:
        bs = binned_stats(d.z_max[m], d.rel_amp[m], AMP_BIN_EDGES, np.median)
        ax_a.errorbar(bs.z_med, bs.est, yerr=[bs.est - bs.ci_lo, bs.ci_hi - bs.est], color=color, ls=ls, lw=1.6,
                      marker="o", ms=4, capsize=2, zorder=3, label=f"{name}: median per bin, 95 % CI")
        if ax_b is not None:
            sd = binned_stats(d.z_max[m], d.rel_amp[m], AMP_BIN_EDGES, robust_sd)
            ax_b.errorbar(sd.z_med, sd.est, yerr=[sd.est - sd.ci_lo, sd.ci_hi - sd.est], color=color, ls=ls, lw=1.6,
                          marker="o", ms=4, capsize=2, label=name)
        if pooled and name.startswith("open"):
            for r in bs.itertuples():
                ax_a.text(r.z_med, r.ci_hi + 0.01, f"{r.n}", fontsize=6.5, color=INK_MUTED, ha="center", va="bottom")
    for ax in (ax_a, ax_b):
        if ax is None:
            continue
        for t in thr_lines:
            ax.axvline(t, color=INK, ls="--", lw=1.0)
        ax.set_xlim(-0.8, 8.2)
    ax_a.axhline(1.0, color=BASE_C, lw=0.9, zorder=0)
    if not pooled:
        ax_a.set_ylim(0, 3.05)
    ol = d[(d.category == "open_loop")]
    kept = ol[ol.z_max <= max(thr_lines)]
    return spearmanr(kept.z_max, kept.rel_amp)


def _sweep_panel(ax, d, thr_lines, tag, legend):
    sw = amplitude_sweep(d)
    ax.plot(sw.cutoff, sw.mean_ratio, color="#eb6834", lw=1.8, label="mean of the kept pulses")
    ax.plot(sw.cutoff, sw.sem_ratio, color="#2a78d6", lw=1.8, label="SEM of that mean")
    ax.axhline(1.0, color=BASE_C, lw=0.9)
    for t in thr_lines:
        ax.axvline(t, color=INK, ls="--", lw=1.0)
    ax.set_xlim(0, 6)
    lo = np.nanmin([sw.mean_ratio[sw.cutoff >= 0.25].min(), 0.9])
    hi = np.nanmax([sw.sem_ratio[sw.cutoff >= 0.25].max(), 1.1])
    ax.set_ylim(lo - 0.02, hi + 0.02)
    ax.set_ylabel("ratio to no ocular gate")
    kept = ", ".join(f"{t:g}: {sw.loc[(sw.cutoff - t).abs().idxmin(), 'kept_pct']:.0f} %" for t in (1.0, 1.5, 2.0, 2.5))
    ax.set_title(f"{tag} · cutoff sweep · pulses kept at {kept}", fontsize=8.5)
    if legend:
        ax.legend(loc="upper right", fontsize=7.5)


def fig_amplitude(R, setup):
    plt = setup_mpl()
    subs = [s for s in SUBJECTS if s in R]
    fig = plt.figure(figsize=(15, 3.0 * (1 + len(subs)) + 1.2), layout="constrained")
    gs = fig.add_gridspec(1 + len(subs), 3)
    D = pd.concat([rel_amplitude(R[s]["df"]).assign(subject=s) for s in subs], ignore_index=True)
    thr_all = sorted({R[s]["thr"] for s in subs})
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, j]) for j in range(3))
    rho = _amp_panels(ax_a, ax_b, D, thr_all, pooled=True)
    ax_a.set_title(f"{len(subs)} sessions pooled · median N45 amplitude per ocular-z bin\n(÷ each session's clean "
                   f"open-loop median; numbers: open-loop pulses per bin)", fontsize=8.5)
    ax_a.set_ylabel("relative N45 amplitude")
    ax_a.legend(loc="lower right", fontsize=7)
    ax_b.set_title("trial-to-trial spread: robust SD per ocular-z bin", fontsize=8.5)
    ax_b.set_ylabel("robust SD (relative units)")
    _sweep_panel(ax_c, D, thr_all, f"{len(subs)} sessions pooled", legend=True)
    for i, s in enumerate(subs, start=1):
        d = rel_amplitude(R[s]["df"])
        ax_a = fig.add_subplot(gs[i, 0:2])
        rho = _amp_panels(ax_a, None, d, [R[s]["thr"]], pooled=False)
        ax_a.set_title(f"{s} · single pulses passing the MAD gates (amplitude > 3 or z > 8 drawn at the edge) · "
                       f"ρ(amplitude, z) among kept open-loop pulses {rho.statistic:+.2f} (p {rho.pvalue:.1g})",
                       fontsize=8.5)
        ax_a.set_ylabel("relative N45 amplitude")
        if i == 1:
            ax_a.legend(loc="upper right", fontsize=7, ncol=5)
        _sweep_panel(fig.add_subplot(gs[i, 2]), d, [R[s]["thr"]], s, legend=False)
        if i == len(subs):
            ax_a.set_xlabel("ocular z")
            fig.axes[-1].set_xlabel("cutoff (kept: ocular z ≤ cutoff)")
    ax_b.set_xlabel("ocular z (bin median)")
    fig.suptitle(f"Does ocular activity leak into the N45? Single-trial dipole amplitude (fit_trial after ICA, SOUND, "
                 f"SSP-SIR; MAD gates applied) against ocular z · {SETUP_STYLE[setup]['label']}", fontsize=10.5)
    save(fig, OUT / "figures" / f"{setup}_5_amplitude_vs_z.png")


def fig_bins(res):
    plt = setup_mpl()
    df, thr, s, setup, meta = res["df"], res["thr"], res["subject"], res["setup"], res["meta"]
    d = df[df.single].copy()
    edges = list(Z_BIN_EDGES)
    labels = [f"z < {edges[0]:g}"] + [f"{a:g} ≤ z < {b:g}" for a, b in zip(edges[:-1], edges[1:])] + [f"z ≥ {edges[-1]:g}"]
    d["bin"] = np.digitize(d.z_max, edges)
    pre = load_arrays(res, "pre_ica", d.stem)
    rem = load_arrays(res, "removed", d.stem)
    fin = load_arrays(res, "final", d.stem)
    info = topo_info(meta["ch_names"])
    ch = meta["ch_names"]
    t_post, t_fin = np.asarray(meta["times_pre_ica_ms"]), np.asarray(meta["times_final_ms"])
    tg = gate_mask_ms(t_post)
    tn = (t_fin >= N45_WINDOW_MS[0]) & (t_fin <= N45_WINDOW_MS[1])
    veog = [ch.index(c) for c in VEOG_CHANNELS]
    roi = [ch.index(c) for c in ROI_CHANNELS]
    nb = len(labels)
    agg = {}
    for b in range(nb):
        m = (d.bin == b).to_numpy()
        if m.sum() < 3:
            continue
        P, Q, F = pre[m], rem[m], fin[m]
        agg[b] = dict(n=int(m.sum()),
                      pre_rms=np.sqrt(np.mean(P[:, :, tg] ** 2, axis=(0, 2))),
                      rem_rms=np.sqrt(np.mean(Q[:, :, tg] ** 2, axis=(0, 2))),
                      fin_mean=F[:, :, tn].mean(axis=(0, 2)),
                      fin_rms=np.sqrt(np.mean(F[:, :, tn] ** 2, axis=(0, 2))),
                      pre_veog_sd=P[:, veog, :].mean(axis=1).std(axis=0),
                      fin_roi_mean=F[:, roi, :].mean(axis=1).mean(axis=0),
                      fin_veog_sd=F[:, veog, :].mean(axis=1).std(axis=0))
    cols = [("pre_rms", "before ICA\nRMS, 15–100 ms", True),
            ("rem_rms", "removed by eye ICs\nRMS, 15–100 ms", True),
            ("fin_mean", "after cleaning\nmean, 38–50 ms", False),
            ("fin_rms", "after cleaning\nRMS, 38–50 ms", True)]
    vmax = {key: (float(np.percentile(np.concatenate([agg[b][key] for b in agg]) if seq else
                                      np.abs(np.concatenate([agg[b][key] for b in agg])), 99)) or 1.0)
            for key, _, seq in cols}
    H = 1.7 * nb + 2.0
    fig = plt.figure(figsize=(15.5, H))
    gs = fig.add_gridspec(nb + 1, 7, width_ratios=[0.95, 1, 1, 1, 1, 0.3, 3.6], height_ratios=[1] * nb + [0.12],
                          hspace=0.35, wspace=0.1, **page(H, 1.0, 0.45))
    ims = {}
    for b in range(nb):
        axl = fig.add_subplot(gs[b, 0])
        axl.axis("off")
        n = agg[b]["n"] if b in agg else 0
        axl.text(0.86, 0.5, f"{labels[b]}\nn = {n}", ha="right", va="center", fontsize=8.5, transform=axl.transAxes)
        axl.plot([0.95], [0.5], marker="s", ms=9, color=BIN_COLORS[b % 5], transform=axl.transAxes, clip_on=False)
        for j, (key, title, seq) in enumerate(cols):
            ax = fig.add_subplot(gs[b, j + 1])
            if b in agg:
                ims[key] = draw_topo(ax, agg[b][key], info, vmax[key], cmap=sequential_cmap() if seq else None,
                                     vmin=0.0 if seq else None)
            else:
                ax.axis("off")
            if b == 0:
                ax.set_title(title, fontsize=8, loc="center")
    for j, (key, _, seq) in enumerate(cols):
        if key in ims:
            cax = fig.add_subplot(gs[nb, j + 1])
            cb = fig.colorbar(ims[key], cax=cax, orientation="horizontal")
            cb.ax.tick_params(labelsize=7)
            cb.set_label("µV", fontsize=7)
    inner = gs[:nb, 6].subgridspec(3, 1, hspace=0.5)
    traces = [("pre_veog_sd", t_post, "Fp1/Fpz/Fp2 mean before ICA: SD over pulses (log scale)", GATE_WINDOW_MS, True),
              ("fin_roi_mean", t_fin, f"left-M1 ROI ({', '.join(ROI_CHANNELS)}) after cleaning: mean", N45_WINDOW_MS, False),
              ("fin_veog_sd", t_fin, "Fp1/Fpz/Fp2 mean after cleaning: SD over pulses", N45_WINDOW_MS, False)]
    for k, (key, t, title, win, logy) in enumerate(traces):
        ax = fig.add_subplot(inner[k])
        ax.axvspan(*win, color=WASH, lw=0, zorder=0)
        for b in sorted(agg):
            ax.plot(t, agg[b][key], color=BIN_COLORS[b % 5], lw=1.5, label=f"{labels[b]} ({agg[b]['n']})")
        ax.axvline(0, color=BASE_C, lw=0.8)
        ax.set_xlim(t[0], 100)
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(1, None)
        ax.set_ylabel("µV")
        ax.set_title(title, fontsize=8.5)
        if k == 0:
            ax.legend(fontsize=7, ncol=2, loc="upper left")
        if k == 2:
            ax.set_xlabel("ms from the pulse")
    fig.suptitle(f"{s} · {SETUP_STYLE[setup]['label']} · single pulses grouped by ocular z (cutoff {thr:g}) · "
                 f"maps share one colour scale per column", y=1 - 0.25 / H, fontsize=10)
    save(fig, OUT / "figures" / f"{setup}_6_bins_{s}.png")


def fig_prime_check(R, setup):
    """Does PRIME's prediction track ocular activity? (it is trained on single-trial amplitudes)"""
    plt = setup_mpl()
    from scipy.stats import spearmanr
    subs = [s for s in SUBJECTS if s in R]
    fig, axes = plt.subplots(1, len(subs), figsize=(4.7 * len(subs), 3.9), squeeze=False, sharey=True,
                             layout="constrained")
    for i, s in enumerate(subs):
        df, thr, ax = R[s]["df"], R[s]["thr"], axes[0, i]
        forced = df.is_forced.astype(str).str.lower().eq("true") if "is_forced" in df else False
        parts = []
        for c in ("predetermined_single", "prime_single_pulse"):
            d = df[(df.category == c) & df.prediction_probability.notna() & ~forced]
            ax.scatter(np.minimum(d.z_max, 8), d.prediction_probability, s=10, color=CAT_COLOR[c], marker=CAT_MARKER[c],
                       alpha=0.55, lw=0, label=f"{CAT_SHORT[c]} (n = {len(d)})")
            r = spearmanr(d.z_max, d.prediction_probability)
            parts.append(f"{CAT_SHORT[c]} ρ {r.statistic:+.2f} (p {r.pvalue:.2g})")
        ax.axvline(thr, color=INK, ls="--", lw=1.0)
        ax.set_xlim(-1, 8.3)
        ax.set_xlabel("ocular z (values > 8 at 8)")
        if i == 0:
            ax.set_ylabel("PRIME prediction probability")
        ax.set_title(f"{s} · " + " · ".join(parts), fontsize=8)
        ax.legend(loc="lower right", fontsize=7.5)
    fig.suptitle(f"Does PRIME's prediction follow ocular activity? Intervention single pulses, forced ones excluded · "
                 f"{SETUP_STYLE[setup]['label']}\npredetermined pulses carry a prediction only if their post window passed "
                 f"online (the decider predicts after the gates), so their ocular z stops at the session's cutoff",
                 fontsize=10)
    save(fig, OUT / "figures" / f"{setup}_7_prime_vs_ocular.png")


BLOCK_RATE_CUTOFFS = [1.0, 1.5, 2.0, 2.5]


def fig_timeline(R, setup):
    plt = setup_mpl()
    subs = [s for s in SUBJECTS if s in R]
    fig = plt.figure(figsize=(16, 2.8 * len(subs) + 1.0), layout="constrained")
    gs = fig.add_gridspec(len(subs), 2, width_ratios=[3.1, 1])
    ramp = [BIN_COLORS[i] for i in (4, 3, 2, 1)]        # stricter cutoff: darker
    for i, s in enumerate(subs):
        df, thr = R[s]["df"], R[s]["thr"]
        ax = fig.add_subplot(gs[i, 0])
        ymax = min(10, np.nanmax(df.z_max) + 0.5)
        blocks = list(dict.fromkeys(df.block))
        for k, b in enumerate(blocks):
            g = df[df.block == b]
            x0, x1 = g.order.min() - 0.5, g.order.max() + 0.5
            if k % 2:
                ax.axvspan(x0, x1, color="#f6f5f2", lw=0, zorder=0)
            ax.text((x0 + x1) / 2, ymax, b, ha="center", va="bottom", fontsize=7.5, color=INK_2)
        for c in CAT_ORDER:
            m = df.category == c
            ax.scatter(df.order[m], np.minimum(df.z_max[m], ymax), s=7, color=CAT_COLOR[c], marker=CAT_MARKER[c],
                       alpha=0.6, lw=0, label=CAT_SHORT[c] if i == 0 else None)
        ax.axhline(thr, color=INK, ls="--", lw=1.0)
        ax.set_ylim(min(-1.5, np.nanmin(df.z_max) - 0.2), ymax)
        ax.set_xlim(-5, df.order.max() + 5)
        ax.set_ylabel("ocular z")
        ax.set_title(f"{s} · cutoff {thr:g} (values above {ymax:g} drawn at {ymax:g})", fontsize=8.5, pad=14)
        if i == len(subs) - 1:
            ax.set_xlabel("pulse, in recording order")
        if i == 0:
            ax.legend(loc="upper left", ncol=4, fontsize=7.5, bbox_to_anchor=(0, 1.32))
        ax2 = fig.add_subplot(gs[i, 1])
        sg = df[df.single]
        for c, col in zip(BLOCK_RATE_CUTOFFS, ramp):
            y = [pct_above(sg.z_max[sg.block == b], c) for b in blocks]
            ax2.plot(range(len(blocks)), y, color=col, marker="o", ms=4, lw=1.5, label=f"z > {c:g}")
        ax2.set_xticks(range(len(blocks)), [BLOCK_SHORT.get(b, b) for b in blocks], rotation=60, fontsize=7.5)
        ax2.set_ylim(0, None)
        ax2.set_ylabel("% of single pulses")
        ax2.set_title("ocular z above each cutoff, per block", fontsize=8.5)
        if i == 0:
            ax2.legend(fontsize=7, ncol=2, loc="best")
    fig.suptitle(f"Ocular z across the session · {SETUP_STYLE[setup]['label']}", fontsize=11)
    save(fig, OUT / "figures" / f"{setup}_8_timeline.png")


def equivalent_cutoff(z_from, c_from, z_to):
    """Cutoff on z_to that rejects the same fraction of pulses as c_from does on z_from."""
    q = np.mean(np.asarray(z_from) > c_from)
    return float(np.quantile(np.asarray(z_to), 1 - q)) if 0 < q < 1 else np.nan


def fig_setup_comparison(RR):
    plt = setup_mpl()
    from scipy.stats import spearmanr
    if not all(k in RR for k in ("online", "current")):
        return
    subs = [s for s in SUBJECTS if s in RR["online"] and s in RR["current"]]
    if not subs:
        return
    fig, axes = plt.subplots(len(subs), 2, figsize=(12.5, 3.8 * len(subs) + 0.9), squeeze=False, layout="constrained",
                             gridspec_kw=dict(width_ratios=[1, 1.3]))
    for i, s in enumerate(subs):
        a, b = RR["online"][s], RR["current"][s]
        m = a["df"][["stem", "z_max", "category", "single"]].merge(b["df"][["stem", "z_max"]], on="stem",
                                                                    suffixes=("_online", "_current"))
        ax = axes[i, 0]
        lim = [min(m.z_max_online.min(), m.z_max_current.min()) - 0.2, 8.2]
        for c in CAT_ORDER:
            mm = m.category == c
            ax.scatter(np.minimum(m.z_max_online[mm], 8), np.minimum(m.z_max_current[mm], 8), s=7, color=CAT_COLOR[c],
                       marker=CAT_MARKER[c], alpha=0.45, lw=0, label=CAT_SHORT[c] if i == 0 else None)
        ax.plot(lim, lim, color=BASE_C, lw=1.0)
        ax.axvline(a["thr"], color=SETUP_STYLE["online"]["color"], ls="--", lw=1.0)
        ax.axhline(b["thr"], color=SETUP_STYLE["current"]["color"], ls="--", lw=1.0)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("ocular z, online setup (values > 8 at 8)")
        ax.set_ylabel("ocular z, current decider")
        rho = spearmanr(m.z_max_online, m.z_max_current).statistic
        both = ((m.z_max_online > a["thr"]) == (m.z_max_current > b["thr"])).mean()
        ax.set_title(f"{s} · Spearman ρ {rho:.3f} · same decision for {100 * both:.1f} % of pulses", fontsize=8.5)
        ax = axes[i, 1]
        ms = m[m.single]
        for setup, col in (("online", "z_max_online"), ("current", "z_max_current")):
            st = SETUP_STYLE[setup]
            ax.plot(SWEEP, [pct_above(ms[col], x) for x in SWEEP], color=st["color"], ls=st["ls"], lw=1.8,
                    label=st["label"] if i == 0 else None)
        eq = equivalent_cutoff(ms.z_max_online, a["thr"], ms.z_max_current)
        p = pct_above(ms.z_max_online, a["thr"])
        ax.plot([a["thr"], eq], [p, p], color=INK_2, lw=0.8)
        ax.plot([a["thr"]], [p], marker="o", color=SETUP_STYLE["online"]["color"], ms=5)
        ax.plot([eq], [p], marker="o", color=SETUP_STYLE["current"]["color"], ms=5)
        ax.text(max(a["thr"], eq) + 0.15, p + 0.5, f"{p:.1f} % of single pulses: online {a['thr']:g} ≙ current {eq:.2f}",
                fontsize=7.5, color=INK_2, va="bottom")
        ax.set_xlim(0, 6)
        ax.set_ylim(0, None)
        ax.set_xlabel("cutoff on ocular z")
        ax.set_ylabel("% of single pulses rejected")
        ax.set_title(f"{s} · rejection curves of both pipelines", fontsize=8.5)
    h, l = [], []
    for ax in axes[0]:
        hh, ll = ax.get_legend_handles_labels()
        h += hh
        l += ll
    fig.legend(h, l, loc="outside lower center", ncol=6)
    fig.suptitle("Same pulses, two pipelines: how the ocular-z scale moved with the current decider", fontsize=11)
    save(fig, OUT / "figures" / "compare_9_online_vs_current.png")


# %% 7 · Tables ----------------------------------------------------------------------------------------------------------

def tables(RR):
    from scipy.stats import spearmanr, mannwhitneyu
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    rej, summ, checks, blocks, ampbins = [], [], [], [], []
    for setup, R in RR.items():
        for s, res in R.items():
            df, thr = res["df"], res["thr"]
            for name, v in res["meta"]["checks"].items():
                if isinstance(v, dict) and "ok" in v:
                    checks.append(dict(setup=setup, subject=s, check=name, ok=v["ok"], detail=v["detail"]))
                elif isinstance(v, dict):
                    for k2, v2 in v.items():
                        checks.append(dict(setup=setup, subject=s, check=f"{name}: {k2}", ok=v2["ok"], detail=v2["detail"]))
            cuts = sorted(set(CANDIDATE_CUTOFFS) | {thr})
            for g, d in [(c, df[df.category == c]) for c in CAT_ORDER] + [("all single pulses", df[df.single])]:
                rej.append(dict(setup=setup, subject=s, group=g, n=len(d), session_cutoff=thr,
                                **{f"% > {c:g}": round(pct_above(d.z_max, c), 2) for c in cuts}))
            sg = df[df.single]
            for b in dict.fromkeys(df.block):
                z = sg.z_max[sg.block == b]
                blocks.append(dict(setup=setup, subject=s, block=b, n_single=len(z),
                                   **{f"% > {c:g}": round(pct_above(z, c), 2) for c in cuts}))
            d = rel_amplitude(df)
            sw = amplitude_sweep(d)
            at = sw.loc[(sw.cutoff - thr).abs().idxmin()]
            ol = d[d.category == "open_loop"]
            r_amp = spearmanr(ol.z_max[ol.z_max <= thr], ol.rel_amp[ol.z_max <= thr])
            forced = df.is_forced.astype(str).str.lower().eq("true")
            pr = df[(df.category == "prime_single_pulse") & df.prediction_probability.notna() & ~forced]
            pd_ = df[(df.category == "predetermined_single") & df.prediction_probability.notna()]
            mw = mannwhitneyu(df.z_max[df.category == "prime_single_pulse"], df.z_max[df.category == "predetermined_single"])
            q = np.percentile(sg.z_max, [50, 75, 90, 95, 99])
            summ.append(dict(setup=setup, subject=s, cutoff=thr, eye_ics=" ".join(map(str, res["meta"]["eye_ics"])),
                             n_single=len(sg), z_median=q[0], z_p75=q[1], z_p90=q[2], z_p95=q[3], z_p99=q[4],
                             robust_line=robust_line(sg.z_max),
                             **{f"pct_single_gt_{c:g}": pct_above(sg.z_max, c) for c in CANDIDATE_CUTOFFS},
                             pct_triplet_at_cutoff=pct_above(df.z_max[~df.single], thr),
                             pct_prime_single_at_cutoff=pct_above(df.z_max[df.category == "prime_single_pulse"], thr),
                             pct_predetermined_at_cutoff=pct_above(df.z_max[df.category == "predetermined_single"], thr),
                             mwu_p_prime_vs_predetermined=mw.pvalue,
                             rho_amp_z_kept_openloop=r_amp.statistic, rho_amp_z_kept_openloop_p=r_amp.pvalue,
                             rho_pred_z_prime=spearmanr(pr.z_max, pr.prediction_probability).statistic,
                             rho_pred_z_predetermined=spearmanr(pd_.z_max, pd_.prediction_probability).statistic,
                             mean_ratio_at_cutoff=at.mean_ratio, sem_ratio_at_cutoff=at.sem_ratio,
                             n_kept_at_cutoff=int(at.n), n_mad_ok=len(d)))
        if R:
            D = pd.concat([rel_amplitude(R[s]["df"]) for s in R], ignore_index=True)
            for grp, dd in (("open-loop singles", D[D.category == "open_loop"]), ("all single pulses", D)):
                med = binned_stats(dd.z_max, dd.rel_amp, AMP_BIN_EDGES, np.median)
                sd = binned_stats(dd.z_max, dd.rel_amp, AMP_BIN_EDGES, robust_sd)
                for a, b in zip(med.itertuples(), sd.itertuples()):
                    ampbins.append(dict(setup=setup, group=grp, z_from=a.lo, z_to=a.hi, n=a.n, median_rel_amp=a.est,
                                        ci_lo=a.ci_lo, ci_hi=a.ci_hi, robust_sd=b.est, sd_ci_lo=b.ci_lo, sd_ci_hi=b.ci_hi))
    if "online" in RR and "current" in RR:
        for r in summ:
            s = r["subject"]
            if r["setup"] == "current" and s in RR["online"] and s in RR["current"]:
                a, b = RR["online"][s], RR["current"][s]
                m = a["df"][["stem", "z_max", "single"]].merge(b["df"][["stem", "z_max"]], on="stem", suffixes=("_on", "_cu"))
                ms = m[m.single]
                r["current_equiv_of_online_cutoff"] = equivalent_cutoff(ms.z_max_on, a["thr"], ms.z_max_cu)
    for name, rows in (("rejection_rates", rej), ("summary", summ), ("checks", checks), ("block_rates", blocks),
                       ("amplitude_by_z_bin", ampbins)):
        pd.DataFrame(rows).to_csv(OUT / "tables" / f"{name}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 60, "display.float_format", "{:.3g}".format):
        bad = [c for c in checks if not c["ok"]]
        print(f"\nChecks: {len(checks) - len(bad)}/{len(checks)} pass" + ("" if not bad else ":"))
        for c in bad:
            print(f"  FAIL {c['setup']} {c['subject']} {c['check']}: {c['detail']}")
        print("\nSummary (single pulses unless stated):")
        print(pd.DataFrame(summ).T.to_string(header=False))
        print("\nRelative N45 amplitude per ocular-z bin (sessions pooled):")
        print(pd.DataFrame(ampbins).to_string(index=False))
    print(f"\ntables written to {OUT / 'tables'}")


# %% 8 · Main ------------------------------------------------------------------------------------------------------------
FIGURES = ["eye_ics", "distribution", "curves", "ladder", "cards", "amplitude", "bins", "prime", "timeline", "compare"]


def make_figures(RR, which=None):
    which = set(which or FIGURES)
    for setup, R in RR.items():
        if not R:
            continue
        print(f"figures · {setup}")
        for name, fn in (("eye_ics", fig_eye_ics), ("distribution", fig_distribution), ("curves", fig_rejection_curves),
                         ("amplitude", fig_amplitude), ("prime", fig_prime_check), ("timeline", fig_timeline)):
            if name in which:
                fn(R, setup)
        for s, res in R.items():
            for name, fn in (("ladder", fig_topo_ladder), ("cards", fig_threshold_cards), ("bins", fig_bins)):
                if name in which:
                    fn(res)
    if "compare" in which:
        fig_setup_comparison(RR)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract", action="store_true", help="child mode: extract one setup x one subject")
    ap.add_argument("--setup", choices=["online", "current"])
    ap.add_argument("--subject")
    ap.add_argument("--budget", type=float, default=None, help="child: stop after about this many seconds (resumable)")
    ap.add_argument("--plots-only", action="store_true")
    ap.add_argument("--setups", nargs="*", default=None)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--figures", nargs="*", default=None, help=f"subset of {FIGURES}")
    ap.add_argument("--no-tables", action="store_true")
    args, _ = ap.parse_known_args(argv)

    if args.extract:
        if not (args.setup and args.subject):
            ap.error("--extract needs --setup and --subject")
        done = extract(args.setup, args.subject, budget=args.budget)
        sys.exit(0 if done else 3)

    setups = args.setups or SETUPS
    subjects = args.subjects or SUBJECTS
    if not args.plots_only:
        for setup in setups:
            for s in subjects:
                if not is_complete(setup, s):
                    run_child(setup, s)
    RR = {setup: load_setup(setup, subjects) for setup in setups}
    missing = [(st, s) for st in setups for s in subjects if s not in RR[st]]
    if missing:
        print("not extracted yet: " + ", ".join(f"{st}/{s}" for st, s in missing))
    if not args.no_tables:
        tables(RR)
    make_figures(RR, args.figures)


if __name__ == "__main__":
    main()
