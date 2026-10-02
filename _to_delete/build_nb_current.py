# -*- coding: utf-8 -*-
"""Generator script for PRIME_online_performance.ipynb. Not part of the deliverable."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

CELLS = []
def M(src): CELLS.append(new_markdown_cell(src.strip("\n")))
def C(src): CELLS.append(new_code_cell(src.strip("\n")))

# ============================================================================
# Title
# ============================================================================
M(r"""
# PRIME online performance — sub-pilot003 / ses-prime / run 103

This notebook extracts and describes what the NeuroSimo/PRIME closed-loop system logged during
one session: the calibration it fit at the start, the dipole and channel setup it used, how its
trial-by-trial predictions behaved, the timing of pulses, and the per-trial processing outcomes
recorded in `trials_intervention.csv`.

It is a **descriptive / diagnostic** notebook: every number below is read directly from the
session's output files (or computed from them with a stated formula) and compared against the
decider source (`prime/decider/`) so the description matches what the code actually does. It does
not run any inferential statistics (no p-values, no hypothesis tests) — for the statistical
question of whether PRIME's prediction discriminates TEP amplitude, see `online-TEPs.ipynb` and
the notebook this one replaced.

Session design (from the project description): a baseline block of open-loop single pulses, a
PRIME calibration block (125 single pulses), four intervention blocks (200 pulses each, a mix of
PRIME-triggered 100 Hz triplets, PRIME-triggered single pulses, and predetermined single pulses),
and evaluation blocks (100 single pulses each) at t0/t15/t30/t60 after the intervention.
""")

# ============================================================================
# 0/1 . Setup and session identification
# ============================================================================
M(r"""
## 1 · Session setup and identification

The session is identified by three strings below (`SUBJECT_ID`, `SESSION_ID`, `RUN_ID`), which
are joined onto a `Loop` root folder to find the NeuroSimo output directory:
`<LOOP_ROOT>/BIDS/sub-<SUBJECT_ID>/<SESSION_ID>/<RUN_ID>/prime`. To point this notebook at a
different session, edit the three strings and re-run — nothing else below refers to Pilot003
by name.

`LOOP_ROOT` is resolved automatically: the `LOOP_ROOT` environment variable if set, else the lab
Windows path `D:\Linus\Loop`, else `~/Desktop/Loop` (Mac), else the first parent of the current
working directory that contains both a `BIDS/` and a `prime/` folder.
""")

C(r"""
from __future__ import annotations

import os, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import mne

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)

%matplotlib inline
plt.rcParams.update({
    "figure.dpi": 100, "figure.figsize": (10, 3.4),
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})

# Colours used consistently whenever predetermined vs. PRIME-triggered trials are compared.
COLOR_PREDETERMINED = "#5b8ff9"   # blue
COLOR_PRIME_SINGLE   = "#e8684a"  # orange
COLOR_PRIME_TRIPLET  = "#9270ca"  # purple

print(f"numpy {np.__version__}  pandas {pd.__version__}  mne {mne.__version__}")
""")

C(r"""
# ---------------------------------------------------------------- session identification
SUBJECT_ID = "pilot003"     # BIDS subject label, without the "sub-" prefix
SESSION_ID = "ses-prime"    # BIDS session label
RUN_ID     = "103"          # NeuroSimo run folder

def _find_loop_root() -> Path:
    env = os.environ.get("LOOP_ROOT")
    candidates = [env] if env else []
    candidates += [r"D:\Linus\Loop", str(Path.home() / "Desktop" / "Loop")]
    for c in candidates:
        if c and (Path(c) / "BIDS").is_dir() and (Path(c) / "prime").is_dir():
            return Path(c)
    for d in (Path.cwd(), *Path.cwd().parents):
        if (d / "BIDS").is_dir() and (d / "prime").is_dir():
            return d
    raise FileNotFoundError(
        "Could not locate the Loop root (a folder containing both 'BIDS' and 'prime'). "
        "Set the LOOP_ROOT environment variable, or edit _find_loop_root()."
    )

LOOP_ROOT = _find_loop_root()
SUBJECT_DIR = LOOP_ROOT / "BIDS" / f"sub-{SUBJECT_ID}" / SESSION_ID / RUN_ID / "prime"
if not SUBJECT_DIR.is_dir():
    raise FileNotFoundError(f"Session folder not found: {SUBJECT_DIR}")

print(f"Loop root : {LOOP_ROOT}")
print(f"session   : sub-{SUBJECT_ID} / {SESSION_ID} / run {RUN_ID}")
print(f"data dir  : {SUBJECT_DIR}")
""")

C(r"""
# ---------------------------------------------------------------- inventory of the session folder
_all_files = sorted(SUBJECT_DIR.iterdir(), key=lambda p: p.name)
_non_trial_files = [p.name for p in _all_files
                     if not p.name.endswith(("_pre_raw.npy", "_post_raw.npy", "_post_proc.npy",
                                              "_pre_proc.npy", "_baseline.npy", "_trial_raw.npy"))
                     and not p.name.startswith("classifier_after_trial_")]
print(f"{len(_all_files)} files in the session folder.")
print(f"{len(_all_files) - len(_non_trial_files)} are per-trial raw/processed EEG buffers or "
      f"per-trial classifier checkpoints (not used below).")
print("\nFiles this notebook reads:")
for name in _non_trial_files:
    print(f"  {name}")
""")

# ============================================================================
# 2. calibration_bunble.npy + calibration_summary.csv
# ============================================================================
M(r"""
## 2 · Calibration bundle (`calibration_bunble.npy`) and `calibration_summary.csv`

*(The file name has a typo — "bunble" not "bundle" — carried over verbatim from
`2_intervention.py`; it is not a mistake in this notebook.)*

At the end of the 125-trial calibration block, `Preprocessor.calibrate()`
(`prime_core/preprocessing/preprocessor.py`) fits every parameter the online pipeline needs for
the rest of the session — bad channels, an artifact-rejection filter, ICA, spatial filters, and
per-stage rejection statistics — from the accumulated calibration trials, and saves the resulting
dict as `calibration_bunble.npy`. `2_intervention.py` prints `"Calibration used N trials"`
straight after, and separately writes that same number to `calibration_summary.csv`.

Trials are dropped from the 125 in three sequential stages (a trial dropped at one stage is not
re-considered at the next):

1. **ocular** — the trial's ICA source activity in the components later marked as eye-blink
   components looks too eye-blink-like (see the "ocular ICA components" table below).
2. **qc** — the pre-stimulus window fails the global/local MAD (median absolute deviation)
   outlier check used for the online QC gate.
3. **post** — the post-stimulus window fails the equivalent post-stimulus MAD check.

Trials surviving all three stages are `trials_kept`, and it is exactly this count that
`calibration_summary.csv` records as `num_calibration_trials`.
""")

C(r"""
calibration_bundle = np.load(SUBJECT_DIR / "calibration_bunble.npy", allow_pickle=True).item()
calibration_summary = pd.read_csv(SUBJECT_DIR / "calibration_summary.csv")

n_delivered = 125  # calibration pulses delivered this session (protocol parameter, not read from disk)
n_used = int(calibration_summary["num_calibration_trials"].iloc[0])
print(f"calibration_summary.csv -> num_calibration_trials = {n_used}")
print(f"calibration pulses delivered this session          = {n_delivered}")
print(f"overall rejection rate                              = {100 * (1 - n_used / n_delivered):.1f}%")
""")

C(r"""
# ---------------------------------------------------------------- rejection cascade
dropped = calibration_bundle["trials_dropped"]
kept = calibration_bundle["trials_kept"]

n0 = n_delivered
n_after_ocular = n0 - len(dropped["ocular"])
n_after_qc = n_after_ocular - len(dropped["qc"])
n_after_post = n_after_qc - len(dropped["post"])

cascade = pd.DataFrame([
    {"stage": "delivered",        "n_remaining": n0,             "n_dropped_here": 0},
    {"stage": "after ocular check","n_remaining": n_after_ocular, "n_dropped_here": len(dropped["ocular"])},
    {"stage": "after qc check",   "n_remaining": n_after_qc,      "n_dropped_here": len(dropped["qc"])},
    {"stage": "after post check", "n_remaining": n_after_post,    "n_dropped_here": len(dropped["post"])},
])
print("Calibration-trial rejection cascade (sequential; a trial dropped at one stage is not",
      "re-evaluated at the next):")
display(cascade)
assert n_after_post == len(kept) == n_used, "cascade does not reconcile with trials_kept / calibration_summary"
print(f"\n{len(kept)} trials kept -> used to fit the dipole location, the TEP normalizer and the "
      f"PRIME classifier (section 3 onward).")
""")

C(r"""
# ---------------------------------------------------------------- bad channels & interpolation
bad_channels = list(calibration_bundle["bad_channels"])
interp = calibration_bundle["channel_interpolation_info"]
print(f"bad channels detected at calibration ({len(bad_channels)}): {', '.join(bad_channels)}")
print(f"these are reconstructed online as a linear combination of the "
      f"{interp['interpolation_matrix'].shape[1]} good channels "
      f"(interpolation matrix shape {interp['interpolation_matrix'].shape}).")
""")

C(r"""
# ---------------------------------------------------------------- QC filter + MAD rejection stats
qc_filter = calibration_bundle["qc_filter"]
print(f"pre-stimulus artifact filter: {len(qc_filter)} coefficient arrays "
      f"(shapes {[np.asarray(a).shape for a in qc_filter]}) — the (b, a) coefficients of the "
      f"session-specific Butterworth band-pass applied to every pre-stimulus window.")

for label, key in [("pre-stimulus (QC) window", "good_trial_stats_qc"),
                    ("post-stimulus window", "good_trial_stats_post")]:
    stats_ = calibration_bundle[key]
    print(f"\n{label} MAD statistics, fit on {len(stats_['mads'])} trials:")
    print(f"  mean(MAD) = {stats_['mads_mean']:.4f}   std(MAD) = {stats_['mads_std']:.4f}")
    print(f"  -> online, a new trial's z-score is (MAD(trial) - {stats_['mads_mean']:.4f}) / {stats_['mads_std']:.4f}, "
          f"compared against a fixed threshold (see section 8.3 for the post-stimulus thresholds).")
""")

C(r"""
# ---------------------------------------------------------------- ocular ICA components
ocular_info = calibration_bundle["ocular_threshold_post"]
rows = []
for comp_idx, info in ocular_info.items():
    rows.append({
        "ICA component": int(comp_idx),
        "n time samples characterised": len(info["time_indices_of_interest"]),
        "calibration mean |activation|": info["mean"],
        "calibration std |activation|": info["std"],
    })
print(f"{len(ocular_info)} ICA component(s) characterised as ocular/eye-blink components at "
      f"calibration. Online, a trial is ocular-rejected if its median z-score in either "
      f"component (over the same time samples) exceeds the ocular z-threshold (2.0, see 8.3):")
display(pd.DataFrame(rows))
""")

C(r"""
# ---------------------------------------------------------------- ICA and spatial filters
ica = calibration_bundle["ica"]
print(f"ICA: method={ica.method!r}  n_components={ica.n_components_}  "
      f"components excluded={list(ica.exclude)}  fit on {ica.n_samples_} samples")
print(f"\nSpatial artifact-suppression filters stored in the bundle (established TMS-EEG methods; "
      f"shapes shown to confirm they were fit, values are session-specific and not printed):")
for key, note in [
    ("sound_filter", "SOUND (Source-estimate Utilizing Noise Discarding) spatial filter"),
    ("sspsir_suppression_matrix_P", "SSP-SIR suppression projector"),
    ("sspsir_filter_kernel", "SSP-SIR temporal filter kernel"),
    ("sspsir_sir_projmat_suppr", "SSP-SIR signal-in-residual projector (suppressed)"),
    ("sspsir_sir_projmat_orig", "SSP-SIR signal-in-residual projector (original)"),
]:
    arr = calibration_bundle[key]
    print(f"  {key:28s} shape={np.asarray(arr).shape}   {note}")
""")
# ============================================================================
# 3. dipole_fitting_info.npy
# ============================================================================
M(r"""
## 3 · Dipole fitting info (`dipole_fitting_info.npy`)

Written by `DipoleFitter.calibrate()` (`prime_core/preprocessing/dipole_fitter.py`) at the end of
calibration. It fixes, for the rest of the session, **where** in the brain, **which orientation**,
and **which time samples** every single-trial dipole amplitude is computed from:

1. A free-orientation dipole is fit independently at **every** fsaverage source-space candidate
   position (20484 ico-5 vertices), at **every** sample of the calibration-averaged evoked
   response (`dipoles_for_indices`) — for each sample, the position with the highest R² wins.
2. `determine_optimal_index_range` then searches over every contiguous window of `min_window_size`
   to `max_window_size` samples (3 to 6 samples, `DipoleFitter`'s constructor defaults) within the
   13-sample [38, 50] ms dipole epoch, and ranks each candidate window on three criteria computed
   across its samples' winning dipoles: its **minimum amplitude** (large is good — the window
   shouldn't include a near-zero moment), its **position stability** (mean distance of each
   sample's position from the window's mean position — small is good), and its **orientation
   stability** (mean angle of each sample's orientation from the window's mean orientation — small
   is good). Each criterion is converted to a rank across all candidate windows, and the winner
   maximizes `size_rank**1.5 / (position_rank + orientation_rank + amplitude_rank)` — so, among
   windows that are similarly stable and strong, a longer window is preferred.
3. Within that winning window, `position_index` is the source-space vertex closest to the
   amplitude-weighted average dipole position across the window's samples, and `orientation` is
   the amplitude-weighted average dipole orientation (unit vector) across the same samples.

Every single-trial `tep_amplitude_raw` afterwards (section 8.8) is this **one fixed dipole**
(position + orientation), projected through the forward-model leadfield onto that trial's
post-stimulus data — not a fresh fit per trial.
""")

C(r"""
dipole_info = np.load(SUBJECT_DIR / "dipole_fitting_info.npy", allow_pickle=True).item()
proc_info_preview = np.load(SUBJECT_DIR / "proc_info.npy", allow_pickle=True).item()  # for sample->ms below

pos_idx = dipole_info["position_index"]
orientation = np.asarray(dipole_info["orientation"])
idx_start, idx_end = dipole_info["index_range"]

dipole_epoch_tmin_ms = proc_info_preview["dipole"]["tmin"] * 1000
dipole_epoch_tmax_ms = proc_info_preview["dipole"]["tmax"] * 1000
dt_ms = 1000.0 / proc_info_preview["sfreq"]
# idx_end is an exclusive slice bound (trial_data[:, start:end]); the *last included* sample is idx_end - 1.
window_tmin_ms = dipole_epoch_tmin_ms + idx_start * dt_ms
window_tmax_ms = dipole_epoch_tmin_ms + (idx_end - 1) * dt_ms

print(f"fsaverage source-space vertex index : {pos_idx}  (of 20484 ico-5 candidate positions)")
print(f"dipole orientation (unit vector, x/y/z): {np.round(orientation, 4)}  "
      f"(norm = {np.linalg.norm(orientation):.4f})")
print(f"optimal fitting window, as sample indices : [{idx_start}, {idx_end})  "
      f"= {idx_end - idx_start} of the {round((dipole_epoch_tmax_ms - dipole_epoch_tmin_ms) / dt_ms) + 1} epoch samples")
print(f"optimal fitting window, as a time range    : {window_tmin_ms:.0f}-{window_tmax_ms:.0f} ms  "
      f"(within the {dipole_epoch_tmin_ms:.0f}-{dipole_epoch_tmax_ms:.0f} ms dipole epoch)")
""")

C(r"""
# ---------------------------------------------------------------- dipole position + orientation figure
fwd_path = LOOP_ROOT / "prime" / "offline_data" / "fsaverage" / "fsaverage-fwd.fif"

if not fwd_path.is_file():
    print(f"forward solution not found at {fwd_path} -- offline_data syncs separately from the rest "
          f"of the Loop tree and may not be present on this machine; skipping the position/orientation figure.")
else:
    dipole_forward = mne.read_forward_solution(str(fwd_path), verbose=False)
    source_rr = dipole_forward["source_rr"]           # (20484, 3) metres, MNE head coordinates
    dipole_pos = source_rr[pos_idx]

    # Arrow length is for visibility only (orientation is a unit vector) -- scaled to the source
    # space's own spatial extent, not to any physical dipole-moment unit.
    arrow_len = 0.15 * np.ptp(source_rr, axis=0).mean()
    arrow_vec = orientation * arrow_len

    views = [("x", "y", "top view (x = right, y = anterior)"),
             ("x", "z", "front view (x = right, z = superior)"),
             ("y", "z", "side view (y = anterior, z = superior)")]
    axis_idx = {"x": 0, "y": 1, "z": 2}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, (h, v, title) in zip(axes, views):
        hi, vi = axis_idx[h], axis_idx[v]
        ax.scatter(source_rr[:, hi], source_rr[:, vi], s=1, color="lightgray", alpha=0.4,
                   label="candidate source positions" if ax is axes[0] else None)
        ax.scatter(dipole_pos[hi], dipole_pos[vi], s=150, color="crimson", zorder=5,
                   label="fitted dipole position" if ax is axes[0] else None)
        ax.annotate("", xy=(dipole_pos[hi] + arrow_vec[hi], dipole_pos[vi] + arrow_vec[vi]),
                    xytext=(dipole_pos[hi], dipole_pos[vi]),
                    arrowprops=dict(arrowstyle="-|>", color="crimson", lw=2, mutation_scale=16))
        ax.set_xlabel(f"{h} (m)")
        ax.set_ylabel(f"{v} (m)")
        ax.set_title(title)
        ax.set_aspect("equal")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Fitted dipole -- source-space vertex {pos_idx}, position {np.round(dipole_pos, 4)} m "
                 f"(MNE head coordinates), orientation {np.round(orientation, 3)}  (arrow direction is exact, "
                 f"its length is not to scale)", y=1.05, fontsize=10)
    fig.tight_layout()
    plt.show()
    print("Red arrow = fitted orientation. Gray cloud = every candidate fsaverage source position "
          "considered during calibration (both hemispheres); the fitted position is a small subset "
          "(one vertex) of this cloud, highlighted in red.")
"""
)

# ============================================================================
# 4. proc_info.npy
# ============================================================================
M(r"""
## 4 · Processing info (`proc_info.npy`)

Written alongside the calibration bundle. Records the channel layout and the four time windows
the online pipeline uses at 1 kHz processed sampling rate (see `prime/decider/configs/prime.yaml`
via `prime_core.prime_config`, and section 8.3 for how the post/dipole windows are used in the
rejection cascade):

- **pre** — the pre-stimulus quality-control window (`qc_tmin`/`qc_tmax`) used for the rolling QC
  gate and for the classifier's input.
- **post** — the post-stimulus window (`post_tmin`/`post_tmax`) that survives the full artifact
  pipeline (baseline, ICA, SOUND, SSP-SIR, stim-artifact interpolation).
- **dipole** — the narrower N45 window (`dipole_tmin`/`dipole_tmax`) the dipole amplitude
  (`tep_amplitude_raw`) is computed from, cropped out of the post window.
""")

C(r"""
proc_info = proc_info_preview
print(f"channels ({len(proc_info['ch_names'])}): {', '.join(proc_info['ch_names'])}")
print(f"\nprocessed sampling rate: {proc_info['sfreq']:.0f} Hz")
print(f"units: {proc_info['units']}")
print(f"array layout returned by preprocessing: {proc_info['proc_shape']}")

windows = pd.DataFrame([
    {"window": k, "tmin_ms": v["tmin"] * 1000, "tmax_ms": v["tmax"] * 1000,
     "n_samples": round((v["tmax"] - v["tmin"]) * proc_info["sfreq"]) + 1}
    for k, v in [("pre (QC)", proc_info["pre"]), ("post", proc_info["post"]), ("dipole", proc_info["dipole"])]
])
display(windows)
""")
# ============================================================================
# 5. prime_predictions.csv
# ============================================================================
M(r"""
## 5 · `prime_predictions.csv`

One row per **prediction attempt**, written by `process_periodic` every time the rolling QC gate
passes (`check_qc()`, section 8.4) during the search phase of a PRIME-guided trial. A trial
normally produces many rows (one per QC-passing 10 ms tick) ending in a single `triggered=True`
row — either a real threshold crossing, or a forced trigger at 5.5 s (`forced=True`,
`probability` is `NaN` because no prediction is made on a forced tick).

This file **only ever contains `prime_triplet` and `prime_single_pulse` trials** —
`predetermined_single` and `calibration` trials are scheduled on a fixed clock and never enter
`process_periodic`'s search loop, so they never appear here. "Only prime-triggered trials" in
the heading below is therefore automatic, not a filter this notebook applies.
""")

C(r"""
prime_predictions = pd.read_csv(SUBJECT_DIR / "prime_predictions.csv")
print(f"prime_predictions.csv: {prime_predictions.shape[0]} prediction attempts, "
      f"{prime_predictions.shape[1]} columns")
print(f"conditions present: {sorted(prime_predictions.condition.unique())}")
print(f"blocks present    : {sorted(prime_predictions.stage.unique())}")
print(f"forced attempts   : {int(prime_predictions.forced.sum())} "
      f"(probability is NaN on these rows by construction)")
""")

M(r"""### 5.1 · Development of PRIME's predicted probability across blocks""")

C(r"""
# ---------------------------------------------------------------- per-block summary
block_order = sorted(prime_predictions.stage.unique(), key=lambda s: int(s.rsplit("_", 1)[1]))
attempts = prime_predictions.dropna(subset=["probability"]).copy()
attempts["stage"] = pd.Categorical(attempts.stage, categories=block_order, ordered=True)

summary = attempts.groupby(["stage", "condition"], observed=True).probability.agg(
    n="count", mean="mean", median="median", std="std").round(3)
print("Predicted probability, every QC-passing attempt (triggered and not), per block and condition:")
display(summary)

triggered_summary = (attempts[attempts.triggered]
                      .groupby(["stage", "condition"], observed=True).probability
                      .agg(n="count", mean="mean", median="median").round(3))
print("\nPredicted probability of the triggering attempt only (the one row per trial that fired the pulse):")
display(triggered_summary)
""")

C(r"""
# ---------------------------------------------------------------- per-block histograms
fig, axes = plt.subplots(1, len(block_order), figsize=(4 * len(block_order), 3.2), sharey=True)
bins = np.linspace(0, 1, 41)
for ax, blk in zip(axes, block_order):
    d = attempts[attempts.stage == blk]
    ax.hist(d.loc[d.condition == "prime_triplet", "probability"], bins=bins,
            alpha=0.55, label="prime_triplet", color=COLOR_PRIME_TRIPLET)
    ax.hist(d.loc[d.condition == "prime_single_pulse", "probability"], bins=bins,
            alpha=0.55, label="prime_single_pulse", color=COLOR_PRIME_SINGLE)
    ax.axvline(d.threshold.mean(), color="k", ls="--", lw=1,
               label=f"mean threshold ({d.threshold.mean():.2f})")
    ax.set_title(blk.replace("intervention_block_", "block "))
    ax.set_xlabel("predicted probability")
axes[0].set_ylabel("count (all QC-passing attempts)")
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Distribution of every predicted probability, per intervention block", y=1.03)
fig.tight_layout()
plt.show()
""")

C(r"""
# ---------------------------------------------------------------- trend across the session
fig, ax = plt.subplots(figsize=(9, 3.4))
for cond, color in [("prime_triplet", COLOR_PRIME_TRIPLET), ("prime_single_pulse", COLOR_PRIME_SINGLE)]:
    m = attempts[attempts.condition == cond].groupby("stage", observed=True).probability.mean()
    s = attempts[attempts.condition == cond].groupby("stage", observed=True).probability.sem()
    x = np.arange(len(block_order))
    ax.errorbar(x, m.reindex(block_order), yerr=s.reindex(block_order), marker="o", color=color,
                label=cond, capsize=3)
thr = attempts.groupby("stage", observed=True).threshold.mean().reindex(block_order)
ax.plot(np.arange(len(block_order)), thr, color="k", ls="--", marker="x", label="mean adaptive threshold")
ax.set_xticks(np.arange(len(block_order)))
ax.set_xticklabels([b.replace("intervention_block_", "block ") for b in block_order])
ax.set_ylabel("predicted probability (mean ± SEM)")
ax.set_title("Mean predicted probability per block, all QC-passing attempts")
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
""")

M(r"""### 5.2 · Predictions on the predetermined singles

`prime_predictions.csv` has no rows for `predetermined_single` trials. But `2_intervention.py`
does still run the classifier retrospectively on those trials — right after the pulse, to assess
PRIME performance against trials PRIME did not choose — and stores the result in the
**`prediction_probability` column of `trials_intervention.csv`** (loaded fully in section 8).
That call only happens if the trial's post-stimulus processing already succeeded, and it is
skipped if the trial's pre-stimulus window itself fails to preprocess — so this is available for
a subset of the 80 predetermined singles delivered, not all of them.
""")

C(r"""
trials_intervention_raw = pd.read_csv(SUBJECT_DIR / "trials_intervention.csv")
predet = trials_intervention_raw[trials_intervention_raw.condition == "predetermined_single"].copy()
n_total = len(predet)
n_postproc_failed = int(predet.postprocessing_failed.sum())
n_preproc_failed = int(predet.preprocessing_failed.sum())
n_with_pred = int(predet.prediction_probability.notna().sum())

print(f"predetermined_single trials delivered      : {n_total}")
print(f"  lost to postprocessing_failed (no attempt reaches the prediction step): {n_postproc_failed}")
print(f"  of the remainder, lost to preprocessing_failed (pre-window unusable)  : {n_preproc_failed}")
print(f"  -> with a retrospective prediction_probability                       : {n_with_pred}")
assert n_with_pred == n_total - n_postproc_failed - n_preproc_failed

pred_predet = predet.prediction_probability.dropna()
print(f"\ndescriptive statistics, retrospective prediction_probability on predetermined singles (n={len(pred_predet)}):")
display(pred_predet.describe().to_frame("prediction_probability").round(3))
""")

C(r"""
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.hist(pred_predet, bins=np.linspace(0, 1, 26), color=COLOR_PREDETERMINED, alpha=0.85)
ax.axvline(pred_predet.mean(), color="k", ls="--", lw=1, label=f"mean = {pred_predet.mean():.3f}")
ax.set_xlabel("retrospective prediction_probability")
ax.set_ylabel("count")
ax.set_title("What the PRIME classifier would have said about the predetermined singles")
ax.legend()
fig.tight_layout()
plt.show()
""")
# ============================================================================
# 6. ITI analysis
# ============================================================================
M(r"""
## 6 · Inter-trial intervals (ITI)

Every **predetermined** trial (baseline, evaluation, and the `predetermined_single` condition in
the intervention blocks) draws its ITI up front from `util/iti.py`'s `sample_iti`, before the
trial starts — the pulse is then scheduled for `trial_start_time + iti` (see also section 8.1).
That function draws from a scaled Beta distribution on `[2.5, 5.5]` s, plus a 2% point mass at
5.5 s, with the Beta shape solved so the whole mixture has mean 3.15 s. **The same distribution
is used for baseline, evaluation, and the intervention blocks' predetermined singles** — they are
independent draws from one generative process, so pooling them below is valid and increases the
sample size.

**PRIME-triggered trials (`prime_triplet`, `prime_single_pulse`) have no drawn `iti`** — their
timing is state-dependent (the trial fires whenever the rolling QC gate and prediction threshold
first agree, or at the 5.5 s forced timeout). There is no `prime_intervention.csv` file (the
notebook's original section title expected one); PRIME-triggered ITI is instead computed here as
the **empirical** gap between consecutive pulses within the same intervention block:
`pulse_time[trial i] - pulse_time[trial i-1]`, restricted to pairs where trial *i* is
PRIME-triggered. This is not the same kind of quantity as the drawn `iti` (one is a plan, the
other a realised outcome) — that distinction is exactly what section 6.2 compares.
""")

C(r"""
# ---------------------------------------------------------------- load the predetermined-ITI sources
trials_baseline = pd.read_csv(SUBJECT_DIR / "trials_baseline.csv")
eval_files = sorted(SUBJECT_DIR.glob("trials_evaluation_*.csv"),
                     key=lambda p: int(p.stem.rsplit("_", 1)[1]))
trials_evaluation = pd.concat([pd.read_csv(p) for p in eval_files], ignore_index=True)
predet_intervention = trials_intervention_raw[trials_intervention_raw.condition == "predetermined_single"]

print(f"trials_baseline.csv            : {len(trials_baseline)} trials")
print(f"trials_evaluation_*.csv ({len(eval_files)} runs) : {len(trials_evaluation)} trials total, "
      f"{trials_evaluation.groupby('run_index').size().to_dict()} per run")
print(f"trials_intervention.csv, predetermined_single    : {len(predet_intervention)} trials")
""")

C(r"""
# ---------------------------------------------------------------- empirical PRIME-triggered ITI
iv_sorted = trials_intervention_raw[trials_intervention_raw.stage.str.startswith("intervention_block")].copy()
iv_sorted = iv_sorted.sort_values(["stage", "trial_in_stage"])
iv_sorted["empirical_iti"] = iv_sorted.groupby("stage")["pulse_time"].diff()
prime_triggered_iti = iv_sorted.loc[
    iv_sorted.condition.isin(["prime_triplet", "prime_single_pulse"]), "empirical_iti"
].dropna()
print(f"empirical inter-pulse interval computed for {len(prime_triggered_iti)} PRIME-triggered trials "
      f"(the first trial of each block has no previous pulse in that block and is excluded).")
""")

M(r"""### 6.1 · Predetermined ITI — baseline, evaluation, and intervention predetermined singles""")

C(r"""
iti_sources = {
    "baseline":               trials_baseline.iti,
    "evaluation (all runs)":  trials_evaluation.iti,
    "intervention, predetermined_single": predet_intervention.iti,
}
desc = pd.DataFrame({name: s.describe() for name, s in iti_sources.items()}).round(3)
print("descriptive statistics, drawn ITI (s), by source:")
display(desc)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), sharey=True, sharex=True)
bins = np.linspace(2.5, 5.5, 31)
for ax, (name, s) in zip(axes, iti_sources.items()):
    ax.hist(s.dropna(), bins=bins, color=COLOR_PREDETERMINED, alpha=0.85)
    ax.set_title(f"{name}\n(n={s.notna().sum()})")
    ax.set_xlabel("ITI (s)")
axes[0].set_ylabel("count")
fig.suptitle("Drawn ITI by source", y=1.03)
fig.tight_layout()
plt.show()
""")

M(r"""### 6.1.1 · Empirical ITI of PRIME-triggered trials""")

C(r"""
print("descriptive statistics, empirical inter-pulse interval (s) ending in a PRIME-triggered pulse:")
display(prime_triggered_iti.describe().to_frame("empirical_iti").round(3))

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.hist(prime_triggered_iti, bins=np.linspace(0, 5.6, 57), color=COLOR_PRIME_SINGLE, alpha=0.85)
ax.axvline(prime_triggered_iti.median(), color="k", ls="--", lw=1,
           label=f"median = {prime_triggered_iti.median():.2f} s")
ax.axvline(5.5, color="grey", ls=":", lw=1, label="5.5 s forced-trigger ceiling")
ax.set_xlabel("empirical inter-pulse interval (s)")
ax.set_ylabel("count")
ax.set_title("PRIME-triggered trials: time since the previous pulse in the block")
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
""")

M(r"""### 6.2 · Overlay: predetermined (drawn) vs. PRIME-triggered (empirical) ITI

The predetermined sources (baseline + evaluation + intervention predetermined singles) are pooled
here since section 6.1 showed they share one generative distribution. The dashed curve is the
theoretical density of `sample_iti` (the Beta component only — the 2% point mass at 5.5 s is
marked separately), computed with the exact parameters solved in `util/iti.py`, as a check that
the pooled empirical draws match the code that generated them.
""")

C(r"""
# ---------------------------------------------------------------- reproduce util/iti.py's Beta parameters
ITI_MIN, ITI_MAX = 2.5, 5.5
TARGET_MEAN, SPIKE_VALUE, SPIKE_WEIGHT, CONCENTRATION = 3.15, 5.5, 0.02, 5.0
_beta_mean = (TARGET_MEAN - SPIKE_WEIGHT * SPIKE_VALUE) / (1 - SPIKE_WEIGHT)
_p = (_beta_mean - ITI_MIN) / (ITI_MAX - ITI_MIN)
_alpha, _beta = _p * CONCENTRATION, (1 - _p) * CONCENTRATION

x = np.linspace(ITI_MIN, ITI_MAX, 400)
pdf_beta_component = (1 - SPIKE_WEIGHT) * stats.beta.pdf((x - ITI_MIN) / (ITI_MAX - ITI_MIN), _alpha, _beta) / (ITI_MAX - ITI_MIN)

predetermined_pooled = pd.concat([trials_baseline.iti, trials_evaluation.iti, predet_intervention.iti], ignore_index=True).dropna()

fig, ax = plt.subplots(figsize=(9, 3.6))
bins = np.linspace(0, 5.6, 57)
ax.hist(predetermined_pooled, bins=bins, density=True, alpha=0.55, color=COLOR_PREDETERMINED,
        label=f"predetermined, drawn iti (n={len(predetermined_pooled)})")
ax.hist(prime_triggered_iti, bins=bins, density=True, alpha=0.55, color=COLOR_PRIME_SINGLE,
        label=f"PRIME-triggered, empirical (n={len(prime_triggered_iti)})")
ax.plot(x, pdf_beta_component, color=COLOR_PREDETERMINED, ls="--", lw=1.5,
        label="sample_iti Beta component (theoretical)")
ax.annotate(f"+{SPIKE_WEIGHT:.0%} point mass\nat {SPIKE_VALUE} s", xy=(5.5, 0), xytext=(4.6, ax.get_ylim()[1]*0.55),
            arrowprops=dict(arrowstyle="->", color=COLOR_PREDETERMINED), fontsize=8, color=COLOR_PREDETERMINED)
ax.set_xlabel("interval (s)")
ax.set_ylabel("density")
ax.set_title("Predetermined (drawn) vs. PRIME-triggered (empirical) inter-trial interval")
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

u = stats.mannwhitneyu(predetermined_pooled, prime_triggered_iti)
print(f"predetermined median {predetermined_pooled.median():.3f} s   "
      f"PRIME-triggered median {prime_triggered_iti.median():.3f} s   "
      f"difference {prime_triggered_iti.median() - predetermined_pooled.median():+.3f} s")
print(f"Mann-Whitney U p = {u.pvalue:.3g} (descriptive check that the two distributions differ; "
      f"not a claim about why)")
""")
# ============================================================================
# 7. profile_*.csv
# ============================================================================
M(r"""
## 7 · Latency profiles (`profile_*.csv`)

Each file holds one column, `duration_ms`: a wall-clock latency sample (in milliseconds) recorded
by `record_profile` every time the named function ran, buffered in memory during the session and
flushed to disk once at teardown (`write_profile` in `2_intervention.py`). What triggers each
sample:

| file | what one row measures | called from |
|---|---|---|
| `profile_process_pulse.csv` | the whole post-stimulus pipeline for one trial (post-processing, dipole fit, TEP normalization, checkpoint save, buffer save) | once per trial (925 = 800 intervention + 125 calibration) |
| `profile_preprocess_pre.csv` | pre-stimulus window preprocessing for one periodic tick | every ~10 ms tick during a PRIME-guided trial's search (`process_periodic`) |
| `profile_check_qc.csv` | one rolling-window QC gate evaluation | every tick once the 500 ms QC history is full |
| `profile_predict.csv` | one classifier forward pass | every tick the QC gate passes |
| `profile_finetune.csv` | one online finetuning step | after a `prime_single_pulse` or (non-open-loop) `predetermined_single` trial, only if its pre-stimulus window preprocessed successfully |
""")

C(r"""
profile_files = sorted(SUBJECT_DIR.glob("profile_*.csv"))
profiles = {p.stem.replace("profile_", ""): pd.read_csv(p)["duration_ms"] for p in profile_files}

desc = pd.DataFrame({name: s.describe(percentiles=[.5, .95, .99]) for name, s in profiles.items()}).round(4)
print("descriptive statistics, latency (ms), per profiled function:")
display(desc)
""")

C(r"""
fig, axes = plt.subplots(1, len(profiles), figsize=(3.4 * len(profiles), 3.2))
for ax, (name, s) in zip(axes, profiles.items()):
    ax.hist(s, bins=50, color="#5b8ff9", alpha=0.85)
    ax.set_title(f"{name}\nmedian {s.median():.3g} ms")
    ax.set_xlabel("duration (ms)")
    ax.set_yscale("log")
axes[0].set_ylabel("count (log scale)")
fig.suptitle("Per-call latency distributions", y=1.04)
fig.tight_layout()
plt.show()
""")
# ============================================================================
# 8. trials_intervention.csv — full breakdown
# ============================================================================
M(r"""
## 8 · `trials_intervention.csv` — per-trial breakdown

The rest of the notebook works from the 800 intervention-block rows of `trials_intervention.csv`
(the 125 calibration rows, already covered in section 2, are excluded from `IV` below).
`trial_order` is the trial's position in the session (0–799, chronological: sorted by block then
`trial_in_stage`); `t_min` is minutes since the first intervention pulse. Vertical dotted lines in
the trend plots mark the boundary between intervention blocks.
""")

C(r"""
IV = trials_intervention_raw[trials_intervention_raw.stage.str.startswith("intervention_block")].copy()
IV = IV.sort_values(["stage", "trial_in_stage"]).reset_index(drop=True)
IV["trial_order"] = np.arange(len(IV))
IV["t_min"] = (IV.pulse_time - IV.pulse_time.min()) / 60
IV["is_forced"] = IV.is_forced.astype(bool)
BLOCK_ORDER = list(IV.stage.unique())
BLOCK_EDGES = IV.groupby("stage", sort=False).size().cumsum().values[:-1]

print(f"IV: {len(IV)} intervention trials across {len(BLOCK_ORDER)} blocks")
print(IV.condition.value_counts().rename("n trials").to_frame())

def trend_plot(ax, df, ycol, conditions, colors, window=25):
    # Scatter of a per-trial value over trial_order, with a rolling mean per condition.
    for cond, color in zip(conditions, colors):
        d = df.loc[df.condition == cond, ["trial_order", ycol]].dropna().sort_values("trial_order")
        if d.empty:
            continue
        roll = d[ycol].rolling(window, min_periods=max(3, window // 3), center=True).mean()
        ax.scatter(d.trial_order, d[ycol], s=7, alpha=0.25, color=color)
        ax.plot(d.trial_order, roll, color=color, lw=1.8, label=f"{cond} (rolling mean, w={window})")
    for edge in BLOCK_EDGES:
        ax.axvline(edge, color="grey", lw=0.7, ls=":")
    ax.set_xlabel("trial order")
    ax.set_ylabel(ycol)
    ax.legend(fontsize=8)
""")

M(r"""### 8.1 · `target_time` vs. `pulse_time`

`target_time` is when the pulse was *supposed* to fire: `trial_start_time + iti` for predetermined
trials, `trigger_time + 0.01 s` for a PRIME trial that crossed threshold. **Forced PRIME trials
(the 5.5 s timeout) never get a `target_time` written** — `2_intervention.py` sets
`trigger_time`, `prime_attempts`, `qc_failures` and `threshold` on a forced trigger, but not
`target_time`, so it stays `NaN` by construction. `pulse_time` is when the pulse actually fired.
The difference is therefore only defined for non-forced trials, and is computed here as
`(pulse_time - target_time) * 1000` in milliseconds.
""")

C(r"""
n_forced = int(IV.is_forced.sum())
print(f"forced trials (excluded from this comparison): {n_forced} / {len(IV)} ({100*n_forced/len(IV):.1f}%)")
print(IV.groupby("condition").is_forced.sum().rename("n forced").to_frame())

nonforced = IV[~IV.is_forced].copy()
nonforced["target_pulse_diff_ms"] = (nonforced.pulse_time - nonforced.target_time) * 1000
assert nonforced.target_time.notna().all()

print("\ntarget_time -> pulse_time difference (ms), non-forced trials only, by condition:")
display(nonforced.groupby("condition").target_pulse_diff_ms.describe().round(3))
""")

C(r"""
fig, ax = plt.subplots(figsize=(8, 3.4))
conds = ["predetermined_single", "prime_single_pulse", "prime_triplet"]
colors = [COLOR_PREDETERMINED, COLOR_PRIME_SINGLE, COLOR_PRIME_TRIPLET]
lo, hi = nonforced.target_pulse_diff_ms.quantile([0.001, 0.999])
bins = np.linspace(lo, hi, 60)
for cond, color in zip(conds, colors):
    ax.hist(nonforced.loc[nonforced.condition == cond, "target_pulse_diff_ms"], bins=bins,
            alpha=0.55, color=color, label=cond, density=True)
ax.set_xlabel("pulse_time - target_time (ms)")
ax.set_ylabel("density")
ax.set_title("Scheduling precision: actual pulse time relative to its target")
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
""")

M(r"""### 8.2 · `preprocessing_failed`

Set when the trial's **pre-stimulus** window (the one used for classifier finetuning / retrospective
prediction) fails to preprocess. This can only happen for a **forced** `prime_single_pulse` trial
(its pre window is re-extracted from the pulse-aligned buffer, since the periodic search never
produced one) or a `predetermined_single` trial (always re-extracted from the pulse-aligned
buffer) — `prime_triplet` trials never attempt this step at all, so `preprocessing_failed` is
always `False` for them.
""")

C(r"""
rate = IV.groupby("condition").preprocessing_failed.agg(["sum", "count"])
rate["rate_%"] = (100 * rate["sum"] / rate["count"]).round(2)
rate.columns = ["n_failed", "n_trials", "rate_%"]
print(f"overall: {int(IV.preprocessing_failed.sum())} / {len(IV)} trials "
      f"({100*IV.preprocessing_failed.mean():.2f}%)")
display(rate)
""")

C(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))

per_block = IV.groupby("stage", sort=False).preprocessing_failed.sum().reindex(BLOCK_ORDER)
axes[0].bar(range(len(BLOCK_ORDER)), per_block.values, color=COLOR_PRIME_SINGLE)
axes[0].set_xticks(range(len(BLOCK_ORDER)))
axes[0].set_xticklabels([b.replace("intervention_block_", "block ") for b in BLOCK_ORDER])
axes[0].set_ylabel("n preprocessing_failed trials")
axes[0].set_title("Per block")

axes[1].step(IV.trial_order, IV.preprocessing_failed.cumsum(), where="post", color=COLOR_PRIME_SINGLE)
for edge in BLOCK_EDGES:
    axes[1].axvline(edge, color="grey", lw=0.7, ls=":")
axes[1].set_xlabel("trial order")
axes[1].set_ylabel("cumulative count")
axes[1].set_title("Cumulative across the session")
fig.suptitle("preprocessing_failed: how it develops across the session", y=1.03)
fig.tight_layout()
plt.show()
""")
M(r"""### 8.3 · `postprocessing_failed` and `post_reject_reason`

Set when the **post-stimulus** window fails the artifact pipeline in `Preprocessor.preprocess_post`
(section 2 walks the same cascade for the calibration trials). Checked in this order, first
failure wins, thresholds from `prime_core/preprocessing/config.py`'s
`trial_reject_opts`:

1. **ocular** — median z-score of either ocular ICA component (over its calibration-characterised
   time samples) exceeds `z_threshold = 2.0`.
2. **global_mad** — the trial's whole-epoch MAD z-score (relative to the calibration
   `good_trial_stats_post` mean/std) falls outside `[-inf, 6.0]` — i.e. an upper-bound-only check.
3. **local_mad** — any single channel's per-channel MAD, z-scored across channels *within that
   trial*, exceeds `5.0`.

A trial that passes all three keeps its `tep_amplitude` / `tep_amplitude_raw`; one that fails any
of them has `post_reject_reason` set to that stage's name and no TEP amplitude.
""")

C(r"""
rate = IV.groupby("condition").postprocessing_failed.agg(["sum", "count"])
rate["rate_%"] = (100 * rate["sum"] / rate["count"]).round(2)
rate.columns = ["n_failed", "n_trials", "rate_%"]
print(f"overall: {int(IV.postprocessing_failed.sum())} / {len(IV)} trials "
      f"({100*IV.postprocessing_failed.mean():.2f}%)")
display(rate)

reason_counts = IV.post_reject_reason.value_counts(dropna=True)
print("\nrejection reason, all intervention trials:")
display(reason_counts.to_frame("n trials"))

reason_by_cond = pd.crosstab(IV.condition, IV.post_reject_reason.fillna("(kept)"))
print("\nreason by condition:")
display(reason_by_cond)
""")

C(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))

axes[0].bar(reason_counts.index, reason_counts.values, color=["#e8684a", "#f6bd16", "#5b8ff9"])
axes[0].set_ylabel("n trials rejected")
axes[0].set_title("Rejection reason (all intervention trials)")

per_block_rate = (IV.groupby("stage", sort=False).postprocessing_failed.mean() * 100).reindex(BLOCK_ORDER)
per_block_rate_by_cond = (IV.groupby(["stage", "condition"], sort=False).postprocessing_failed.mean() * 100
                           ).unstack("condition").reindex(BLOCK_ORDER)
x = np.arange(len(BLOCK_ORDER))
for cond, color in zip(conds, colors):
    if cond in per_block_rate_by_cond.columns:
        axes[1].plot(x, per_block_rate_by_cond[cond], marker="o", color=color, label=cond)
axes[1].plot(x, per_block_rate.values, marker="s", color="k", ls="--", label="all conditions", lw=1)
axes[1].set_xticks(x)
axes[1].set_xticklabels([b.replace("intervention_block_", "block ") for b in BLOCK_ORDER])
axes[1].set_ylabel("postprocessing_failed rate (%)")
axes[1].set_title("Development across the session")
axes[1].legend(fontsize=8)
fig.tight_layout()
plt.show()
""")

M(r"""### 8.4 · `qc_failures`

The number of rolling-window QC gate failures (`check_qc()` in `2_intervention.py`) counted
*during the search for that trial*, reset to 0 at the start of each trial. Only meaningful for
PRIME-guided trials (`prime_triplet`, `prime_single_pulse`) — predetermined trials never run the
search loop, so `qc_failures` is `NaN` for them (and for calibration).

`check_qc()` itself: a candidate tick is accepted only if (1) the most recent 5 pre-stimulus
windows all preprocessed successfully, **and** (2) at least 80% of the last 50 windows (500 ms of
history, at one 10 ms tick each) did. Each tick that fails either condition increments
`qc_failures` by one and blocks prediction for that tick.
""")

C(r"""
prime_guided = IV[IV.condition.isin(["prime_triplet", "prime_single_pulse"])].copy()
has_any = prime_guided.qc_failures > 0
print(f"PRIME-guided trials: {len(prime_guided)}")
print(f"  had >=1 qc failure during the search : {int(has_any.sum())} "
      f"({100*has_any.mean():.1f}%)")
print(f"\nqc_failures, all PRIME-guided trials (0s included):")
display(prime_guided.qc_failures.describe().round(2).to_frame("qc_failures"))
print(f"\nqc_failures, only trials where it happened (>0):")
display(prime_guided.loc[has_any, "qc_failures"].describe().round(2).to_frame("qc_failures (>0 only)"))
""")

C(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
per_block_mean = prime_guided.groupby("stage", sort=False).qc_failures.mean().reindex(BLOCK_ORDER)
axes[0].bar(range(len(BLOCK_ORDER)), per_block_mean.values, color=COLOR_PRIME_TRIPLET)
axes[0].set_xticks(range(len(BLOCK_ORDER)))
axes[0].set_xticklabels([b.replace("intervention_block_", "block ") for b in BLOCK_ORDER])
axes[0].set_ylabel("mean qc_failures per trial")
axes[0].set_title("Per block")

trend_plot(axes[1], prime_guided, "qc_failures", ["prime_triplet", "prime_single_pulse"],
           [COLOR_PRIME_TRIPLET, COLOR_PRIME_SINGLE])
axes[1].set_title("Trend across the session")
fig.suptitle("qc_failures: how often, how many, and how it develops", y=1.03)
fig.tight_layout()
plt.show()
""")
M(r"""### 8.5 · `prediction_probability` (as recorded per trial)

Section 5 analysed the full attempt-by-attempt search log (`prime_predictions.csv`). This is the
single value that ends up attached to each row of `trials_intervention.csv`: for a non-forced
`prime_triplet`/`prime_single_pulse` trial it is the probability of the triggering attempt (set in
`process_periodic` the instant threshold is crossed, so it does **not** depend on what happens to
the trial afterwards); for a forced PRIME trial it is `NaN` (no attempt crossed threshold); for
`predetermined_single` it is the retrospective prediction from section 5.2 (depends on
post-processing and pre-processing both succeeding).
""")

C(r"""
avail = IV.groupby("condition").prediction_probability.apply(lambda s: s.notna().sum())
print("trials with a non-null prediction_probability, by condition:")
display(avail.rename("n").to_frame())

fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
per_block = (IV.groupby(["stage", "condition"], sort=False).prediction_probability.mean()
             .unstack("condition").reindex(BLOCK_ORDER))
x = np.arange(len(BLOCK_ORDER))
for cond, color in zip(conds, colors):
    if cond in per_block.columns:
        axes[0].plot(x, per_block[cond], marker="o", color=color, label=cond)
axes[0].set_xticks(x)
axes[0].set_xticklabels([b.replace("intervention_block_", "block ") for b in BLOCK_ORDER])
axes[0].set_ylabel("mean prediction_probability")
axes[0].set_title("Per block")
axes[0].legend(fontsize=8)

trend_plot(axes[1], IV, "prediction_probability", conds, colors)
axes[1].set_title("Trend across the session")
fig.suptitle("prediction_probability: development across the session", y=1.03)
fig.tight_layout()
plt.show()
""")

M(r"""### 8.6 · `prime_attempts`

The number of QC-passing prediction attempts made before the trial triggered (forced or not) —
`self.prime_attempt_count` at the moment of trigger. Populated for every PRIME-guided trial
(`prime_triplet`, `prime_single_pulse`), forced or not; `NaN` for predetermined/calibration
trials, which never search.
""")

C(r"""
print("prime_attempts, PRIME-guided trials, by condition:")
display(prime_guided.groupby("condition").prime_attempts.describe().round(2))

fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
per_block_mean = prime_guided.groupby("stage", sort=False).prime_attempts.mean().reindex(BLOCK_ORDER)
axes[0].bar(range(len(BLOCK_ORDER)), per_block_mean.values, color=COLOR_PRIME_TRIPLET)
axes[0].set_xticks(range(len(BLOCK_ORDER)))
axes[0].set_xticklabels([b.replace("intervention_block_", "block ") for b in BLOCK_ORDER])
axes[0].set_ylabel("mean prime_attempts per trial")
axes[0].set_title("Per block")

trend_plot(axes[1], prime_guided, "prime_attempts", ["prime_triplet", "prime_single_pulse"],
           [COLOR_PRIME_TRIPLET, COLOR_PRIME_SINGLE])
axes[1].set_title("Trend across the session")
fig.suptitle("prime_attempts: how it develops across the session", y=1.03)
fig.tight_layout()
plt.show()
""")

M(r"""### 8.7 · `search_time`

`update_threshold()` in `2_intervention.py`: `search_time = pulse_time - search_start_time`, where
`search_start_time` is the moment the rolling 500 ms QC history first fills (i.e. the earliest the
trial *could* have triggered). It is this quantity — with QC-rejected time subtracted off — that
drives the adaptive threshold: the threshold rises when a trial resolves faster than the
`TARGET_WAIT` of 0.65 s and falls when it resolves slower, bounded to `[0.50, 0.95]`. Forced trials
are the ones that hit the 5.5 s ceiling (`PRIME_MAX_WAIT`) without ever crossing threshold.
""")

C(r"""
print("search_time (s), PRIME-guided trials, by condition and forced status:")
display(prime_guided.groupby(["condition", "is_forced"]).search_time.describe()[["count", "mean", "50%", "std", "max"]].round(3))

fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
per_block_mean = prime_guided.groupby("stage", sort=False).search_time.mean().reindex(BLOCK_ORDER)
axes[0].bar(range(len(BLOCK_ORDER)), per_block_mean.values, color=COLOR_PRIME_TRIPLET)
axes[0].axhline(0.65, color="k", ls="--", lw=1, label="TARGET_WAIT (0.65 s)")
axes[0].set_xticks(range(len(BLOCK_ORDER)))
axes[0].set_xticklabels([b.replace("intervention_block_", "block ") for b in BLOCK_ORDER])
axes[0].set_ylabel("mean search_time (s)")
axes[0].set_title("Per block")
axes[0].legend(fontsize=8)

trend_plot(axes[1], prime_guided, "search_time", ["prime_triplet", "prime_single_pulse"],
           [COLOR_PRIME_TRIPLET, COLOR_PRIME_SINGLE])
axes[1].axhline(5.5, color="grey", ls=":", lw=1, label="PRIME_MAX_WAIT (5.5 s)")
axes[1].set_title("Trend across the session")
axes[1].legend(fontsize=8)
fig.suptitle("search_time: how it develops across the session", y=1.03)
fig.tight_layout()
plt.show()
""")
M(r"""### 8.8 · `tep_amplitude`: `prime_single_pulse` vs. `predetermined_single`

`tep_amplitude` is the CDF-normalized label from `TEPNormalizer.transform_verbose`
(`prime_core/tep_normalizer.py`): the raw dipole amplitude (`tep_amplitude_raw`, section 3) is
first detrended against its own EWMA trend (span 25 trials) to remove slow drift, then z-scored
against the calibration mean/std of that detrended signal, then mapped through the calibration
empirical CDF — so `tep_amplitude` is a number in `[0, 1]`, "how large was this TEP relative to
the calibration distribution, after removing slow drift". **`prime_triplet` never gets this label**
(`update_normalizer=False` for triplets in `2_intervention.py`, since a triplet's post-stimulus
window is dominated by the 2nd/3rd pulse of the burst, not a clean single-pulse N45) — only its
`tep_amplitude_raw` is stored. That is why the comparison below, like the project's own outcome
variable, is restricted to the two single-pulse conditions.
""")

C(r"""
singles = IV[IV.condition.isin(["predetermined_single", "prime_single_pulse"])].copy()
tep_avail = singles.groupby("condition").tep_amplitude.apply(lambda s: s.notna().sum())
print("trials with a non-null tep_amplitude, by condition:")
display(tep_avail.rename("n").to_frame())

print("\ntep_amplitude, descriptive statistics:")
display(singles.groupby("condition").tep_amplitude.describe().round(3))
""")

C(r"""
fig, axes = plt.subplots(1, 3, figsize=(15, 3.4))

bins = np.linspace(0, 1, 26)
for cond, color in [("predetermined_single", COLOR_PREDETERMINED), ("prime_single_pulse", COLOR_PRIME_SINGLE)]:
    axes[0].hist(singles.loc[singles.condition == cond, "tep_amplitude"].dropna(), bins=bins,
                 alpha=0.55, color=color, label=cond, density=True)
axes[0].set_xlabel("tep_amplitude")
axes[0].set_ylabel("density")
axes[0].set_title("Distribution")
axes[0].legend(fontsize=8)

data_box = [singles.loc[singles.condition == c, "tep_amplitude"].dropna()
            for c in ["predetermined_single", "prime_single_pulse"]]
bp = axes[1].boxplot(data_box, labels=["predetermined_\nsingle", "prime_\nsingle_pulse"], patch_artist=True)
for patch, color in zip(bp["boxes"], [COLOR_PREDETERMINED, COLOR_PRIME_SINGLE]):
    patch.set_facecolor(color); patch.set_alpha(0.6)
axes[1].set_ylabel("tep_amplitude")
axes[1].set_title("Side by side")

trend_plot(axes[2], singles, "tep_amplitude", ["predetermined_single", "prime_single_pulse"],
           [COLOR_PREDETERMINED, COLOR_PRIME_SINGLE], window=20)
axes[2].set_title("Trend across the session")
fig.suptitle("tep_amplitude: prime_single_pulse vs. predetermined_single", y=1.04)
fig.tight_layout()
plt.show()

a = singles.loc[singles.condition == "prime_single_pulse", "tep_amplitude"].dropna()
b = singles.loc[singles.condition == "predetermined_single", "tep_amplitude"].dropna()
print(f"prime_single_pulse   median = {a.median():.3f}  (n={len(a)})")
print(f"predetermined_single median = {b.median():.3f}  (n={len(b)})")
print(f"difference in medians (prime - predetermined) = {a.median() - b.median():+.3f}")
""")

M(r"""### 8.9 · Correlation: `prediction_probability` vs. `tep_amplitude`

Computed on trials where both are present — mechanically, that means `prime_triplet` trials drop
out (no `tep_amplitude`) and so do forced PRIME trials (no `prediction_probability`) and any trial
lost to post/pre-processing. **Caveat worth stating plainly**: for `prime_single_pulse` trials,
`prediction_probability >= threshold` is *why the pulse fired at all* — low-probability PRIME
moments are under-represented by design (range restriction), which this notebook does not correct
for. `predetermined_single` trials have no such selection (they fire on a clock regardless of the
predicted probability), so they are shown separately as well as combined.
""")

C(r"""
corr_df = singles.dropna(subset=["prediction_probability", "tep_amplitude"])
print(f"trials with both prediction_probability and tep_amplitude: {len(corr_df)}")
display(corr_df.groupby("condition").size().rename("n").to_frame())

fig, ax = plt.subplots(figsize=(7, 5))
for cond, color in [("predetermined_single", COLOR_PREDETERMINED), ("prime_single_pulse", COLOR_PRIME_SINGLE)]:
    d = corr_df[corr_df.condition == cond]
    ax.scatter(d.prediction_probability, d.tep_amplitude, s=14, alpha=0.5, color=color, label=cond)
    if len(d) > 2:
        b1, b0 = np.polyfit(d.prediction_probability, d.tep_amplitude, 1)
        xs = np.linspace(d.prediction_probability.min(), d.prediction_probability.max(), 20)
        ax.plot(xs, b0 + b1 * xs, color=color, lw=1.5)
ax.set_xlabel("prediction_probability")
ax.set_ylabel("tep_amplitude")
ax.set_title("prediction_probability vs. tep_amplitude")
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

for label, d in [("combined", corr_df), ("predetermined_single only", corr_df[corr_df.condition == "predetermined_single"]),
                 ("prime_single_pulse only", corr_df[corr_df.condition == "prime_single_pulse"])]:
    if len(d) < 3:
        continue
    r, p = stats.pearsonr(d.prediction_probability, d.tep_amplitude)
    rho, p_s = stats.spearmanr(d.prediction_probability, d.tep_amplitude)
    print(f"{label:28s} n={len(d):4d}   Pearson r = {r:+.3f} (p={p:.3g})   Spearman rho = {rho:+.3f} (p={p_s:.3g})")
""")
# ============================================================================
# Chapter 2 -- offline replay of Baseline and Evaluation, an intervention comparison,
# classifier predictions, and sensor-space TEPs
# ============================================================================
M(r"""
# Chapter 2 · Offline replay, intervention comparison, and sensor-space TEPs

Baseline and the four Evaluation runs are **open-loop, predetermined-timing** blocks:
`prime/decider/1_baseline.py` and `3_evaluation.py` only crop and save each trial's raw pre- and
post-stimulus buffers (see their module docstrings, verbatim: *"the raw pre/post EEG buffers are
saved for each trial so the ... TEP can be computed offline"*) — online, they never call the
preprocessing, dipole-fitting, or prediction code at all. That deferred offline computation is what
Item 1 does, using the session's own saved calibration state (`calibration_bunble.npy`,
`dipole_fitting_info.npy`) and the **actual `prime_core` classes** the intervention blocks used
online — the same "exact replay" approach as the dipole figure in section 3 and
`claude/online_TEP_replay.md` — so the numbers below are what the online system would have produced
had it processed these trials in real time.

**Scope — the 5 blocks that only saved raw buffers:** `baseline` (100 trials) and `evaluation`
runs 1-4 (100 trials each). The evaluation runs are numbered by the order the operator started them
(`3_evaluation.py`: *"detects how many evaluation runs already exist ... writes to its own
trials_evaluation_<N>.csv"*), not explicitly labelled t0/t15/t30/t60 in the data — but the protocol
calls for evaluations at t0/t15/t30/t60 post-intervention, and both the `trials_evaluation_<N>.csv`
write timestamps and the runs' own raw-buffer file timestamps increase strictly in that order, with
gaps consistent with a t0→t15→t30→t60 spacing (checked both ways — see
`claude/PRIME_online_performance_notes.md` and `claude/TEP_diagnostics_notes.md`). Runs 1-4 are
labelled `evaluation_t0` .. `evaluation_t60` below on that basis.

Nothing in this chapter names Pilot003 or any other subject directly: everything (bad channels, the
dipole window, block trial counts, the condition mix, which channels turn out interpolated) is
re-derived from whichever session section 1 points at. Item 1.1 additionally pulls in the
intervention blocks' `predetermined_single` trials — the one intervention condition whose timing is
open-loop like baseline/evaluation — from the post-stimulus buffers `2_intervention.py` already
preprocessed and saved **online** (real recorded output, not a replay), so `TEP_amplitude` can be
compared across baseline, intervention, and evaluation on one consistent axis.

Order in this chapter: restore the calibrated state, Item 1 (`TEP_amplitude` from the replay) and
1.1 (the same, plus intervention), Item 2 (classifier predictions) and 2.1 (correlation), Item 3
(save the per-trial table), Item 4 (sensor-space TEPs).
""")

C(r"""
# ---------------------------------------------------------------- import the actual online code
sys.path.insert(0, str(LOOP_ROOT / "prime" / "decider"))
from prime_core.preprocessing.preprocessor import Preprocessor
from prime_core.preprocessing.dipole_fitter import DipoleFitter
from prime_core.prime_config import (
    get_raw_sfreq, get_qc_time_range, get_post_initial_time_range,
    get_post_time_range, get_dipole_time_range, epoch_n_times, get_processed_sfreq,
)

RAW_SFREQ = get_raw_sfreq()
QC_TMIN, QC_TMAX = get_qc_time_range()                     # pre-stim saved-buffer window (s)
POST_I_TMIN, POST_I_TMAX = get_post_initial_time_range()   # post-stim saved-buffer window (s)
POST_TMIN, POST_TMAX = get_post_time_range()                # preprocess_post's final crop (s)
DIPOLE_TMIN, DIPOLE_TMAX = get_dipole_time_range()          # preprocess_post's *return-value* crop (s)

print(f"raw_sfreq = {RAW_SFREQ:.0f} Hz")
print(f"pre-stim saved buffers span  [{QC_TMIN}, {QC_TMAX}] s   (qc_tmin / qc_tmax)")
print(f"post-stim saved buffers span [{POST_I_TMIN}, {POST_I_TMAX}] s   (post_initial_tmin / tmax)")
print(f"preprocess_post's final window [{POST_TMIN}, {POST_TMAX}] s   (post_tmin / post_tmax)")
print(f"dipole-fitting window [{DIPOLE_TMIN}, {DIPOLE_TMAX}] s   (get_dipole_time_range -- a fixed "
      f"protocol constant, not calibration-derived; matches section 3's dipole_epoch_tmin_ms/tmax_ms)")
""")

C(r"""
# ---------------------------------------------------------------- restore the calibrated state
# Preprocessor: same from_bundle() restoration as section 3 (no recalibration).
preprocessor = Preprocessor.from_bundle(calibration_bundle, fwd_path)
preprocessor.keep_full_epochs = True   # populates last_post_full / last_post_baselines, used below

# DipoleFitter: fit_trial() only needs self._fitting_info (position_index, orientation, index_range)
# -- exactly the dict DipoleFitter.calibrate() produces and what was saved to
# dipole_fitting_info.npy (2_intervention.py, ~line 805). There is no public restore method (its
# own docstring mentions a from_fitting_info() that was never implemented -- grep confirms it
# exists only in a docstring and an error message), so it is restored by assigning the private
# attribute directly to the same dict already loaded as `dipole_info` in section 3.
dipole_fitter = DipoleFitter(fwd_path)
dipole_fitter._fitting_info = dipole_info

# Channel names + the post-stimulus time axis: both needed from here through item 4, so computed
# once, here, rather than re-derived in whichever item first happens to need them.
ch_names = list(preprocessor._info["ch_names"])
assert ch_names == list(proc_info["ch_names"]), \
    "channel order mismatch between the restored Preprocessor and proc_info.npy"
n_post_times = epoch_n_times(POST_TMIN, POST_TMAX, get_processed_sfreq())
post_time_ms = np.linspace(POST_TMIN * 1000, POST_TMAX * 1000, n_post_times)
n_dipole_times = epoch_n_times(DIPOLE_TMIN, DIPOLE_TMAX, get_processed_sfreq())
dipole_window_mask = (post_time_ms >= DIPOLE_TMIN * 1000 - 1e-6) & (post_time_ms <= DIPOLE_TMAX * 1000 + 1e-6)
assert dipole_window_mask.sum() == n_dipole_times, \
    f"dipole-window crop gave {dipole_window_mask.sum()} samples of post_time_ms, expected {n_dipole_times}"

aw1 = preprocessor._cfg.artifact_window_1
aw2 = preprocessor._cfg.artifact_window_2
print("Preprocessor  restored from calibration_bunble.npy   (Preprocessor.from_bundle)")
print("DipoleFitter  restored from dipole_fitting_info.npy  (position_index, orientation, index_range)")
print(f"{len(ch_names)} channels; bad_channels ({len(bad_channels)}): {', '.join(bad_channels) or '(none)'}")
print(f"artifact_window_1 (1st stim-artifact interpolation): [{aw1[0]*1000:.0f}, {aw1[1]*1000:.0f}] ms")
print(f"artifact_window_2 (2nd stim-artifact interpolation): [{POST_TMIN*1000:.0f}, {aw2[1]*1000:.0f}] ms")
""")

M(r"""
## Item 1 · `TEP_amplitude` from the post-stimulus buffers

For each trial: `preprocessor.preprocess_post(post_raw, post_time_offsets)` runs the full online
post-stimulus pipeline (baseline-correct → fix stim artifact → interpolate bad channels → average
reference → ICA ocular check → apply ICA → baseline-correct again → crop to
`[post_tmin, post_tmax]` → SOUND → SSP-SIR → fix stim artifact again → average reference), returning
`None` (rejected: `preprocessor.last_post_reject_reason` is `'ocular'`, `'global_mad'`, or
`'local_mad'`) or the epoch cropped a second time, down to this session's dipole-fitting window
(`[DIPOLE_TMIN, DIPOLE_TMAX]` above, fixed at calibration — section 3). On success,
`dipole_fitter.fit_trial(post)` fits a free-orientation dipole amplitude at that session's fixed
position/window — this **is** `TEP_amplitude`, with no further normalisation (unlike
`trials_intervention.csv`'s `tep_amplitude`, which is additionally EWMA-detrended by
`TEPNormalizer` — a step baseline/evaluation trials never went through online, since the normalizer
is not calibrated until *after* baseline and is only used for PRIME's real-time trigger decision
during intervention). This chapter sticks to real, saved buffers and this session's own online
parameters throughout — nowhere does it synthesize or approximate an input.

`preprocessor.preprocess_pre(pre_raw, pre_time_offsets, from_pulse=True)` is run alongside on the
saved pre-stimulus buffer — not needed for `TEP_amplitude` itself, but its success/failure is one of
the columns item 3 asks for, and it costs nothing extra to compute here (item 2's classifier
predictions, further below, reuse its output).

Saved buffers carry no timestamp array, only samples; `crop_eeg_buffer` (used inside both
`preprocess_pre`/`preprocess_post`) only ever reads `time_offsets[0]` as an anchor, so a synthetic
uniform grid at `raw_sfreq` starting at the buffer's known crop `tmin` (`qc_tmin` for the pre-stim
buffer, `post_initial_tmin` for the post-stim buffer) reproduces it exactly.

The two baseline-correction magnitudes recorded below (`baseline_correction_1_mean_abs` /
`baseline_correction_2_mean_abs`) differ by orders of magnitude by design, not error: the first
`apply_baseline` runs before `fix_stim_artifact`, on an epoch that still contains the raw TMS pulse
artifact, so its correction absorbs that artifact's DC offset; the second `apply_baseline` runs
after the artifact has been interpolated and ICA has been applied, so it is correcting only a small
residual drift.

Block/trial-file helpers, then baseline and the four evaluation runs are all processed into one
table, and the resulting grand-average sensor-space epoch per block (`evoked_by_block`, used from
item 1.1 through item 4) is built right away, from the same trials:
""")

C(r"""
BLOCKS = [
    ("baseline",   None, "baseline"),
    ("evaluation", 1,    "evaluation_t0"),
    ("evaluation", 2,    "evaluation_t15"),
    ("evaluation", 3,    "evaluation_t30"),
    ("evaluation", 4,    "evaluation_t60"),
]
BLOCK_LABELS = [label for _, _, label in BLOCKS]

def _stem(stage, run, trial):
    prefix = stage if run is None else f"{stage}_{run}"
    return f"{prefix}_{trial:04d}"

def _n_trials(stage, run):
    prefix = stage if run is None else f"{stage}_{run}"
    return len(sorted(SUBJECT_DIR.glob(f"{prefix}_*_pre_raw.npy")))
""")

C(r"""
records = []
sensor_traces = {label: [] for _, _, label in BLOCKS}

for stage, run, label in BLOCKS:
    for trial in range(_n_trials(stage, run)):
        stem = _stem(stage, run, trial)
        pre_raw = np.load(SUBJECT_DIR / f"{stem}_pre_raw.npy")
        post_raw = np.load(SUBJECT_DIR / f"{stem}_post_raw.npy")
        pre_ts = QC_TMIN + np.arange(pre_raw.shape[0]) / RAW_SFREQ
        post_ts = POST_I_TMIN + np.arange(post_raw.shape[0]) / RAW_SFREQ

        model_buf = preprocessor.preprocess_pre(pre_raw, pre_ts, from_pulse=True)
        post = preprocessor.preprocess_post(post_raw, post_ts)

        rec = {
            "block": label, "run_index": run, "trial_in_stage": trial,
            "preprocessing_success": model_buf is not None,
            "postprocessing_success": post is not None,
            "postprocessing_reject_reason": preprocessor.last_post_reject_reason,
            "tep_amplitude": np.nan,
            "baseline_correction_1_mean_abs": np.nan,
            "baseline_correction_2_mean_abs": np.nan,
        }
        if preprocessor.last_post_baselines is not None:
            b1, b2 = preprocessor.last_post_baselines
            rec["baseline_correction_1_mean_abs"] = float(np.abs(b1).mean())
            rec["baseline_correction_2_mean_abs"] = float(np.abs(b2).mean())
        if post is not None:
            rec["tep_amplitude"] = float(dipole_fitter.fit_trial(post))
            sensor_traces[label].append(preprocessor.last_post_full)
        records.append(rec)

replay = pd.DataFrame.from_records(records)
evoked_by_block = {b: np.mean(np.stack(sensor_traces[b], axis=0), axis=0)
                    for b in BLOCK_LABELS if sensor_traces[b]}   # (60, n_post_times) per block
print(f"processed {len(replay)} trials across {len(BLOCKS)} blocks\n")

summary = replay.groupby("block", sort=False).agg(
    n_trials=("trial_in_stage", "size"),
    preprocessing_failures=("preprocessing_success", lambda s: int((~s).sum())),
    postprocessing_failures=("postprocessing_success", lambda s: int((~s).sum())),
    tep_amplitude_n=("tep_amplitude", "count"),
    tep_amplitude_median=("tep_amplitude", "median"),
).reindex(BLOCK_LABELS)
display(summary)

print("\npostprocessing rejection reasons by block:")
rej = replay.loc[~replay.postprocessing_success]
if len(rej):
    display(rej.groupby(["block", "postprocessing_reject_reason"]).size().unstack(fill_value=0).reindex(BLOCK_LABELS))
else:
    print("(no postprocessing rejections)")
""")

M(r"""
### 1.1 · `TEP_amplitude` across baseline, intervention (predetermined singles), and evaluation

`predetermined_single` is the intervention condition added here — like baseline/evaluation, its
timing is drawn from the same open-loop ITI process rather than PRIME-triggered, so it is the
fairest single-pulse comparison point between the intervention blocks and the blocks flanking them.
Unlike baseline/evaluation, `predetermined_single` trials already had `preprocess_post()` run on
them **online** (`2_intervention.py`), so this entry is read from the real, online-computed
`<stem>_post_proc.npy` buffers directly — the full `[post_tmin, post_tmax]` epoch (the same array
`preprocessor.last_post_full` captures, used again in item 4), cropped here to the dipole window and
fit with `dipole_fitter.fit_trial()`, the same final step item 1 applies to its replayed
baseline/evaluation epochs — rather than replayed from raw buffers. Trials `trials_intervention.csv`
already flags `postprocessing_failed` are excluded, matching the replay's own accept/reject logic
for the other blocks.
""")

C(r"""
intervention_predetermined = IV[
    (IV.condition == "predetermined_single") & (~IV.postprocessing_failed)
].copy()
intervention_predetermined["stem"] = (
    intervention_predetermined.stage + "_"
    + intervention_predetermined.trial_in_stage.astype(int).map(lambda i: f"{i:04d}")
)

intervention_tep_amplitude = []
n_missing = 0
for stem in intervention_predetermined.stem:
    fp = SUBJECT_DIR / f"{stem}_post_proc.npy"
    if not fp.is_file():
        n_missing += 1
        continue
    full_epoch = np.load(fp)
    intervention_tep_amplitude.append(float(dipole_fitter.fit_trial(full_epoch[:, dipole_window_mask])))
intervention_tep_amplitude = np.array(intervention_tep_amplitude)

print(f"intervention predetermined_single: {len(intervention_predetermined)} / "
      f"{int((IV.condition == 'predetermined_single').sum())} trials with postprocessing_failed==False, "
      f"{n_missing} missing files -- {len(intervention_tep_amplitude)} TEP_amplitude values computed")
""")

C(r"""
BOXPLOT_LABELS = ["baseline", "intervention_predetermined_single"] + BLOCK_LABELS[1:]
data_by_block = (
    [replay.loc[replay.block == "baseline", "tep_amplitude"].dropna().values]
    + [intervention_tep_amplitude]
    + [replay.loc[replay.block == b, "tep_amplitude"].dropna().values for b in BLOCK_LABELS[1:]]
)

fig, ax = plt.subplots(figsize=(10, 4.5))
bp = ax.boxplot(data_by_block, positions=range(len(BOXPLOT_LABELS)), widths=0.5,
                 showfliers=False, patch_artist=True)
for patch in bp["boxes"]:
    patch.set_facecolor(COLOR_PREDETERMINED)
    patch.set_alpha(0.35)
rng = np.random.default_rng(0)
for i, vals in enumerate(data_by_block):
    jitter = rng.uniform(-0.12, 0.12, size=len(vals))
    ax.scatter(np.full(len(vals), i) + jitter, vals, s=10, alpha=0.5, color=COLOR_PREDETERMINED)
ax.set_xticks(range(len(BOXPLOT_LABELS)))
ax.set_xticklabels(BOXPLOT_LABELS, rotation=15, ha="right")
ax.set_ylabel("TEP_amplitude  (free-orientation dipole moment norm, a.u.)")
ax.set_title("TEP_amplitude by block  (DipoleFitter.fit_trial; baseline/evaluation replayed here,\n"
             "intervention from its own real online-computed buffers)")
fig.tight_layout()
plt.show()

for label, vals in zip(BOXPLOT_LABELS, data_by_block):
    print(f"{label:32s} n={len(vals):3d}   median={np.median(vals):.4f}   "
          f"mean={np.mean(vals):.4f}   sd={np.std(vals):.4f}")
""")

M(r"""
## Item 2 · `prediction_probability` from the pre-stimulus buffers, via the 5 stored classifiers

`OnlinePredictor` (`prime_core/online_predictor.py`) hardcodes `torch.device("cuda")` with no CPU
fallback, so the cell below only *computes* anything when this notebook is run in a kernel with a
CUDA-capable PyTorch install. Otherwise it detects that up front, loads a previously-written
`<SUBJECT_ID>_classifier_predictions.csv` if one already exists for this subject, or explains what
to install if neither is available -- either way the notebook still runs end-to-end. Needed beyond
section 1's setup, if not already installed: `torch` (a CUDA build matching the GPU driver -- see
https://pytorch.org/get-started/locally/), `omegaconf`, `scikit-learn`, and `mne_icalabel` (pulled in
transitively by `prime_core.preprocessing.preprocessor`, even though this notebook never fits a
fresh ICA -- it only applies the one already saved in `calibration_bunble.npy`).

**Preprocessing** (pre-stimulus, classifier-independent): identical to `preprocessor.preprocess_pre`
already used for item 1's pre-stim success/failure flag -- recomputed here, this time keeping the
actual preprocessed arrays, so this section stands on its own the way items 1 and 4 do.
""")

C(r"""
pre_keys = []            # (block, run_index, trial_in_stage) for successfully preprocessed trials
pre_model_buffers = []   # matching arrays, in the same order as pre_keys

for stage, run, label in BLOCKS:
    for trial in range(_n_trials(stage, run)):
        stem = _stem(stage, run, trial)
        pre_raw = np.load(SUBJECT_DIR / f"{stem}_pre_raw.npy")
        pre_ts = QC_TMIN + np.arange(pre_raw.shape[0]) / RAW_SFREQ
        model_buf = preprocessor.preprocess_pre(pre_raw, pre_ts, from_pulse=True)
        if model_buf is not None:
            pre_keys.append((label, run, trial))
            pre_model_buffers.append(model_buf)

pre_model_buffers = np.stack(pre_model_buffers, axis=0)   # (n_ok_total, 60, n_pre_times)
print(f"{len(pre_keys)} / {len(replay)} trials preprocessed successfully for classifier prediction "
      f"(matches item 1's preprocessing_success count: {int(replay.preprocessing_success.sum())})")
""")

M(r"""
**Classifiers**: each of the 5 stored checkpoints (`classifier_after_calibration.pt`,
`_after_trial_200/400/600/800.pt`) is loaded as a **frozen** snapshot --
`OnlinePredictor(global_backrotation, model_path=checkpoint, seed=201)` then `predict_batch()`,
with no `.finetune()` / `.adapt_alignment()` calls in between trials, so each checkpoint's own saved
`alignment_transform` is used unchanged for every trial rather than continuing to adapt as it did
online. `predict_batch()`'s own docstring describes it as numerically identical to calling
`.predict()` per trial. This evaluates every replayed trial against each of the 5 snapshots
independently -- "what would this snapshot have predicted" -- rather than re-simulating the online
session's one continuously-adapting predictor (these 5 files are just periodic saves of its state;
redoing that continuous sequence is not what was asked for). `global_backrotation_all.npy`
(`prime/decider/offline_results/train/` -- a shared, cross-subject artifact, not subject-specific)
is loaded once and applied unchanged to every trial, confirmed in `tta_wrapper.py`:
`global_backrotation_torch` is set once in `TTAWrapper.__init__` and never reassigned.
""")

C(r"""
PRED_PATH = LOOP_ROOT / "scripts_analysis" / f"{SUBJECT_ID}_classifier_predictions.csv"
CLASSIFIER_COLS = [
    "prediction_probability_after_calibration",
    "prediction_probability_after_trial_200",
    "prediction_probability_after_trial_400",
    "prediction_probability_after_trial_600",
    "prediction_probability_after_trial_800",
]
SEED_PREDICTOR = 201   # prime/decider/2_intervention.py, module-level SEED constant

try:
    import torch
    _cuda_ok = torch.cuda.is_available()
except ImportError:
    torch = None
    _cuda_ok = False

if PRED_PATH.is_file():
    predictions = pd.read_csv(PRED_PATH)
    print(f"loaded {len(predictions)} rows from {PRED_PATH.name} (already computed for this subject)")
    display(predictions.head())
elif not _cuda_ok:
    predictions = None
    reason = "PyTorch is not installed" if torch is None else "torch.cuda.is_available() is False"
    print(f"{reason} in this kernel -- OnlinePredictor hardcodes torch.device('cuda') with no CPU "
          f"fallback, so this cell cannot compute predictions here.")
    print("Install a CUDA-capable PyTorch build (https://pytorch.org/get-started/locally/) plus "
          "omegaconf, scikit-learn, and mne_icalabel, and re-run this notebook on that machine -- "
          "this cell will then compute and cache the predictions itself. Until then, item 2.1 and "
          "the classifier columns of item 3's CSV stay empty/NaN.")
else:
    from prime_core.online_predictor import OnlinePredictor

    GLOBAL_BACKROTATION_PATH = LOOP_ROOT / "prime" / "decider" / "offline_results" / "train" / "global_backrotation_all.npy"
    if not GLOBAL_BACKROTATION_PATH.is_file():
        raise FileNotFoundError(f"global_backrotation_all.npy not found at {GLOBAL_BACKROTATION_PATH}")
    global_backrotation = np.load(GLOBAL_BACKROTATION_PATH)
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"global_backrotation loaded, shape {global_backrotation.shape}")

    CLASSIFIERS = {
        "after_calibration": SUBJECT_DIR / "classifier_after_calibration.pt",
        "after_trial_200":   SUBJECT_DIR / "classifier_after_trial_200.pt",
        "after_trial_400":   SUBJECT_DIR / "classifier_after_trial_400.pt",
        "after_trial_600":   SUBJECT_DIR / "classifier_after_trial_600.pt",
        "after_trial_800":   SUBJECT_DIR / "classifier_after_trial_800.pt",
    }
    for name, path in CLASSIFIERS.items():
        if not path.is_file():
            raise FileNotFoundError(f"classifier checkpoint not found: {path}")

    predictions = pd.DataFrame(pre_keys, columns=["block", "run_index", "trial_in_stage"])
    for name, path in CLASSIFIERS.items():
        predictor = OnlinePredictor(global_backrotation, model_path=path, seed=SEED_PREDICTOR)
        probs = predictor.predict_batch(pre_model_buffers, batch_size=50)
        predictions[f"prediction_probability_{name}"] = probs
        print(f"  {name:20s} predicted {len(probs)} trials  (mean = {probs.mean():.3f}, "
              f"range [{probs.min():.3f}, {probs.max():.3f}])")
        del predictor
        torch.cuda.empty_cache()

    predictions.to_csv(PRED_PATH, index=False)
    print(f"wrote {len(predictions)} rows to {PRED_PATH.name}")
""")
M(r"""### 2.1 · Correlating predictions with `TEP_amplitude`, per classifier""")

C(r"""
if predictions is None:
    print(f"skipped -- {PRED_PATH.name} not present yet (see item 2 above).")
else:
    merged = replay.merge(predictions, on=["block", "run_index", "trial_in_stage"], how="left")
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.2), sharey=True)
    for ax, col in zip(axes, CLASSIFIER_COLS):
        d = merged.dropna(subset=[col, "tep_amplitude"])
        ax.scatter(d[col], d.tep_amplitude, s=10, alpha=0.4, color=COLOR_PREDETERMINED)
        title = col.replace("prediction_probability_", "")
        if len(d) > 2:
            r, p = stats.pearsonr(d[col], d.tep_amplitude)
            b1, b0 = np.polyfit(d[col], d.tep_amplitude, 1)
            xs = np.linspace(d[col].min(), d[col].max(), 20)
            ax.plot(xs, b0 + b1 * xs, color=COLOR_PRIME_SINGLE, lw=1.5)
            title += f"\nr={r:+.2f}  (n={len(d)}, p={p:.3g})"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("prediction_probability")
    axes[0].set_ylabel("TEP_amplitude")
    fig.suptitle("prediction_probability vs. TEP_amplitude, per stored classifier snapshot  (baseline + evaluation trials)")
    fig.tight_layout()
    plt.show()
""")

M(r"""
## Item 3 · Per-trial CSV across the 5 replayed blocks

One row per trial (baseline + 4 evaluation runs), combining everything item 1 computed:
pre-stimulus preprocessing outcome, post-stimulus preprocessing outcome (+ rejection reason),
`TEP_amplitude`, the two baseline-correction intermediate-step diagnostics, and, once item 2
has been computed, the 5 classifiers' `prediction_probability`. (Item 1.1's intervention
comparison is a boxplot addition, not part of this table -- its trials didn't go through the same
replay, so they don't carry the same per-trial diagnostic columns.) Saved under this session's
subject ID so re-running for a different subject never overwrites -- or is mistaken for -- another
subject's cached table.
""")

C(r"""
final = replay.copy()
if predictions is not None:
    final = final.merge(predictions, on=["block", "run_index", "trial_in_stage"], how="left")
else:
    for col in CLASSIFIER_COLS:
        final[col] = np.nan

OUT_CSV = LOOP_ROOT / "scripts_analysis" / f"{SUBJECT_ID}_trial_table.csv"
final.to_csv(OUT_CSV, index=False)
print(f"wrote {len(final)} rows x {len(final.columns)} columns to {OUT_CSV.name}")
display(final.head(8))
""")

M(r"""
## Item 4 · Sensor-space TEPs per block

Four views, in this order: a butterfly plot per block, then GFP per block on one shared axis, then
a fixed set of six motor/premotor channels shown individually (blocks overlaid), then those same
six channels shown together (per block). All four are built from the same
`evoked_by_block` grand averages item 1 computed (`preprocessor.last_post_full`, averaged across
every trial that passed post-stimulus preprocessing, per block) -- trial-to-trial noise is averaged
out before any channel reduction or single-channel read, not after.

**Channels of interest** — `FC3, C1, C3, C5, CP3, Cz` — are fixed across subjects, not derived from
any one session's bad-channel list: a left frontocentral/centroparietal cluster around the
(typically left) M1 stimulation site, plus `Cz` as a spatially distant reference. Any of the six may
be interpolated rather than measured for a given subject; each cell below checks this session's live
`bad_channels` list and flags it explicitly rather than assuming either way.

The epoch's early lead-in is synthetic, not measured: `artifact_window_1` and `artifact_window_2`
(printed above, where the calibrated state was restored) are both linearly interpolated
(`mne.preprocessing.fix_stim_artifact`) — near-zero amplitude there reflects that, not a null
response. The dipole-fitting window (shaded on every plot below) sits safely outside both.
""")

M(r"""### 4.1 · Butterfly plot per block""")

C(r"""
fig, axes = plt.subplots(1, len(BLOCK_LABELS), figsize=(3.6 * len(BLOCK_LABELS), 3.6),
                          sharey=True, sharex=True)
for ax, b in zip(axes, BLOCK_LABELS):
    if b not in evoked_by_block:
        ax.set_visible(False)
        continue
    evoked = evoked_by_block[b]   # (60, n_post_times)
    gfp = evoked.std(axis=0, ddof=0)
    ax.plot(post_time_ms, evoked.T, lw=0.5, color="grey", alpha=0.5)
    ax.plot(post_time_ms, gfp, lw=1.8, color="crimson", label="GFP")
    ax.axvspan(dipole_epoch_tmin_ms, dipole_epoch_tmax_ms, color="grey", alpha=0.12)
    ax.axhline(0, color="black", lw=0.4)
    ax.set_title(f"{b}\n(n={len(sensor_traces[b])})", fontsize=9)
    ax.set_xlabel("ms")
axes[0].set_ylabel("amplitude (a.u.)")
axes[0].legend(fontsize=7, loc="upper right")
fig.suptitle("Butterfly plot per block -- all 60 channels (grey), GFP (crimson)", y=1.03)
fig.tight_layout()
plt.show()
""")

M(r"""
### 4.2 · GFP per block

**Global Field Power** (GFP) reduces the 60 preprocessed channels to a single trace per block --
the standard scalp-ERP summary that avoids the sign cancellation a plain across-channel average
would suffer (channels on opposite sides of the fitted dipole have opposite polarity):
`GFP(t) = std across the 60 channels of the grand-average at time t` (population std, `ddof=0`).
It is polarity-blind by construction (a standard deviation, always ≥ 0), so on its own it can't show
a negative-going component like N45 -- only that *something* happens near a given latency; item 4.3
below shows the actual signed voltage. Whether GFP continuing to rise toward the buffer's 100 ms
edge (if it does, for this session) is a genuine later component or a filter/epoch-boundary effect
from SOUND/SSP-SIR isn't something the saved 100 ms buffers can settle either way.
""")

C(r"""
gfp_by_block = {b: evoked_by_block[b].std(axis=0, ddof=0) for b in evoked_by_block}

fig, ax = plt.subplots(figsize=(10, 4.5))
for b in BLOCK_LABELS:
    if b not in gfp_by_block:
        continue
    ax.plot(post_time_ms, gfp_by_block[b], lw=1.8, label=f"{b}  (n={len(sensor_traces[b])})")
ax.axvspan(dipole_epoch_tmin_ms, dipole_epoch_tmax_ms, color="grey", alpha=0.12,
           label="dipole-fitting window")
ax.set_xlabel("time after TMS pulse (ms)")
ax.set_ylabel("GFP  (std across 60 channels, a.u.)")
ax.set_title("Sensor-space TEP per block — GFP of the grand-average post-stimulus epoch")
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

for b in BLOCK_LABELS:
    if b in gfp_by_block:
        peak_i = int(np.argmax(gfp_by_block[b]))
        print(f"{b:16s} GFP peak = {gfp_by_block[b][peak_i]:.4f}  at {post_time_ms[peak_i]:.0f} ms")
""")

M(r"""
### 4.3 · `FC3`, `C1`, `C3`, `C5`, `CP3`, `Cz` individually

One panel per channel, every block overlaid (same grand-average-then-reduce order as GFP above),
the actual signed voltage rather than GFP's polarity-blind summary -- the way TEPs are
conventionally displayed in the TMS-EEG literature. Candidate P30 / N45 / P60 peaks are marked
where a local extremum of the expected sign (positive / negative / positive) occurs inside a fixed
latency window; the N45 window is **this session's own dipole-fitting window** (item 1's
`TEP_amplitude` source), so the marked N45 here is the sensor-space counterpart of that metric, not
an independently-chosen latency. A block whose window is dominated by the opposite polarity is left
blank (`NaN`) there rather than reporting a same-sign-mismatched number under a component label.
""")

C(r"""
CHANNELS_OF_INTEREST = ["FC3", "C1", "C3", "C5", "CP3", "Cz"]
_missing = [c for c in CHANNELS_OF_INTEREST if c not in ch_names]
if _missing:
    print(f"WARNING: not in this montage, skipped: {_missing}")
CHANNELS_OF_INTEREST = [c for c in CHANNELS_OF_INTEREST if c in ch_names]

_interpolated_here = [c for c in CHANNELS_OF_INTEREST if c in bad_channels]
if _interpolated_here:
    print(
        f"WARNING: interpolated (not measured) in this session: {', '.join(_interpolated_here)} "
        f"-- reconstructed from neighbouring good channels (channel_interpolation_info), not "
        f"measured EEG. A neighbour-driven reconstruction is least trustworthy exactly where the "
        f"true local signal is most likely to differ from its neighbours -- e.g. near the "
        f"stimulation site."
    )
else:
    print("None of these six channels are interpolated in this session -- all measured directly.")

COMPONENT_WINDOWS = {
    "P30": (20.0, 35.0, "max"),
    "N45": (dipole_epoch_tmin_ms, dipole_epoch_tmax_ms, "min"),
    "P60": (55.0, 75.0, "max"),
}

def _local_extremum(trace_1d, lo_ms, hi_ms, kind):
    # Only reports a peak when the window's own extremum has the expected sign -- a window
    # dominated by the opposite polarity has no candidate component to report, and a
    # same-sign-mismatched number under a component label would be misleading rather than
    # merely uncertain.
    win_idx = np.where((post_time_ms >= lo_ms) & (post_time_ms <= hi_ms))[0]
    seg = trace_1d[win_idx]
    i = win_idx[int(np.argmax(seg) if kind == "max" else np.argmin(seg))]
    lat, amp = post_time_ms[i], trace_1d[i]
    if (kind == "max" and amp < 0) or (kind == "min" and amp > 0):
        return np.nan, np.nan
    return lat, amp

n_cols = min(3, len(CHANNELS_OF_INTEREST))
n_rows = int(np.ceil(len(CHANNELS_OF_INTEREST) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.0 * n_rows), sharex=True)
axes = np.atleast_1d(axes).ravel()

all_peak_records = []
for ax, ch in zip(axes, CHANNELS_OF_INTEREST):
    idx = ch_names.index(ch)
    for b in BLOCK_LABELS:
        if b not in evoked_by_block:
            continue
        trace = evoked_by_block[b][idx]
        line, = ax.plot(post_time_ms, trace, lw=1.6, label=b)
        row = {"channel": ch, "block": b}
        for comp, (lo, hi, kind) in COMPONENT_WINDOWS.items():
            lat, amp = _local_extremum(trace, lo, hi, kind)
            if not np.isnan(amp):
                ax.plot(lat, amp, marker="o", ms=4, color=line.get_color())
            row[f"{comp}_latency_ms"], row[f"{comp}_amplitude"] = lat, amp
        all_peak_records.append(row)
    ax.axvspan(dipole_epoch_tmin_ms, dipole_epoch_tmax_ms, color="grey", alpha=0.12)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(ch + ("  (interpolated)" if ch in bad_channels else ""), fontsize=10)
    ax.set_xlabel("ms")
for ax in axes[len(CHANNELS_OF_INTEREST):]:
    ax.set_visible(False)
axes[0].set_ylabel("amplitude (a.u.)")
axes[0].legend(fontsize=7, loc="upper right")
fig.suptitle("Sensor-space TEP per block, six channels individually", y=1.02)
fig.tight_layout()
plt.show()

peak_df = pd.DataFrame.from_records(all_peak_records).set_index(["channel", "block"])
display(peak_df)
""")

M(r"""### 4.4 · `FC3`, `C1`, `C3`, `C5`, `CP3`, `Cz` together

The same six channels, overlaid on each other within each block's own panel, so their relative
shapes and magnitudes can be compared directly -- including whether an interpolated channel's shape
is actually shared by its real, measured neighbours in this cluster, rather than something the
interpolation alone could have manufactured. Dashed lines mark whichever of the six are interpolated
in this session (flagged above); solid lines are measured directly.
""")

C(r"""
fig, axes = plt.subplots(1, len(BLOCK_LABELS), figsize=(3.6 * len(BLOCK_LABELS), 3.6),
                          sharey=True, sharex=True)
channel_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for ax, b in zip(axes, BLOCK_LABELS):
    if b not in evoked_by_block:
        ax.set_visible(False)
        continue
    for ci, ch in enumerate(CHANNELS_OF_INTEREST):
        idx = ch_names.index(ch)
        style = "--" if ch in bad_channels else "-"
        ax.plot(post_time_ms, evoked_by_block[b][idx], lw=1.4, ls=style,
                 color=channel_colors[ci % len(channel_colors)], label=ch)
    ax.axvspan(dipole_epoch_tmin_ms, dipole_epoch_tmax_ms, color="grey", alpha=0.12)
    ax.axhline(0, color="black", lw=0.4)
    ax.set_title(f"{b}\n(n={len(sensor_traces[b])})", fontsize=9)
    ax.set_xlabel("ms")
axes[0].set_ylabel("amplitude (a.u.)")
axes[0].legend(fontsize=7, loc="upper right")
fig.suptitle("Sensor-space TEP per block -- FC3 / C1 / C3 / C5 / CP3 / Cz together "
             "(dashed = interpolated in this session)", y=1.03)
fig.tight_layout()
plt.show()
""")

# ============================================================================
# Footer
# ============================================================================
M(r"""
---
### Notes for reading this on a different session

Everything above is read from files in `SUBJECT_DIR`, or computed from them with a formula
grounded in `prime/decider/` (cited by file and function throughout). Re-point section 1 at a
different `SUBJECT_ID` / `SESSION_ID` / `RUN_ID` and re-run; if that session predates a given
column or file (e.g. `dipole_fitting_info.npy`, `prediction_probability` on predetermined trials,
or the `threshold` column in `prime_predictions.csv` — all added partway through this project, see
`claude/online_TEP_replay.md`), the corresponding cell will raise a clear `KeyError` /
`FileNotFoundError` rather than silently showing nothing.

Chapter 2 additionally needs `prime/decider/offline_results/train/global_backrotation_all.npy`
(a shared, cross-subject training artifact, not subject-specific -- see item 2's markdown cell).
Item 2 / 2.1 / the classifier columns of item 3's CSV compute themselves in place, in item 2's
own cell, the first time this notebook is run for a given subject in a kernel with a CUDA-capable
PyTorch install (plus omegaconf, scikit-learn, mne_icalabel) -- otherwise that cell loads a
previously-written `<SUBJECT_ID>_classifier_predictions.csv` if one exists for this subject, or
explains what to install if not. Items 1, 1.1, most of item 3, and item 4 need neither and run in
any kernel.
""")

# ============================================================================
# Assemble, execute, save
# ============================================================================
nb = new_notebook()
nb.cells = CELLS
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

import nbformat
nbformat.validate(nb)
print(f"assembled {len(nb.cells)} cells "
      f"({sum(1 for c in nb.cells if c.cell_type=='code')} code, "
      f"{sum(1 for c in nb.cells if c.cell_type=='markdown')} markdown)")

OUT_PATH = "PRIME_online_performance.ipynb"
nbformat.write(nb, OUT_PATH)
print(f"wrote unexecuted notebook to {OUT_PATH}")
