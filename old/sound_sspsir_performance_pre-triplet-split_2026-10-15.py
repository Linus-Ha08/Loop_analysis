#!/usr/bin/env python
"""
sound_sspsir_performance.py
===========================

What do SOUND and SSP-SIR actually do to the online post-stimulus epochs?

The online post chain (``Preprocessor.preprocess_post``) is replayed here from the
saved raw buffers with the session's own ``calibration_bundle.npy``, stopping at four
points so every step can be measured separately:

    pre_sound : baseline -> artifact window -> [500 Hz low-pass] -> interpolation
                -> avg ref -> ICA -> baseline -> avg ref -> crop to [1, 100] ms
                                                                    (SOUND's input)
    sound     : + SOUND spatial filter, + avg ref
    sspsir    : + SSP-SIR
    final     : + artifact window [1, 15] ms, + avg ref            (== *_post_proc.npy)

Nothing in prime/ is imported for its side effects and nothing there is written to;
the decider package is only put on sys.path (with bytecode writing disabled).

Sampling rate: the post-stimulus chain runs at the rate the session's calibration bundle
was made at (Preprocessor.from_bundle) -- 1 kHz for pilot001-005, 5 kHz since pilot006,
whose bundles also switch on the 500 Hz post low-pass after the first artifact blank.
All windows are given in ms and converted to samples at that rate.

Measured per trial and stage
    RMS in the early / mid / N45 / late windows, GFP time course,
    >100 Hz "muscle" residual, single-dipole amplitude (free and fixed) and R^2
    at the session's calibrated vertex, gain and decorrelation introduced by the step.
Measured per session
    evoked GFP and SNR per stage, effective dimensionality, SOUND eigen-spectrum and
    channel weights, SSP-SIR kernel / components removed / rank / per-sample gain.

Self-check: for trials that have a saved ``*_post_proc.npy`` the replayed ``final``
stage must reproduce it to ~1e-12. The number is printed and put on page 1.

Usage
-----
    python sound_sspsir_performance.py                       # all three pilots, all trials
    python sound_sspsir_performance.py --limit 60            # 60 trials per stage group
    python sound_sspsir_performance.py --subjects pilot005 --groups intervention

Outputs (default ``scripts_analysis/sound_sspsir_outputs/``)
    sound_sspsir_performance.pdf   multi-page figure set
    trial_metrics.csv              one row per trial x stage
    stage_summary.csv              medians per session x group x stage
    filter_metrics.csv             one row per session (filter-level numbers)
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Paths / constants
# --------------------------------------------------------------------------------------

BIDS_ROOT = Path("D:/Linus/Loop/BIDS")
DECIDER = Path("D:/Linus/Loop/prime/decider")
OUT_DIR = Path("D:/Linus/Loop/scripts_analysis/sound_sspsir_pilot006")
SUBJECTS = ["pilot006"]
GROUPS = ["baseline", "calibration", "intervention", "evaluation"]

STAGES = ["pre_sound", "sound", "sspsir", "final"]
STAGE_LABEL = {
    "pre_sound": "in (pre-SOUND)",
    "sound": "+ SOUND",
    "sspsir": "+ SSP-SIR",
    "final": "final",
}
# Okabe-Ito subset, validated for CVD separation; each stage also gets its own marker
# and is direct-labelled, so identity never rests on colour alone.
STAGE_COLOR = {
    "pre_sound": "#0072B2",
    "sound": "#009E73",
    "sspsir": "#D55E00",
    "final": "#CC79A7",
}
STAGE_MARKER = {"pre_sound": "o", "sound": "s", "sspsir": "^", "final": "D"}
# final sits exactly on sspsir after 15 ms — dash it so both stay visible
STAGE_LS = {"pre_sound": "-", "sound": "-", "sspsir": "-", "final": "--"}
INK, MUTED, GRID = "#1a1a1a", "#5b5b5b", "#d8d8d8"

# Windows in ms inside the [1, 100] ms post window (N45 is replaced per session by the
# session's own dipole window from proc_info.npy).
WINDOWS_MS = {"early": (1, 15), "mid": (15, 38), "n45": (38, 50), "late": (60, 100)}


# --------------------------------------------------------------------------------------
# Session discovery and loading
# --------------------------------------------------------------------------------------


def find_session(bids_root: Path, subject: str) -> Path:
    """Directory holding the decider outputs, e.g. BIDS/sub-pilot003/ses-prime/103/prime."""
    hits = sorted(bids_root.glob(f"sub-{subject}/ses-*/*/prime"))
    hits = [h for h in hits if list(h.glob("calibration_bun*le.npy"))]
    if not hits:
        raise FileNotFoundError(f"no session with a calibration bundle for {subject}")
    return hits[-1]


def load_session_files(run_dir: Path) -> dict:
    bundle_path = next(iter(sorted(run_dir.glob("calibration_bun*le.npy"))))  # bundle/bunble
    out = {
        "bundle": np.load(bundle_path, allow_pickle=True).item(),
        "dipole": np.load(run_dir / "dipole_fitting_info.npy", allow_pickle=True).item(),
        "proc_info": np.load(run_dir / "proc_info.npy", allow_pickle=True).item(),
    }
    csv = run_dir / "trials_intervention.csv"
    out["trials"] = pd.read_csv(csv) if csv.exists() else None
    return out


def list_trials(run_dir: Path, groups, limit=None) -> list[tuple[str, str, Path]]:
    """(group, stem, post_raw path) for the requested trial groups."""
    found = []
    for path in sorted(run_dir.glob("*_post_raw.npy")):
        stem = path.name[: -len("_post_raw.npy")]
        m = re.match(r"^(baseline|calibration|intervention_block_\d+|evaluation_\d+)_(\d{4})$", stem)
        if not m:
            continue
        group = re.sub(r"_(block_)?\d+$", "", m.group(1))
        if group in groups:
            found.append((group, stem, path))
    if limit:
        keep, seen = [], {}
        for group, stem, path in found:
            seen[group] = seen.get(group, 0) + 1
            if seen[group] <= limit:
                keep.append((group, stem, path))
        found = keep
    return found


# --------------------------------------------------------------------------------------
# Replay of the online post chain, stage by stage
# --------------------------------------------------------------------------------------


def build_replayer(decider: Path, forward_path: Path, sess: dict):
    """Return (stage_fn, fitter, sfreq) built from the unmodified decider code."""
    sys.dont_write_bytecode = True
    if str(decider) not in sys.path:
        sys.path.insert(0, str(decider))

    import mne
    from prime_core.preprocessing.preprocessor import (
        Preprocessor,
        _interpolate_bad_channels,
        _numpy_to_epochs_array,
    )
    from prime_core.preprocessing.utils.resampling import resample_buffer_polyphase
    from prime_core.preprocessing.utils.post_lowpass import lowpass_post
    from prime_core.preprocessing.utils.ssp_sir_python import ssp_sir_single_trial
    from prime_core.preprocessing.dipole_fitter import DipoleFitter

    # from_bundle takes the post / ICA rates from the bundle itself (1 kHz for pilot001-005,
    # 5 kHz since pilot006), not from prime.yaml, and checks the bundle against them.
    pp = Preprocessor.from_bundle(sess["bundle"], forward_path)
    pp.keep_full_epochs = True
    post_sfreq = pp.post_sfreq

    # index_range in dipole_fitting_info is in samples at the fit's rate -> same rate here.
    fitter = DipoleFitter(forward_path, sfreq=post_sfreq)
    fitter._fitting_info = sess["dipole"]

    cp, cfg = sess["bundle"], pp._cfg
    raw_sfreq = pp._info["sfreq"]
    post_lowpass = bool(cp.get("post_lowpass", False))  # absent in pre-pilot006 bundles

    def stages(raw_post: np.ndarray) -> dict[str, np.ndarray]:
        """raw_post: (n_samples, n_channels) at raw_sfreq over [post_initial_tmin, tmax]."""
        # 5 kHz -> 5 kHz is a plain copy (no anti-alias filter); legacy bundles go to 1 kHz.
        buf = resample_buffer_polyphase(raw_post, sfreq_from=raw_sfreq, sfreq_to=post_sfreq)
        ep = _numpy_to_epochs_array(buf, pp._info_post, pp._post_initial_tmin)

        ep.apply_baseline(cfg.baseline, verbose=False)
        ep = mne.preprocessing.fix_stim_artifact(
            ep, tmin=cfg.artifact_window_1[0], tmax=cfg.artifact_window_1[1], mode="window"
        )
        if post_lowpass:  # after the first blank, exactly as preprocess_post does
            data = lowpass_post(ep.get_data(copy=True), post_sfreq)
            ep = mne.EpochsArray(data, ep.info, events=ep.events, tmin=ep.times[0], verbose=False)
        if cp["bad_channels"]:
            ep, _ = _interpolate_bad_channels(ep, cp["bad_channels"], cp["channel_interpolation_info"])
        ep.set_eeg_reference("average", projection=False, verbose=False)
        cp["ica"].apply(ep, verbose=False)
        ep.apply_baseline(cfg.baseline, verbose=False).set_eeg_reference(
            "average", projection=False, verbose=False
        )
        ep.crop(pp._post_tmin, pp._post_tmax, include_tmax=True)
        out = {"pre_sound": ep.get_data(copy=True)[0]}

        data = np.matmul(cp["sound_filter"], ep.get_data(copy=True))
        ep = mne.EpochsArray(data, ep.info, events=ep.events, tmin=ep.times[0], verbose=False)
        ep.set_eeg_reference("average", projection=False, verbose=False)
        out["sound"] = ep.get_data(copy=True)[0]

        data = ssp_sir_single_trial(
            ep.get_data(copy=True)[0],
            cp["sspsir_suppression_matrix_P"],
            cp["sspsir_sir_projmat_suppr"],
            cp["sspsir_sir_projmat_orig"],
            cp["sspsir_filter_kernel"],
        )
        ep = mne.EpochsArray(data[None], ep.info, events=ep.events, tmin=ep.times[0], verbose=False)
        out["sspsir"] = ep.get_data(copy=True)[0]

        ep = mne.preprocessing.fix_stim_artifact(
            ep, tmin=pp._post_tmin, tmax=cfg.artifact_window_2[1], mode="window"
        )
        ep.set_eeg_reference("average", projection=False, verbose=False)
        out["final"] = ep.get_data(copy=True)[0]
        return out

    return stages, fitter, post_sfreq, post_lowpass


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def gfp(x: np.ndarray) -> np.ndarray:
    return np.sqrt((x ** 2).mean(axis=0))


def win_rms(x: np.ndarray, sl: slice) -> float:
    return float(np.sqrt((x[:, sl] ** 2).mean()))


def hf_rms(x: np.ndarray, sfreq: float, sl: slice, cutoff=100.0) -> float:
    """RMS of the >cutoff Hz residual — the muscle-artifact proxy SSP-SIR targets."""
    from scipy.signal import butter, filtfilt

    b, a = butter(2, cutoff / (sfreq / 2), btype="high")
    return win_rms(filtfilt(b, a, x, axis=-1), sl)


def dipole_stats(fitter, epoch: np.ndarray, info: dict, dip_slice: slice) -> dict:
    """Amplitude (free + fixed) and the R^2 of the sample the fit selects.

    ``index_range`` is relative to the dipole crop (38-50 ms), exactly as
    ``DipoleFitter.fit_trial`` uses it — not to the full [1, 100] ms epoch.
    """
    L = fitter._forward["sol"]["data"]
    L = L - L.mean(axis=0)
    p = info["position_index"] * 3
    Lp = L[:, p : p + 3]
    s, e = info["index_range"]
    seg = epoch[:, dip_slice][:, s:e]
    pinv, Lori = np.linalg.pinv(Lp), Lp @ info["orientation"]

    best = {"amp_free": np.nan, "r2_free": -np.inf, "amp_fixed": np.nan, "r2_fixed": -np.inf}
    for k in range(seg.shape[1]):
        y = seg[:, k]
        denom = ((y - y.mean()) ** 2).sum()
        if denom <= 0:  # a zeroed/constant sample carries no fit
            continue
        Q = pinv @ y
        r2 = 1 - ((y - Lp @ Q) ** 2).sum() / denom
        if r2 > best["r2_free"]:
            best["r2_free"], best["amp_free"] = r2, float(np.linalg.norm(Q))
        a = float(Lori @ y / (Lori @ Lori))
        r2f = 1 - ((y - Lp @ (info["orientation"] * a)) ** 2).sum() / denom
        if r2f > best["r2_fixed"]:
            best["r2_fixed"], best["amp_fixed"] = r2f, abs(a)
    for key in ("r2_free", "r2_fixed"):
        if best[key] == -np.inf:
            best[key] = np.nan
    return best


def step_change(x_in: np.ndarray, x_out: np.ndarray, sl: slice) -> tuple[float, float]:
    """Amplitude gain and 1 - corr between a step's input and output in one window."""
    a, b = x_in[:, sl].ravel(), x_out[:, sl].ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    gain = float(nb / na) if na else np.nan
    corr = float(a @ b / (na * nb)) if na and nb else np.nan
    return gain, 1 - corr


def participation_ratio(cov: np.ndarray) -> float:
    """Effective number of spatial dimensions carrying the variance."""
    ev = np.linalg.eigvalsh(cov)
    ev = ev[ev > 0]
    return float(ev.sum() ** 2 / (ev ** 2).sum()) if ev.size else np.nan


def filter_metrics(sess: dict) -> dict:
    """Everything that can be read off the calibration bundle alone."""
    b = sess["bundle"]
    F, P = b["sound_filter"], b["sspsir_suppression_matrix_P"]
    n_ch = F.shape[0]
    sv = np.linalg.svd(F, compute_uv=False)
    kernel = np.asarray(b["sspsir_filter_kernel"])[0]
    Msup, Morig = b["sspsir_sir_projmat_suppr"], b["sspsir_sir_projmat_orig"]
    A_gain = np.array(
        [
            np.linalg.norm(k * (Msup @ P) + (1 - k) * Morig, ord="fro") / np.sqrt(n_ch)
            for k in kernel
        ]
    )
    return {
        "n_channels": n_ch,
        "bad_channels": ",".join(b["bad_channels"]),
        "sound_dof": float(np.trace(F)),
        "sound_sv": sv,
        "sound_diag": np.diag(F).copy(),
        "sound_dev_identity": float(np.linalg.norm(F - np.eye(n_ch)) / np.sqrt(n_ch)),
        "sspsir_n_pcs": int(round(n_ch - np.trace(P))),
        "sspsir_rank_M": int(np.linalg.matrix_rank(Msup)),
        "sspsir_kernel": kernel,
        "sspsir_gain": A_gain,
    }


# --------------------------------------------------------------------------------------
# Per-session run
# --------------------------------------------------------------------------------------


def run_session(subject: str, args) -> tuple[pd.DataFrame, dict]:
    run_dir = find_session(Path(args.bids_root), subject)
    sess = load_session_files(run_dir)
    forward = Path(args.forward or Path(args.decider) / "offline_data/fsaverage/fsaverage-fwd.fif")
    stages, fitter, sfreq, post_lowpass = build_replayer(Path(args.decider), forward, sess)

    pinfo = sess["proc_info"]
    # proc_info has one rate per window since pilot006; older sessions one top-level 'sfreq'.
    pinfo_sfreq = float(pinfo["post"].get("sfreq", pinfo.get("sfreq", np.nan)))
    if abs(pinfo_sfreq - sfreq) > 1e-6:
        raise ValueError(f"[{subject}] bundle post rate {sfreq} Hz != proc_info post rate {pinfo_sfreq} Hz")

    dip_ms = pinfo["dipole"]
    win = dict(WINDOWS_MS, n45=(dip_ms["tmin"] * 1000, dip_ms["tmax"] * 1000))
    post_t0 = pinfo["post"]["tmin"] * 1000  # 1 ms
    n_post = int(round((pinfo["post"]["tmax"] * 1000 - post_t0) * sfreq / 1000)) + 1
    t_ms = post_t0 + np.arange(n_post) * 1000.0 / sfreq  # time axis of the [1, 100] ms epoch

    def _idx(ms: float) -> int:
        return int(round((ms - post_t0) * sfreq / 1000))

    slices = {k: slice(_idx(a), _idx(b) + 1) for k, (a, b) in win.items()}
    n_dip = slices["n45"].stop - slices["n45"].start
    if n_dip != fitter._post_n_times:
        raise ValueError(f"[{subject}] dipole slice has {n_dip} samples, DipoleFitter expects {fitter._post_n_times}")

    reasons, online_amp = {}, {}
    if sess["trials"] is not None:
        for _, r in sess["trials"].iterrows():
            key = f"{r['stage']}_{int(r['trial_in_stage']):04d}"
            reasons[key] = r.get("post_reject_reason")
            online_amp[key] = r.get("tep_amplitude_raw")

    trials = list_trials(run_dir, args.groups, args.limit)
    print(f"[{subject}] {run_dir}  ({len(trials)} trials, post at {sfreq:.0f} Hz, "
          f"500 Hz low-pass {'on' if post_lowpass else 'off'})")

    rows, evoked_acc, cov_acc, repro = [], {}, {}, []
    for i, (group, stem, path) in enumerate(trials, 1):
        if i % 100 == 0 or i == len(trials):
            print(f"  {i}/{len(trials)}", flush=True)
        try:
            st = stages(np.load(path))
        except Exception as exc:  # a malformed buffer should not kill the session
            warnings.warn(f"{stem}: {exc}")
            continue

        proc = path.with_name(f"{stem}_post_proc.npy")
        if proc.exists():
            repro.append(float(np.max(np.abs(st["final"] - np.load(proc)))))

        prev = None
        for stage in STAGES:
            x = st[stage]
            row = {
                "subject": subject,
                "group": group,
                "stem": stem,
                "stage": stage,
                "post_reject_reason": reasons.get(stem, np.nan),
            }
            row.update({f"rms_{k}": win_rms(x, s) for k, s in slices.items()})
            row["hf_early"] = hf_rms(x, sfreq, slices["early"])
            row["hf_n45"] = hf_rms(x, sfreq, slices["n45"])
            row.update(dipole_stats(fitter, x, sess["dipole"], slices["n45"]))
            if stage == "final":
                # cross-check: the decider's own fitter on the same epoch, and the
                # amplitude that was logged online for this trial
                row["amp_fit_trial"] = float(fitter.fit_trial(x[:, slices["n45"]]))
                row["amp_online"] = online_amp.get(stem, np.nan)
            if prev is not None:
                row["gain_n45"], row["dissim_n45"] = step_change(prev, x, slices["n45"])
            rows.append(row)
            prev = x

            key = (group, stage)
            acc = evoked_acc.setdefault(key, [np.zeros_like(x), 0])
            acc[0] += x
            acc[1] += 1
            cov_acc.setdefault(stage, np.zeros((x.shape[0], x.shape[0])))
            seg = x[:, slices["n45"]]
            cov_acc[stage] += seg @ seg.T

    df = pd.DataFrame(rows)
    evoked = {k: (v[0] / v[1], v[1]) for k, v in evoked_acc.items()}

    # Evoked / single-trial SNR per stage, from the accumulated averages.
    snr = {}
    for stage in STAGES:
        ev = [evoked[(g, stage)] for g in args.groups if (g, stage) in evoked]
        if not ev:
            continue
        n_tot = sum(n for _, n in ev)
        mean_ev = sum(e * n for e, n in ev) / n_tot
        sub = df[df.stage == stage]
        snr[stage] = {
            "evoked_rms_n45": win_rms(mean_ev, slices["n45"]),
            "trial_rms_n45": float(sub["rms_n45"].median()),
            "snr_n45": win_rms(mean_ev, slices["n45"]) / float(sub["rms_n45"].median()),
            "pr_n45": participation_ratio(cov_acc[stage]),
        }

    fin = df[df.stage == "final"]
    amp_err = float((fin["amp_free"] - fin["amp_fit_trial"]).abs().max()) if len(fin) else np.nan
    online = fin.dropna(subset=["amp_online"])
    amp_online_err = float((online["amp_fit_trial"] - online["amp_online"]).abs().max()) if len(online) else np.nan

    info = filter_metrics(sess)
    info.update(
        subject=subject,
        run_dir=str(run_dir),
        n_trials=len(trials),
        repro_max_abs=max(repro) if repro else np.nan,
        repro_n=len(repro),
        amp_err=amp_err,
        amp_online_err=amp_online_err,
        dipole_window_ms=f"{win['n45'][0]:.0f}-{win['n45'][1]:.0f}",
        # index_range counts samples from the dipole window start, at the post rate
        fit_window_ms="{:.1f}-{:.1f}".format(
            win["n45"][0] + sess["dipole"]["index_range"][0] * 1000 / sfreq,
            win["n45"][0] + (sess["dipole"]["index_range"][1] - 1) * 1000 / sfreq,
        ),
        slices=slices,
        t_ms=t_ms,
        evoked=evoked,
        snr=snr,
        sfreq=sfreq,
        post_lowpass=post_lowpass,
    )
    return df, info


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def make_pdf(df: pd.DataFrame, infos: list[dict], out_pdf: Path, groups):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams.update(
        {
            "figure.figsize": (11.7, 8.3),
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlesize": 11,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.size": 9,
            "legend.frameon": False,
            "axes.axisbelow": True,
        }
    )

    def tidy(ax):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        return ax

    subs = [i["subject"] for i in infos]
    with PdfPages(out_pdf) as pdf:
        # ---- page 1: what was run, and the self-check ------------------------------
        fig = plt.figure()
        fig.suptitle("SOUND and SSP-SIR — online post-stimulus chain, replayed", y=0.97)
        lines = [
            "stages: pre_sound (SOUND input) -> sound -> sspsir -> final (== *_post_proc.npy)",
            f"trial groups: {', '.join(groups)}",
            "",
            f"{'session':<10}{'trials':>7}{'replication':>13}{'amp check':>11}{'vs online':>11}"
            f"{'post Hz':>9}{'crop win':>10}{'fit win':>12}{'SOUND dof':>11}{'|F-I|':>8}{'SSP PCs':>9}"
            f"{'rank M':>8}{'kernel@fit':>12}  bad channels",
        ]
        for i in infos:
            lines.append(
                f"{i['subject']:<10}{i['n_trials']:>7}{i['repro_max_abs']:>13.2e}"
                f"{i['amp_err']:>11.2e}{i['amp_online_err']:>11.2e}"
                f"{i['sfreq']:>9.0f}{i['dipole_window_ms']:>10}{i['fit_window_ms']:>12}{i['sound_dof']:>11.1f}"
                f"{i['sound_dev_identity']:>8.3f}{i['sspsir_n_pcs']:>9}{i['sspsir_rank_M']:>8}"
                f"{np.mean(np.asarray(i['sspsir_kernel'])[i['slices']['n45']]):>12.3f}"
                f"  {i['bad_channels']}"
            )
        lines += [
            "",
            "replication = max |replayed final − saved *_post_proc.npy|   (expect ~1e-12 or 0)",
            "amp check   = max |own dipole fit − DipoleFitter.fit_trial|  (expect ~1e-15)",
            "vs online   = max |fit_trial − tep_amplitude_raw in the trial table|",
            "SOUND dof = trace of the filter, out of 60 channels; |F-I| = RMS distance from identity",
            "post Hz = rate of the post-stimulus chain (from the calibration bundle); "
            + "; ".join(f"{i['subject']} 500 Hz low-pass {'on' if i['post_lowpass'] else 'off'}" for i in infos),
        ]
        fig.text(0.06, 0.86, "\n".join(lines), family="monospace", va="top", fontsize=8.5)
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 2: SOUND filter ---------------------------------------------------
        fig, axes = plt.subplots(1, 3)
        fig.suptitle("SOUND filter — how much of sensor space survives", y=0.96)
        for i in infos:
            axes[0].plot(np.arange(1, len(i["sound_sv"]) + 1), i["sound_sv"], lw=2, label=i["subject"])
        axes[0].set(xlabel="component", ylabel="singular value (gain)", title="eigen-spectrum")
        axes[0].axhline(1, color=MUTED, lw=1, ls=":")
        axes[0].legend()
        for i in infos:
            axes[1].plot(np.sort(i["sound_diag"]), lw=2, label=i["subject"])
        axes[1].set(xlabel="channel (sorted)", ylabel="diagonal of F", title="per-channel self-weight")
        axes[2].bar(subs, [i["sound_dof"] for i in infos], color="#0072B2", width=0.55)
        axes[2].axhline(infos[0]["n_channels"], color=MUTED, ls=":", lw=1)
        axes[2].set(ylabel="trace(F)", title=f"effective dimensions (of {infos[0]['n_channels']})")
        for ax in axes:
            tidy(ax)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 3: SSP-SIR filter -------------------------------------------------
        fig, axes = plt.subplots(1, 3)
        fig.suptitle("SSP-SIR — when it acts, how much it removes", y=0.96)
        for i in infos:
            t = i["t_ms"]
            axes[0].plot(t, i["sspsir_kernel"], lw=2, label=i["subject"])
            axes[1].plot(t, i["sspsir_gain"], lw=2, label=i["subject"])
        for ax, ttl, yl in (
            (axes[0], "artifact kernel (1 = full suppression)", "kernel"),
            (axes[1], "per-sample operator gain", "‖A(t)‖ / √n"),
        ):
            a, b = infos[0]["dipole_window_ms"].split("-")
            ax.axvspan(float(a), float(b), color="#0072B2", alpha=0.08, lw=0)
            ax.set(xlabel="time (ms)", ylabel=yl, title=ttl)
            ax.legend()
        x = np.arange(len(subs))
        axes[2].bar(x - 0.2, [i["sspsir_n_pcs"] for i in infos], 0.38, color="#D55E00", label="PCs removed")
        axes[2].bar(x + 0.2, [i["sspsir_rank_M"] for i in infos], 0.38, color="#0072B2", label="rank M")
        axes[2].set_xticks(x, subs)
        axes[2].set(title="components removed / retained rank")
        axes[2].legend()
        for ax in axes:
            tidy(ax)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 4: evoked GFP per stage ------------------------------------------
        fig, axes = plt.subplots(2, len(infos))
        axes = np.atleast_2d(axes).reshape(2, len(infos))
        fig.suptitle("Evoked GFP by stage — whole window (log) and the TEP range", y=0.97)
        for c, i in enumerate(infos):
            curves = {}
            for stage in STAGES:
                ev = [(e, n) for (g, s), (e, n) in i["evoked"].items() if s == stage]
                if not ev:
                    continue
                mean_ev = sum(e * n for e, n in ev) / sum(n for _, n in ev)
                curves[stage] = gfp(mean_ev)
            t = i["t_ms"]
            for stage, y in curves.items():
                for r in (0, 1):
                    axes[r, c].plot(t, y, lw=2, ls=STAGE_LS[stage], color=STAGE_COLOR[stage],
                                    label=STAGE_LABEL[stage])
            a, b = i["dipole_window_ms"].split("-")
            stacked = np.vstack(list(curves.values()))
            for r, xlim in ((0, (1, 100)), (1, (25, 100))):
                ax = axes[r, c]
                ax.axvspan(float(a), float(b), color="#0072B2", alpha=0.08, lw=0)
                ax.set(xlabel="time (ms)", xlim=xlim)
                # the artifact window is zeroed, so scale to the data that is actually there
                seg = stacked[:, (t >= xlim[0]) & (t <= xlim[1])]
                hi = float(seg.max()) * 1.15
                ax.set_ylim(hi / 1e3 if r == 0 else 0, hi)
                tidy(ax)
            axes[0, c].set_yscale("log")
            axes[0, c].set_title(i["subject"])
        axes[0, 0].set_ylabel("GFP (µV), log")
        axes[1, 0].set_ylabel("GFP (µV)")
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 5: single-trial RMS by window ------------------------------------
        wins = list(WINDOWS_MS)
        fig, axes = plt.subplots(len(infos), len(wins), sharex=True)
        axes = np.atleast_2d(axes)
        fig.suptitle("Single-trial RMS by window and stage (median, IQR)", y=0.97)
        for r, i in enumerate(infos):
            sub = df[df.subject == i["subject"]]
            for c, w in enumerate(wins):
                ax = axes[r, c]
                med = [sub[sub.stage == s][f"rms_{w}"].median() for s in STAGES]
                q1 = [sub[sub.stage == s][f"rms_{w}"].quantile(0.25) for s in STAGES]
                q3 = [sub[sub.stage == s][f"rms_{w}"].quantile(0.75) for s in STAGES]
                ax.bar(range(4), med, 0.6, color=[STAGE_COLOR[s] for s in STAGES])
                ax.errorbar(range(4), med, yerr=[np.subtract(med, q1), np.subtract(q3, med)],
                            fmt="none", ecolor=MUTED, lw=1)
                ax.set_xticks(range(4), [STAGE_LABEL[s] for s in STAGES], rotation=35, ha="right")
                if r == 0:
                    ax.set_title(f"{w}  {WINDOWS_MS[w][0]}-{WINDOWS_MS[w][1]} ms")
                if c == 0:
                    ax.set_ylabel(f"{i['subject']}\nRMS (µV)")
                tidy(ax)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 6: muscle residual, SNR, dimensionality ---------------------------
        fig, axes = plt.subplots(1, 3)
        fig.suptitle("Where each step helps", y=0.96)
        x = np.arange(len(subs))
        for k, stage in enumerate(STAGES):
            off = (k - 1.5) * 0.2
            hf = [df[(df.subject == s) & (df.stage == stage)]["hf_early"].median() for s in subs]
            axes[0].bar(x + off, hf, 0.19, color=STAGE_COLOR[stage], label=STAGE_LABEL[stage])
            snr = [next((i["snr"][stage]["snr_n45"] for i in infos if i["subject"] == s), np.nan) for s in subs]
            axes[1].bar(x + off, snr, 0.19, color=STAGE_COLOR[stage])
            pr = [next((i["snr"][stage]["pr_n45"] for i in infos if i["subject"] == s), np.nan) for s in subs]
            axes[2].bar(x + off, pr, 0.19, color=STAGE_COLOR[stage])
        for ax, ttl, yl in (
            (axes[0], ">100 Hz residual, 1-15 ms", "RMS (µV)"),
            (axes[1], "evoked / single-trial RMS in dipole window", "SNR"),
            (axes[2], "effective dimensions, dipole window", "participation ratio"),
        ):
            ax.set_xticks(x, subs)
            ax.set(title=ttl, ylabel=yl)
            tidy(ax)
        handles = [
            plt.Line2D([], [], marker="s", ls="", ms=9, color=STAGE_COLOR[s], label=STAGE_LABEL[s])
            for s in STAGES
        ]
        fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.93))
        fig.tight_layout(rect=(0, 0, 1, 0.89))
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 7: the outcome measure itself ------------------------------------
        fig, axes = plt.subplots(2, len(infos), sharex=True)
        axes = np.atleast_2d(axes).reshape(2, len(infos))
        fig.suptitle("Effect on the fitted dipole (free orientation) and fit quality", y=0.97)
        for c, i in enumerate(infos):
            sub = df[df.subject == i["subject"]]
            amp = [sub[sub.stage == s]["amp_free"].dropna().values for s in STAGES]
            r2 = [sub[sub.stage == s]["r2_free"].dropna().values for s in STAGES]
            for ax, vals, yl in ((axes[0, c], amp, "amplitude (µA·m)"), (axes[1, c], r2, "R²")):
                bp = ax.boxplot(vals, widths=0.55, showfliers=False, patch_artist=True)
                for patch, s in zip(bp["boxes"], STAGES):
                    patch.set(facecolor=STAGE_COLOR[s], alpha=0.75, edgecolor=MUTED, lw=0.8)
                for med in bp["medians"]:
                    med.set(color=INK, lw=1.4)
                ax.set_xticks(range(1, 5), [STAGE_LABEL[s] for s in STAGES], rotation=35, ha="right")
                ax.set_ylabel(yl if c == 0 else "")
                tidy(ax)
            axes[0, c].set_title(i["subject"])
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 8: what each step does to a single trial --------------------------
        fig, axes = plt.subplots(1, 2)
        fig.suptitle("Per-trial effect of each step inside the dipole window", y=0.96)
        steps = [("sound", "SOUND"), ("sspsir", "SSP-SIR"), ("final", "window")]
        for j, (stage, name) in enumerate(steps):
            for k, s in enumerate(subs):
                sub = df[(df.subject == s) & (df.stage == stage)]
                pos = j * (len(subs) + 1) + k
                for ax, col in ((axes[0], "gain_n45"), (axes[1], "dissim_n45")):
                    vals = sub[col].dropna().values
                    if not len(vals):
                        continue
                    bp = ax.boxplot(vals, positions=[pos], widths=0.7, showfliers=False, patch_artist=True)
                    bp["boxes"][0].set(facecolor=STAGE_COLOR[stage], alpha=0.75, edgecolor=MUTED, lw=0.8)
                    bp["medians"][0].set(color=INK, lw=1.4)
        ticks = [j * (len(subs) + 1) + k for j in range(len(steps)) for k in range(len(subs))]
        labels = [f"{n}\n{s.replace('pilot', '')}" for _, n in steps for s in subs]
        for ax, ttl, yl, ref in (
            (axes[0], "amplitude gain of the step", "‖out‖ / ‖in‖", 1.0),
            (axes[1], "signal change introduced", "1 − cos(in, out)", 0.0),
        ):
            ax.axhline(ref, color=MUTED, ls=":", lw=1)
            ax.set_xticks(ticks, labels, fontsize=7)
            ax.set(title=ttl, ylabel=yl)
            tidy(ax)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 9: stability across blocks ----------------------------------------
        fig, axes = plt.subplots(1, len(infos), sharey=True)
        axes = np.atleast_1d(axes)
        fig.suptitle("Dipole-window RMS by trial group — does the calibrated filter keep working?", y=0.96)
        for ax, i in zip(axes, infos):
            sub = df[df.subject == i["subject"]]
            present = [g for g in groups if (sub.group == g).any()]
            for stage in STAGES:
                med = [sub[(sub.group == g) & (sub.stage == stage)]["rms_n45"].median() for g in present]
                ax.plot(present, med, lw=2, ls=STAGE_LS[stage], marker=STAGE_MARKER[stage], ms=6,
                        color=STAGE_COLOR[stage], label=STAGE_LABEL[stage])
            ax.set(title=i["subject"], xlabel="")
            ax.tick_params(axis="x", rotation=25)
            tidy(ax)
        axes[0].set_ylabel("median RMS in dipole window (µV)")
        axes[0].legend()
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        pdf.savefig(fig)
        plt.close(fig)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bids-root", default=str(BIDS_ROOT))
    ap.add_argument("--decider", default=str(DECIDER))
    ap.add_argument("--forward", default=None, help="fsaverage-fwd.fif (default: under --decider)")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--subjects", nargs="+", default=SUBJECTS)
    ap.add_argument("--groups", nargs="+", default=GROUPS, choices=GROUPS)
    ap.add_argument("--limit", type=int, default=None, help="max trials per group per session")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frames, infos = [], []
    for subject in args.subjects:
        df, info = run_session(subject, args)
        frames.append(df)
        infos.append(info)

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out / "trial_metrics.csv", index=False)

    num = df.select_dtypes("number").columns
    summary = df.groupby(["subject", "group", "stage"])[list(num)].median().reset_index()
    summary.to_csv(out / "stage_summary.csv", index=False)

    pd.DataFrame(
        [
            {
                k: v
                for k, v in i.items()
                if k not in {"sound_sv", "sound_diag", "sspsir_kernel", "sspsir_gain", "evoked", "snr", "slices", "t_ms"}
            }
            | {
                "kernel_mean_dipole_window": float(
                    np.mean(np.asarray(i["sspsir_kernel"])[i["slices"]["n45"]])
                ),
                "snr_pre_sound": i["snr"].get("pre_sound", {}).get("snr_n45", np.nan),
                "snr_final": i["snr"].get("final", {}).get("snr_n45", np.nan),
            }
            for i in infos
        ]
    ).to_csv(out / "filter_metrics.csv", index=False)

    make_pdf(df, infos, out / "sound_sspsir_performance.pdf", args.groups)

    print("\n=== summary ===")
    for i in infos:
        k = np.asarray(i["sspsir_kernel"])[i["slices"]["n45"]]
        print(
            f"{i['subject']}: replication {i['repro_max_abs']:.2e} (n={i['repro_n']}), "
            f"SOUND dof {i['sound_dof']:.1f}/{i['n_channels']}, "
            f"SSP-SIR {i['sspsir_n_pcs']} PCs removed, rank {i['sspsir_rank_M']}, "
            f"kernel in dipole window {k.mean():.3f}, "
            f"SNR {i['snr'].get('pre_sound', {}).get('snr_n45', float('nan')):.2f} -> "
            f"{i['snr'].get('final', {}).get('snr_n45', float('nan')):.2f}"
        )
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
