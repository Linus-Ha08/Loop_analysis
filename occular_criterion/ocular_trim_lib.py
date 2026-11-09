# -*- coding: utf-8 -*-
"""ocular_trim_lib.py - loaders and gate arithmetic for the [30,50] / trimmed-reference review.
Reads (read-only) what ocular_window_review.py stored in Loop/offline_ocular_window; writes nothing into prime/.
Pulse sources are stored at 1 kHz (every 5th sample of the 5 kHz epoch, -30..100 ms); the 125 calibration trials'
sources at the native 5 kHz. So a reference (mean / SD of |s| over a window) is exact; a pulse's median |s| over a
window is computed from the 1 kHz samples (a sub-sample of the 5 kHz ones) and validated against the stored exact scores.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home() / "mnt" / "Loop"
WIN = ROOT / "offline_ocular_window"
SESSIONS = ["pilot003", "pilot004", "pilot005", "pilot006"]
BLOCKS = ["baseline", "calibration", "iv1", "iv2", "iv3", "iv4", "t0", "t15", "t30", "t60"]
EVIDENCE_K, STRONG_K = 3.0, 6.0


def robust_sd(x):
    x = np.asarray(x, dtype=float)
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def _ms(t):
    """stored time axes are in seconds in some files, ms in others"""
    return t * 1e3 if np.nanmax(np.abs(t)) < 5 else t


def load_session(s):
    w = WIN / "windows" / f"sub-{s}"
    r = WIN / "replay" / f"sub-{s}"
    meta = json.load(open(w / "meta.json"))
    cal = np.load(r / "calibration_sources.npz", allow_pickle=True)
    # pulse sources (1 kHz) in recording order
    stems, src = [], []
    for b in BLOCKS:
        f = r / "blocks" / f"{b}.npz"
        if not f.is_file():
            continue
        z = np.load(f, allow_pickle=True)
        stems += [str(x) for x in z["stems"]]
        src.append(np.asarray(z["src"], dtype=float))
    src = np.concatenate(src)
    # table: one row per pulse
    ws = pd.read_csv(w / "scores.csv")
    base = ws[ws.window == "20-60"].drop_duplicates("stem").set_index("stem")
    pre = pd.read_csv(w / "pre" / "scores.csv").drop_duplicates("stem").set_index("stem")
    pl = pd.read_csv(WIN / "windows" / "tables" / f"pulses_{s}.csv").set_index("stem")
    df = pd.DataFrame(index=pd.Index(stems, name="stem"))
    for c in ["block", "trial_in_stage", "category", "condition", "order", "single"]:
        df[c] = base.loc[df.index, c].to_numpy()
    df["single"] = df["single"].astype(bool)
    for c in ["veog_mean_n45", "heog_mean_n45"]:
        df[c] = pre.loc[df.index, c].to_numpy(dtype=float)
    # pre-pulse eye activity (raw -210..-10 ms p-p) lives in the pulses table
    df["veog_pre_ptp"] = pl.loc[df.index, "veog_pre_ptp"].to_numpy(dtype=float)
    df["heog_pre_ptp"] = pl.loc[df.index, "heog_pre_ptp"].to_numpy(dtype=float)
    df["z_now"] = pl.loc[df.index, "z_max__current"].to_numpy(dtype=float)       # the decider as it runs (HEAD)
    df["amp"] = pl.loc[df.index, "amp__current"].to_numpy(dtype=float)
    df["order"] = df["order"].astype(int)
    return dict(s=s, meta=meta, df=df, src=src, cal=np.asarray(cal["src"], float), cal_t=_ms(np.asarray(cal["times"], float)),
                eye_ics=[int(i) for i in cal["eye_ics"]], ws=ws, pl=pl)


def pulse_t_ms():
    return np.arange(-30, 101, dtype=float)               # 1 kHz grid of the stored pulse sources


def win_mask(t, a, b):
    return (t >= a - 1e-9) & (t <= b + 1e-9)


def reference(cal_abs, mask, keep=None):
    """mean / SD of |s| over every sample of the (kept) calibration trials inside the window, per eye IC."""
    x = cal_abs[:, :, mask] if keep is None else cal_abs[keep][:, :, mask]
    flat = x.transpose(1, 0, 2).reshape(x.shape[1], -1)
    return flat.mean(axis=1), flat.std(axis=1)           # np.std = population SD, as scipy.stats.zscore


def cal_scores(cal_abs, mask, mean, sd):
    """score of every calibration trial: max over eye ICs of the median z over the window"""
    z = (np.median(cal_abs[:, :, mask], axis=2) - mean[None]) / sd[None]
    return z.max(axis=1)


def trimmed_reference(cal_abs, mask, cutoff, max_iter=10):
    """drop calibration trials scoring above `cutoff`, refit, repeat until the kept set stops changing"""
    keep = np.ones(len(cal_abs), bool)
    for it in range(max_iter):
        mean, sd = reference(cal_abs, mask, keep)
        sc = cal_scores(cal_abs, mask, mean, sd)
        new = sc <= cutoff
        if (new == keep).all():
            break
        keep = new
    return mean, sd, keep, sc
