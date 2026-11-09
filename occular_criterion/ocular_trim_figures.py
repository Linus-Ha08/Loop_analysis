# -*- coding: utf-8 -*-
"""ocular_trim_figures.py - figures for ocular_trim_review.py (reads its tables and cache; writes only to
Loop/offline_ocular_window/trim/figures)."""
import sys
sys.dont_write_bytecode = True
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import ocular_trim_lib as L
import ocular_trim_review as T

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COL = {"now": "#52514e", "20-60": "#2a78d6", "30-60": "#eb6834", "30-50": "#1baf7a"}
MK = {"now": "o", "20-60": "s", "30-60": "D", "30-50": "^"}
FIG = T.OUT / "figures"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS,
                     "font.family": ["DejaVu Sans"], "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.grid": True, "axes.axisbelow": True})
LABEL = {"now": "now (whole-epoch reference, 15–100 ms)", "20-60|all": "20–60 ms, all calibration trials", "20-60|trim": "20–60 ms, trimmed",
         "30-60|all": "30–60 ms, all calibration trials", "30-60|trim": "30–60 ms, trimmed",
         "30-50|all": "30–50 ms, all calibration trials", "30-50|trim": "30–50 ms, trimmed"}


def sty(des):
    w, _, m = des.partition("|")
    return dict(color=COL[w], marker=MK[w], ls="--" if m == "trim" else "-", mfc="none" if m == "trim" else COL[w])


def fig_equal_cost():
    mt = pd.read_csv(T.TAB / "matched.csv")
    g = mt.groupby(["design", "q"])[["hit_n45", "precision_n45", "clean_rejected_pct"]].mean().reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.3), sharex=True)
    for des in T.DESIGNS:
        d = g[g.design == des]
        for a, c in zip(ax, ["hit_n45", "precision_n45"]):
            a.plot(d.q, d[c], lw=1.8, ms=5.5, mew=1.4, **sty(des))
    ax[0].set_title("Frontal-offset pulses caught")
    ax[0].set_ylabel("% of pulses with a 38–50 ms offset that are rejected")
    ax[1].set_title("Precision of the rejections")
    ax[1].set_ylabel("% of rejected pulses with a 38–50 ms offset")
    ax[1].set_ylim(84, 101)
    for a in ax:
        a.set_xlabel("% of single pulses rejected (cutoff set per session)")
        a.set_xticks(T.MATCH)
    handles = [Line2D([], [], lw=1.8, ms=5.5, mew=1.4, label=LABEL[d], **sty(d)) for d in T.DESIGNS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.6, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Equal cost: the window and the trimming change little", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(FIG / "t2_equal_cost.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_global_cutoff(q=4):
    C = pickle.load(open(T.OUT / "cache.pkl", "rb"))
    S = list(C)
    designs = ["now", "20-60|all", "30-60|all", "30-50|all", "30-50|trim", "30-60|trim"]
    res = {}
    for des in designs:
        rej = np.array([[(C[s]["D"][des][i] & C[s]["single"]).sum() / C[s]["single"].sum() * 100 for i in range(len(T.Z_FINE))] for s in S])
        iz = int(np.argmin(np.abs(rej.mean(0) - q)))
        res[des] = (T.Z_FINE[iz], [T.metrics(C[s]["D"][des][iz], C[s]["ev"], C[s]["single"]) for s in S])
    rows = []
    for des in designs:
        for s, m in zip(S, res[des][1]):
            rows.append(dict(design=des, session=s, z=res[des][0], **m))
    pd.DataFrame(rows).to_csv(T.TAB / f"global_cutoff_q{q}.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.3))
    x = np.arange(len(S))
    off = np.linspace(-0.27, 0.27, len(designs))
    for k, des in enumerate(designs):
        for a, c in zip(ax, ["pct_rejected", "hit_n45"]):
            a.plot(x + off[k], [m[c] for m in res[des][1]], ls="none", ms=6.5, mew=1.4, **{kk: v for kk, v in sty(des).items() if kk != "ls"})
    ev = pd.read_csv(L.WIN / "windows" / "tables" / "evidence.csv")
    evn = ev[ev.evidence == "n45"].set_index("subject").pct_singles
    ax[0].plot(x, [evn[s] for s in S], ls="none", marker="_", ms=26, mew=2, color=INK)
    ax[0].annotate("frontal offset\n(evidence rate)", (x[0] - 0.42, evn[S[0]] + 0.2), fontsize=8, color=INK2, ha="left", va="bottom")
    ax[0].set_title(f"Rejected single pulses at one global cutoff (mean {q} %)")
    ax[0].set_ylabel("% of single pulses rejected")
    ax[1].set_title("Frontal-offset pulses caught")
    ax[1].set_ylabel("% of pulses with a 38–50 ms offset that are rejected")
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(S)
        a.set_xlim(-0.6, len(S) - 0.4)
        a.set_ylim(bottom=0)
        a.grid(axis="x", visible=False)
    handles = [Line2D([], [], ls="none", ms=6.5, mew=1.4, label=f"{LABEL[d]}  (z {res[d][0]:.2f})",
                      **{kk: v for kk, v in sty(d).items() if kk != "ls"}) for d in designs]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.6, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("One cutoff for everyone: sessions differ in how much gets rejected", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(FIG / f"t3_global_cutoff_q{q}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_reference():
    r = pd.read_csv(T.TAB / "reference.csv")
    prof = r[r.window.str.startswith("profile")]
    ref = r[~r.window.str.startswith("profile")]
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.3), gridspec_kw=dict(width_ratios=[1, 1.35]))
    bins = ["profile 15-20", "profile 20-30", "profile 30-50", "profile 50-60", "profile 60-100"]
    xs = np.arange(len(bins))
    ics = prof.groupby(["session", "ic"]).size().index.tolist()
    for (s, ic) in ics:
        d = prof[(prof.session == s) & (prof.ic == ic)].set_index("window").loc[bins]["mean"].to_numpy()
        hot = s == "pilot006" or (s == "pilot004" and ic == 0)
        ax[0].plot(xs, d, color="#e34948" if hot else MUTED, lw=1.8 if hot else 1.0, alpha=1 if hot else 0.7, marker="o", ms=3.5)
        if hot:
            ax[0].annotate(f"{s[-3:]} IC{ic}", (xs[0], d[0]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK2, va="center")
    ax[0].set_yscale("log")
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels([b.replace("profile ", "") for b in bins])
    ax[0].set_xlabel("ms after the pulse")
    ax[0].set_ylabel("median |s| of the calibration trials")
    ax[0].set_title("TMS decay sits in some eye ICs until about 30 ms")
    ax[0].axvspan(1.5, 2.5, color=GRID, alpha=0.6, lw=0)
    # reference SD by IC
    lab, k = [], 0
    for (s, ic) in ics:
        for j, W in enumerate(["20-60", "30-60", "30-50"]):
            for m, mf in (("all", COL[W]), ("trim", "none")):
                v = ref[(ref.session == s) & (ref.ic == ic) & (ref.window == W) & (ref["mode"] == m)].sd.iloc[0]
                ax[1].plot(k + (j - 1) * 0.22, v, marker=MK[W], ms=6.2, mew=1.4, color=COL[W], mfc=mf, ls="none")
        lab.append(f"{s[-3:]}\nIC{ic}")
        k += 1
    ax[1].set_xticks(range(len(lab)))
    ax[1].set_xticklabels(lab, fontsize=8)
    ax[1].set_ylabel("SD of |s| in the reference window")
    ax[1].set_title("Reference SD: trimming shrinks it; 30 ms start removes the decay")
    ax[1].grid(axis="x", visible=False)
    handles = [Line2D([], [], ls="none", marker=MK[w], ms=6.2, mew=1.4, color=COL[w], label=f"{w.replace('-', '–')} ms") for w in ["20-60", "30-60", "30-50"]]
    handles += [Line2D([], [], ls="none", marker="o", ms=6.2, mew=1.4, color=INK2, label="all calibration trials"),
                Line2D([], [], ls="none", marker="o", ms=6.2, mew=1.4, color=INK2, mfc="none", label="trimmed (z 3)")]
    ax[1].legend(handles=handles, frameon=False, fontsize=8.4, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "t1_reference.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_reference()
    fig_equal_cost()
    fig_global_cutoff(4)
    fig_global_cutoff(5)
    print("figures written")
