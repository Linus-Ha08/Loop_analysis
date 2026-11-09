"""Concentration sweep for the predetermined-trial ITI generator (prime/decider/util/iti.py).

TARGET_WAIT went 0.65 -> 0.50 s, so the Beta + point-mass mixture is re-solved for
TARGET_MEAN = 3.0 s and drawn for a range of CONCENTRATION values, each against the ITIs
PRIME actually produced in pilot006 -- the only session run with TARGET_WAIT = 0.50.

Read-only with respect to prime/: iti.py's constants are imported, its mixture algebra is
mirrored here (and checked against its own _ALPHA/_BETA), and every candidate is also drawn
through iti.sample_iti itself by temporarily swapping its module-level parameters.

PRIME ITI = pulse_time - trial_start_time over prime_triplet + prime_single_pulse trials,
forced trials pinned to SPIKE_VALUE. PRIME fires on the 10 ms periodic grid, so the KS
distance is taken at the tick midpoints (2.505, 2.515, ...) instead of at sub-ms jitter.

    python iti_concentration_sweep.py

Writes next to this script:
    iti_concentration_sweep_densities.png   one panel per concentration, pilot006 underneath
    iti_concentration_sweep_fit.png         CDFs, KS vs concentration, forced rate per session
    iti_concentration_sweep_table.csv       the numbers behind both figures
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator
from scipy import stats

HERE = Path(__file__).resolve().parent
LOOP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LOOP / "prime" / "decider"))
import util.iti as iti  # noqa: E402  (read-only import of the decider's sampler)

# ---------------------------------------------------------------------------
# Sweep settings
# ---------------------------------------------------------------------------
TARGET_MEAN = 3.0                      # s: 2.5 s to search start + TARGET_WAIT 0.50
SPIKE_WEIGHT = iti.SPIKE_WEIGHT        # kept as in iti.py; override here to test others
PANEL_CONCENTRATIONS = (2, 3, 4, 5, 7, 10, 15, 25)
CDF_CONCENTRATIONS = (2, 3, 5, 10, 25)
SWEEP = np.geomspace(1.0, 40.0, 400)

# session -> TARGET_WAIT it ran with (only pilot006 used 0.50)
SESSIONS = {"pilot003": 0.65, "pilot004": 0.65, "pilot005": 0.65, "pilot006": 0.50}
REFERENCE = "pilot006"
PRIME_CONDITIONS = ("prime_triplet", "prime_single_pulse")
PERIODIC_INTERVAL = 0.01               # s, process_periodic tick (2_intervention.py)
N_DRAWS = 100_000
SEED = 0

LO, HI, SPIKE = iti.ITI_MIN, iti.ITI_MAX, iti.SPIKE_VALUE
SPAN = HI - LO
TICKS = np.arange(LO + PERIODIC_INTERVAL / 2, HI, PERIODIC_INTERVAL)
GRID = np.linspace(LO, HI, 30001)

# Chart tokens (light surface)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID_LINE, AXIS_LINE, DATA_FILL = "#e1e0d9", "#c3c2b7", "#d3d2cb"
MODEL, ALT = "#2a78d6", "#eb6834"
RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")   # ordinal, one per CDF_CONCENTRATIONS


# ---------------------------------------------------------------------------
# Mixture algebra (mirrors util/iti.py)
# ---------------------------------------------------------------------------
def solve_beta(target_mean, concentration, w=SPIKE_WEIGHT):
    """Beta(alpha, beta) on [ITI_MIN, ITI_MAX] such that the mixture mean is target_mean."""
    beta_mean = (target_mean - w * SPIKE) / (1 - w)
    p = (beta_mean - LO) / SPAN
    if not 0 < p < 1:
        raise ValueError(f"target_mean {target_mean} puts the Beta mean at {beta_mean:.3f}, "
                         f"outside [{LO}, {HI}]")
    return p * concentration, (1 - p) * concentration


def mixture_cdf(t, a, b, w=SPIKE_WEIGHT):
    t = np.asarray(t, dtype=float)
    cont = stats.beta.cdf(np.clip((t - LO) / SPAN, 0, 1), a, b)
    return (1 - w) * cont + w * (t >= SPIKE)


def continuous_pdf(t, a, b, w=SPIKE_WEIGHT):
    """Density of the Beta part, scaled by its share (1 - w) of the mixture."""
    return (1 - w) * stats.beta.pdf((np.asarray(t) - LO) / SPAN, a, b) / SPAN


def mixture_quantile(q, a, b, w=SPIKE_WEIGHT):
    q = np.asarray(q, dtype=float)
    return np.where(q <= 1 - w, LO + SPAN * stats.beta.ppf(np.minimum(q / (1 - w), 1), a, b), SPIKE)


def mixture_sd(a, b, w=SPIKE_WEIGHT):
    p = a / (a + b)
    mu, var = LO + SPAN * p, SPAN ** 2 * p * (1 - p) / (a + b + 1)
    m1 = (1 - w) * mu + w * SPIKE
    m2 = (1 - w) * (var + mu ** 2) + w * SPIKE ** 2
    return float(np.sqrt(m2 - m1 ** 2))


def ks_tick(sorted_iti, a, b, w=SPIKE_WEIGHT):
    """Max CDF gap at the 10 ms tick midpoints -- where PRIME can and cannot fire."""
    fn = np.searchsorted(sorted_iti, TICKS, side="right") / len(sorted_iti)
    return float(np.max(np.abs(fn - mixture_cdf(TICKS, a, b, w))))


def w1(sorted_iti, a, b, w=SPIKE_WEIGHT):
    """Wasserstein-1 = mean |quantile difference| between model and session, in s."""
    fn = np.searchsorted(sorted_iti, GRID, side="right") / len(sorted_iti)
    return float(np.trapezoid(np.abs(fn - mixture_cdf(GRID, a, b, w)), GRID))


def draw_through_sample_iti(a, b, w=SPIKE_WEIGHT, n=N_DRAWS, seed=SEED):
    """Draw with the decider's own sample_iti, parameters swapped in for the call only."""
    saved = iti._ALPHA, iti._BETA, iti.SPIKE_WEIGHT
    iti._ALPHA, iti._BETA, iti.SPIKE_WEIGHT = a, b, w
    try:
        rng = np.random.default_rng(seed)
        return np.array([iti.sample_iti(rng) for _ in range(n)])
    finally:
        iti._ALPHA, iti._BETA, iti.SPIKE_WEIGHT = saved


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


# ---------------------------------------------------------------------------
# Recorded PRIME ITIs
# ---------------------------------------------------------------------------
@dataclass
class Session:
    name: str
    target_wait: float
    iti: np.ndarray          # sorted, forced pinned to SPIKE
    forced: np.ndarray       # bool, aligned with iti
    active_mean: float
    qc_blocked_mean: float

    @property
    def n(self):
        return len(self.iti)

    @property
    def n_forced(self):
        return int(self.forced.sum())


def load_session(name, target_wait):
    num = name[-2:]                                       # "pilot006" -> "06" -> folder "106"
    path = LOOP / "BIDS" / f"sub-{name}" / "ses-prime" / f"1{num}" / "prime" / "trials_intervention.csv"
    d = pd.read_csv(path)
    d = d[d["condition"].isin(PRIME_CONDITIONS)]
    forced = d["is_forced"].astype(str).str.lower().eq("true").to_numpy()
    raw = (d["pulse_time"] - d["trial_start_time"]).to_numpy(float)
    itis = np.where(forced, SPIKE, np.clip(raw, LO, HI))
    order = np.argsort(itis, kind="stable")
    qc_blocked = d["qc_failures"].to_numpy(float) * PERIODIC_INTERVAL
    active = np.maximum(d["search_time"].to_numpy(float) - qc_blocked, 0)
    return Session(name, target_wait, itis[order], forced[order], active.mean(), qc_blocked.mean())


def moments_concentration(session):
    """Concentration whose Beta spread matches the session's non-forced ITIs (method of moments)."""
    x = (session.iti[~session.forced] - LO) / SPAN
    m, v = x.mean(), x.var(ddof=1)
    return m * (1 - m) / v - 1


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def style(ax, ygrid=True):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_LINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8.5, length=3, width=0.8, color=AXIS_LINE)
    if ygrid:
        ax.grid(axis="y", color=GRID_LINE, linewidth=0.6)
    ax.set_axisbelow(True)


def iti_axis(ax):
    ax.set_xlim(LO - 0.05, SPIKE + 0.1)
    ax.xaxis.set_major_locator(FixedLocator(np.arange(LO, HI + 1e-9, 0.5)))


def plot_densities(ref, rows):
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.6), sharex=True, sharey=True, facecolor=SURFACE)
    bins = np.arange(LO, HI + 1e-9, 0.05)
    nonforced = ref.iti[~ref.forced]
    t = np.linspace(LO + 1e-5, HI - 1e-5, 3000)
    ymax = 3.2
    for k, (ax, c) in enumerate(zip(axes.flat, PANEL_CONCENTRATIONS)):
        style(ax)
        ax.hist(nonforced, bins=bins, weights=np.full(len(nonforced), 1 / (ref.n * 0.05)),
                color=DATA_FILL, edgecolor=SURFACE, linewidth=0.8)
        r = rows[(TARGET_MEAN, c)]
        ax.plot(t, np.minimum(continuous_pdf(t, r["alpha"], r["beta"]), ymax * 1.5),
                color=MODEL, lw=1.8, solid_capstyle="round", solid_joinstyle="round")
        ax.vlines(SPIKE, 0, SPIKE_WEIGHT / 0.05, color=MODEL, lw=1.8)
        ax.plot([SPIKE], [SPIKE_WEIGHT / 0.05], "o", color=MODEL, ms=6, mec=SURFACE, mew=1.5)
        tag = "  (current)" if np.isclose(c, iti.CONCENTRATION) else ""
        ax.set_title(f"c = {c:g}{tag}", loc="left", fontsize=10.5, color=INK,
                     fontweight="bold", pad=17)
        ax.text(0, 1.035, f"α {r['alpha']:.2f} · β {r['beta']:.2f} · SD {r['sd_s']:.2f} s · "
                          f"{r['frac_below_2p6']:.0%} < 2.6 s",
                transform=ax.transAxes, fontsize=8, color=INK2)
        if r["alpha"] < 1:
            ax.text(LO + 0.07, ymax * 0.955, "↑ ∞ at 2.5 s (α < 1)", fontsize=7.5, color=MUTED, va="top")
        if k == 0:
            ax.annotate(f"{SPIKE_WEIGHT:.0%} spike at {SPIKE:g} s\n({REFERENCE} forced: {ref.n_forced / ref.n:.1%})",
                        xy=(SPIKE - 0.01, SPIKE_WEIGHT / 0.05 + 0.1),
                        xytext=(SPIKE - 0.05, 1.25), fontsize=7.5, color=INK2, ha="right",
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
        iti_axis(ax)
        ax.set_ylim(0, ymax)
    for ax in axes[1]:
        ax.set_xlabel("ITI (s)", fontsize=9, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("density (1/s)", fontsize=9, color=INK2)

    fig.suptitle(f"Predetermined-trial ITI at TARGET_MEAN = {TARGET_MEAN:.1f} s, swept over CONCENTRATION",
                 x=0.012, ha="left", fontsize=13, color=INK, fontweight="bold", y=0.995)
    fig.text(0.012, 0.935,
             f"Blue: Beta part on [{LO}, {HI}] s plus a {SPIKE_WEIGHT:.1%} spike at {SPIKE} s "
             f"(stem height = its mass per 50-ms bin).   Grey: {REFERENCE} non-forced PRIME ITIs of n = {ref.n} "
             f"(SD {ref.iti.std(ddof=1):.2f} s · {np.mean(ref.iti < 2.6):.0%} < 2.6 s · "
             f"{ref.n_forced / ref.n:.1%} forced), TARGET_WAIT {ref.target_wait:.2f}.",
             fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.2, w_pad=1.0)
    return fig


def plot_fit(ref, sessions, rows, ks_curves, best):
    fig = plt.figure(figsize=(15.5, 5.3), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.1, 0.9], wspace=0.3,
                          left=0.05, right=0.985, top=0.86, bottom=0.25)

    # (a) CDFs at TARGET_MEAN against the session
    ax = fig.add_subplot(gs[0])
    style(ax)
    ax.grid(axis="x", color=GRID_LINE, linewidth=0.6)
    t = np.r_[np.linspace(LO - 0.05, SPIKE - 1e-9, 4000), SPIKE, SPIKE + 0.1]
    for c, color in zip(CDF_CONCENTRATIONS, RAMP):
        r = rows[(TARGET_MEAN, c)]
        tag = " (current)" if np.isclose(c, iti.CONCENTRATION) else ""
        ax.plot(t, mixture_cdf(t, r["alpha"], r["beta"]), color=color, lw=1.6,
                label=f"mean {TARGET_MEAN:.1f}, c = {c:g}{tag}")
    c_alt = best[ref.iti.mean()]
    a_alt, b_alt = solve_beta(ref.iti.mean(), c_alt)
    ax.plot(t, mixture_cdf(t, a_alt, b_alt), color=ALT, lw=1.6,
            label=f"mean {ref.iti.mean():.2f}, c = {c_alt:.1f} (best at that mean)")
    xs = np.r_[LO - 0.05, ref.iti, SPIKE + 0.1]
    ys = np.r_[0, np.arange(1, ref.n + 1) / ref.n, 1]
    ax.step(xs, ys, where="post", color=INK, lw=1.8, label=f"{REFERENCE} PRIME ITIs (n = {ref.n})")
    iti_axis(ax)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("ITI (s)", fontsize=9, color=INK2)
    ax.set_ylabel("cumulative fraction of trials", fontsize=9, color=INK2)
    ax.set_title(f"CDFs vs {REFERENCE}", loc="left", fontsize=11, color=INK, fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    order = [len(handles) - 1] + list(range(len(handles) - 1))
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower right",
              frameon=False, fontsize=8, labelcolor=INK2, handlelength=1.6)

    # (b) KS vs concentration, at TARGET_MEAN and at the session's own PRIME mean
    ax = fig.add_subplot(gs[1])
    style(ax)
    crit = 1.36 / np.sqrt(ref.n)
    ax.axhline(crit, color=MUTED, lw=0.8)
    ax.text(SWEEP[-1], crit + 0.006, f"KS 5% critical value, n = {ref.n}", fontsize=7.5,
            color=MUTED, ha="right", va="bottom")
    ax.vlines(iti.CONCENTRATION, 0, 0.37, color=MUTED, lw=0.8)
    ax.text(iti.CONCENTRATION * 1.04, 0.37, f"current c = {iti.CONCENTRATION:g}", fontsize=7.5,
            color=MUTED, va="top")
    for (m, ks), color in zip(ks_curves.items(), (MODEL, ALT)):
        i = int(np.argmin(ks))
        what = "" if np.isclose(m, TARGET_MEAN) else f" ({REFERENCE} PRIME)"
        ax.plot(SWEEP, ks, color=color, lw=1.8,
                label=f"mean {m:.2f} s{what}: best c = {SWEEP[i]:.1f}, KS {ks[i]:.3f}")
        ax.plot(SWEEP[i], ks[i], "o", color=color, ms=6, mec=SURFACE, mew=1.5, zorder=5)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 3, 5, 10, 20, 40]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["1", "2", "3", "5", "10", "20", "40"])
    ax.set_xlim(SWEEP[0], SWEEP[-1])
    ax.set_ylim(0, 0.47)
    ax.set_xlabel("CONCENTRATION (log scale)", fontsize=9, color=INK2)
    ax.set_ylabel(f"KS distance to {REFERENCE}", fontsize=9, color=INK2)
    ax.set_title("Which concentration matches PRIME", loc="left", fontsize=11, color=INK,
                 fontweight="bold")
    ax.legend(loc="upper center", frameon=False, fontsize=8, labelcolor=INK2, handlelength=1.6)

    # (c) Forced rate per session vs SPIKE_WEIGHT
    ax = fig.add_subplot(gs[2])
    style(ax)
    for i, s in enumerate(sessions):
        k, n = s.n_forced, s.n
        lo_ci, hi_ci = wilson(k, n)
        pct = 100 * k / n
        color = INK if s.name == REFERENCE else MUTED
        ax.errorbar(i, pct, yerr=[[pct - 100 * lo_ci], [100 * hi_ci - pct]], fmt="o", color=color,
                    ms=6, mec=SURFACE, mew=1.5, elinewidth=1.4, capsize=0, zorder=4)
        ax.annotate(f"{pct:.1f}%", (i, pct), xytext=(9, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2)
    ax.axhline(100 * SPIKE_WEIGHT, color=MODEL, lw=1.4, zorder=3)
    ax.text(-0.42, 100 * SPIKE_WEIGHT + 0.3, f"SPIKE_WEIGHT {SPIKE_WEIGHT:g}",
            fontsize=8, color=INK2, ha="left", va="bottom")
    ax.set_xticks(range(len(sessions)))
    ax.set_xticklabels([f"{s.name}\nwait {s.target_wait:.2f} s\nQC {s.qc_blocked_mean:.2f} s"
                        for s in sessions], fontsize=7.5)
    ax.set_xlabel("TARGET_WAIT · mean QC-blocked time per PRIME trial", fontsize=8, color=MUTED)
    ax.set_xlim(-0.5, len(sessions) - 0.5)
    ax.set_ylim(0, 14.5)
    ax.set_ylabel("forced PRIME trials (%)", fontsize=9, color=INK2)
    ax.set_title("Point mass: forced rate (95% CI)", loc="left", fontsize=11, color=INK,
                 fontweight="bold")

    fig.suptitle("Matching the ITI mixture to what PRIME delivered", x=0.012, ha="left", fontsize=13,
                 color=INK, fontweight="bold", y=0.985)
    fig.text(0.012, 0.035,
             f"PRIME ITI = 2.49 s to search start + active wait + QC-blocked ticks; the threshold "
             f"controller only drives the active part to TARGET_WAIT. {REFERENCE}: 2.49 + "
             f"{ref.active_mean:.2f} + {ref.qc_blocked_mean:.2f} = {ref.iti.mean():.2f} s.\n"
             f"KS taken at the 10-ms tick midpoints (PRIME fires on the periodic grid); the critical "
             f"value is a yardstick only, since c is tuned on the same trials.",
             fontsize=8, color=INK2, va="bottom")
    return fig


# ---------------------------------------------------------------------------
def main():
    a0, b0 = solve_beta(iti.TARGET_MEAN, iti.CONCENTRATION, iti.SPIKE_WEIGHT)
    assert np.isclose(a0, iti._ALPHA) and np.isclose(b0, iti._BETA), \
        "mixture algebra here no longer matches util/iti.py"

    sessions = [load_session(name, tw) for name, tw in SESSIONS.items()]
    ref = next(s for s in sessions if s.name == REFERENCE)

    print("PRIME-trial ITIs per session")
    print(f"  {'session':9s} {'wait':>5s} {'n':>4s} {'forced':>14s} {'95% CI':>13s} {'mean':>6s} "
          f"{'SD':>5s} {'active':>6s} {'QC-blk':>6s} {'c(mom)':>6s}")
    for s in sessions:
        lo_ci, hi_ci = wilson(s.n_forced, s.n)
        print(f"  {s.name:9s} {s.target_wait:5.2f} {s.n:4d} {s.n_forced:4d} = {s.n_forced / s.n:6.2%} "
              f"[{lo_ci:5.1%}, {hi_ci:5.1%}] {s.iti.mean():6.3f} {s.iti.std(ddof=1):5.3f} "
              f"{s.active_mean:6.3f} {s.qc_blocked_mean:6.3f} {moments_concentration(s):6.2f}")

    ref_mean = float(ref.iti.mean())
    ks_curves, best = {}, {}
    for m in (TARGET_MEAN, ref_mean):
        ks_curves[m] = np.array([ks_tick(ref.iti, *solve_beta(m, c)) for c in SWEEP])
        best[m] = float(SWEEP[np.argmin(ks_curves[m])])

    rows, records = {}, []
    for m in (TARGET_MEAN, ref_mean):
        for c in sorted(set(PANEL_CONCENTRATIONS) | {round(best[m], 2)}):
            a, b = solve_beta(m, c)
            r = dict(target_mean_s=round(m, 4), concentration=c, alpha=a, beta=b,
                     beta_mean_s=LO + SPAN * a / (a + b), spike_weight=SPIKE_WEIGHT,
                     sd_s=mixture_sd(a, b),
                     median_s=float(mixture_quantile(0.5, a, b)),
                     p90_s=float(mixture_quantile(0.9, a, b)),
                     frac_first_tick=float(mixture_cdf(LO + PERIODIC_INTERVAL / 2, a, b)),
                     frac_below_2p6=float(mixture_cdf(2.6, a, b)),
                     ks_tick=ks_tick(ref.iti, a, b), w1_ms=1000 * w1(ref.iti, a, b))
            draws = draw_through_sample_iti(a, b)
            r.update(draw_mean_s=draws.mean(), draw_spike_frac=np.mean(draws == SPIKE))
            assert abs(r["draw_mean_s"] - m) < 0.01, f"sample_iti mean off at c={c}"
            rows[(m, c)] = r
            records.append(r)
    records.append(dict(target_mean_s=round(ref_mean, 4), concentration=np.nan,
                        sd_s=ref.iti.std(ddof=1), median_s=float(np.median(ref.iti)),
                        p90_s=float(np.quantile(ref.iti, 0.9)),
                        frac_first_tick=float(np.mean(ref.iti < LO + PERIODIC_INTERVAL / 2)),
                        frac_below_2p6=float(np.mean(ref.iti < 2.6)),
                        spike_weight=ref.n_forced / ref.n, source=f"{REFERENCE} PRIME trials"))
    table = pd.DataFrame(records)
    table["source"] = table.get("source", pd.Series(dtype=object)).fillna("mixture")

    print(f"\nMixture vs {REFERENCE} (spike weight {SPIKE_WEIGHT:g}; draws through sample_iti, "
          f"n = {N_DRAWS:,} each)")
    print(f"  {'mean':>5s} {'c':>6s} {'alpha':>6s} {'beta':>6s} {'SD':>5s} {'1st tick':>8s} "
          f"{'<2.6s':>6s} {'KS':>6s} {'W1 ms':>6s} {'draw mean':>9s}")
    best_c = {round(m, 4): round(c, 2) for m, c in best.items()}
    for r in records[:-1]:
        flag = ""
        if np.isclose(r["concentration"], best_c[r["target_mean_s"]]):
            flag = "  <- best KS at this mean"
        elif np.isclose(r["concentration"], iti.CONCENTRATION) and np.isclose(r["target_mean_s"], TARGET_MEAN):
            flag = "  <- current c"
        print(f"  {r['target_mean_s']:5.2f} {r['concentration']:6.2f} {r['alpha']:6.3f} {r['beta']:6.3f} "
              f"{r['sd_s']:5.3f} {r['frac_first_tick']:8.1%} {r['frac_below_2p6']:6.1%} "
              f"{r['ks_tick']:6.3f} {r['w1_ms']:6.1f} {r['draw_mean_s']:9.4f}{flag}")
    e = records[-1]
    print(f"  {REFERENCE} PRIME: SD {e['sd_s']:.3f}  first tick {e['frac_first_tick']:.1%}  "
          f"<2.6 s {e['frac_below_2p6']:.1%}  forced {e['spike_weight']:.1%}")
    a_now, b_now = solve_beta(iti.TARGET_MEAN, iti.CONCENTRATION, iti.SPIKE_WEIGHT)
    print(f"  as run in {REFERENCE} (mean {iti.TARGET_MEAN}, c {iti.CONCENTRATION:g}): "
          f"KS {ks_tick(ref.iti, a_now, b_now, iti.SPIKE_WEIGHT):.3f}, "
          f"W1 {1000 * w1(ref.iti, a_now, b_now, iti.SPIKE_WEIGHT):.1f} ms")
    c_alpha1 = 1 / ((solve_beta(TARGET_MEAN, 1.0)[0]))
    print(f"  alpha crosses 1 (mode leaves ITI_MIN) at c = {c_alpha1:.2f} for mean {TARGET_MEAN}")

    out_csv = HERE / "iti_concentration_sweep_table.csv"
    table.to_csv(out_csv, index=False, float_format="%.5g")
    figs = {"densities": plot_densities(ref, rows), "fit": plot_fit(ref, sessions, rows, ks_curves, best)}
    for name, fig in figs.items():
        out = HERE / f"iti_concentration_sweep_{name}.png"
        fig.savefig(out, dpi=150, facecolor=SURFACE)
        print(f"saved {out}")
    print(f"saved {out_csv}")


if __name__ == "__main__":
    main()
