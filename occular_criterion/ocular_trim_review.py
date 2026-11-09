# -*- coding: utf-8 -*-
"""ocular_trim_review.py - window reference [20,60] / [30,60] / [30,50], untrimmed vs trimmed, singles only.

Question (2026-10-27): keep the gate as it is (median of z over a window, z of |s| against mean / SD of |s|), but take the
window, for the score AND for the reference, to be [30,50] or [20,60]; with the reference as it is (all 125 calibration
trials) or trimmed (calibration trials scoring above the cutoff are dropped and the statistics refitted until stable).
Triplets are not considered. pilot003-006.

Inputs (read-only): Loop/offline_ocular_window/{windows,replay}/sub-<ID> (stored by ocular_window_review.py).
Outputs: Loop/offline_ocular_window/trim/{tables,figures}. Nothing is written into prime/.
Evidence (gate-independent, no ICA) = the definitions of ocular_window_review.py:
  n45: Fp1/Fpz/Fp2 mean or F7-F8 mean over 38-50 ms (baseline -15..-5 ms), > 3 robust SD from the session median of
       the single pulses (either sign); strong: > 6.   pre: raw pre-window (-210..-10 ms) VEOG/HEOG p-p > median + 3 robust SD.
  clean = single pulse with neither.
Exactness: references are exact (5 kHz calibration sources); a pulse's median |s| over a window comes from its 1 kHz
sources (every 5th sample). Validated against the stored exact 20-60 / 30-60 scores (see tables/validation.csv).
"""
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import ocular_trim_lib as L

OUT = L.WIN / "trim"
TAB = OUT / "tables"
TAB.mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)

WINDOWS = {"20-60": (20, 60), "30-60": (30, 60), "30-50": (30, 50)}
MODES = ["all", "trim"]
DESIGNS = ["now"] + [f"{w}|{m}" for w in WINDOWS for m in MODES]
Z_FINE = np.round(np.arange(1.0, 6.0001, 0.05), 2)
Z_TAB = [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]
MATCH = [2, 3, 4, 5, 6, 8]
N_BOOT = 500
MIN_KEEP = 60


def evidence(df):
    single = df.single.to_numpy()
    out = {}
    for k, (cv_, ch_, two) in {"n45": ("veog_mean_n45", "heog_mean_n45", True),
                               "pre": ("veog_pre_ptp", "heog_pre_ptp", False)}.items():
        v, h = df[cv_].to_numpy(float), df[ch_].to_numpy(float)
        cv, ch = np.median(v[single]), np.median(h[single])
        sv, sh = L.robust_sd(v[single]), L.robust_sd(h[single])
        for tag, kk in (("", L.EVIDENCE_K), ("_strong", L.STRONG_K)):
            f = ((np.abs(v - cv) > kk * sv) | (np.abs(h - ch) > kk * sh)) if two else ((v > cv + kk * sv) | (h > ch + kk * sh))
            out[f"{k}{tag}"] = f & single
    out["clean"] = single & ~out["n45"] & ~out["pre"]
    return out


def trimmed(calabs, mask, z):
    keep = np.ones(len(calabs), bool)
    for _ in range(20):
        mean, sd = L.reference(calabs, mask, keep)
        sc = L.cal_scores(calabs, mask, mean, sd)
        new = sc <= z
        if new.sum() < MIN_KEEP or (new == keep).all():
            break
        keep = new
    mean, sd = L.reference(calabs, mask, keep)
    return mean, sd, keep


def design_matrix(R, W, mode):
    """rejection decisions (cutoff x pulse) and per-cutoff calibration info"""
    a, b = WINDOWS[W]
    m1 = L.win_mask(L.pulse_t_ms(), a, b)
    med = np.median(np.abs(R["src"][:, :, m1]), axis=2)
    mask5 = L.win_mask(R["cal_t"], a, b)
    calabs = np.abs(R["cal"])
    ref_all = L.reference(calabs, mask5)
    D = np.zeros((len(Z_FINE), len(med)), bool)
    info = []
    for i, z in enumerate(Z_FINE):
        if mode == "all":
            mean, sd = ref_all
            keep = np.ones(len(calabs), bool)
        else:
            mean, sd, keep = trimmed(calabs, mask5, z)
        score = ((med - mean[None]) / sd[None]).max(axis=1)
        D[i] = score > z
        csc = L.cal_scores(calabs, mask5, mean, sd)
        info.append(dict(z=z, cal_dropped=int((csc > z).sum()), cal_in_ref=int(keep.sum()), mean=mean, sd=sd))
    return D, info


def metrics(rej, ev, sel):
    """rej: decisions; ev: evidence dict; sel: boolean selector of the pulses counted (singles)"""
    r = rej & sel
    n = int(sel.sum())
    nr = int(r.sum())
    n45 = ev["n45"] & sel
    strong = ev["n45_strong"] & sel
    clean = ev["clean"] & sel
    return dict(pct_rejected=100 * nr / n, n_rejected=nr,
                precision_n45=100 * (r & n45).sum() / nr if nr else np.nan,
                hit_n45=100 * (r & n45).sum() / max(1, n45.sum()),
                hit_n45_strong=100 * (r & strong).sum() / max(1, strong.sum()),
                n_kept_strong_n45=int((~r & strong).sum()),
                clean_rejected_pct=100 * (r & clean).sum() / max(1, clean.sum()),
                precision_pre=100 * (r & ev["pre"] & sel).sum() / nr if nr else np.nan)


def main():
    data = {}
    val_rows, ref_rows = [], []
    ev_chk = pd.read_csv(L.WIN / "windows" / "tables" / "evidence.csv")
    for s in L.SESSIONS:
        R = L.load_session(s)
        ev = evidence(R["df"])
        single = R["df"].single.to_numpy()
        e0 = ev_chk[(ev_chk.subject == s) & (ev_chk.evidence == "n45")].iloc[0]
        val_rows.append(dict(session=s, check="n45 evidence % of singles (this vs stored)", this=100 * ev["n45"][single].mean(),
                             stored=e0.pct_singles))
        val_rows.append(dict(session=s, check="n45 strong % of singles (this vs stored)", this=100 * ev["n45_strong"][single].mean(),
                             stored=e0.pct_singles_strong))
        D = {"now": (R["df"].z_now.to_numpy(float)[None] > Z_FINE[:, None]), }
        info = {"now": None}
        for W in WINDOWS:
            for m in MODES:
                D[f"{W}|{m}"], info[f"{W}|{m}"] = design_matrix(R, W, m)
        # validation of the 1 kHz pulse scores against the stored exact ones
        for W in ("20-60", "30-60"):
            a, b = WINDOWS[W]
            mask5 = L.win_mask(R["cal_t"], a, b)
            mean, sd = L.reference(np.abs(R["cal"]), mask5)
            z = ((np.median(np.abs(R["src"][:, :, L.win_mask(L.pulse_t_ms(), a, b)]), axis=2) - mean[None]) / sd[None]).max(axis=1)
            w = R["ws"][R["ws"].window == W].drop_duplicates("stem").set_index("stem").loc[R["df"].index]
            zs = w.z_max.to_numpy(float)
            for zc in (2.0, 2.5, 3.0):
                val_rows.append(dict(session=s, check=f"{W}: pulses decided differently at z {zc} (of 1425, 1 kHz vs exact 5 kHz)",
                                     this=int(((z > zc) != (zs > zc)).sum()), stored=0))
        # reference table at z 3.0
        for W in WINDOWS:
            for m in MODES:
                iz = int(np.argmin(np.abs(Z_FINE - 3.0)))
                inf = info[f"{W}|{m}"][iz]
                for j, ic in enumerate(R["eye_ics"]):
                    ref_rows.append(dict(session=s, window=W, mode=m, ic=ic, mean=inf["mean"][j], sd=inf["sd"][j],
                                         cal_in_ref=inf["cal_in_ref"]))
        # decay profile of the eye ICs in the calibration trials (5 kHz), median |s|
        calabs = np.abs(R["cal"])
        for j, ic in enumerate(R["eye_ics"]):
            for (a, b) in [(15, 20), (20, 30), (30, 50), (50, 60), (60, 100)]:
                mk = L.win_mask(R["cal_t"], a, b)
                ref_rows.append(dict(session=s, window=f"profile {a}-{b}", mode="", ic=ic,
                                     mean=float(np.median(calabs[:, j, :][:, mk])), sd=np.nan, cal_in_ref=np.nan))
        data[s] = dict(R=R, ev=ev, D=D, info=info, single=single)
    pd.DataFrame(val_rows).to_csv(TAB / "validation.csv", index=False)
    pd.DataFrame(ref_rows).to_csv(TAB / "reference.csv", index=False)

    # ---------- fixed cutoffs
    rows = []
    for s, d in data.items():
        for des in DESIGNS:
            for z in Z_TAB:
                iz = int(np.argmin(np.abs(Z_FINE - z)))
                m = metrics(d["D"][des][iz], d["ev"], d["single"])
                inf = d["info"][des]
                m.update(session=s, design=des, z=z,
                         cal_dropped=np.nan if inf is None else inf[iz]["cal_dropped"],
                         cal_in_ref=np.nan if inf is None else inf[iz]["cal_in_ref"])
                # sensitivity: out-of-sample singles (calibration block excluded)
                oos = d["single"] & (d["R"]["df"].block.to_numpy() != "calibration")
                m["pct_rejected_oos"] = metrics(d["D"][des][iz], d["ev"], oos)["pct_rejected"]
                rows.append(m)
    fx = pd.DataFrame(rows)
    fx.to_csv(TAB / "fixed_cutoff.csv", index=False)

    # ---------- equal cost (per session cutoff set to reject q % of singles)
    rows = []
    for s, d in data.items():
        for des in DESIGNS:
            rej = np.array([(d["D"][des][i] & d["single"]).sum() / d["single"].sum() * 100 for i in range(len(Z_FINE))])
            for q in MATCH:
                iz = int(np.argmin(np.abs(rej - q)))
                m = metrics(d["D"][des][iz], d["ev"], d["single"])
                m.update(session=s, design=des, q=q, z_used=Z_FINE[iz])
                rows.append(m)
    mt = pd.DataFrame(rows)
    mt.to_csv(TAB / "matched.csv", index=False)

    # ---------- pooled equal cost with paired bootstrap (mean over sessions)
    rng = np.random.default_rng(0)
    cols = ["precision_n45", "hit_n45", "clean_rejected_pct"]
    boot = {(des, q): [] for des in DESIGNS for q in MATCH}
    for b in range(N_BOOT):
        idx = {s: rng.integers(0, int(data[s]["single"].sum()), int(data[s]["single"].sum())) for s in data}
        for des in DESIGNS:
            for q in MATCH:
                vals = []
                for s, d in data.items():
                    sel = np.flatnonzero(d["single"])[idx[s]]
                    Dsub = d["D"][des][:, sel]
                    rej = Dsub.mean(axis=1) * 100
                    r = Dsub[int(np.argmin(np.abs(rej - q)))]
                    n45, clean = d["ev"]["n45"][sel], d["ev"]["clean"][sel]
                    vals.append([100 * (r & n45).sum() / max(1, r.sum()), 100 * (r & n45).sum() / max(1, n45.sum()),
                                 100 * (r & clean).sum() / max(1, clean.sum())])
                boot[(des, q)].append(np.mean(vals, axis=0))
    rows = []
    ref_des = "20-60|all"
    for des in DESIGNS:
        for q in MATCH:
            B = np.array(boot[(des, q)])
            Bd = B - np.array(boot[(ref_des, q)])
            for j, c in enumerate(cols):
                rows.append(dict(design=des, q=q, metric=c, mean_over_sessions=float(mt[(mt.design == des) & (mt.q == q)][c].mean()),
                                 ci_lo=float(np.percentile(B[:, j], 2.5)), ci_hi=float(np.percentile(B[:, j], 97.5)),
                                 diff_vs_20_60_all=float(np.mean(Bd[:, j])), diff_lo=float(np.percentile(Bd[:, j], 2.5)),
                                 diff_hi=float(np.percentile(Bd[:, j], 97.5))))
    pd.DataFrame(rows).to_csv(TAB / "matched_pooled.csv", index=False)

    # ---------- per block rejection at the matched 4 % cutoff (session-wise z)
    rows = []
    for s, d in data.items():
        df = d["R"]["df"]
        for des in DESIGNS:
            z_used = float(mt[(mt.session == s) & (mt.design == des) & (mt.q == 4)].z_used.iloc[0])
            iz = int(np.argmin(np.abs(Z_FINE - z_used)))
            for blk in L.BLOCKS:
                sel = d["single"] & (df.block.to_numpy() == blk)
                if sel.sum():
                    rows.append(dict(session=s, design=des, block=blk, z_used=z_used, n=int(sel.sum()),
                                     pct_rejected=100 * (d["D"][des][iz] & sel).sum() / sel.sum()))
    pd.DataFrame(rows).to_csv(TAB / "blocks_q4.csv", index=False)
    import pickle
    pickle.dump({s: dict(D=d["D"], ev=d["ev"], single=d["single"], info={k: (None if v is None else [{kk: vv for kk, vv in x.items()} for x in v]) for k, v in d["info"].items()},
                         block=d["R"]["df"].block.to_numpy(), category=d["R"]["df"].category.to_numpy(), amp=d["R"]["df"].amp.to_numpy(),
                         eye_ics=d["R"]["eye_ics"]) for s, d in data.items()}, open(OUT / "cache.pkl", "wb"))
    print("done")


if __name__ == "__main__":
    main()
