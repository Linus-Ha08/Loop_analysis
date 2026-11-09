"""[30,60] vs [30,100] gate window, same machinery as ocular_trim_review.py (analysis only; `prime` untouched).
Score = max over eye ICs of the median z over the window; reference = mean/SD of |s| over the same window of the
125 calibration trials (untrimmed 'all', or iteratively trimmed). Singles only. Evidence (frontal proxy, NOT true EOG):
n45 = Fp mean / F7-F8 over 38-50 ms > 3 robust SD (strong: 6). Equal cost = per-session cutoff set to reject q % of singles.
Also: persistence strata for the n45-evidence pulses (is the frontal offset still there 60-100 ms?).
Outputs offline_ocular_window/trim/tables/w100_*.csv
"""
import sys, numpy as np, pandas as pd
sys.dont_write_bytecode = True; sys.path.insert(0, '.')
import ocular_trim_lib as L, ocular_trim_review as T
T.WINDOWS.clear(); T.WINDOWS.update({"30-60": (30, 60), "30-100": (30, 100), "15-100": (15, 100)})
DES = ["now", "30-60|all", "30-100|all", "15-100|all", "30-60|trim", "30-100|trim"]
MATCH = [2, 3, 4, 5, 6, 8]; NB = 500
data = {}
for s in L.SESSIONS:
    R = L.load_session(s); ev = T.evidence(R["df"]); sg = R["df"].single.to_numpy()
    D = {"now": R["df"].z_now.to_numpy(float)[None] > T.Z_FINE[:, None]}
    for d in DES[1:]:
        W, m = d.split("|"); D[d], _ = T.design_matrix(R, W, m)
    # frontal-proxy persistence (needs veog/heog at 1 kHz from the replay blocks)
    vz, hz = [], []
    for b in L.BLOCKS:
        f = L.WIN / "replay" / f"sub-{s}" / "blocks" / f"{b}.npz"
        if f.is_file():
            z = np.load(f, allow_pickle=True); vz.append(np.asarray(z["veog"], float)); hz.append(np.asarray(z["heog"], float))
    data[s] = dict(R=R, ev=ev, D=D, single=sg, veog=np.concatenate(vz), heog=np.concatenate(hz))
    print(s, "veog", data[s]["veog"].shape, "pulses", len(sg), flush=True)

def at_q(d, des, q, sel=None):
    sel = d["single"] if sel is None else sel
    rej = d["D"][des][:, sel].mean(axis=1) * 100
    return int(np.argmin(np.abs(rej - q)))

# ---- equal cost per session
rows = []
for s, d in data.items():
    for des in DES:
        for q in MATCH:
            iz = at_q(d, des, q); m = T.metrics(d["D"][des][iz], d["ev"], d["single"]); m.update(session=s, design=des, q=q, z_used=T.Z_FINE[iz]); rows.append(m)
mt = pd.DataFrame(rows); mt.to_csv(T.TAB / "w100_matched.csv", index=False)

# ---- pooled + paired bootstrap, difference vs 30-60|all
rng = np.random.default_rng(1); cols = ["precision_n45", "hit_n45", "hit_n45_strong", "clean_rejected_pct"]; boot = {(d, q): [] for d in DES for q in MATCH}
for b in range(NB):
    idx = {s: rng.integers(0, int(d["single"].sum()), int(d["single"].sum())) for s, d in data.items()}
    for des in DES:
        for q in MATCH:
            vals = []
            for s, d in data.items():
                sel = np.flatnonzero(d["single"])[idx[s]]; Dsub = d["D"][des][:, sel]
                r = Dsub[int(np.argmin(np.abs(Dsub.mean(axis=1) * 100 - q)))]; n45, st, cl = d["ev"]["n45"][sel], d["ev"]["n45_strong"][sel], d["ev"]["clean"][sel]
                vals.append([100 * (r & n45).sum() / max(1, r.sum()), 100 * (r & n45).sum() / max(1, n45.sum()), 100 * (r & st).sum() / max(1, st.sum()), 100 * (r & cl).sum() / max(1, cl.sum())])
            boot[(des, q)].append(np.mean(vals, axis=0))
rows = []
for des in DES:
    for q in MATCH:
        B = np.array(boot[(des, q)]); Bd = B - np.array(boot[("30-60|all", q)])
        for j, c in enumerate(cols):
            rows.append(dict(design=des, q=q, metric=c, mean_over_sessions=float(mt[(mt.design == des) & (mt.q == q)][c].mean()), ci_lo=float(np.percentile(B[:, j], 2.5)), ci_hi=float(np.percentile(B[:, j], 97.5)),
                             diff_vs_30_60_all=float(Bd[:, j].mean()), diff_lo=float(np.percentile(Bd[:, j], 2.5)), diff_hi=float(np.percentile(Bd[:, j], 97.5))))
pp = pd.DataFrame(rows); pp.to_csv(T.TAB / "w100_matched_pooled.csv", index=False)

# ---- persistence strata: n45-evidence singles split by whether the frontal offset persists over 60-100 ms
t = L.pulse_t_ms(); m38, m60 = L.win_mask(t, 38, 50), L.win_mask(t, 60, 100); rows = []
for s, d in data.items():
    sg = d["single"]; ev = d["ev"]["n45"]
    def zz(x):
        v = x[:, m60].mean(axis=1); return (v - np.median(v[sg])) / L.robust_sd(v[sg])
    persist = ((np.abs(zz(d["veog"])) > 3) | (np.abs(zz(d["heog"])) > 3)) & ev      # offset still >3 SD at 60-100 ms
    brief = ev & ~persist
    for des in ("30-60|all", "30-100|all"):
        iz = at_q(d, des, 4); r = d["D"][des][iz]
        rows.append(dict(session=s, design=des, n_persistent=int(persist.sum()), caught_persistent_pct=100 * (r & persist).sum() / max(1, persist.sum()),
                         n_brief=int(brief.sum()), caught_brief_pct=100 * (r & brief).sum() / max(1, brief.sum()),
                         late_only_pct_of_singles=np.nan))
    # pulses with a late (60-100) offset but NO 38-50 offset: what each window does with them
    late_only = sg & ~ev & ((np.abs(zz(d["veog"])) > 3) | (np.abs(zz(d["heog"])) > 3))
    for des in ("30-60|all", "30-100|all"):
        iz = at_q(d, des, 4); r = d["D"][des][iz]
        rows.append(dict(session=s, design=des + " [late-only offset]", n_persistent=int(late_only.sum()), caught_persistent_pct=100 * (r & late_only).sum() / max(1, late_only.sum())))
st = pd.DataFrame(rows); st.to_csv(T.TAB / "w100_persistence.csv", index=False)

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
print("\n=== equal cost, mean over sessions (q = % of singles rejected) ===")
g = mt.groupby(["q", "design"])[["hit_n45", "precision_n45", "hit_n45_strong", "n_kept_strong_n45", "clean_rejected_pct"]].mean().round(1)
print(g.unstack("design").reindex(columns=pd.MultiIndex.from_product([["hit_n45", "precision_n45", "hit_n45_strong", "n_kept_strong_n45", "clean_rejected_pct"], DES])).loc[[3, 4, 5]].T.to_string())
print("\n=== paired bootstrap: 30-100|all minus 30-60|all (95 % CI) ===")
print(pp[(pp.design == "30-100|all")][["q", "metric", "mean_over_sessions", "diff_vs_30_60_all", "diff_lo", "diff_hi"]].round(1).to_string(index=False))
print("\n=== persistence strata at q=4 ===")
print(st.round(1).to_string(index=False))
