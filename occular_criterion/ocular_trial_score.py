"""How to summarise a trial's eye-IC time course into one score? median (current) vs mean / rms / max / percentiles / smoothed peak / excess area / run length /
signed-source mean ('DC'). Same z(t) as the gate: per eye IC, z(t) = (|s(t)| - mean) / SD with the window reference of the 125 calibration trials (untrimmed, 30-60 ms).
Window = 30-60 ms (gate) or 38-50 ms (N45). Scores combined over eye ICs by max. Pulses from the stored 1 kHz sources. Singles only. Analysis only.
Labels (frontal proxy, NOT true EOG): n45 = Fp / F7-F8 offset 38-50 ms > 3 robust SD; sustained = n45 and still > 3 SD at 60-100 ms (blink-like);
brief = n45 and not sustained; pre-only = eye activity before the pulse but no n45 offset; clean = neither n45 nor pre.
Consequence: C = dipole-space size of what the eye ICs remove in 38-50 ms (as in ocular_uv_gate.py). Outputs tables/score_*.csv
"""
import sys, glob, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from numpy.lib.stride_tricks import sliding_window_view as swv
sys.dont_write_bytecode = True; sys.path.insert(0, '.')
import ocular_trim_lib as L, ocular_trim_review as T
WINS = {'30-60': (30, 60), '38-50': (38, 50)}; QS = (3, 4, 6)
def runlen(B):
    out = np.zeros(B.shape[:2], int); cur = np.zeros(B.shape[:2], int)
    for i in range(B.shape[2]): cur = np.where(B[..., i], cur + 1, 0); out = np.maximum(out, cur)
    return out
SC = {  # name -> f(Z (n,k,w) z of |s|, Zs (n,k,w) signed s / SD)
 'median (current)': lambda Z, Zs: np.median(Z, axis=2),
 'mean': lambda Z, Zs: Z.mean(axis=2),
 'rms of positive z': lambda Z, Zs: np.sqrt((np.maximum(Z, 0) ** 2).mean(axis=2)),
 '75th percentile': lambda Z, Zs: np.percentile(Z, 75, axis=2),
 '90th percentile': lambda Z, Zs: np.percentile(Z, 90, axis=2),
 'max (single sample)': lambda Z, Zs: Z.max(axis=2),
 'peak of 5 ms mean': lambda Z, Zs: swv(Z, 5, axis=2).mean(axis=-1).max(axis=-1),
 'excess area over z 2': lambda Z, Zs: np.maximum(Z - 2, 0).sum(axis=2),
 'longest run above z 2': lambda Z, Zs: runlen(Z > 2).astype(float),
 'DC: |mean of signed source|': lambda Z, Zs: np.abs(Zs.mean(axis=2)),
 'DC: peak |5 ms mean of signed|': lambda Z, Zs: np.abs(swv(Zs, 5, axis=2).mean(axis=-1)).max(axis=-1),
}
t = L.pulse_t_ms(); S = {}
for s in L.SESSIONS:
    R = L.load_session(s); df = R['df']; ev = T.evidence(df); sg = df.single.to_numpy()
    m5 = L.win_mask(R['cal_t'], 30, 60); ca = np.abs(R['cal']); mean, sd = L.reference(ca, m5)
    cs = R['cal'][:, :, m5]; sd_s = cs.transpose(1, 0, 2).reshape(cs.shape[1], -1).std(axis=1)
    Zall = (np.abs(R['src']) - mean[None, :, None]) / sd[None, :, None]; Zsall = R['src'] / sd_s[None, :, None]
    vz, hz = [], []
    for b in L.BLOCKS:
        f = L.WIN / 'replay' / f'sub-{s}' / 'blocks' / f'{b}.npz'
        if f.is_file(): z = np.load(f, allow_pickle=True); vz.append(np.asarray(z['veog'], float)); hz.append(np.asarray(z['heog'], float))
    ve, he = np.concatenate(vz), np.concatenate(hz); m60 = L.win_mask(t, 60, 100)
    def zz(x): v = x[:, m60].mean(axis=1); return (v - np.median(v[sg])) / L.robust_sd(v[sg])
    late = (np.abs(zz(ve)) > 3) | (np.abs(zz(he)) > 3)
    lab = dict(n45=ev['n45'], strong=ev['n45_strong'], sustained=ev['n45'] & late, brief=ev['n45'] & ~late, pre_only=ev['pre'] & ~ev['n45'], clean=ev['clean'])
    # consequence on the N45 estimate (what the eye ICs remove in 38-50 ms, in dipole space)
    p = L.WIN / 'windows' / f'sub-{s}' / 'pre'; rows = {}
    for f in glob.glob(str(p / 'blocks' / '*.npz')):
        z = np.load(f, allow_pickle=True)
        for i, st in enumerate(z['stems']): rows[str(st)] = z['map_n45_removed'][i].astype(float)
    rem = np.array([rows[st] for st in df.index]); fz = np.load(p / 'fitting.npz', allow_pickle=True)
    Lp = fz[[k for k in fz.files if k.startswith('L_pos') and k.endswith('z3.00')][0]].astype(float); Cc = np.linalg.norm(rem @ np.linalg.pinv(Lp).T, axis=1)
    S[s] = dict(Zall=Zall, Zsall=Zsall, lab=lab, sg=sg, C=Cc, ev=ev, late=late)
    print(s, 'loaded', flush=True)
def auc(x, pos, neg):
    r = rankdata(np.concatenate([x[pos], x[neg]])); n1, n0 = pos.sum(), neg.sum(); return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
rows, eq = [], []
STORE = {}
for wn, (a, b) in WINS.items():
    w = L.win_mask(t, a, b)
    for s, d in S.items():
        Z, Zs, sg = d['Zall'][:, :, w], d['Zsall'][:, :, w], d['sg']
        med_ref = np.median(Z, axis=2).max(axis=1)
        for name, f in SC.items():
            sc = f(Z, Zs).max(axis=1); sc = sc + 1e-9 * med_ref                        # tie-break
            STORE[(wn, name, s)] = sc
            r = dict(window=wn, score=name, session=s)
            for lb in ('n45', 'strong', 'sustained', 'brief', 'pre_only'): r['auc_' + lb] = auc(sc, d['lab'][lb] & sg, d['lab']['clean'] & sg)
            r['rho_consequence'] = spearmanr(sc[sg], d['C'][sg])[0]
            top = sc > np.percentile(sc[sg], 96); topC = d['C'] > np.percentile(d['C'][sg], 96); r['overlap_top4_consequence'] = 100 * (top & topC & sg).sum() / max(1, (top & sg).sum())
            rows.append(r)
            for q in QS:
                thr = np.percentile(sc[sg], 100 - q); rej = (sc > thr) & sg; m = T.metrics(rej, d['ev'], sg)
                for lb in ('sustained', 'brief'): m['caught_' + lb] = 100 * (rej & d['lab'][lb]).sum() / max(1, d['lab'][lb].sum())
                m.update(window=wn, score=name, session=s, q=q, cut=thr); eq.append(m)
A = pd.DataFrame(rows); E = pd.DataFrame(eq); A.to_csv(T.TAB / 'score_auc.csv', index=False); E.to_csv(T.TAB / 'score_equal_cost.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
order = list(SC)
for wn in WINS:
    a = A[A.window == wn].groupby('score')[['auc_n45', 'auc_strong', 'auc_sustained', 'auc_brief', 'auc_pre_only', 'rho_consequence', 'overlap_top4_consequence']].mean().reindex(order).round(3)
    e = E[(E.window == wn) & (E.q == 4)].groupby('score')[['hit_n45', 'caught_sustained', 'caught_brief', 'hit_n45_strong', 'n_kept_strong_n45', 'precision_n45', 'precision_pre', 'clean_rejected_pct']].mean().reindex(order).round(1)
    cutr = E[(E.window == wn) & (E.q == 4)].groupby('score').cut.apply(lambda x: x.max() / x.min()).reindex(order).round(2)
    print(f'\n===== window {wn} ms: ranking quality (AUC vs clean singles; mean of 4 pilots) ====='); print(a.to_string())
    print(f'\n===== window {wn} ms: equal cost, 4 % of singles rejected per pilot (mean of 4) ====='); print(e.assign(cutoff_max_over_min=cutr).to_string())
# ---- diagnostics: where is the max, and what do the eye ICs do for the brief offsets?
print('\n===== diagnostics (30-60 ms window) =====')
w = L.win_mask(t, 30, 60); tw = t[w]
for s, d in S.items():
    Z = d['Zall'][:, :, w]; sg = d['sg']; pk = Z.max(axis=1); at = tw[pk.argmax(axis=1)]
    cl = d['lab']['clean'] & sg; su = d['lab']['sustained'] & sg; br = d['lab']['brief'] & sg
    w2 = L.win_mask(t, 38, 50); pz = d['Zall'][:, :, w2].max(axis=(1, 2))
    print(f"{s}: time of the max z within 30-60 ms is <= 34 ms in {100*(at[cl]<=34).mean():.0f}% of clean, {100*(at[su]<=34).mean():.0f}% of sustained, {100*(at[br]<=34).mean():.0f}% of brief pulses | "
          f"peak z in 38-50 ms (median; % > 3): clean {np.median(pz[cl]):.1f}; {100*(pz[cl]>3).mean():.0f}% | brief {np.median(pz[br]):.1f}; {100*(pz[br]>3).mean():.0f}% | sustained {np.median(pz[su]):.1f}; {100*(pz[su]>3).mean():.0f}%  (n clean/brief/sustained = {cl.sum()}/{br.sum()}/{su.sum()})")

# ---------- paired bootstrap of the contrasts (resample single pulses within each pilot, mean over pilots)
rng = np.random.default_rng(11); NB = 500
PAIRS = [('median (current)', '30-60', 'mean', '30-60'), ('median (current)', '30-60', 'max (single sample)', '30-60'), ('median (current)', '30-60', '90th percentile', '30-60'),
         ('median (current)', '30-60', 'DC: |mean of signed source|', '30-60'), ('median (current)', '30-60', 'median (current)', '38-50'), ('median (current)', '30-60', 'mean', '38-50'),
         ('median (current)', '30-60', 'DC: |mean of signed source|', '38-50'), ('median (current)', '38-50', 'DC: |mean of signed source|', '38-50'), ('median (current)', '38-50', 'mean', '38-50')]
names = sorted({(a, b) for a, b, _, _ in PAIRS for a, b in ((a, b),)} | {(c, d) for _, _, c, d in PAIRS})
sg_idx = {s: np.flatnonzero(d['sg']) for s, d in S.items()}
def stats(sc, lab, sel, q=4):
    cl = lab['clean'][sel]; out = []
    for l in ('n45', 'sustained', 'brief'): out.append(auc(sc[sel], lab[l][sel] & True, cl) if False else None)
    return out
def one(sc, d, sel):
    """AUC n45 / sustained / brief, and at 4 % cost: caught n45, precision, strong caught"""
    x = sc[sel]; lab = {k: v[sel] for k, v in d['lab'].items()}; cl = lab['clean']; res = []
    for l in ('n45', 'sustained', 'brief'):
        pos = lab[l]; res.append(auc(x, pos, cl) if pos.sum() and cl.sum() else np.nan)
    thr = np.percentile(x, 96); rej = x > thr
    res += [100 * (rej & lab['n45']).sum() / max(1, lab['n45'].sum()), 100 * (rej & lab['n45']).sum() / max(1, rej.sum()), 100 * (rej & lab['strong']).sum() / max(1, lab['strong'].sum())]
    return np.array(res)
need = sorted({(w_, n_) for _, _, _, _ in [(0, 0, 0, 0)] for w_, n_ in []}) 
keys = sorted({(w1, n1) for n1, w1, n2, w2 in PAIRS} | {(w2, n2) for n1, w1, n2, w2 in PAIRS})
B = {k: [] for k in keys}
for b in range(NB):
    per = {k: [] for k in keys}
    for s, d in S.items():
        sel = sg_idx[s][rng.integers(0, len(sg_idx[s]), len(sg_idx[s]))]
        for k in keys: per[k].append(one(STORE[(k[0], k[1], s)], d, sel))
    for k in keys: B[k].append(np.nanmean(per[k], axis=0))
B = {k: np.array(v) for k, v in B.items()}; cols = ['AUC n45', 'AUC sustained', 'AUC brief', 'caught n45 (4 %)', 'precision (4 %)', 'strong caught (4 %)']; rows = []
for n1, w1, n2, w2 in PAIRS:
    D = B[(w2, n2)] - B[(w1, n1)]
    rows.append(dict(contrast=f'{n2} [{w2}]  minus  {n1} [{w1}]', **{c: f'{D[:, j].mean():+.3f} [{np.percentile(D[:, j], 2.5):+.3f}, {np.percentile(D[:, j], 97.5):+.3f}]' if j < 3 else f'{D[:, j].mean():+.1f} [{np.percentile(D[:, j], 2.5):+.1f}, {np.percentile(D[:, j], 97.5):+.1f}]' for j, c in enumerate(cols)}))
BT = pd.DataFrame(rows); BT.to_csv(T.TAB / 'score_bootstrap.csv', index=False)
print('\n===== paired bootstrap of the contrasts (mean over 4 pilots; 95 % CI) ====='); print(BT.to_string(index=False))
