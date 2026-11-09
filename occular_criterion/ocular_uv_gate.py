"""Feasibility of an absolute (physical) gate: size of what the eye ICs remove, instead of a z-score.
Metrics per pulse, from the stored replay (`windows/sub-*/pre`, HEAD processing, eye ICs removed, mean over 38-50 ms):
  A  |Fp mean of removed activity|          (uV)   'eye IC projected back to Fp'
  B  RMS of removed activity, all channels  (uV)
  C  dipole-space size of the removed scalp map = ||pinv(L) m_removed||, as % of the session-median single-trial N45 size
     (same projection of the cleaned 38-50 ms map)   'eye IC projected back to the dipole'
Compared with the z gate [30,60] untrimmed ('30-60|all') and the gate as it runs ('now'). Singles only. Analysis only.
Evidence (frontal proxy, not true EOG): n45 = Fp / F7-F8 offset 38-50 ms > 3 robust SD (strong: 6); pre = eye activity -210..-10 ms.
CAUTION: A and B are measured on the same Fp channels as the n45 evidence -> hit / precision for A are partly circular.
Outputs offline_ocular_window/trim/tables/uv_gate_*.csv
"""
import sys, glob, pickle, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.dont_write_bytecode = True; sys.path.insert(0, '.')
import ocular_trim_lib as L, ocular_trim_review as T
C = pickle.load(open(T.OUT / 'cache.pkl', 'rb'))
Q = 4.0
data = {}
for s in L.SESSIONS:
    R = L.load_session(s); df = R['df']; ev = T.evidence(df); sg = df.single.to_numpy()
    p = L.WIN / 'windows' / f'sub-{s}' / 'pre'
    sc = pd.read_csv(p / 'scores.csv').drop_duplicates('stem').set_index('stem').loc[df.index]
    rows = {}
    for f in glob.glob(str(p / 'blocks' / '*.npz')):
        z = np.load(f, allow_pickle=True)
        for i, st in enumerate(z['stems']): rows[str(st)] = (z['map_n45_removed'][i].astype(float), z['dip_pre'][i].astype(float).mean(axis=1))
    miss = [st for st in df.index if st not in rows]
    rem = np.array([rows[st][0] for st in df.index]); fin = np.array([rows[st][1] for st in df.index])
    fz = np.load(p / 'fitting.npz', allow_pickle=True); key = [k for k in fz.files if k.startswith('L_pos') and k.endswith('z3.00')][0]
    Lp = fz[key].astype(float); P = np.linalg.pinv(Lp)                      # (3, 60)
    dip_rem = np.linalg.norm(rem @ P.T, axis=1); dip_fin = np.linalg.norm(fin @ P.T, axis=1)
    scale = np.median(dip_fin[sg])
    m = dict(A=np.abs(sc.removed_veog_n45.to_numpy(float)), B=sc.removed_rms_n45.to_numpy(float), C=100 * dip_rem / scale)
    # consistency of my rebuild with the stored column
    meta = pd.read_json(p / 'meta.json', typ='series'); chn = list(meta['ch_names']); fp = [chn.index(c) for c in ['Fp1', 'Fpz', 'Fp2']]
    chk = np.nanmax(np.abs(rem[:, fp].mean(axis=1) - sc.removed_veog_n45.to_numpy(float)))
    data[s] = dict(df=df, ev=ev, sg=sg, m=m, scale=scale, miss=len(miss), chk=chk, sc=sc,
                   z_now=C[s]['D']['now'], z3060=C[s]['D']['30-60|all'])
    print(f"{s}: pulses without stored maps {len(miss)}; rebuilt Fp-mean vs stored column max |diff| {chk:.2e} uV; N45 scale (median |q| of cleaned map) {scale:.3g}", flush=True)

# ---------- 1 distributions (singles) and cross-session comparability
rows = []
for s, d in data.items():
    cl = d['ev']['clean']
    for k, v in d['m'].items():
        for nm, sel in (('all singles', d['sg']), ('clean singles', cl)):
            rows.append(dict(session=s, metric=k, set=nm, n=int(sel.sum()), median=np.median(v[sel]), p90=np.percentile(v[sel], 90), p95=np.percentile(v[sel], 95), p97=np.percentile(v[sel], 97)))
dist = pd.DataFrame(rows); dist.to_csv(T.TAB / 'uv_gate_distribution.csv', index=False)

# ---------- 2 cutoff needed for Q % per session, spread across sessions; rejection under one global cutoff
rows = []
def z_curve(des, s):  # rejection % of singles per Z_FINE index
    return np.array([(C[s]['D'][des][i] & C[s]['single']).sum() / C[s]['single'].sum() * 100 for i in range(len(T.Z_FINE))])
for name in ['A', 'B', 'C', 'z now', 'z 30-60 all']:
    cut, rate = {}, {}
    for s, d in data.items():
        if name in d['m']: v = d['m'][name][d['sg']]; cut[s] = np.percentile(v, 100 - Q)
        else:
            des = 'now' if name == 'z now' else '30-60|all'; cr = z_curve(des, s); cut[s] = T.Z_FINE[int(np.argmin(np.abs(cr - Q)))]
    # global cutoff: the one whose mean rejection over sessions is Q %
    if name in data['pilot003']['m']:
        grid = np.unique(np.concatenate([d['m'][name][d['sg']] for d in data.values()])); grid = grid[::max(1, len(grid) // 4000)]
        mean_rate = np.array([np.mean([100 * (d['m'][name][d['sg']] > g).mean() for d in data.values()]) for g in grid])
        g = grid[int(np.argmin(np.abs(mean_rate - Q)))]
        rate = {s: 100 * (d['m'][name][d['sg']] > g).mean() for s, d in data.items()}
    else:
        des = 'now' if name == 'z now' else '30-60|all'
        mr = np.array([np.mean([z_curve(des, s)[i] for s in data]) for i in range(len(T.Z_FINE))]); i0 = int(np.argmin(np.abs(mr - Q))); g = T.Z_FINE[i0]
        rate = {s: z_curve(des, s)[i0] for s in data}
    c = np.array(list(cut.values()))
    rows.append(dict(gate=name, **{f'cut_{s}': cut[s] for s in data}, cut_max_over_min=c.max() / c.min(), global_cut=g, **{f'rate_at_global_{s}': rate[s] for s in data},
                     rate_min=min(rate.values()), rate_max=max(rate.values())))
cons = pd.DataFrame(rows); cons.to_csv(T.TAB / 'uv_gate_cross_session.csv', index=False)

# ---------- 3 fixed physical cutoffs: what share of singles is rejected in each session
rows = []
for name, cuts in (('A', [1, 2, 3, 5, 10]), ('B', [0.5, 1, 2, 3, 5]), ('C', [10, 25, 50, 100, 200])):
    for g in cuts:
        rows.append(dict(metric=name, cutoff=g, **{s: 100 * (d['m'][name][d['sg']] > g).mean() for s, d in data.items()}))
fixed = pd.DataFrame(rows); fixed.to_csv(T.TAB / 'uv_gate_fixed_cutoffs.csv', index=False)

# ---------- 4 equal cost (Q %) per session: performance, overlap with z gate, circularity
rows = []
for s, d in data.items():
    sg, ev = d['sg'], d['ev']; zr = d['z3060'][int(np.argmin(np.abs(z_curve('30-60|all', s) - Q)))] & sg
    cands = {k: (v > np.percentile(v[sg], 100 - Q)) & sg for k, v in d['m'].items()}; cands['z 30-60 all'] = zr
    cands['z now'] = d['z_now'][int(np.argmin(np.abs(z_curve('now', s) - Q)))] & sg
    for k, r in cands.items():
        mt = T.metrics(r, ev, sg); mt.update(session=s, gate=k, overlap_with_z3060=(r & zr).sum() / max(1, (r | zr).sum()) * 100, n_rej=int(r.sum())); rows.append(mt)
perf = pd.DataFrame(rows); perf.to_csv(T.TAB / 'uv_gate_equal_cost.csv', index=False)

# ---------- 5 circularity / redundancy: rank correlations among the metrics and with the frontal evidence value
rows = []
for s, d in data.items():
    sg = d['sg']; df = d['df']; zs = np.max(np.array([d['sc'].z_eye1, d['sc'].z_eye2]), axis=0) if 'z_eye2' in d['sc'] else d['sc'].z_eye1
    rows.append(dict(session=s, A_vs_frontal_offset=spearmanr(d['m']['A'][sg], np.abs(df.veog_mean_n45.to_numpy(float))[sg])[0], C_vs_frontal_offset=spearmanr(d['m']['C'][sg], np.abs(df.veog_mean_n45.to_numpy(float))[sg])[0],
                     A_vs_C=spearmanr(d['m']['A'][sg], d['m']['C'][sg])[0], C_vs_B=spearmanr(d['m']['C'][sg], d['m']['B'][sg])[0],
                     C_vs_zgate_score=spearmanr(d['m']['C'][sg], np.nan_to_num(zs)[sg])[0]))
red = pd.DataFrame(rows); red.to_csv(T.TAB / 'uv_gate_redundancy.csv', index=False)

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print('\n=== A  distribution (clean singles vs all singles); units: A,B uV; C % of N45 ==='); print(dist[dist.metric.isin(['A', 'C'])].pivot_table(index=['metric', 'set'], columns='session', values=['median', 'p95']).round(1).to_string())
print('\n=== cutoff needed for 4 % per session, and rejection rate under ONE global cutoff ==='); print(cons.round(2).T.to_string())
print('\n=== rejection % of singles at FIXED physical cutoffs ==='); print(fixed.round(1).to_string(index=False))
print('\n=== equal cost 4 %: mean over sessions ===')
print(perf.groupby('gate')[['hit_n45', 'precision_n45', 'hit_n45_strong', 'n_kept_strong_n45', 'clean_rejected_pct', 'precision_pre', 'overlap_with_z3060']].mean().round(1).to_string())
print('\n=== rank correlations (singles) ==='); print(red.round(2).to_string(index=False))
