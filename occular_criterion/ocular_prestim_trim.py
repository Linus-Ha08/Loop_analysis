"""Pre-stim window-level reference with iterative dropping, vs the references tested so far. Analysis only, `prime` untouched.
Reference: eye-IC |s| of the 125 calibration trials in the ICA window (-1100..-5 ms, 1 kHz, as stored in windows/sub-*/pre/pre_sources.npz), cut into
non-overlapping 30 ms windows (first/last 100 ms excluded: filter-edge transients), statistic per window = median |s| (same statistic as the gate).
Reference = mean / SD of the window statistics per eye IC. Score of a pulse = max over eye ICs of (median |s| over 30-60 ms post - mean) / SD; reject if > z.
Iterative dropping: score every reference window the same way (max over ICs), drop windows above the dropping cutoff, refit, repeat until stable.
  A  dropping cutoff = the gate's z (self-consistent; one reference per z)
  B  fixed dropping cutoff c (2.5 / 3 / 4), one reference for all z
Comparators: W0 window-level without dropping; S0 sample-level pre reference (the earlier 'pre30-60' design); '30-60|all' (post-pulse window reference of the
calibration trials); 'now' (the gate as it runs). Singles only; equal cost = per-pilot cutoff set to reject q % of singles. Frontal-proxy evidence, not true EOG.
Outputs offline_ocular_window/trim/tables/prestim_*.csv
"""
import sys, pickle, numpy as np, pandas as pd
sys.dont_write_bytecode = True; sys.path.insert(0, '.')
import ocular_trim_lib as L, ocular_trim_review as T
C = pickle.load(open(T.OUT / 'cache.pkl', 'rb')); Z = T.Z_FINE; QS = [3, 4, 5, 6]; MIN_KEEP = 500; NB = 500; W = 30
EDGE = int(sys.argv[1]) if len(sys.argv) > 1 else 100
TAG = '' if EDGE == 100 else f'_edge{EDGE}'
rng = np.random.default_rng(21)
from pathlib import Path
ONLY = sys.argv[2] if len(sys.argv) > 2 else None
CF = lambda s_: Path.home() / f'prestim_cache_{EDGE}_{s_}.pkl'
def win_stats(pa, edge):
    n = (pa.shape[2] - 2 * edge) // W; X = pa[:, :, edge: edge + n * W].reshape(pa.shape[0], pa.shape[1], n, W)
    return np.median(X, axis=3).transpose(0, 2, 1).reshape(-1, pa.shape[1])         # (125 n, k)
def trim(Wf, cut, max_iter=60):
    keep = np.ones(len(Wf), bool)
    for _ in range(max_iter):
        mu, sd = Wf[keep].mean(0), Wf[keep].std(0); new = ((Wf - mu) / sd).max(1) <= cut
        if new.sum() < MIN_KEEP or (new == keep).all(): break
        keep = new
    return Wf[keep].mean(0), Wf[keep].std(0), keep
DES = ['now', '30-60|all', 'S0 sample-level, no dropping', 'W0 window-level, no dropping', 'A dropping cutoff = gate z', 'B dropping cutoff 2.5', 'B dropping cutoff 3', 'B dropping cutoff 4']
S = {}; refrows = []; valrows = []; refcsv = pd.read_csv(L.WIN / 'windows' / 'tables' / 'reference.csv')
t = L.pulse_t_ms(); m60 = L.win_mask(t, 60, 100)
for s in L.SESSIONS:
    if CF(s).is_file():
        S[s], rr_, vv_ = pickle.load(open(CF(s), 'rb')); refrows += rr_; valrows += vv_; continue
    if ONLY and s != ONLY: continue
    n_r, n_v = len(refrows), len(valrows)
    R = L.load_session(s); df = R['df']; ev = T.evidence(df); sg = df.single.to_numpy()
    pre = np.load(L.WIN / 'windows' / f'sub-{s}' / 'pre' / 'pre_sources.npz', allow_pickle=True); pa = np.abs(pre['src']).astype(float)
    assert [int(i) for i in pre['eye_ics']] == list(R['eye_ics']), (pre['eye_ics'], R['eye_ics'])
    m1 = L.win_mask(t, 30, 60); post = np.median(np.abs(R['src'][:, :, m1]), axis=2); k = post.shape[1]
    vz, hz = [], []
    for b in L.BLOCKS:
        f = L.WIN / 'replay' / f'sub-{s}' / 'blocks' / f'{b}.npz'
        if f.is_file(): z_ = np.load(f, allow_pickle=True); vz.append(np.asarray(z_['veog'], float)); hz.append(np.asarray(z_['heog'], float))
    ve, he = np.concatenate(vz), np.concatenate(hz)
    def zz(x): v = x[:, m60].mean(axis=1); return (v - np.median(v[sg])) / L.robust_sd(v[sg])
    late = (np.abs(zz(ve)) > 3) | (np.abs(zz(he)) > 3)
    ev['sustained'] = ev['n45'] & late; ev['brief'] = ev['n45'] & ~late
    flat = pa.transpose(1, 0, 2).reshape(k, -1); mu_s, sd_s = flat.mean(1), flat.std(1)
    rc = refcsv[(refcsv.subject == s) & (refcsv.window == 'pre20-60')].drop_duplicates('ic').set_index('ic')
    for j, ic in enumerate(R['eye_ics']):
        if ic in rc.index: valrows.append(dict(session=s, ic=ic, mean_this=mu_s[j], mean_stored=rc.loc[ic, 'ref_mean'], sd_this=sd_s[j], sd_stored=rc.loc[ic, 'ref_sd']))
    sc_S0 = ((post - mu_s) / sd_s).max(1)
    Wf = win_stats(pa, EDGE)
    mu0, sd0 = Wf.mean(0), Wf.std(0); sc_W0 = ((post - mu0) / sd0).max(1)
    D = {'now': C[s]['D']['now'], '30-60|all': C[s]['D']['30-60|all'], 'S0 sample-level, no dropping': sc_S0[None] > Z[:, None], 'W0 window-level, no dropping': sc_W0[None] > Z[:, None]}
    DA = np.zeros((len(Z), len(post)), bool); kinfo = {}
    for i, z in enumerate(Z):
        mu, sd, keep = trim(Wf, z); DA[i] = ((post - mu) / sd).max(1) > z
        if abs(z - 3.0) < 1e-9: kinfo['A'] = (mu, sd, keep)
    D['A dropping cutoff = gate z'] = DA
    for c in (2.5, 3, 4):
        mu, sd, keep = trim(Wf, c); D[f'B dropping cutoff {c:g}'] = (((post - mu) / sd).max(1))[None] > Z[:, None]; kinfo[f'B{c:g}'] = (mu, sd, keep)
    for j, ic in enumerate(R['eye_ics']):
        r = dict(session=s, ic=ic, n_windows=len(Wf), sample_ref_mean=mu_s[j], sample_ref_sd=sd_s[j], window_ref_mean=mu0[j], window_ref_sd=sd0[j])
        for nm, (mu, sd, keep) in kinfo.items(): r[f'{nm}: dropped %'] = 100 * (1 - keep.mean()); r[f'{nm}: mean'] = mu[j]; r[f'{nm}: sd'] = sd[j]; r[f'{nm}: sd ratio'] = sd0[j] / sd[j]
        refrows.append(r)
    S[s] = dict(df=df, ev=ev, sg=sg, D=D)
    pickle.dump((S[s], refrows[n_r:], valrows[n_v:]), open(CF(s), 'wb'))
    print(s, 'done', flush=True)
if ONLY: sys.exit()
pd.DataFrame(valrows).to_csv(T.TAB / f'prestim_validation{TAG}.csv', index=False); ref = pd.DataFrame(refrows); ref.to_csv(T.TAB / f'prestim_reference{TAG}.csv', index=False)
def rate_curve(s, des): d = S[s]; return np.array([(d['D'][des][i] & d['sg']).sum() / d['sg'].sum() * 100 for i in range(len(Z))])
RC = {(s, des): rate_curve(s, des) for s in S for des in DES}
rows = []
for s, d in S.items():
    for des in DES:
        rc_ = RC[(s, des)]
        for q in QS:
            iz = int(np.argmin(np.abs(rc_ - q))); r = d['D'][des][iz] & d['sg']; m = T.metrics(r, d['ev'], d['sg'])
            for lb in ('sustained', 'brief'): m['caught_' + lb] = 100 * (r & d['ev'][lb]).sum() / max(1, d['ev'][lb].sum())
            m.update(session=s, design=des, q=q, z_used=Z[iz], realised=rc_[iz]); rows.append(m)
mt = pd.DataFrame(rows); mt.to_csv(T.TAB / f'prestim_matched{TAG}.csv', index=False)
rows = []
for des in DES:
    zs = {s: Z[int(np.argmin(np.abs(RC[(s, des)] - 4)))] for s in S}
    mr = np.mean([RC[(s, des)] for s in S], axis=0); ig = int(np.argmin(np.abs(mr - 4))); zg = Z[ig]
    r = dict(design=des, **{f'z4_{s[-3:]}': zs[s] for s in S}, z4_max_over_min=max(zs.values()) / min(zs.values()), global_z=zg, **{f'rate_global_{s[-3:]}': RC[(s, des)][ig] for s in S})
    r['rate_min'] = min(r[f'rate_global_{s[-3:]}'] for s in S); r['rate_max'] = max(r[f'rate_global_{s[-3:]}'] for s in S)
    for zf in (2.5, 3.0, 4.0): i = int(np.argmin(np.abs(Z - zf))); r[f'rate_at_z{zf:g}'] = ' / '.join(f'{RC[(s, des)][i]:.1f}' for s in S)
    rows.append(r)
port = pd.DataFrame(rows); port.to_csv(T.TAB / f'prestim_portability{TAG}.csv', index=False)
cols = ['precision_n45', 'hit_n45', 'hit_n45_strong', 'clean_rejected_pct']; boot = {des: [] for des in DES}
for b in range(NB):
    idx = {s: rng.integers(0, int(S[s]['sg'].sum()), int(S[s]['sg'].sum())) for s in S}
    for des in DES:
        vals = []
        for s, d in S.items():
            sel = np.flatnonzero(d['sg'])[idx[s]]; Ds = d['D'][des][:, sel]; r = Ds[int(np.argmin(np.abs(Ds.mean(axis=1) * 100 - 4)))]
            n45, st, cl = d['ev']['n45'][sel], d['ev']['n45_strong'][sel], d['ev']['clean'][sel]
            vals.append([100 * (r & n45).sum() / max(1, r.sum()), 100 * (r & n45).sum() / max(1, n45.sum()), 100 * (r & st).sum() / max(1, st.sum()), 100 * (r & cl).sum() / max(1, cl.sum())])
        boot[des].append(np.mean(vals, axis=0))
rows = []
for des in DES:
    Bd = np.array(boot[des]) - np.array(boot['30-60|all'])
    rows.append(dict(design=des, **{c: f'{Bd[:, j].mean():+.1f} [{np.percentile(Bd[:, j], 2.5):+.1f}, {np.percentile(Bd[:, j], 97.5):+.1f}]' for j, c in enumerate(cols)}))
bt = pd.DataFrame(rows); bt.to_csv(T.TAB / f'prestim_bootstrap_q4{TAG}.csv', index=False)
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 60)
print(f'\n##### EDGE = {EDGE} ms excluded at each end of the ICA window #####')
print('\n=== validation: sample-level reference rebuilt from pre_sources vs stored table (pre20-60) ==='); print(pd.DataFrame(valrows).round(3).to_string(index=False))
print('\n=== window-level reference per eye IC (30 ms median |s|): untrimmed and after dropping ==='); 
cols_ = ['session', 'ic', 'n_windows', 'sample_ref_mean', 'sample_ref_sd', 'window_ref_mean', 'window_ref_sd', 'A: dropped %', 'A: sd ratio', 'B3: dropped %', 'B3: sd ratio', 'B4: dropped %', 'B4: sd ratio']
print(ref[cols_].round(2).to_string(index=False))
print('\n=== equal cost: mean over 4 pilots (q = % of singles rejected per pilot) ===')
for q in QS:
    g = mt[mt.q == q].groupby('design').agg(caught=('hit_n45', 'mean'), sustained=('caught_sustained', 'mean'), brief=('caught_brief', 'mean'), precision=('precision_n45', 'mean'), strong_caught=('hit_n45_strong', 'mean'), strong_left=('n_kept_strong_n45', 'mean'),
                                         clean_rej=('clean_rejected_pct', 'mean'), precision_pre=('precision_pre', 'mean')).reindex(DES).round(1)
    print(f'\n-- q = {q} % --'); print(g.to_string())
print('\n=== paired bootstrap at q = 4: difference vs 30-60|all (mean of 4 pilots, 95 % CI) ==='); print(bt.to_string(index=False))
print('\n=== portability: z needed for 4 % per pilot; rate under one global z; rate at fixed z (pilot003/004/005/006) ==='); print(port.round(2).to_string(index=False))
