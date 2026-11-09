"""Reasoning for the cutoff z on the [30,60] ms ocular gate (untrimmed window reference): z 2.0-3.5, pilot003-006, singles only. Analysis only.
Per cutoff: cost (rate per pilot, pulses per session, calibration trials dropped, cutoff noise), benefit (caught, strong left in), and the marginal
band = pulses rejected at this z but kept at z+0.25: how often are THEY contaminated? ('price' = clean pulses lost per contaminated pulse removed).
Evidence (frontal proxy, not true EOG): n45 = Fp / F7-F8 offset 38-50 ms > 3 robust SD; sustained = n45 and still > 3 SD at 60-100 ms (blink-like);
pre = eye activity -210..-10 ms (independent of the post-pulse window). Outputs offline_ocular_window/trim/tables/zsweep_*.csv
"""
import sys, pickle, numpy as np, pandas as pd
sys.dont_write_bytecode = True; sys.path.insert(0, '.')
import ocular_trim_lib as L, ocular_trim_review as T
C = pickle.load(open(T.OUT / 'cache.pkl', 'rb')); rng = np.random.default_rng(5); NB = 500
ZS = [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5]
S = {}
for s in L.SESSIONS:
    R = L.load_session(s); df = R['df']; ev = T.evidence(df); sg = df.single.to_numpy(); cb = (df.block.to_numpy() == 'calibration') & sg; oos = sg & ~cb
    m5 = L.win_mask(R['cal_t'], 30, 60); m1 = L.win_mask(L.pulse_t_ms(), 30, 60); ca = np.abs(R['cal']); mean, sd = L.reference(ca, m5)
    cal_z = L.cal_scores(ca, m5, mean, sd); pul_raw = np.median(np.abs(R['src'][:, :, m1]), axis=2); pul_z = ((pul_raw - mean[None]) / sd[None]).max(axis=1)
    # sustained offset: still > 3 robust SD at 60-100 ms
    vz, hz = [], []
    for b in L.BLOCKS:
        f = L.WIN / 'replay' / f'sub-{s}' / 'blocks' / f'{b}.npz'
        if f.is_file(): z = np.load(f, allow_pickle=True); vz.append(np.asarray(z['veog'], float)); hz.append(np.asarray(z['heog'], float))
    ve, he = np.concatenate(vz), np.concatenate(hz); t = L.pulse_t_ms(); m60 = L.win_mask(t, 60, 100)
    def zz(x): v = x[:, m60].mean(axis=1); return (v - np.median(v[sg])) / L.robust_sd(v[sg])
    late = (np.abs(zz(ve)) > 3) | (np.abs(zz(he)) > 3)
    ev['sustained'] = ev['n45'] & late
    Dz = C[s]['D']['30-60|all']; i3 = int(np.argmin(np.abs(T.Z_FINE - 3.0)))
    S[s] = dict(df=df, ev=ev, sg=sg, oos=oos, cb=cb, ca=ca, m5=m5, pul_raw=pul_raw, pul_z=pul_z, cal_z=cal_z, Dz=Dz, agree3=((pul_z > 3.0) & sg == Dz[i3] & sg)[sg].mean() * 100)
    print(s, f"score vs stored decisions at z 3.0: {S[s]['agree3']:.2f} % agree", flush=True)

# ---------- cost / benefit per cutoff
rows = []
for z in ZS:
    iz = int(np.argmin(np.abs(T.Z_FINE - z))); r = {}
    for s, d in S.items():
        rej = (d['pul_z'] > z) & d['sg']; m = T.metrics(rej, d['ev'], d['sg'])
        S1 = d['ca'][:, :, d['m5']].sum(axis=2); S2 = (d['ca'][:, :, d['m5']] ** 2).sum(axis=2); ns = int(d['m5'].sum()); n_ = len(d['ca']) * ns
        idx = rng.integers(0, len(d['ca']), (NB, len(d['ca']))); mu = S1[idx].sum(axis=1) / n_; sdv = np.sqrt(np.maximum(S2[idx].sum(axis=1) / n_ - mu ** 2, 1e-18))
        zb = ((d['pul_raw'][d['oos']][None] - mu[:, None, :]) / sdv[:, None, :]).max(axis=2); rb = (zb > z).mean(axis=1) * 100
        rows.append(dict(z=z, session=s, rate_oos=100 * (rej & d['oos']).sum() / d['oos'].sum(), rate_all=m['pct_rejected'], pulses_per_session=int(rej.sum()), cal_dropped=int((d['cal_z'] > z).sum()),
                         caught=m['hit_n45'], precision=m['precision_n45'], strong_caught=m['hit_n45_strong'], strong_kept=m['n_kept_strong_n45'], clean_rej=m['clean_rejected_pct'],
                         boot_sd=rb.std(), boot_hw=(np.percentile(rb, 97.5) - np.percentile(rb, 2.5)) / 2))
cost = pd.DataFrame(rows); cost.to_csv(T.TAB / 'zsweep_cutoffs.csv', index=False)

# ---------- marginal bands (pooled over pilots)
edges = [-np.inf] + ZS + [np.inf]; rows = []
def wilson(k, n, zc=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; den = 1 + zc ** 2 / n; c = (p + zc ** 2 / (2 * n)) / den; h = zc * np.sqrt(p * (1 - p) / n + zc ** 2 / (4 * n ** 2)) / den; return (100 * (c - h), 100 * (c + h))
for lo, hi in zip(edges[:-1], edges[1:]):
    pooled = {k: [0, 0] for k in ('n45', 'sustained', 'pre', 'strong')}; n_tot = 0; per = {k: [] for k in ('n45', 'sustained')}; nper = []
    for s, d in S.items():
        sel = d['sg'] & (d['pul_z'] > lo) & (d['pul_z'] <= hi); n_tot += sel.sum(); nper.append(int(sel.sum()))
        for k, key in (('n45', 'n45'), ('sustained', 'sustained'), ('pre', 'pre'), ('strong', 'n45_strong')):
            pooled[k][0] += int((d['ev'][key] & sel).sum()); pooled[k][1] += int(sel.sum())
        for k in per: per[k].append(100 * (d['ev'][k] & sel).sum() / max(1, sel.sum()))
    rows.append(dict(band=f"{lo:g} to {hi:g}", z_lo=lo, z_hi=hi, n_pooled=int(n_tot), n_per_pilot='/'.join(map(str, nper)),
                     **{f'pct_{k}': 100 * v[0] / max(1, v[1]) for k, v in pooled.items()},
                     n45_ci_lo=wilson(*pooled['n45'])[0], n45_ci_hi=wilson(*pooled['n45'])[1], sustained_ci_lo=wilson(*pooled['sustained'])[0], sustained_ci_hi=wilson(*pooled['sustained'])[1],
                     n45_per_pilot='/'.join(f'{v:.0f}' for v in per['n45']), sustained_per_pilot='/'.join(f'{v:.0f}' for v in per['sustained'])))
bands = pd.DataFrame(rows); bands['price_clean_per_contaminated_n45'] = (100 - bands.pct_n45) / bands.pct_n45; bands['price_clean_per_contaminated_sustained'] = (100 - bands.pct_sustained) / bands.pct_sustained
bands.to_csv(T.TAB / 'zsweep_bands.csv', index=False)

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
g = cost.groupby('z')
tab = pd.DataFrame({'rate oos 003/004/005/006': g.rate_oos.apply(lambda x: ' / '.join(f'{v:.1f}' for v in x)), 'mean rate': g.rate_all.mean(), 'spread': g.rate_oos.apply(lambda x: x.max() - x.min()),
    'pulses/session': g.pulses_per_session.mean(), 'cal dropped 003/004/005/006': g.cal_dropped.apply(lambda x: '/'.join(str(v) for v in x)),
    'caught': g.caught.mean(), 'precision': g.precision.mean(), 'strong caught': g.strong_caught.mean(), 'strong kept': g.strong_kept.mean(), 'clean rej': g.clean_rej.mean(), 'cutoff noise 95% hw': g.boot_hw.mean()}).round(1)
print('\n=== cost / benefit per cutoff (mean of 4 pilots unless listed) ==='); print(tab.to_string())
print('\n=== marginal bands: pulses with score in (z_lo, z_hi], pooled over 4 pilots ===')
print(bands[['band', 'n_pooled', 'n_per_pilot', 'pct_n45', 'n45_ci_lo', 'n45_ci_hi', 'n45_per_pilot', 'pct_sustained', 'sustained_per_pilot', 'pct_strong', 'pct_pre', 'price_clean_per_contaminated_n45', 'price_clean_per_contaminated_sustained']].round(1).to_string(index=False))

# ---------- calibration impact check against the stored per-z calibrations, and outcome sensitivity for context (not a selection criterion)
cal = pd.read_csv(L.WIN / 'windows' / 'tables' / 'calibration.csv'); ev_ = pd.read_csv(L.WIN / 'windows' / 'tables' / 'evoked.csv')
c = cal[cal.window == '30-60'][['subject', 'z', 'dropped_ocular', 'trials_for_prime']]
print('\n=== stored per-z calibrations, window 30-60: ocular drops in calibration / trials left for PRIME ===')
print(c.pivot(index='z', columns='subject', values=['dropped_ocular', 'trials_for_prime']).to_string())
print('\nstored evoked table: processing values', ev_.processing.unique(), 'windows', ev_.window.unique())
e = ev_[(ev_.window == '30-60') & (ev_.block.isin(['t0', 't15', 't30', 't60']))]
for proc in e.processing.unique():
    p = e[e.processing == proc].groupby(['subject', 'z']).change_pct.mean().unstack('z').round(1); print(f'\nevoked N45 change from baseline, mean of t0-t60 (%), processing={proc}'); print(p.to_string())
