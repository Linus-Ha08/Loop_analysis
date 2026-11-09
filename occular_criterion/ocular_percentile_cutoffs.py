"""Percentile cutoffs on the [30,60] ms ocular score, calibrated on the session's 125 calibration trials. Analysis only.
P1  score = max over eye ICs of the median z over 30-60 ms (untrimmed window reference, as in '30-60|all'); cutoff = p-th percentile of
    the 125 calibration trials' scores  (= a per-session z chosen so that calibration rejects ~(100-p) %).
P2  score per eye IC = median |s| over 30-60 ms (no normalisation); cutoff per IC = p-th percentile of that IC's calibration scores;
    reject if ANY eye IC is above its cutoff.
Pulses are scored from the stored 1 kHz sources, calibration trials from the 5 kHz sources (as calibrate() would see them).
Singles only. 'oos' = singles outside the calibration block. Evidence = frontal proxy (not true EOG).
Outputs offline_ocular_window/trim/tables/pct_*.csv
"""
import sys, pickle, numpy as np, pandas as pd
sys.dont_write_bytecode = True; sys.path.insert(0, '.')
import ocular_trim_lib as L, ocular_trim_review as T
C = pickle.load(open(T.OUT / 'cache.pkl', 'rb'))
PS = [85, 90, 92.5, 95, 96, 97, 97.5, 98, 99]; NB = 500; rng = np.random.default_rng(3)
rows, brow, blk = [], [], []
chk = []
for s in L.SESSIONS:
    R = L.load_session(s); df = R['df']; ev = T.evidence(df); sg = df.single.to_numpy()
    cb = (df.block.to_numpy() == 'calibration') & sg; oos = sg & ~cb
    m5 = L.win_mask(R['cal_t'], 30, 60); m1 = L.win_mask(L.pulse_t_ms(), 30, 60)
    ca = np.abs(R['cal']); mean, sd = L.reference(ca, m5)
    cal_raw = np.median(ca[:, :, m5], axis=2)                                   # (125, k)
    cal_z = ((cal_raw - mean[None]) / sd[None]).max(axis=1)                     # (125,)
    pul_raw = np.median(np.abs(R['src'][:, :, m1]), axis=2)                     # (1425, k)
    pul_z = ((pul_raw - mean[None]) / sd[None]).max(axis=1)
    Dz = C[s]['D']['30-60|all']; zr = np.array([(Dz[i] & sg).sum() / sg.sum() * 100 for i in range(len(T.Z_FINE))])
    for p in PS:
        # ---- P1
        cut = np.percentile(cal_z, p); r1 = (pul_z > cut) & sg
        # ---- P2
        cuts2 = np.percentile(cal_raw, p, axis=0); r2 = (pul_raw > cuts2[None]).any(axis=1) & sg
        for name, r, ccount, cutdesc in (('P1 max-z', r1, int((cal_z > cut).sum()), cut), ('P2 per-IC raw', r2, int((cal_raw > cuts2[None]).any(axis=1).sum()), float(np.mean(cuts2)))):
            mt = T.metrics(r, ev, sg); mt.update(session=s, rule=name, p=p, nominal=100 - p, cut=cutdesc, cal_rejected=ccount,
                                                 pct_oos=100 * (r & oos).sum() / oos.sum(), pct_cal_block_1khz=100 * (r & cb).sum() / cb.sum())
            if name == 'P1 max-z':   # the z gate at the same realised cost (should be the same decisions up to the grid)
                iz = int(np.argmin(np.abs(zr - mt['pct_rejected']))); mz = T.metrics(Dz[iz], ev, sg)
                mt.update(z_equiv=T.Z_FINE[iz], hit_z_same_cost=mz['hit_n45'], prec_z_same_cost=mz['precision_n45'], agree_pct=100 * ((Dz[iz] & sg)[sg] == r1[sg]).mean())
                # bootstrap of the 125 calibration trials -> cutoff uncertainty -> rejection rate of later singles
                idx = rng.integers(0, len(cal_z), (NB, len(cal_z))); cb_cut = np.percentile(cal_z[idx], p, axis=1)
                rates = (pul_z[oos][None, :] > cb_cut[:, None]).mean(axis=1) * 100
                mt.update(boot_rate_sd=rates.std(), boot_rate_lo=np.percentile(rates, 2.5), boot_rate_hi=np.percentile(rates, 97.5), boot_cut_sd=cb_cut.std())
                if p in (95, 97.5):   # per block rate
                    for b in L.BLOCKS:
                        sel = sg & (df.block.to_numpy() == b)
                        if sel.sum(): blk.append(dict(session=s, p=p, block=b, n=int(sel.sum()), pct=100 * (r & sel).sum() / sel.sum()))
            rows.append(mt)
    # fixed-z reference rows (same window); bootstrap re-estimates the window reference (mean / SD) from resampled calibration trials
    S1 = ca[:, :, m5].sum(axis=2); S2 = (ca[:, :, m5] ** 2).sum(axis=2); ns = int(m5.sum())
    for zc in (2.5, 3.0):
        iz = int(np.argmin(np.abs(T.Z_FINE - zc))); mt = T.metrics(Dz[iz], ev, sg)
        idx = rng.integers(0, len(ca), (NB, len(ca))); n_ = len(ca) * ns
        mu = S1[idx].sum(axis=1) / n_; sdv = np.sqrt(np.maximum(S2[idx].sum(axis=1) / n_ - mu ** 2, 1e-18))
        zb = ((pul_raw[oos][None, :, :] - mu[:, None, :]) / sdv[:, None, :]).max(axis=2); rb = (zb > zc).mean(axis=1) * 100
        mt.update(boot_rate_sd=rb.std(), boot_rate_lo=np.percentile(rb, 2.5), boot_rate_hi=np.percentile(rb, 97.5))
        mt.update(session=s, rule=f'fixed z {zc}', p=np.nan, nominal=np.nan, cut=zc, cal_rejected=int((cal_z > zc).sum()), pct_oos=100 * (Dz[iz] & oos).sum() / oos.sum(),
                  pct_cal_block_1khz=100 * (Dz[iz] & cb).sum() / cb.sum()); rows.append(mt)
    mt = T.metrics(C[s]['D']['now'][int(np.argmin(np.abs(T.Z_FINE - 3.0)))], ev, sg); mt.update(session=s, rule='gate as it runs (z 3.0, whole-epoch ref)', p=np.nan, nominal=np.nan, cut=3.0,
        cal_rejected=np.nan, pct_oos=100 * (C[s]['D']['now'][int(np.argmin(np.abs(T.Z_FINE - 3.0)))] & oos).sum() / oos.sum(), pct_cal_block_1khz=np.nan); rows.append(mt)
    print(s, 'done', flush=True)
res = pd.DataFrame(rows); res.to_csv(T.TAB / 'pct_overview.csv', index=False); pd.DataFrame(blk).to_csv(T.TAB / 'pct_blocks.csv', index=False)

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
def summ(rule):
    d = res[res.rule == rule]
    g = d.groupby('p', sort=True)
    out = pd.DataFrame({'nominal %': g.nominal.first(),
        'cal rejected (of 125) 3/4/5/6': g.cal_rejected.apply(lambda x: '/'.join(str(int(v)) for v in x)),
        'rate oos % 003/004/005/006': g.pct_oos.apply(lambda x: ' / '.join(f'{v:.1f}' for v in x)),
        'mean rate %': g.pct_rejected.mean(), 'spread (max-min)': g.pct_oos.apply(lambda x: x.max() - x.min()),
        'caught': g.hit_n45.mean(), 'precision': g.precision_n45.mean(), 'strong caught': g.hit_n45_strong.mean(),
        'strong kept': g.n_kept_strong_n45.mean(), 'clean rej %': g.clean_rejected_pct.mean()})
    if rule == 'P1 max-z':
        out['z used 003/004/005/006'] = g.cut.apply(lambda x: ' / '.join(f'{v:.2f}' for v in x))
        out['boot SD of rate (mean)'] = g.boot_rate_sd.mean(); out['boot 95% half-width (mean)'] = g.apply(lambda x: ((x.boot_rate_hi - x.boot_rate_lo) / 2).mean())
        out['same decisions as z at same cost %'] = g.agree_pct.mean()
    return out.round(1)
for r in ('P1 max-z', 'P2 per-IC raw'):
    print(f'\n=== {r} ==='); print(summ(r).to_string())
print('\n=== reference rows (mean over sessions; per-session oos rate in brackets) ===')
ref = res[res.rule.str.startswith(('fixed', 'gate'))]
print(ref.groupby('rule').agg(mean_rate=('pct_rejected', 'mean'), caught=('hit_n45', 'mean'), precision=('precision_n45', 'mean'), strong_caught=('hit_n45_strong', 'mean'), strong_kept=('n_kept_strong_n45', 'mean'), clean_rej=('clean_rejected_pct', 'mean'),
      oos_rates=('pct_oos', lambda x: ' / '.join(f'{v:.1f}' for v in x)), boot_sd=('boot_rate_sd', 'mean'), boot_halfwidth=('boot_rate_lo', lambda x: float(((ref.loc[x.index, 'boot_rate_hi'] - x) / 2).mean()))).round(1).to_string())
b = pd.DataFrame(blk); print('\n=== P1 per-block rate (% of singles) at p=95 and 97.5: min-max over blocks, per session ===')
print(b.groupby(['p', 'session']).pct.agg(['min', 'max']).round(1).unstack('session').to_string())
