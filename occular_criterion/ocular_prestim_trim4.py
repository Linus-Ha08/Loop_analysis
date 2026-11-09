"""Is the ranking gain of variant B (iterative drop at fixed c, tiny SD) real, and what z would the gate need? Bootstrap vs 30-60|all at q = 4 %, z needed per pilot, rate under one global z."""
import sys, pickle, numpy as np, pandas as pd
from pathlib import Path
sys.dont_write_bytecode = True; sys.path.insert(0, '.')
import ocular_trim_lib as L, ocular_trim_review as T
exec(open('ocular_prestim_trim3.py').read().split('ZA = np.arange')[0])         # SP, trim, FIX
C = pickle.load(open(T.OUT / 'cache.pkl', 'rb')); rng = np.random.default_rng(9); NB = 1000
NMS = ['iterative c=3 (B)', 'iterative c=4 (B)', 'W0 no dropping']
sc = {s: {nm: (lambda mu_sd: ((SP[s]['post'] - mu_sd[0]) / mu_sd[1]).max(1))(FIX[nm](SP[s]['W100'])) for nm in NMS} for s in SP}
# z needed per pilot for 4 %, and under one global z
for nm in NMS:
    zs = {s: np.sort(sc[s][nm][SP[s]['sg']])[::-1][int(round(0.04 * SP[s]['sg'].sum()))] for s in SP}
    zg = np.mean(list(zs.values())); print(f'{nm}: z for 4 % per pilot', {k[-3:]: round(v, 1) for k, v in zs.items()}, f' max/min {max(zs.values())/min(zs.values()):.2f}',
          '| rate under mean z=%.1f:' % zg, {k[-3:]: round(100 * (sc[k][nm][SP[k]['sg']] > zg).mean(), 1) for k in SP})
    for zf in (5, 10, 15, 20): print(f'   rate at z={zf}:', {k[-3:]: round(100 * (sc[k][nm][SP[k]['sg']] > zf).mean(), 1) for k in SP})
bs = {nm: [] for nm in ['30-60|all'] + NMS}
for b in range(NB):
    idx = {s: rng.integers(0, int(SP[s]['sg'].sum()), int(SP[s]['sg'].sum())) for s in SP}
    for nm in bs:
        vals = []
        for s, d in SP.items():
            sel = np.flatnonzero(d['sg'])[idx[s]]; n = len(sel)
            if nm == '30-60|all': Ds = C[s]['D']['30-60|all'][:, sel]; r = Ds[int(np.argmin(np.abs(Ds.mean(axis=1) * 100 - 4)))]
            else: x = sc[s][nm][sel]; r = x > np.sort(x)[::-1][int(round(0.04 * n))]
            n45, st, cl = d['ev']['n45'][sel], d['ev']['n45_strong'][sel], d['ev']['clean'][sel]
            vals.append([100 * (r & n45).sum() / max(1, r.sum()), 100 * (r & n45).sum() / max(1, n45.sum()), 100 * (r & st).sum() / max(1, st.sum()), 100 * (r & cl).sum() / max(1, cl.sum())])
        bs[nm].append(np.mean(vals, axis=0))
print('\npaired bootstrap at q = 4 vs 30-60|all: precision, caught, strong caught, clean rejected (mean diff [95 % CI])')
for nm in NMS:
    Bd = np.array(bs[nm]) - np.array(bs['30-60|all']); print(f'{nm:20s}', '  '.join(f'{Bd[:, j].mean():+.1f} [{np.percentile(Bd[:, j], 2.5):+.1f}, {np.percentile(Bd[:, j], 97.5):+.1f}]' for j in range(4)))
# per pilot: strong caught at q=4 for the three
print('\nper pilot at q = 4 (caught / strong caught / precision):')
for s, d in SP.items():
    out = []
    for nm in NMS:
        x = sc[s][nm]; sg = d['sg']; r = (x > np.sort(x[sg])[::-1][int(round(0.04 * sg.sum()))]) & sg; m = T.metrics(r, d['ev'], sg); out.append(f"{nm}: {m['hit_n45']:.0f}/{m['hit_n45_strong']:.0f}/{m['precision_n45']:.0f}")
    Dm = C[s]['D']['30-60|all']; i = int(np.argmin(np.abs(np.array([(Dm[k] & d['sg']).sum() / d['sg'].sum() * 100 for k in range(len(T.Z_FINE))]) - 4))); m = T.metrics(Dm[i] & d['sg'], d['ev'], d['sg'])
    print(s, ' | '.join(out), f"| 30-60|all: {m['hit_n45']:.0f}/{m['hit_n45_strong']:.0f}/{m['precision_n45']:.0f}")
