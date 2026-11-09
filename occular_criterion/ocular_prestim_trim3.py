"""Fair equal-cost version of the pre-stim reference variants. The z grid of the first two scripts stops at z = 6, so designs whose reference collapses (small SD) could not reach the target
rejection rate and ran into the ceiling; here a fixed-score design rejects exactly the top q % of singles by score (no grid), and variant A (reference depends on z) uses a grid to z = 20."""
import sys, pickle, numpy as np, pandas as pd
from pathlib import Path
sys.dont_write_bytecode = True; sys.path.insert(0, '.')
import ocular_trim_lib as L, ocular_trim_review as T
MIN_KEEP = 500; QS = [3, 4, 5, 6]
SP = {}
for s in L.SESSIONS: SP[s] = pickle.load(open(Path.home() / f'prestim_cache_100_{s}.pkl', 'rb'))[0]; SP[s].update(pickle.load(open(Path.home() / f'prestim2_cache_{s}.pkl', 'rb')))
def trim(Wf, cut, max_iter=60):
    keep = np.ones(len(Wf), bool)
    for _ in range(max_iter):
        mu, sd = Wf[keep].mean(0), Wf[keep].std(0); new = ((Wf - mu) / sd).max(1) <= cut
        if new.sum() < MIN_KEEP or (new == keep).all(): break
        keep = new
    return Wf[keep].mean(0), Wf[keep].std(0), keep
def one_pass(Wf, c): mu, sd = Wf.mean(0), Wf.std(0); keep = ((Wf - mu) / sd).max(1) <= c; return Wf[keep].mean(0), Wf[keep].std(0)
def top_p(Wf, p): mu, sd = Wf.mean(0), Wf.std(0); sc = ((Wf - mu) / sd).max(1); keep = sc <= np.percentile(sc, 100 - p); return Wf[keep].mean(0), Wf[keep].std(0)
FIX = {'W0 no dropping': lambda W: (W.mean(0), W.std(0)), 'one pass c=3': lambda W: one_pass(W, 3), 'one pass c=4': lambda W: one_pass(W, 4), 'drop top 10% once': lambda W: top_p(W, 10),
       'iterative c=3 (B)': lambda W: trim(W, 3)[:2], 'iterative c=4 (B)': lambda W: trim(W, 4)[:2]}
ZA = np.arange(1.0, 20.01, 0.25); rows = []; ar = []
for s, d in SP.items():
    sg, ev, post, Wf = d['sg'], d['ev'], d['post'], d['W100']
    for nm, fn in FIX.items():
        mu, sd = fn(Wf); sc = ((post - mu) / sd).max(1); thr_all = np.sort(sc[sg])[::-1]
        for q in QS:
            n = int(round(q / 100 * sg.sum())); rej = (sc > thr_all[n]) & sg                  # top-n singles by score
            m = T.metrics(rej, ev, sg)
            for lb in ('sustained', 'brief'): m['caught_' + lb] = 100 * (rej & ev[lb]).sum() / max(1, ev[lb].sum())
            m.update(session=s, design=nm, q=q); rows.append(m)
    refs = {z: trim(Wf, z) for z in ZA}
    rate = np.array([((((post - refs[z][0]) / refs[z][1]).max(1) > z) & sg).sum() / sg.sum() * 100 for z in ZA])
    for q in QS:
        iz = int(np.argmin(np.abs(rate - q))); z = ZA[iz]; rej = ((((post - refs[z][0]) / refs[z][1]).max(1) > z)) & sg; m = T.metrics(rej, ev, sg)
        for lb in ('sustained', 'brief'): m['caught_' + lb] = 100 * (rej & ev[lb]).sum() / max(1, ev[lb].sum())
        m.update(session=s, design='A iterative, cutoff = gate z', q=q, z_used=z, realised=rate[iz], dropped_pct=100 * (1 - refs[z][2].mean())); rows.append(m)
    ar.append(dict(session=s, **{f'rate z={zf:g}': rate[int(np.argmin(np.abs(ZA - zf)))] for zf in (3, 5, 7, 10, 15)}, **{f'dropped% z={zf:g}': 100 * (1 - refs[zf][2].mean()) for zf in (3, 5, 7, 10, 15)}))
mt = pd.DataFrame(rows); mt.to_csv(T.TAB / 'prestim_equalcost_fair.csv', index=False)
cmp_ = pd.read_csv(T.TAB / 'prestim_matched.csv'); cmp_ = cmp_[cmp_.design.isin(['now', '30-60|all', 'S0 sample-level, no dropping'])]
allm = pd.concat([cmp_.assign(design=cmp_.design.map({'now': 'now (gate as it runs)', '30-60|all': '30-60|all (post-cal reference)', 'S0 sample-level, no dropping': 'S0 pre sample-level'})), mt], ignore_index=True)
order = ['now (gate as it runs)', '30-60|all (post-cal reference)', 'S0 pre sample-level'] + list(FIX) + ['A iterative, cutoff = gate z']
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
for q in (3, 4, 5):
    g = allm[allm.q == q].groupby('design').agg(caught=('hit_n45', 'mean'), sustained=('caught_sustained', 'mean'), brief=('caught_brief', 'mean'), precision=('precision_n45', 'mean'), strong_caught=('hit_n45_strong', 'mean'), strong_left=('n_kept_strong_n45', 'mean'),
                                              clean_rej=('clean_rejected_pct', 'mean'), precision_pre=('precision_pre', 'mean')).reindex(order).round(1)
    print(f'\n-- fair equal cost, q = {q} % (mean of 4 pilots) --'); print(g.to_string())
print('\n=== variant A: rate (% singles) at gate z and share of pre-stim windows dropped (extended grid) ==='); print(pd.DataFrame(ar).round(1).to_string(index=False))
a = mt[(mt.design.str.startswith('A ')) & (mt.q == 4)][['session', 'z_used', 'realised', 'dropped_pct']]; print('\nA at q=4:'); print(a.round(2).to_string(index=False))
# paired bootstrap vs 30-60|all at q = 4 on the fair designs
rng = np.random.default_rng(5); import ocular_trim_review as TT
C = pickle.load(open(T.OUT / 'cache.pkl', 'rb')); Z = T.Z_FINE; NB = 500
def sel_rej_fixed(d, nm):
    mu, sd = FIX[nm](d['W100']); return ((d['post'] - mu) / sd).max(1)
bs = {nm: [] for nm in ['30-60|all'] + ['W0 no dropping', 'one pass c=3', 'drop top 10% once']}
scs = {s: {nm: sel_rej_fixed(SP[s], nm) for nm in ['W0 no dropping', 'one pass c=3', 'drop top 10% once']} for s in SP}
for b in range(NB):
    idx = {s: rng.integers(0, int(SP[s]['sg'].sum()), int(SP[s]['sg'].sum())) for s in SP}
    for nm in bs:
        vals = []
        for s, d in SP.items():
            sel = np.flatnonzero(d['sg'])[idx[s]]; n = len(sel)
            if nm == '30-60|all': Ds = C[s]['D']['30-60|all'][:, sel]; r = Ds[int(np.argmin(np.abs(Ds.mean(axis=1) * 100 - 4)))]
            else: sc = scs[s][nm][sel]; r = sc > np.sort(sc)[::-1][int(round(0.04 * n))]
            n45, st, cl = d['ev']['n45'][sel], d['ev']['n45_strong'][sel], d['ev']['clean'][sel]
            vals.append([100 * (r & n45).sum() / max(1, r.sum()), 100 * (r & n45).sum() / max(1, n45.sum()), 100 * (r & st).sum() / max(1, st.sum()), 100 * (r & cl).sum() / max(1, cl.sum())])
        bs[nm].append(np.mean(vals, axis=0))
print('\n=== paired bootstrap at q = 4 vs 30-60|all (precision, caught, strong caught, clean rejected) ===')
for nm in bs:
    Bd = np.array(bs[nm]) - np.array(bs['30-60|all']); print(f'{nm:22s}', '  '.join(f'{Bd[:, j].mean():+.1f} [{np.percentile(Bd[:, j], 2.5):+.1f}, {np.percentile(Bd[:, j], 97.5):+.1f}]' for j in range(4)))
