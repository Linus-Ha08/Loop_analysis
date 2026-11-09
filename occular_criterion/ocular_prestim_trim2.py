"""Follow-up to ocular_prestim_trim.py: why does iterative dropping on the pre-stim windows collapse, and do milder variants help? Reads the caches of the first script.
Diagnostic: trimmed pre-stim reference vs the post-pulse 30-60 ms level of CLEAN singles (domain mismatch). Variants: single-pass drop at c SD (refit once), drop top p % of windows once, edge 0 vs 100 ms.
Same equal-cost evaluation (singles, per-pilot cutoff for q % rejected)."""
import sys, pickle, numpy as np, pandas as pd
from pathlib import Path
sys.dont_write_bytecode = True; sys.path.insert(0, '.')
import ocular_trim_lib as L, ocular_trim_review as T
Z = T.Z_FINE; W = 30; MIN_KEEP = 500; ONLY = sys.argv[1] if len(sys.argv) > 1 else None
CP = lambda s_: Path.home() / f'prestim2_cache_{s_}.pkl'
def win_stats(pa, edge):
    n = (pa.shape[2] - 2 * edge) // W; X = pa[:, :, edge: edge + n * W].reshape(pa.shape[0], pa.shape[1], n, W)
    return np.median(X, axis=3).transpose(0, 2, 1).reshape(-1, pa.shape[1])
t = L.pulse_t_ms(); m1 = L.win_mask(t, 30, 60)
for s in L.SESSIONS:
    if CP(s).is_file() or (ONLY and s != ONLY): continue
    r = L.WIN / 'replay' / f'sub-{s}'; src = np.concatenate([np.asarray(np.load(r / 'blocks' / f'{b}.npz', allow_pickle=True)['src'], float) for b in L.BLOCKS if (r / 'blocks' / f'{b}.npz').is_file()])
    pre = np.load(L.WIN / 'windows' / f'sub-{s}' / 'pre' / 'pre_sources.npz', allow_pickle=True); pa = np.abs(pre['src']).astype(float)
    pickle.dump(dict(post=np.median(np.abs(src[:, :, m1]), axis=2), W100=win_stats(pa, 100), W0=win_stats(pa, 0)), open(CP(s), 'wb')); print(s, 'cached', flush=True)
if ONLY: sys.exit()
def trim(Wf, cut, max_iter=60):
    keep = np.ones(len(Wf), bool)
    for _ in range(max_iter):
        mu, sd = Wf[keep].mean(0), Wf[keep].std(0); new = ((Wf - mu) / sd).max(1) <= cut
        if new.sum() < MIN_KEEP or (new == keep).all(): break
        keep = new
    return Wf[keep].mean(0), Wf[keep].std(0), keep
def one_pass(Wf, c): mu, sd = Wf.mean(0), Wf.std(0); keep = ((Wf - mu) / sd).max(1) <= c; return Wf[keep].mean(0), Wf[keep].std(0), keep
def top_p(Wf, p): mu, sd = Wf.mean(0), Wf.std(0); sc = ((Wf - mu) / sd).max(1); keep = sc <= np.percentile(sc, 100 - p); return Wf[keep].mean(0), Wf[keep].std(0), keep
SP = {}; 
for s in L.SESSIONS: SP[s] = pickle.load(open(Path.home() / f'prestim_cache_100_{s}.pkl', 'rb'))[0]; SP[s].update(pickle.load(open(CP(s), 'rb')))
VAR = {'S0/W0 no dropping (edge 100)': lambda d: (d['W100'], None), }
def designs(d):
    out = {}
    for tag, Wf in (('e100', d['W100']), ('e0', d['W0'])):
        out[f'W0 {tag}: no dropping'] = (Wf.mean(0), Wf.std(0))
        for c in (2.5, 3, 4, 5): m, sd, _ = one_pass(Wf, c); out[f'one pass c={c:g} {tag}'] = (m, sd)
        for p in (5, 10, 20): m, sd, _ = top_p(Wf, p); out[f'drop top {p}% once {tag}'] = (m, sd)
        for c in (3, 4): m, sd, _ = trim(Wf, c); out[f'iterative c={c:g} {tag}'] = (m, sd)
    return out
DN = list(designs(SP['pilot003']).keys()); rows = []; diag = []
for s, d in SP.items():
    sg, ev = d['sg'], d['ev']; post = d['post']; des = designs(d)
    cl = ev['clean'] & sg
    for nm, (mu, sd) in des.items():
        sc = ((post - mu) / sd).max(1)
        rc = np.array([(sc > z)[sg].mean() * 100 for z in Z])
        r3 = (sc > 3)[sg].mean() * 100
        diag.append(dict(session=s, design=nm, rate_at_z3=r3, clean_z_median=np.median(sc[cl]), clean_z_p95=np.percentile(sc[cl], 95), ref_mean=' / '.join(f'{x:.2f}' for x in mu), ref_sd=' / '.join(f'{x:.2f}' for x in sd),
                         clean_post_median=' / '.join(f'{x:.2f}' for x in np.median(post[cl], axis=0))))
        for q in (3, 4, 5, 6):
            iz = int(np.argmin(np.abs(rc - q))); rej = (sc > Z[iz]) & sg; m = T.metrics(rej, ev, sg); m.update(session=s, design=nm, q=q, z_used=Z[iz]); rows.append(m)
dg = pd.DataFrame(diag); dg.to_csv(T.TAB / 'prestim_variants_diagnostic.csv', index=False); mt = pd.DataFrame(rows); mt.to_csv(T.TAB / 'prestim_variants_matched.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_colwidth', 40)
print('=== domain check: reference (mean / sd per eye IC) vs post-pulse 30-60 ms level of CLEAN singles; rate = % singles rejected at z 3 ===')
for s in ('pilot003', 'pilot006'):
    print(s); print(dg[(dg.session == s) & dg.design.str.contains('e100')][['design', 'rate_at_z3', 'clean_z_median', 'clean_z_p95', 'ref_mean', 'ref_sd', 'clean_post_median']].round(2).to_string(index=False))
print('\n=== rate at z 3 (% singles), per pilot (003/004/005/006), and mean clean z ===')
g = dg.pivot(index='design', columns='session', values='rate_at_z3').reindex(DN).round(1); g['mean clean z (median)'] = dg.groupby('design').clean_z_median.mean().reindex(DN).round(2); print(g.to_string())
print('\n=== equal cost, mean over 4 pilots, q = 4 % ===')
x = mt[mt.q == 4].groupby('design').agg(caught=('hit_n45', 'mean'), precision=('precision_n45', 'mean'), strong_caught=('hit_n45_strong', 'mean'), strong_left=('n_kept_strong_n45', 'mean'), clean_rej=('clean_rejected_pct', 'mean'), precision_pre=('precision_pre', 'mean'), z=('z_used', 'mean')).reindex(DN).round(2); print(x.to_string())
