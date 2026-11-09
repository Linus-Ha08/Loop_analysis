"""EOG headroom bound from stored pilot003-006 data (analysis only, nothing in `prime` touched).
Gate = [30,60] ms, untrimmed reference ('30-60|all'), cutoff per session set to reject 4 % of singles.
Evidence = frontal proxies (VEOG = mean Fp1/Fpz/Fp2, HEOG = F7-F8), NOT true EOG:
  n45 = frontal offset 38-50 ms > 3 robust SD; pre = raw -210..-10 ms p-p above median + 3 robust SD.
Outputs offline_ocular_window/trim/tables/eog_headroom.csv
"""
import sys, pickle, numpy as np, pandas as pd
sys.dont_write_bytecode = True; sys.path.insert(0, '.')
import ocular_trim_lib as L, ocular_trim_review as T
C = pickle.load(open(T.OUT / 'cache.pkl', 'rb'))
DES, Q = '30-60|all', 4.0
rows = []
for s in L.SESSIONS:
    R = L.load_session(s); df = R['df']; sg = C[s]['single']; ev = C[s]['ev']
    curve = np.array([(C[s]['D'][DES][i] & sg).sum() / sg.sum() * 100 for i in range(len(T.Z_FINE))])
    r = C[s]['D'][DES][int(np.argmin(np.abs(curve - Q)))]
    n45, strong, pre = ev['n45'], ev['n45_strong'], ev['pre']; nopre = sg & ~pre; N = sg.sum()
    v = df.veog_mean_n45.to_numpy(float); h = df.heog_mean_n45.to_numpy(float)
    zv = (v - np.median(v[sg])) / L.robust_sd(v[sg]); zh = (h - np.median(h[sg])) / L.robust_sd(h[sg])
    vd = ((np.abs(zv) > 3) | (np.abs(zh) > 3)) & (np.abs(zv) >= np.abs(zh))
    pos, neg = vd & (zv > 3) & sg, vd & (zv < -3) & sg
    rows.append(dict(session=s, n_singles=int(N), gate_reject_pct=100 * (r & sg).sum() / N,
        offset_no_gate_pct=100 * (n45 & sg).sum() / N,
        offset_gate_only_pct=100 * (n45 & ~r & sg).sum() / N,
        offset_holdoff_only_pct=100 * (n45 & nopre).sum() / nopre.sum(),
        offset_gate_plus_holdoff_pct=100 * (n45 & ~r & nopre).sum() / nopre.sum(),
        strong_gate_only_pct=100 * (strong & ~r & sg).sum() / N,
        strong_gate_plus_holdoff_pct=100 * (strong & ~r & nopre).sum() / nopre.sum(),
        pre_flagged_pct=100 * (pre & sg).sum() / N,
        p_offset_given_pre=100 * (n45 & pre & sg).sum() / (pre & sg).sum(),
        p_offset_given_nopre=100 * (n45 & nopre).sum() / nopre.sum(),
        leaks_preflagged_pct=100 * (n45 & ~r & pre & sg).sum() / (n45 & ~r & sg).sum(),
        gate_reject_after_holdoff_pct=100 * (r & nopre).sum() / nopre.sum(),
        pos_offset_n=int(pos.sum()), pos_caught_pct=100 * (r & pos).sum() / max(1, pos.sum()),
        neg_offset_n=int(neg.sum()), neg_caught_pct=100 * (r & neg).sum() / max(1, neg.sum()),
        pos_leaked_pct_of_singles=100 * (pos & ~r).sum() / N))
t = pd.DataFrame(rows); out = T.OUT / 'tables' / 'eog_headroom.csv'; t.to_csv(out, index=False)
pd.set_option('display.width', 250); print(t.round(1).T); print('mean over sessions:\n', t.drop(columns='session').mean().round(1)); print('wrote', out)
