"""
Does the TEP-label EWMA track the amplitudes PRIME selects for?  Checked against the predetermined
(random-timed) single pulses.  Read-only: reads BIDS/sub-*/ses-prime/*/prime/, writes nothing.

Online, the EWMA history is: calibration amplitudes, then every labelled single pulse in order
(PRIME singles incl. forced + predetermined singles; triplets never enter).  The EWMA is linear, so if
PRIME-condition trials are delta larger than unselected trials, the trend at trial t sits delta * p_w(t)
above an unselected trend, where p_w(t) is the EWMA-weighted share of PRIME trials in the history
(calibration counts as unselected).  Counterfactual detrended value = logged detrended + delta * p_w(t).

Per session it prints: delta (PRIME - random, raw amplitude, bootstrap CI stratified by block), the trend
bias, mean labels as run vs with an unselected trend, a model-free check (trend from calibration + random
singles only), and delta adjusted for the gap since the previous pulse.

Sessions without stored calibration amplitudes (bundle 'normalizer' entry, from a1d8593 on) are rebuilt
from the logged tep_ewma (exact) and the label function from logged (detrended, label) pairs (interpolated;
0.0015 mean error on pilot006, where the exact ECDF is available).

Usage:  python ewma_selection_check.py
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

LOOP = Path(__file__).resolve().parents[1]
SPAN = 25; ALPHA = 2 / (SPAN + 1); Q = 1 - ALPHA; WARM = SPAN
NBOOT = 5000; SEED = 0


def load_bundle(path):
    try:
        return np.load(path, allow_pickle=True).item()
    except Exception:                       # no mne here: unpickle with stand-ins for missing classes
        class Stub:
            def __init__(self, *a, **k): pass
            def __setstate__(self, s): pass
            def __setitem__(self, k, v): pass
            def append(self, x): pass
            def extend(self, x): pass
            def update(self, *a, **k): pass

        class U(pickle.Unpickler):
            def find_class(self, mod, name):
                try:
                    return super().find_class(mod, name)
                except Exception:
                    return type(name, (Stub,), {})
        with open(path, 'rb') as f:
            v = np.lib.format.read_magic(f)
            (np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0)(f)
            o = U(f).load()
        return o.item() if isinstance(o, np.ndarray) else o


def run_ewma(N0, D0, x):
    N, D, out = N0, D0, []
    for a in x:
        N = a + Q * N; D = 1 + Q * D; out.append(N / D)
    return np.array(out)


def analyse(pdir, rng):
    t = pd.read_csv(pdir / 'trials_intervention.csv')
    s = t[t.tep_amplitude.notna()].reset_index(drop=True)
    a = s.tep_amplitude_raw.to_numpy(float); y_log = s.tep_ewma.to_numpy(float)
    d_log = s.tep_detrended.to_numpy(float); lab = s.tep_amplitude.to_numpy(float)
    prime = (s.condition == 'prime_single_pulse').to_numpy(); blk = s.stage.str[-1].to_numpy()
    n_cal = int(pd.read_csv(pdir / 'calibration_summary.csv').num_calibration_trials[0])

    nz = None
    try:
        nz = load_bundle(pdir / 'calibration_bundle.npy').get('normalizer')
    except Exception:
        pass
    if nz is not None:                      # exact: calibration amplitudes stored in the bundle
        cal = np.asarray(nz['cal_amplitudes'], float)
        N0 = D0 = 0.0
        for c in cal:
            N0 = c + Q * N0; D0 = 1 + Q * D0
        dc = (pd.Series(cal) - pd.Series(cal).ewm(span=SPAN, adjust=True).mean()).to_numpy()[WARM:]
        ref = np.sort(dc[~np.isnan(dc)])
        F = lambda x: np.searchsorted(ref, x, side='right') / len(ref)
        src = 'bundle'
    else:                                   # rebuilt from the logged trend and logged labels
        D0 = (1 - Q ** n_cal) / (1 - Q)
        N0 = (y_log[0] * (1 + Q * D0) - a[0]) / Q
        order = np.argsort(d_log)
        F = lambda x: np.interp(x, d_log[order], lab[order])
        src = 'log'
    trend_err = np.max(np.abs(run_ewma(N0, D0, a) - y_log)) / np.mean(np.abs(y_log))

    delta = a[prime].mean() - a[~prime].mean()
    groups = {(c, b): np.where((prime == c) & (blk == b))[0] for c in (True, False) for b in np.unique(blk)}
    boots = []
    for _ in range(NBOOT):
        idx = {k: rng.choice(v, len(v)) for k, v in groups.items()}
        boots.append(a[np.concatenate([idx[k] for k in idx if k[0]])].mean()
                     - a[np.concatenate([idx[k] for k in idx if not k[0]])].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])

    zz = s.tep_zscore.to_numpy(float)       # sigma of calibration detrended values (z = (d - mu) / sigma)
    sigma = np.linalg.lstsq(np.c_[np.ones_like(zz), zz], d_log, rcond=None)[0][1]
    p_w = run_ewma(0.0, D0, prime.astype(float))
    lab_cf = F(d_log + delta * p_w)
    lab_cf_lo, lab_cf_hi = F(d_log + lo * p_w), F(d_log + hi * p_w)

    N, D, tr = N0, D0, []                   # model-free: trend from calibration + random singles only
    for ai, isp in zip(a, prime):
        if not isp:
            N = ai + Q * N; D = 1 + Q * D
        tr.append(N / D)
    lab_rand_only = F(a - np.array(tr))

    pulses = t[t.pulse_time.notna()].sort_values('pulse_time')
    gap = t.join(pulses.groupby('stage').pulse_time.diff().rename('gap'))
    g = gap[gap.tep_amplitude.notna()].gap.to_numpy(float); ok = np.isfinite(g) & (g > 0)
    beta = np.linalg.lstsq(np.c_[np.ones(ok.sum()), prime[ok], np.log(g[ok])], a[ok], rcond=None)[0]

    return dict(src=src, trend_err=trend_err, n_prime=int(prime.sum()), n_rand=int((~prime).sum()),
                delta=delta, delta_lo=lo, delta_hi=hi, delta_pct=100 * delta / a[~prime].mean(),
                delta_sd=delta / sigma, delta_gap_adj=beta[1], pw_mean=p_w.mean(),
                trend_bias_sd=(delta * p_w).mean() / sigma,
                lab_rand=lab[~prime].mean(), lab_rand_cf=lab_cf[~prime].mean(), lab_rand_randonly=lab_rand_only[~prime].mean(),
                lab_prime=lab[prime].mean(), lab_prime_cf=lab_cf[prime].mean(),
                shift=(lab - lab_cf).mean(), shift_ci_lo=(lab - lab_cf_hi).mean(), shift_ci_hi=(lab - lab_cf_lo).mean(),
                gap_med_prime=np.median(g[ok & prime]), gap_med_rand=np.median(g[ok & ~prime]))


if __name__ == '__main__':
    rng = np.random.default_rng(SEED)
    rows = {}
    for csv in sorted((LOOP / 'BIDS').glob('sub-*/ses-prime/*/prime/trials_intervention.csv')):
        rows[csv.parts[-5]] = analyse(csv.parent, rng)
    R = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    print(R.to_string(float_format=lambda v: f'{v:.4g}'))
    num = R.drop(['src']).astype(float)
    print('\nMean across sessions:')
    print(num.mean(axis=1).to_string(float_format=lambda v: f'{v:.4g}'))
