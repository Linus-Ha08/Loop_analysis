"""Draw 10000 ITIs from util/iti.py and show the distribution.

Sanity check for the sampler the deciders use. Run it after changing any of
TARGET_MEAN / SPIKE_VALUE / SPIKE_WEIGHT / CONCENTRATION.

    python test_iti_generator.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# util/iti.py lives in the decider, which is a sibling of scripts_analysis.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'prime' / 'decider'))
from util.iti import (sample_iti, ITI_MIN, ITI_MAX, TARGET_MEAN,
                      SPIKE_VALUE, SPIKE_WEIGHT, _ALPHA, _BETA)

N = 10_000
SEED = 0

rng = np.random.default_rng(SEED)
iti = np.array([sample_iti(rng) for _ in range(N)])
spike = iti == SPIKE_VALUE

print(f"n = {N}, seed = {SEED}")
print(f"  mean   {iti.mean():.4f}   (target {TARGET_MEAN})")
print(f"  sd     {iti.std():.4f}")
print(f"  median {np.median(iti):.4f}")
print(f"  range  [{iti.min():.4f}, {iti.max():.4f}]   (support [{ITI_MIN}, {ITI_MAX}])")
print(f"  at spike {spike.sum()} = {spike.mean():.4f}   (target {SPIKE_WEIGHT})")
print(f"  Beta(alpha={_ALPHA:.3f}, beta={_BETA:.3f}), "
      f"{'mode at the lower bound' if _ALPHA < 1 else 'interior mode'}")
print(f"  quantiles " + "  ".join(
    f"q{q:g}={np.quantile(iti, q):.2f}" for q in (0.05, 0.25, 0.5, 0.75, 0.95)))

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(iti, bins=np.linspace(ITI_MIN, ITI_MAX, 61), color='#0072B2', edgecolor='none')
ax.axvline(iti.mean(), color='#D55E00', lw=2,
           label=f"mean {iti.mean():.2f} s (target {TARGET_MEAN})")
ax.annotate(f"{spike.sum()} draws at {SPIKE_VALUE} s\n(forced-PRIME timeout)",
            xy=(SPIKE_VALUE, spike.sum()), xytext=(-12, -6),
            textcoords='offset points', ha='right', va='top', fontsize=8, color='#555555')
ax.set(xlabel='ITI (s)', ylabel=f'count of {N} draws',
       title='Predetermined-trial ITI distribution (util/iti.py)')
ax.grid(alpha=0.25)
ax.set_axisbelow(True)
ax.legend(frameon=False)
fig.tight_layout()

out = Path(__file__).resolve().parent / 'test_iti_generator.png'
fig.savefig(out, dpi=150)
print(f"\nsaved {out}")
plt.show()
