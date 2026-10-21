#!/usr/bin/env python3
"""
Table 5, Hartigan part -- Hartigan dip test rejection rate under bootstrap
resampling of the Johnson SU sample (Section 4.5).
Standalone: run it in a terminal, no project imports needed.

    pip install diptest numpy          # one-time
    python3 bootstrap_hartigan.py

WHAT IT DOES
    Reads the pinned original realization (Johnson SU, N=200, seed 200110),
    draws the same 1000 bootstrap resamples as bootstrap_section.bootstrap_resample
    (used by bootstrap_sweep.py and bootstrap_silverman.py), applies
    Hartigan's dip test to each, and reports the rejection rate at alpha=0.05.
    Writes tables/Table05_hartigan_pvals_seed200110.csv. Takes well under a
    minute.

WHY THE PINNED CSV AND NOT A FRESH DRAW
    The realization must come from jpse's JohnsonSU (legacy np.random.seed +
    inverse-CDF). scipy.stats.johnsonsu.rvs with a default_rng gives a
    DIFFERENT sample from the same seed. Reading the pinned CSV avoids needing
    jpse installed and avoids that mistake.

WHY UNSORTED
    Resampling draws rng.choice(x_orig, ...) from the sample in GENERATION
    order. choice() picks indices, so sorting first would silently produce a
    different set of resamples.
"""
import os
import numpy as np

try:
    import diptest
except ImportError:
    raise SystemExit('Missing dependency. Run:  pip install diptest')

HERE  = os.path.dirname(os.path.abspath(__file__))
SEED  = 200110
N_BOOT = 1000
ALPHA = 0.05
X_UNSORTED = os.path.join(HERE, '..', 'data', f'su_n200_seed{SEED}_x_orig_unsorted.csv')

x_orig = np.loadtxt(X_UNSORTED, skiprows=2)          # generation order
print(f'realization: Johnson SU N={len(x_orig)}, seed {SEED}')
print(f'  range [{x_orig.min():.4f}, {x_orig.max():.4f}]  mean {x_orig.mean():.4f}')

pvals = np.empty(N_BOOT)
for b in range(N_BOOT):
    rng = np.random.default_rng(SEED * 10_000 + b)
    x_boot = rng.choice(x_orig, size=len(x_orig), replace=True)
    pvals[b] = diptest.diptest(x_boot)[1]
    if (b + 1) % 200 == 0:
        print(f'  {b+1}/{N_BOOT}', flush=True)

rate = 100.0 * (pvals < ALPHA).mean()
out = os.path.join(HERE, '..', 'tables', f'Table05_hartigan_pvals_seed{SEED}.csv')
os.makedirs(os.path.dirname(out), exist_ok=True)
np.savetxt(out, pvals, fmt='%.10g', header='hartigan_pval', comments='')

print()
print(f'Hartigan dip test, {N_BOOT} bootstrap resamples, alpha={ALPHA}')
print(f'  rejection rate : {rate:.1f}%   ({int((pvals < ALPHA).sum())}/{N_BOOT})')
print(f'  median p       : {np.median(pvals):.4f}')
print(f'  min p          : {pvals.min():.4f}')
print(f'  p-values saved : {out}')
print()
print('This rate is the Hartigan result reported in Table 5.')
