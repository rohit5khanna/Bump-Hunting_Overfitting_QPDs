"""
Figure 6 -- Empirical and fitted Metalog quantile functions (K = 10) with the
95% bootstrap interval, Johnson SU sample (seed 200110), Section 4.5.

The realization and the resamples come from bootstrap_section (SEED, the
pinned sample, bootstrap_resample), never written in here, so this figure
shows the same sample as the rest of the section.

The bootstrap band is refitted from the pinned realization: all 1000
resamples, Metalog K = 10, valid fits only, pointwise 2.5/97.5 percentiles.

Panel data (the plotted curves) is written to tables/ as CSV.

FRAMEWORK (code/common/eval_framework.py): every curve is drawn on the
evaluation grid p = 0.001..0.999 only; validity = Q increasing on that grid;
the empirical QF is drawn as the order statistics at i/(N+1), linear in
between, flat beyond (drawing only; W1 uses the order statistics themselves).
"""
import os, sys, json, datetime
import numpy as np
import figure_style as FS, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib as mpl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', 'code', 'common'))
from bootstrap_section import (SEED, N, DIST_PARAMS, load_original_sample, load_original_sample_unsorted, bootstrap_resample)
from jpse import JohnsonSU
from metalog.metalog_v2 import Metalog
import eval_framework as EF

FIG_DIR = os.path.join(HERE, '..', 'figures')
TAB_DIR = os.path.join(HERE, '..', 'tables')
LOG_DIR = os.path.join(HERE, '..', 'logs')
P_GRID  = EF.GRID
K_VALS  = [10]      # single panel: the order the section's argument rests on

mpl.rcParams.update({'font.family': 'sans-serif', 'font.size': 11,
                     'axes.titlesize': 11, 'axes.labelsize': 10,
                     'figure.dpi': 150, 'savefig.dpi': 300,
                     'axes.grid': True, 'grid.alpha': 0.3})

N_BOOT_FIG4 = 1000
x = load_original_sample()                       # sorted, pinned
x_unsorted = load_original_sample_unsorted()     # generation order, for resampling
p_emp = np.arange(1, N + 1) / (N + 1)            # Weibull, Equation 3
true_qf = np.asarray(JohnsonSU(**DIST_PARAMS).quantile(P_GRID), dtype=float)
print(f'realization seed {SEED}: range [{x.min():.4f}, {x.max():.4f}]')

full_fits = {}
for k in K_VALS:
    m = Metalog(x, p_emp, terms=k)
    full_fits[k] = m
    print(f'  Metalog K={k}: valid={EF.is_valid(m)}')

# The band is REFITTED here: bootstrap_sweep.csv stores only scalars per fit
# (W1, mode counts), not the quantile function a pointwise band needs.
# Refitting costs ~15 s.
ci = {}
for k in K_VALS:
    _curves = []
    for _b in range(N_BOOT_FIG4):
        _xb = np.sort(bootstrap_resample(x_unsorted, _b))
        try:
            _m = Metalog(_xb, p_emp, terms=k)
        except Exception:
            continue
        if EF.is_valid(_m):
            _curves.append(np.asarray(_m.quantile(P_GRID), float))
    arr = np.vstack(_curves)
    ci[k] = {'lo': np.percentile(arr, 2.5, axis=0),
             'hi': np.percentile(arr, 97.5, axis=0), 'n': len(arr)}
    print(f'  K={k}: {len(arr)} valid bootstrap resamples (refitted)')

fig, axes = plt.subplots(1, len(K_VALS), figsize=(7.5, 5.5), squeeze=False)
axes = axes.ravel()
fig.suptitle(f'Johnson SU  |  N={N}  |  Metalog Bootstrap QF Stability',
             fontsize=13, fontweight='bold', y=1.01)
fig.subplots_adjust(wspace=0.08)
for ax, k in zip(axes, K_VALS):
    ax.fill_between(P_GRID, ci[k]['lo'], ci[k]['hi'], color=FS.OTHER['band'],
                    alpha=FS.BAND_ALPHA, label='95% Bootstrap Interval')
    ax.plot(P_GRID, true_qf, color=FS.OTHER['truth'], linewidth=2.0, linestyle='--',
            label='True QF (Johnson SU)', zorder=3)
    ax.plot(P_GRID, EF.eqf(x), color='black', linewidth=1.4,
            label='Empirical QF', zorder=4)
    if EF.is_valid(full_fits[k]):
        ax.plot(P_GRID, full_fits[k].quantile(P_GRID), color=FS.QPD['Metalog'],
                linewidth=2, label=f'Metalog K={k}', zorder=5)
    ax.set_ylim(-4, 4)
    ax.set_xlabel('Probability p')
    if ax is axes[0]:
        ax.set_ylabel('Quantile (x)')
    ax.legend(loc='upper left', fontsize=8)

fig.tight_layout()
os.makedirs(FIG_DIR, exist_ok=True)
name = 'Figure06_SU_bootstrap_QF_interval'
fig.savefig(FS.figure_path(name) + '.png', dpi=300, bbox_inches='tight')
fig.savefig(FS.figure_path(name) + '.pdf', bbox_inches='tight')
plt.close(fig)

os.makedirs(TAB_DIR, exist_ok=True)
panel = {'p': P_GRID, 'true_qf': true_qf}
for k in K_VALS:
    panel[f'K{k}_ci_lo'] = ci[k]['lo']; panel[f'K{k}_ci_hi'] = ci[k]['hi']
    panel[f'K{k}_metalog_fulldata'] = full_fits[k].quantile(P_GRID)
pd.DataFrame(panel).to_csv(os.path.join(TAB_DIR, f'{name}_paneldata.csv'), index=False)
pd.DataFrame({'p_emp': p_emp, 'x_sorted': x}).to_csv(
    os.path.join(TAB_DIR, f'{name}_empiricalqf.csv'), index=False)

os.makedirs(LOG_DIR, exist_ok=True)
json.dump({'generated': datetime.datetime.now().isoformat(timespec='seconds'),
           'seed': SEED, 'N': N, 'K_values': K_VALS,
           'x_orig_range': [float(x.min()), float(x.max())],
           'n_valid_bootstrap': {str(k): ci[k]['n'] for k in K_VALS},
           'band': '2.5/97.5 percentile of valid bootstrap QFs on the 0.001..0.999 grid',
           'framework': 'code/common/eval_framework.py',
           'band_source': 'refitted from the pinned realization via bootstrap_section.bootstrap_resample'},
          open(os.path.join(LOG_DIR, f'{name}.json'), 'w'), indent=2)
print(f'  saved {name}.png/.pdf + panel data')
