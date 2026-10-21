"""
Hydrology -- the single bootstrap sweep behind Table 8, the bootstrap band of
Figure 12, the mode scatter of Figure 13, and the median-bootstrap fits
(hydrology_figures.py; not in the paper).

1000 paired replicates x {GEV, Log Metalog K=4/7/10/13, Log QFlex-A+ K=4/7/10/13}.

ONE sweep, and every model sees the SAME replicate b (hydrology_common.resample),
so GEV and the QPDs are compared on identical bootstrap samples, and the EQF
band is computed from those same samples. See hydrology_common.

Recorded per fit: feasibility, mode count, mode locations/heights, and W1 in
both forms (feet and Equation 6), at the sample points, BOTH ways: *_own vs the
resample the fit was made to, *_orig vs the observed sample; w1_ft / w1_eq6 =
the one eval_framework.BOOTSTRAP_W1_REFERENCE selects. Curves are NOT stored -- the figures refit
the handful of replicates they need, which keeps this file small and means a
figure can never drift from the sweep it claims to come from.

Also written: the pointwise 95% EQF band, computed from the SAME replicates.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')          # must precede numpy

import sys, json, time, datetime, warnings, argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
from scipy.stats import genextreme
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'code', 'common'))
sys.path.insert(0, HERE)

from hydrology_common import (load_hydrology, resample, replicate_seed, P_GRID,
                              N_BOOT, JITTER, LOWER_BOUND, SPECS,
                              empirical_qf, interdecile, w1, data_fingerprint)
from metalog.metalog_v2 import LogMetalog
from qflex.transforms import LogQFlex
from qflex.constraints import ConstraintType
import eval_framework as EF

TAB = os.path.join(HERE, '..', 'tables')
LOG = os.path.join(HERE, '..', 'logs')
import tplus_mode as TP                 # QPD_TPLUS_ONLY=1: supplementary QFlex-T+ run
TAB, LOG = TP.table_dir(TAB), TP.log_dir(LOG)
for d in (TAB, LOG):
    os.makedirs(d, exist_ok=True)

_X = _P = _EQF = _L = None


def _init():
    global _X, _P, _EQF, _L
    _X, p = load_hydrology()
    _P = np.arange(1, len(_X) + 1) / (len(_X) + 1)
    _EQF = empirical_qf(_X, p)
    _L = interdecile(_X, p)


def fit_curve(key, s, K):
    """(Q, density, Q-function) -- Q and density on the evaluation grid, the
    Q-function for W1 at the sample points -- or None if the fit is invalid.
    The density is ALWAYS the grid difference of Q (eval_framework), for GEV
    as well as the QPDs, so every model is measured the same way."""
    try:
        if key == 'gev':
            c, loc, scale = genextreme.fit(s, method='MLE')
            qf = genextreme.ppf(P_GRID, c, loc=loc, scale=scale)
            qf = np.asarray(qf, float)
            if not (np.all(np.isfinite(qf)) and np.all(np.diff(qf) > 0)):
                return None
            qfun = lambda pp, c=c, loc=loc, scale=scale: genextreme.ppf(pp, c, loc=loc, scale=scale)
            return qf, EF.density_from_quantile(qf), qfun
        m = (LogMetalog(s, _P, lower_bound=LOWER_BOUND, terms=K) if key == 'metalog'
             else LogQFlex(s, _P, lower_bound=LOWER_BOUND, terms=K,
                           constraint_type=(ConstraintType.TL if key == 'qflex_tl'
                                            else ConstraintType.A)))
        if not EF.is_valid(m):
            return None
        return EF.density(m) + (m,)
    except Exception:
        return None


ALL_SPECS = ([] if TP.TPLUS_ONLY else [('GEV', 'gev', 0)]) + list(SPECS)


def run_block(args):
    lo, hi = args
    if _X is None:
        _init()
    rows, bands = [], []
    for b in range(lo, hi):
        s = np.sort(resample(_X, b=b))
        bands.append(np.interp(P_GRID, _P, s))
        for lab, key, K in ALL_SPECS:
            r = dict(model=lab, K=K, b=b, seed=replicate_seed(b=b), feasible=False,
                     n_modes=np.nan, n_peaks=np.nan, w1_ft=np.nan, w1_eq6=np.nan,
                     w1_own_ft=np.nan, w1_own_eq6=np.nan, w1_orig_ft=np.nan, w1_orig_eq6=np.nan,
                     mode_locs='', mode_heights='')
            out = fit_curve(key, s, K)
            if out is not None:
                qf, pdf, qfun = out
                try:
                    nm, locs, hgts = EF.modes_from_density(qf, pdf)
                except Exception:
                    nm, locs, hgts = None, None, None
                ww = EF.w1_both(qfun, s, _X)
                r.update(feasible=True,
                         n_modes=EF.n_modes(nm), n_peaks=(int(nm) if nm is not None else np.nan),
                         w1_ft=ww['main'][0], w1_eq6=ww['main'][1],
                         w1_own_ft=ww['own'][0], w1_own_eq6=ww['own'][1],
                         w1_orig_ft=ww['orig'][0], w1_orig_eq6=ww['orig'][1],
                         mode_locs=';'.join(f'{v:.6f}' for v in (locs if locs is not None else [])),
                         mode_heights=';'.join(f'{v:.8f}' for v in (hgts if hgts is not None else [])))
            rows.append(r)
    return rows, np.asarray(bands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    ap.add_argument('--procs', type=int, default=4)
    a = ap.parse_args()

    t0 = time.time()
    print(f'Hydrology sweep: {a.n_boot} replicates x {len(ALL_SPECS)} models '
          f'= {a.n_boot*len(ALL_SPECS):,} fits (no jitter -- continuous data)',
          flush=True)

    edges = np.linspace(0, a.n_boot, a.procs * 4 + 1).astype(int)
    blocks = [(int(edges[i]), int(edges[i+1])) for i in range(len(edges)-1)
              if edges[i+1] > edges[i]]
    rows, bands = [], []
    with mp.Pool(a.procs, initializer=_init) as pool:
        for i, (r, bd) in enumerate(pool.imap_unordered(run_block, blocks), 1):
            rows.extend(r); bands.append(bd)
            print(f'  block {i}/{len(blocks)}  {time.time()-t0:6.1f}s', flush=True)

    df = pd.DataFrame(rows).sort_values(['model', 'K', 'b']).reset_index(drop=True)
    out = os.path.join(TAB, 'hydrology_sweep.csv')
    df.to_csv(out, index=False)
    print(f'\nSaved -> {out}  ({len(df):,} rows)', flush=True)

    M = np.vstack(bands)
    band = pd.DataFrame({'p': P_GRID,
                         'eqf_observed': _EQF if _EQF is not None else np.nan,
                         'q025': np.percentile(M, 2.5, axis=0),
                         'q975': np.percentile(M, 97.5, axis=0)})
    x, p = load_hydrology()
    band['eqf_observed'] = empirical_qf(x, p)
    band.to_csv(os.path.join(TAB, 'Figure12_hydrology_EQF_bootstrap_interval_paneldata.csv'), index=False)
    print(f'EQF band -> tables/Figure12_hydrology_EQF_bootstrap_interval_paneldata.csv '
          f'(from the same {a.n_boot} replicates)', flush=True)

    print('\n% feasible')
    print(df.groupby(['model', 'K'])['feasible'].mean().mul(100).round(1)
            .unstack(0).to_string())

    fn, n, sha = data_fingerprint()
    json.dump({
        'generated': datetime.datetime.now().isoformat(timespec='seconds'),
        'data_file': fn, 'data_N': n, 'data_sha256': sha,
        'n_bootstrap': a.n_boot, 'jitter': JITTER,
        'models': [f'{l} K={K}' if K else l for l, _k, K in ALL_SPECS],
        'seeding': 'hydrology_common.resample -- default_rng(42 + 40_000_000 + b); '
                   'GEV and every QPD see the SAME replicate b',
        'w1': 'Equation 6 at the sample points (eval_framework.w1), no interpolation; '
              '_own vs the resample, _orig vs the observed sample; divisor = the '
              'reference sample\'s interdecile',
        'w1_main_reference': EF.BOOTSTRAP_W1_REFERENCE,
        'framework': 'code/common/eval_framework.py -- validity, grid-difference density '
                     '(GEV too), interior modes',
        'runtime_sec': round(time.time() - t0, 1),
    }, open(os.path.join(LOG, 'hydrology_sweep_provenance.json'), 'w'), indent=2)
    print(f'Total {time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
