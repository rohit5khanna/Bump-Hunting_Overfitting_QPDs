"""
Bootstrap diagnostics (Section 4.5) -- the sweep behind Figures 7 and 8,
Figures O5-O7 (Online Supplement Section O3), Table 6 (via bootstrap_summary.py)
and the valid-unimodal shifted-modes figure (not in the paper).

One pinned realization: Johnson SU, N = 200, seed 200110 (see
bootstrap_section.py, which owns the sample and the resampling rule). 1000
bootstrap resamples, fitted with Metalog, QFlex-U, QFlex-A+, QFlex-TA+ and
QFlex-T+ at K = 3..13. (Table 6 reports the first four; QFlex-T+ appears in
Figure O6.)

Recorded per fit: validity, mode count, mode locations/heights, and W1, all
on the paper's evaluation framework (code/common/eval_framework.py: grid
0.001..0.999, validity = Q increasing on it, density from quantile
differences, interior peaks with relative prominence >= 1% (of the peak's own height), W1 =
Equation 6 at the sample points, stored BOTH ways: w1_own* against the
resample the fit was made to, w1_orig* against the ORIGINAL realization;
w1 / w1_norm = the one EF.BOOTSTRAP_W1_REFERENCE selects). Curves are not stored; the
shifted-modes figure refits the handful of replicates it needs, so a figure
cannot drift from the sweep it came from.

"Persist" (tables/bootstrap_persist.csv, carried into Table 6 as Persist_pct)
is the share of bootstrap fits whose mode count equals the ORIGINAL
realization's at that K.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')          # must precede numpy

import sys, json, time, datetime, warnings, argparse
import multiprocessing as mp
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'code', 'common')); sys.path.insert(0, HERE)

from bootstrap_section import (load_original_sample, load_original_sample_unsorted,
                               bootstrap_resample, SEED, N, DIST_PARAMS)
from metalog.metalog_v2 import Metalog
from qflex.core import QFlex
from qflex.constraints import ConstraintType
import eval_framework as EF

TAB = os.path.join(HERE, '..', 'tables'); LOG = os.path.join(HERE, '..', 'logs')
for d in (TAB, LOG): os.makedirs(d, exist_ok=True)

N_BOOT = 1000
K_ALL = list(range(3, 14))   # 3..13; the paper reports K = 4, 7, 10, 13
SPECS = [('Metalog', 'metalog', None),
         ('QFlex-U', 'qflex', ConstraintType.NONE),
         ('QFlex-A+', 'qflex', ConstraintType.A),
         ('QFlex-TA+', 'qflex', ConstraintType.TA),
         ('QFlex-T+', 'qflex', ConstraintType.TL)]

_XS = _XU = _P = None


def _init():
    global _XS, _XU, _P
    _XS = load_original_sample()
    _XU = load_original_sample_unsorted()
    _P = np.arange(1, N + 1) / (N + 1)


def _fit(kind, ct, x, K):
    try:
        m = (Metalog(x, _P, terms=K) if kind == 'metalog'
             else QFlex(x, _P, terms=K, constraint_type=ct))
        return m if EF.is_valid(m) else None
    except Exception:
        return None


def _row(lab, K, b, m, s):
    """s = the sample m was fitted to (the resample; the original for b = -1)."""
    r = dict(model=lab, K=K, b=b, feasible=False, n_modes=np.nan, n_peaks=np.nan,
             w1=np.nan, w1_norm=np.nan, w1_own=np.nan, w1_own_norm=np.nan,
             w1_orig=np.nan, w1_orig_norm=np.nan, mode_locs='', mode_heights='')
    if m is None:
        return r
    try:
        nm, locs, hgts = EF.modes(m)
        ww = EF.w1_both(m, s, _XS)
        r.update(feasible=True,
                 n_modes=EF.n_modes(nm), n_peaks=int(nm),     # n_modes = max(1, peaks)
                 w1=ww['main'][0], w1_norm=ww['main'][1],
                 w1_own=ww['own'][0], w1_own_norm=ww['own'][1],
                 w1_orig=ww['orig'][0], w1_orig_norm=ww['orig'][1],
                 mode_locs=';'.join(f'{v:.6f}' for v in (locs if locs is not None else [])),
                 mode_heights=';'.join(f'{v:.8f}' for v in (hgts if hgts is not None else [])))
    except Exception:
        pass
    return r


def run_block(args):
    lo, hi = args
    if _XS is None:
        _init()
    rows = []
    for b in range(lo, hi):
        s = np.sort(bootstrap_resample(_XU, b))
        for lab, kind, ct in SPECS:
            for K in K_ALL:
                rows.append(_row(lab, K, b, _fit(kind, ct, s, K), s))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    ap.add_argument('--procs', type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    print(f'Bootstrap sweep: Johnson SU N={N} seed={SEED}, {a.n_boot} resamples x '
          f'{len(SPECS)} models x {len(K_ALL)} orders = '
          f'{a.n_boot*len(SPECS)*len(K_ALL):,} fits', flush=True)

    _init()
    # the ORIGINAL realization's own fits -- what "Persist" is measured against
    orig = []
    for lab, kind, ct in SPECS:
        for K in K_ALL:
            r = _row(lab, K, -1, _fit(kind, ct, _XS, K), _XS)
            orig.append(r)
    pd.DataFrame(orig).to_csv(os.path.join(TAB, 'bootstrap_original_fits.csv'), index=False)

    edges = np.linspace(0, a.n_boot, a.procs * 4 + 1).astype(int)
    blocks = [(int(edges[i]), int(edges[i+1])) for i in range(len(edges)-1)
              if edges[i+1] > edges[i]]
    rows = []
    with mp.Pool(a.procs, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(run_block, blocks), 1):
            rows.extend(r)
            print(f'  block {i}/{len(blocks)}  {time.time()-t0:6.1f}s', flush=True)
    df = pd.DataFrame(rows).sort_values(['model', 'K', 'b']).reset_index(drop=True)
    out = os.path.join(TAB, 'bootstrap_sweep.csv')
    df.to_csv(out, index=False)
    print(f'\nSaved -> {out}  ({len(df):,} rows)', flush=True)

    od = pd.DataFrame(orig).set_index(['model', 'K'])['n_modes'].to_dict()
    print('\nPersist %  (share of bootstrap fits whose mode count equals the '
          'original realization\'s)')
    print(f'{"model":<11}' + ''.join(f'{"K="+str(K):>9}' for K in K_ALL))
    persist = []
    for lab, _k, _c in SPECS:
        line = f'{lab:<11}'
        for K in K_ALL:
            g = df[(df['model'] == lab) & (df['K'] == K) & df['feasible'] & df['n_modes'].notna()]
            o = od.get((lab, K), np.nan)
            pv = 100 * float(np.mean(g['n_modes'] == o)) if len(g) and not pd.isna(o) else np.nan
            persist.append(dict(model=lab, K=K, original_n_modes=o,
                                n_valid=len(g), persist_pct=round(pv, 1) if pv == pv else np.nan,
                                max_modes=int(g['n_modes'].max()) if len(g) else np.nan))
            line += f'{pv:>9.1f}' if pv == pv else f'{"-":>9}'
        print(line)
    pd.DataFrame(persist).to_csv(os.path.join(TAB, 'bootstrap_persist.csv'), index=False)
    print('\n(Persist_pct is carried into Table 6 by bootstrap_summary.py)')

    json.dump({
        'generated': datetime.datetime.now().isoformat(timespec='seconds'),
        'realization': f'Johnson SU N={N}, seed={SEED}, params {DIST_PARAMS}',
        'sample': f'reproduction/data/su_n200_seed{SEED}_x_orig*.csv (checksummed)',
        'n_bootstrap': a.n_boot, 'K_values': K_ALL,
        'models': [s[0] for s in SPECS],
        'resampling': 'bootstrap_section.bootstrap_resample -- default_rng(SEED*10_000 + b)',
        'framework': 'code/common/eval_framework.py (grid 0.001..0.999 for validity/modes; '
                     'W1 at the sample points, both vs the resample and vs the original)',
        'w1_main_reference': EF.BOOTSTRAP_W1_REFERENCE,
        'persist_definition': "share of valid bootstrap fits whose mode count equals "
                              "the ORIGINAL realization's at that K",
        'runtime_sec': round(time.time() - t0, 1),
    }, open(os.path.join(LOG, 'bootstrap_sweep_provenance.json'), 'w'), indent=2)
    print(f'Total {time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
