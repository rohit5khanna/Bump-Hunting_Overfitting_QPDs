"""
Geyser -- the single bootstrap sweep that Table 13 is built from
(geyser_tables.py, geyser_mode_iqr.py). geyser_modality_paper_replicates.py
(Table 12) checks its replicate seeds against this file.

K = 4..13 x {Log Metalog, Log QFlex-U, Log QFlex-TA+} x 1000 jittered replicates.

ONE sweep. Table 13's summary columns, its mode-IQR columns and the
mode-dispersion detail (geyser_mode_dispersion.csv, not a paper table) all
derive from it, so they are consistent by construction: the "valid fits" and
"bimodal fits" of the mode-dispersion slice ARE the summary's feasibility and
modality counts, not a separate experiment that happens to agree.

Every fit is recorded, feasible or not, with its mode count, mode locations
and heights, and both W1 forms. Downstream scripts only read this file.

Conventions come from geyser_common: 0.5-min jitter applied, Weibull
plotting positions, 1000-point grid,
Equation 6 W1 at the sample points, BOTH ways: *_own vs the jittered resample
the fit was made to, *_orig vs the ORIGINAL sample; w1_raw / w1_eq6 = the one
eval_framework.BOOTSTRAP_W1_REFERENCE selects.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')          # must precede numpy

import sys, json, time, datetime, warnings, argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'code', 'common'))
sys.path.insert(0, HERE)

from geyser_common import (load_geyser, resample, replicate_seed, P_GRID, N_BOOT,
                           JITTER, LOWER_BOUND, K_VALUES, SPECS,
                           empirical_qf, interdecile, w1, data_fingerprint)
from metalog.metalog_v2 import LogMetalog
from qflex.transforms import LogQFlex
from qflex.constraints import ConstraintType
import eval_framework as EF

TAB = os.path.join(HERE, '..', 'tables')
LOG = os.path.join(HERE, '..', 'logs')
for d in (TAB, LOG):
    os.makedirs(d, exist_ok=True)

# Log QFlex-U is unconstrained (ConstraintType.NONE), as in mc_tables_rebuild
# and fish_summary.
_CT = {'qflex_u': ConstraintType.NONE, 'qflex_ta': ConstraintType.TA}

_X = _P = _EQF = _L = None


def _init():
    global _X, _P, _EQF, _L
    _X, p = load_geyser()
    _P = np.arange(1, len(_X) + 1) / (len(_X) + 1)
    _EQF = empirical_qf(_X, p)
    _L = interdecile(_X, p)


def _fit(key, s, K):
    try:
        m = (LogMetalog(s, _P, lower_bound=LOWER_BOUND, terms=K) if key == 'metalog'
             else LogQFlex(s, _P, lower_bound=LOWER_BOUND, terms=K,
                           constraint_type=_CT[key]))
        return m if EF.is_valid(m) else None
    except Exception:
        return None


def run_block(args):
    lo, hi = args
    if _X is None:
        _init()
    rows = []
    for b in range(lo, hi):
        s = np.sort(resample(_X, JITTER, b))
        for lab, key in SPECS:
            for K in K_VALUES:
                r = dict(model=lab, K=K, b=b, seed=replicate_seed(JITTER, b),
                         feasible=False, n_modes=np.nan,
                         w1_raw=np.nan, w1_eq6=np.nan,
                         w1_own_raw=np.nan, w1_own_eq6=np.nan,
                         w1_orig_raw=np.nan, w1_orig_eq6=np.nan,
                         mode_locs='', mode_heights='')
                m = _fit(key, s, K)
                if m is not None:
                    try:
                        nm, locs, hgts = EF.modes(m)
                        ww = EF.w1_both(m, s, _X)
                        r.update(feasible=True, n_modes=(int(nm) if nm is not None and nm >= 0 else np.nan),
                                 w1_raw=ww['main'][0], w1_eq6=ww['main'][1],
                                 w1_own_raw=ww['own'][0], w1_own_eq6=ww['own'][1],
                                 w1_orig_raw=ww['orig'][0], w1_orig_eq6=ww['orig'][1],
                                 mode_locs=';'.join(f'{v:.6f}' for v in (locs if locs is not None else [])),
                                 mode_heights=';'.join(f'{v:.8f}' for v in (hgts if hgts is not None else [])))
                    except Exception:
                        pass
                rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    ap.add_argument('--procs', type=int, default=4)
    a = ap.parse_args()

    t0 = time.time()
    n_fits = a.n_boot * len(SPECS) * len(K_VALUES)
    print(f'Geyser sweep: {a.n_boot} replicates x {len(SPECS)} models x '
          f'{len(K_VALUES)} orders = {n_fits:,} fits (jitter +/-{JITTER} min)',
          flush=True)

    edges = np.linspace(0, a.n_boot, a.procs * 4 + 1).astype(int)
    blocks = [(int(edges[i]), int(edges[i + 1])) for i in range(len(edges) - 1)
              if edges[i + 1] > edges[i]]
    rows = []
    with mp.Pool(a.procs, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(run_block, blocks), 1):
            rows.extend(r)
            print(f'  block {i}/{len(blocks)}  {time.time()-t0:6.1f}s', flush=True)

    df = pd.DataFrame(rows).sort_values(['model', 'K', 'b']).reset_index(drop=True)
    out = os.path.join(TAB, 'geyser_sweep.csv')
    df.to_csv(out, index=False)
    print(f'\nSaved -> {out}  ({len(df):,} rows)', flush=True)

    feas = df.groupby(['model', 'K'])['feasible'].mean().mul(100).round(1)
    print('\n% feasible')
    print(feas.unstack(0).to_string())

    json.dump({
        'generated': datetime.datetime.now().isoformat(timespec='seconds'),
        'data_file': data_fingerprint()[0],
        'data_N': data_fingerprint()[1],
        'data_sha256': data_fingerprint()[2],
        'n_bootstrap': a.n_boot, 'jitter_half_width_min': JITTER,
        'K_values': K_VALUES, 'models': [s[0] for s in SPECS],
        'seeding': 'geyser_common.resample -- '
                   'default_rng(42 + 20_000_000 + round(jitter*100)*10_000 + b)',
        'w1': 'Equation 6 at the sample points (eval_framework.w1), no interpolation; '
              '_own vs the resample, _orig vs the original sample',
        'w1_main_reference': EF.BOOTSTRAP_W1_REFERENCE,
        'mode_grid': 'eval_framework.GRID = 0.001..0.999',
        'runtime_sec': round(time.time() - t0, 1),
    }, open(os.path.join(LOG, 'geyser_sweep_provenance.json'), 'w'), indent=2)
    print(f'Total {time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
