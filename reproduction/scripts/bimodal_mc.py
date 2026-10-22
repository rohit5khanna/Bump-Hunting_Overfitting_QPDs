"""
Bimodal recovery (Section 4.6) -- the Monte Carlo behind Table 7
(bimodal_summary.py) and Figure 9 (bimodal_figure.py).

1000 INDEPENDENT samples of N = 200 drawn from the analytic 60/40 Johnson SU
mixture, fitted with Metalog and QFlex-TA+ at K = 4..14. See bimodal_common
for the design, the mixture and the seeding.

Before fitting, main() calls bimodal_common.check_against_source(), which
compares the constants with the earlier sampler source (code/MC_Cloud_Bimodal)
when that file is present; it is not in this repository, so the check is
skipped.

Chunkable: a shell on the user's machine is capped at 180 s, so the run writes
per-block partials to tables/bimodal_partials_framework/ and merges once every
replicate is present. `--lo/--hi` runs a slice; `--merge` combines without
fitting. Output: tables/bimodal_mc.csv (one row per fit).
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

from bimodal_common import (draw, true_quantile, empirical_qf, interdecile_from_grid,
                            w1, P_GRID, K_VALUES, SPECS, N_REPLICATES, SAMPLE_SIZE,
                            SEED, BIMODAL_BASE, check_against_source)
from metalog.metalog_v2 import Metalog
from qflex.core import QFlex
from qflex.constraints import ConstraintType
import eval_framework as EF

TAB = os.path.join(HERE, '..', 'tables')
LOG = os.path.join(HERE, '..', 'logs')
PART = os.path.join(TAB, 'bimodal_partials_framework')   # per-block partials
for d in (TAB, LOG, PART):
    os.makedirs(d, exist_ok=True)

_TQ = _TL = None


def _init():
    global _TQ, _TL
    _TQ = true_quantile()
    _TL = interdecile_from_grid(_TQ)


def _fit(key, x, p, K):
    try:
        m = (Metalog(x, p, terms=K) if key == 'metalog'
             else QFlex(x, p, terms=K, constraint_type=ConstraintType.TA))
        return m if EF.is_valid(m) else None
    except Exception:
        return None


def run_block(args):
    lo, hi = args
    if _TQ is None:
        _init()
    rows = []
    for r in range(lo, hi):
        x = draw(r, SAMPLE_SIZE)
        p = np.arange(1, len(x) + 1) / (len(x) + 1)
        for lab, key in SPECS:
            for K in K_VALUES:
                row = dict(model=lab, K=K, replication=r,
                           seed=SEED + BIMODAL_BASE + r, feasible=False,
                           n_modes=np.nan, n_peaks=np.nan, w1_eq6_sample=np.nan,
                           w1_raw_sample=np.nan,
                           mode_locs='', mode_heights='')
                m = _fit(key, x, p, K)
                if m is not None:
                    try:
                        nm, locs, hgts = EF.modes(m)
                        raw_s, eq6_s = w1(m, x)
                        row.update(feasible=True,
                                   n_modes=EF.n_modes(nm), n_peaks=int(nm),
                                   w1_raw_sample=raw_s, w1_eq6_sample=eq6_s,
                                   mode_locs=';'.join(f'{v:.6f}' for v in (locs if locs is not None else [])),
                                   mode_heights=';'.join(f'{v:.8f}' for v in (hgts if hgts is not None else [])))
                    except Exception:
                        pass
                rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lo', type=int, default=0)
    ap.add_argument('--hi', type=int, default=N_REPLICATES)
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--merge', action='store_true')
    a = ap.parse_args()
    t0 = time.time()

    if not a.merge:
        check_against_source()
        n = (a.hi - a.lo) * len(SPECS) * len(K_VALUES)
        print(f'Bimodal MC: replicates [{a.lo}, {a.hi}) x {len(SPECS)} models x '
              f'{len(K_VALUES)} orders = {n:,} fits (N={SAMPLE_SIZE}, '
              f'independent draws)', flush=True)
        edges = np.linspace(a.lo, a.hi, a.procs * 3 + 1).astype(int)
        blocks = [(int(edges[i]), int(edges[i+1])) for i in range(len(edges)-1)
                  if edges[i+1] > edges[i]]
        rows = []
        with mp.Pool(a.procs, initializer=_init) as pool:
            for i, r in enumerate(pool.imap_unordered(run_block, blocks), 1):
                rows.extend(r)
                print(f'  block {i}/{len(blocks)}  {time.time()-t0:6.1f}s', flush=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(PART, f'bimodal_part_{a.lo:05d}_{a.hi:05d}.csv'), index=False)
        print(f'partial saved: replicates [{a.lo}, {a.hi})', flush=True)

    have = set()
    for f in os.listdir(PART):
        if f.endswith('.csv'):
            have |= set(pd.read_csv(os.path.join(PART, f), usecols=['replication'])
                        ['replication'].unique())
    left = sorted(set(range(N_REPLICATES)) - have)
    if left:
        print(f'still to run: {len(left)} replicates, next {left[0]}..{left[min(9,len(left)-1)]}',
              flush=True)
        print(f'Total {time.time()-t0:.1f}s', flush=True)
        return

    df = pd.concat([pd.read_csv(os.path.join(PART, f))
                    for f in sorted(os.listdir(PART)) if f.endswith('.csv')],
                   ignore_index=True)
    df = df.drop_duplicates(subset=['model', 'K', 'replication'], keep='last')
    df = df.sort_values(['model', 'K', 'replication']).reset_index(drop=True)
    out = os.path.join(TAB, 'bimodal_mc.csv')
    df.to_csv(out, index=False)
    print(f'\nSaved -> {out}  ({len(df):,} rows)', flush=True)
    print('\n% feasible')
    print(df.groupby(['model', 'K'])['feasible'].mean().mul(100).round(1)
            .unstack(0).to_string(), flush=True)

    json.dump({
        'generated': datetime.datetime.now().isoformat(timespec='seconds'),
        'design': 'genuine Monte Carlo -- 1000 INDEPENDENT draws of N=200 from '
                  'the analytic 60/40 Johnson SU mixture',
        'n_replicates': N_REPLICATES, 'sample_size': SAMPLE_SIZE,
        'K_values': K_VALUES, 'models': [s[0] for s in SPECS],
        'seeding': f'default_rng({SEED} + {BIMODAL_BASE} + replication)',
        'w1_sample': 'Equation 6 at the replicate\'s own sample points (eval_framework.w1; '
                     'no interpolation). No W1 against the true mixture is computed.',
        'framework': 'code/common/eval_framework.py (grid 0.001..0.999 for validity/modes)',
        'runtime_sec': round(time.time() - t0, 1),
    }, open(os.path.join(LOG, 'bimodal_mc_provenance.json'), 'w'), indent=2)
    print(f'Total {time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
