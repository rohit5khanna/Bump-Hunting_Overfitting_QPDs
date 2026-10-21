"""
Paper Table 3 -- Hartigan and Silverman unimodality rejection rates on the
Monte Carlo reference distributions (Johnson SU / SL / SB at N = 50 and 200).
Output: tables/Table03_mc_modality_tests.csv (per-replicate p-values in
tables/mc_modality_tests_raw.csv).

REPLICATE STREAM. mc_tables_rebuild.draw, i.e. dist.rvs(size=n,
random_state=42 + n*1000 + rep) through jpse's legacy inverse-CDF path -- the
same stream every other Monte Carlo artifact in reproduction/ uses, so
replication `rep` here is replication `rep` everywhere else.

WHAT IS BEING TESTED. All three references are UNIMODAL by construction, so H0
is true in every cell and these rates are the tests' realised size. Expect them
near or below the nominal 0.05: Hall & York (2001) measure the uncalibrated
Silverman test's true level at 0.010 against a nominal 0.05, so values well
under 5% are the documented behaviour, not a defect.

Silverman here is k = 1 only. The nested k = 2, 3 question does not arise for a
unimodal reference.

Usage
  python3 mc_modality_tests.py                 # all 6 cells, 1000 reps each
  python3 mc_modality_tests.py --n-reps 50     # a short look
  python3 mc_modality_tests.py --dists su --sizes 200
Resumable: each completed chunk is appended; rerun to continue. Refuses to
resume across a change of Silverman implementation.
"""
import argparse, multiprocessing, os, sys, time
from pathlib import Path

if multiprocessing.get_start_method(allow_none=True) != 'fork':
    multiprocessing.set_start_method('fork', force=True)

import numpy as np, pandas as pd

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / 'code' / 'common'))

from stats_tests import apply_silverman_test, apply_hartigan_dip_test, SILVERMAN_IMPL
from mc_tables_rebuild import DISTS, DIST_PARAMS, mc_seed, draw, N_REPS

OUT = HERE.parent / 'tables'
ALPHA, K = 0.05, 1
SIZES = [50, 200]


def _chunk(args):
    key, n, r0, nr, n_inner = args
    label, cls, _support = DISTS[key]
    dist = cls(**DIST_PARAMS)
    rows = []
    for rep in range(r0, r0 + nr):
        x = draw(dist, n, rep)
        # apply_silverman_test's inner bootstrap draws on numpy's legacy global
        # state; seed per cell so any single replicate is reproducible alone.
        np.random.seed((mc_seed(n, rep) * 7 + K) % (2 ** 32 - 1))
        rows.append(dict(dist=key, label=label, n=n, rep=rep,
                         hartigan_p=apply_hartigan_dip_test(x),
                         silverman_p=apply_silverman_test(x, n_bootstrap=n_inner, k=K)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-reps', type=int, default=N_REPS)
    ap.add_argument('--n_inner', type=int, default=199)   # the MC/bootstrap convention
    ap.add_argument('--dists', default=','.join(DISTS))
    ap.add_argument('--sizes', default=','.join(str(s) for s in SIZES))
    ap.add_argument('--chunk_size', type=int, default=25)
    ap.add_argument('--n_workers', type=int, default=max(1, multiprocessing.cpu_count()))
    a = ap.parse_args()

    dists = [d.strip() for d in a.dists.split(',')]
    sizes = [int(s) for s in a.sizes.split(',')]
    for d in dists:
        assert d in DISTS, f'{d} is not one of {list(DISTS)}'

    part = OUT / 'mc_modality_tests_partials.csv'
    print(f'Monte Carlo modality tests | dists={dists} sizes={sizes} '
          f'n_reps={a.n_reps} n_inner={a.n_inner} workers={a.n_workers}', flush=True)
    print(f'impl={SILVERMAN_IMPL}', flush=True)

    done = set()
    if part.exists():
        d = pd.read_csv(part)
        if 'n_inner' in d and set(d.n_inner.unique()) - {a.n_inner}:
            raise SystemExit(f'{part.name} holds a different n_inner; delete it or match it.')
        if 'impl' in d and set(d.impl.dropna().unique()) - {SILVERMAN_IMPL}:
            raise SystemExit(
                f'{part.name} was written by a DIFFERENT Silverman implementation\n'
                f'  file: {sorted(set(d.impl.dropna().unique()))}\n'
                f'  now : {SILVERMAN_IMPL}\n'
                'Resuming would mix two tests in one result. Move the file aside and rerun.')
        done = set(zip(d.dist, d.n, d.rep))
        print(f'resuming: {len(done)} replicates already done', flush=True)

    tasks = []
    for key in dists:
        for n in sizes:
            for r0 in range(0, a.n_reps, a.chunk_size):
                nr = min(a.chunk_size, a.n_reps - r0)
                if all((key, n, r) in done for r in range(r0, r0 + nr)):
                    continue
                tasks.append((key, n, r0, nr, a.n_inner))
    print(f'{len(tasks)} chunks to run', flush=True)

    t0 = time.time()
    if tasks:
        with multiprocessing.Pool(a.n_workers) as pool:
            for i, rows in enumerate(pool.imap_unordered(_chunk, tasks, chunksize=1), 1):
                df = pd.DataFrame(rows)
                df['n_inner'] = a.n_inner
                df['impl'] = SILVERMAN_IMPL
                df.to_csv(part, mode='a', header=not part.exists(), index=False)
                el = time.time() - t0
                print(f'  chunk {i}/{len(tasks)}  elapsed={el/60:.1f}min  '
                      f'eta={(len(tasks)-i)*el/i/60:.1f}min', flush=True)

    d = pd.read_csv(part).drop_duplicates(['dist', 'n', 'rep']).sort_values(['dist', 'n', 'rep'])
    d.to_csv(OUT / 'mc_modality_tests_raw.csv', index=False)

    rows = []
    for (key, n), g in d.groupby(['dist', 'n']):
        rows.append(dict(
            distribution=DISTS[key][0], n=n, n_reps=len(g),
            hartigan_reject=round(float((g.hartigan_p < ALPHA).mean()), 4),
            hartigan_p_mean=round(float(g.hartigan_p.mean()), 5),
            silverman_reject=round(float((g.silverman_p < ALPHA).mean()), 4),
            silverman_p_median=round(float(g.silverman_p.median()), 4),
            n_silverman_failed=int(g.silverman_p.isna().sum())))
    s = pd.DataFrame(rows).sort_values(['distribution', 'n'])
    s.to_csv(OUT / 'Table03_mc_modality_tests.csv', index=False)
    print('\n' + s.to_string(index=False))
    print('\nAll three references are unimodal, so these are realised sizes '
          'against a nominal 0.05.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
