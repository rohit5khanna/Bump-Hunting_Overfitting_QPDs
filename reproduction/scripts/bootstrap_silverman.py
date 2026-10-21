"""
Silverman critical-bandwidth test on the BOOTSTRAP Johnson SU experiment
(Table 5, Section 4.5), at H0: unimodal (k = 1).

Table 5 reports a Hartigan result and a Silverman result. bootstrap_hartigan.py
computes the Hartigan half on the section's replicates; this script computes
the Silverman half.

REPLICATE STREAM. The outer sample is bootstrap_section.bootstrap_resample(
x_orig_unsorted, b) -- rng = default_rng(SEED*10_000 + b), drawn from the
GENERATION-ORDER array. That is the identical stream used by bootstrap_sweep.py
for the QPD fits behind Table 6 and Figures 7 and 8 and by bootstrap_hartigan.py
for the Hartigan result, so every number in the section describes the same
1000 samples.

SEED comes from bootstrap_section and is never written in this file.

INNER BOOTSTRAP. apply_silverman_test's smoothed bootstrap draws from numpy's
legacy global state, which _chunk seeds per replicate with
np.random.seed((SEED*10_000 + b) mod (2^32 - 1)). The outer resample has its
own default_rng, so consuming the inner stream cannot perturb the outer
sample, and any single replicate's p-value is reproducible on its own without
replaying the sweep.

No jitter: Johnson SU is continuous and the realization has no ties, so the
tie-breaking jitter the fish and geyser sections need does not arise here.

Hartigan is recomputed alongside at no meaningful cost. It is a check, not a
new result: its rejection rate must match Table05_hartigan_pvals_seed{SEED}.csv.

Usage
  python3 bootstrap_silverman.py                  # 1000 replicates, k = 1
  python3 bootstrap_silverman.py --n_boot 50      # a short look
Resumable: each completed chunk is appended at once; rerun to continue.
Writes tables/Table05_silverman_seed{SEED}_summary.csv (plus _raw and
_partials CSVs).
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
from bootstrap_section import (SEED, N, N_BOOTSTRAP,
                               load_original_sample_unsorted, bootstrap_resample)
from stats_tests import apply_silverman_test, apply_hartigan_dip_test, SILVERMAN_IMPL

# IMPLEMENTATION. apply_silverman_test (code/common/stats_tests.py): a
# 2000-point KDE grid, modes = peaks with prominence >= 1% of the tallest
# (KDE_PROMINENCE_FRAC), critical bandwidth by bisection from [0.01*h0, 5*h0]
# with an expanding upper bracket, null = the KDE at h_crit sampled by the
# variance-rescaled smoothed bootstrap. It is the one the Monte Carlo
# (Table 3), bootstrap and hydrology sections use, and the single
# implementation this repo keeps. Its p-values are RAW: no Hall & York
# calibration is applied.

OUT = HERE.parent / 'tables'
# k = 1 only: Table 5 reports the Silverman test at H0: unimodal.
ALPHA, K = 0.05, 1


def _chunk(args):
    xu, b0, nb, n_inner = args
    rows = []
    for b in range(b0, b0 + nb):
        s = np.sort(bootstrap_resample(xu, b))
        # apply_silverman_test's smoothed bootstrap goes through scipy's
        # gaussian_kde.resample, which draws on numpy's legacy global state.
        # Seed it per replicate so each cell is reproducible on its own.
        np.random.seed((SEED * 10_000 + b) % (2 ** 32 - 1))
        rows.append(dict(b=b,
                         hartigan_p=apply_hartigan_dip_test(s),
                         silverman_p=apply_silverman_test(s, n_bootstrap=n_inner, k=K)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_boot', type=int, default=N_BOOTSTRAP)
    ap.add_argument('--n_inner', type=int, default=199)   # the MC/bootstrap convention
    ap.add_argument('--chunk_size', type=int, default=25)
    ap.add_argument('--n_workers', type=int, default=max(1, multiprocessing.cpu_count()))
    a = ap.parse_args()

    part = OUT / f'Table05_silverman_seed{SEED}_partials.csv'
    xu = load_original_sample_unsorted()
    print(f'Johnson SU N={N} seed={SEED} | n_boot={a.n_boot} n_inner={a.n_inner} '
          f'workers={a.n_workers}', flush=True)

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
        done = set(d.b.astype(int))
        print(f'resuming: {len(done)} replicates already done', flush=True)

    tasks = []
    for b0 in range(0, a.n_boot, a.chunk_size):
        nb = min(a.chunk_size, a.n_boot - b0)
        if all(b in done for b in range(b0, b0 + nb)):
            continue
        tasks.append((xu, b0, nb, a.n_inner))
    print(f'{len(tasks)} chunks to run', flush=True)

    t0 = time.time()
    if tasks:
        with multiprocessing.Pool(a.n_workers) as pool:
            for i, rows in enumerate(pool.imap_unordered(_chunk, tasks, chunksize=1), 1):
                df = pd.DataFrame(rows); df['n_inner'] = a.n_inner; df['impl'] = SILVERMAN_IMPL
                df.to_csv(part, mode='a', header=not part.exists(), index=False)
                el = time.time() - t0
                print(f'  chunk {i}/{len(tasks)}  elapsed={el/60:.1f}min  '
                      f'eta={(len(tasks)-i)*el/i/60:.1f}min', flush=True)

    d = pd.read_csv(part).drop_duplicates('b').sort_values('b')
    d.to_csv(OUT / f'Table05_silverman_seed{SEED}_raw.csv', index=False)
    summ = pd.DataFrame([dict(
        seed=SEED, n=len(d), k=K, n_inner=int(d.n_inner.iloc[0]),
        silverman_reject=float((d.silverman_p < ALPHA).mean() * 100),
        silverman_p_median=float(d.silverman_p.median()),
        hartigan_reject=float((d.hartigan_p < ALPHA).mean() * 100),
        n_silverman_failed=int(d.silverman_p.isna().sum()))])
    summ.to_csv(OUT / f'Table05_silverman_seed{SEED}_summary.csv', index=False)
    print('\n' + summ.T.to_string())

    ref = OUT / f'Table05_hartigan_pvals_seed{SEED}.csv'
    if ref.exists() and len(d) == a.n_boot:
        want = float((np.loadtxt(ref, skiprows=1) < ALPHA).mean() * 100)
        got = float((d.hartigan_p < ALPHA).mean() * 100)
        print(f'\nHartigan cross-check vs {ref.name}: {got:.1f}% here, {want:.1f}% there '
              f'-> {"MATCH" if abs(got-want) < 1e-9 else "DIFFER -- streams disagree"}')


if __name__ == '__main__':
    main()
