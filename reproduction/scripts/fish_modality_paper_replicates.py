"""
Fish modality tests on THE PAPER'S OWN BOOTSTRAP REPLICATES -- paper Table 9
(modality tests and sequential Silverman outcomes across bootstrap resamples,
by jitter), written as Table09_fish_modality_tests.csv.

The outer sample is fish_common.resample, so replicate b here is the same
array that the other fish tables fit a QPD to (Table 10 at 0.5 lb), and the
results are comparable replicate by replicate, not just in the margin.

WHAT IS COMPUTED, per (jitter width, replicate b):
  Hartigan dip p-value, and the Silverman p-value for each requested k. The
  paper uses k = 1, 2 and 3 (run with --k 1,2,3); all three k share the one
  outer sample, so the sequential mode-count estimator can be read off
  directly (the seq_* columns, written only when k = 1, 2, 3 are all run).

SEEDING.
  outer : fish_common.resample(x, jitter, b)  -- rng per replicate,
          seed = 42 + round(jitter*100)*10_000 + b
  inner : np.random.default_rng(INNER_BASE + replicate_seed(jitter, b)), a
          SEPARATE stream, so consuming it cannot perturb the outer sample and
          any single cell is reproducible on its own. INNER_BASE = 50_000_000
          is clear of every stream fish_common defines (replicate_seed tops out
          near 1.0e6; jitter_original's STREAM_BASE is 5e6).

The test functions are imported unchanged from code/common/stats_tests.py
(apply_silverman_test, apply_hartigan_dip_test). Jitter 0 is not in the
default sweep -- see WIDTHS below.

Usage
  python3 fish_modality_paper_replicates.py --k 1,2,3                 # all positive widths
  python3 fish_modality_paper_replicates.py --k 1,2,3 --jitters 0.9,1.0
Resumable: every completed chunk is appended immediately; rerun to continue.
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
from fish_common import load_fish, resample, replicate_seed, PAPER_JITTERS, N_BOOT
from stats_tests import apply_silverman_test, apply_hartigan_dip_test, SILVERMAN_IMPL

OUT = HERE.parent / 'tables'
ALPHA = 0.05
# GRID_SIZE. apply_silverman_test defaults to 2000, which suits N = 200. At
# N = 3474 over a 30 lb range, 512 gives identical mode counts at every
# bandwidth down to h = 0.17 -- comfortably below the smallest critical
# bandwidth fish produces (h3 ~ 0.19) -- and is 7x faster. The two differ only
# where the count is already in the dozens, which the k <= 3 bisection never
# visits. This is the one per-section parameter; the implementation is the same
# apply_silverman_test the Monte Carlo, bootstrap and hydrology sections use.
GRID_SIZE = 512
# The DEFAULT sweep is the positive widths. Jitter 0 is still a paper width --
# it is Table 9's first column -- and can be run with --jitters 0.0; it is out
# of the default only because the raw fish weights are a 0.5-lb lattice (56
# distinct values over 3474 observations, 98.4% ties), so at zero jitter the
# critical bandwidth measures the recording grid rather than the distribution.
WIDTHS = [j for j in PAPER_JITTERS if j > 0]


def _chunk(args):
    x, jw, b0, nb, n_inner, ks = args
    rows = []
    for b in range(b0, b0 + nb):
        s = np.sort(resample(x, jw, b))
        r = dict(jitter=jw, b=b, hartigan_p=apply_hartigan_dip_test(s))
        for k in ks:
            # apply_silverman_test's smoothed bootstrap runs through scipy's
            # gaussian_kde.resample, which draws on numpy's legacy global state.
            # Seed it per (replicate, k) so every cell is reproducible alone.
            np.random.seed((replicate_seed(jw, b) * 7 + k) % (2 ** 32 - 1))
            r[f'p{k}'] = apply_silverman_test(s, n_bootstrap=n_inner, k=k,
                                              grid_size=GRID_SIZE)
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jitters', default='')
    ap.add_argument('--n_boot', type=int, default=N_BOOT)
    ap.add_argument('--n_inner', type=int, default=199)   # the MC/bootstrap convention
    ap.add_argument('--k', default='1',
                    help="comma-separated null orders, e.g. '1' or '1,2,3'. "
                         "Table 9 uses k = 1,2,3 (sequential Silverman).")
    ap.add_argument('--chunk_size', type=int, default=25)
    ap.add_argument('--n_workers', type=int, default=max(1, multiprocessing.cpu_count()))
    a = ap.parse_args()
    widths = [round(float(s), 2) for s in a.jitters.split(',')] if a.jitters else WIDTHS
    ks = tuple(int(v) for v in a.k.split(','))
    for w in widths:
        assert any(abs(w - p) < 1e-9 for p in PAPER_JITTERS), f'{w} is not a paper jitter width'

    part = OUT / 'fish_modality_paper_replicates_partials.csv'
    x, _ = load_fish()
    print(f'N={len(x)} | widths={widths} | n_boot={a.n_boot} n_inner={a.n_inner} '
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
        done = {(round(float(r.jitter), 2), int(r.b)) for r in d.itertuples()}
        print(f'resuming: {len(done)} replicates already done', flush=True)

    tasks = []
    for jw in widths:
        for b0 in range(0, a.n_boot, a.chunk_size):
            nb = min(a.chunk_size, a.n_boot - b0)
            if all((round(jw, 2), b) in done for b in range(b0, b0 + nb)):
                continue
            tasks.append((x, jw, b0, nb, a.n_inner, ks))
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

    d = pd.read_csv(part).drop_duplicates(['jitter', 'b']).sort_values(['jitter', 'b'])
    d.to_csv(OUT / 'fish_modality_paper_replicates_raw.csv', index=False)
    rows = []
    for jw in widths:
        g = d[np.isclose(d.jitter, jw)]
        row = dict(jitter=jw, n=len(g),
                   hartigan_reject=float((g.hartigan_p < ALPHA).mean() * 100))
        for k in ks:
            row[f'sil_reject_le{k}'] = float((g[f'p{k}'] < ALPHA).mean() * 100)
            row[f'sil_p_median_k{k}'] = float(g[f'p{k}'].median())
        if set(ks) >= {1, 2, 3}:
            r1, r2, r3 = g.p1 < ALPHA, g.p2 < ALPHA, g.p3 < ALPHA
            row.update(seq_1mode=float((~r1).mean() * 100),
                       seq_2modes=float((r1 & ~r2).mean() * 100),
                       seq_3modes=float((r1 & r2 & ~r3).mean() * 100),
                       seq_gt3=float((r1 & r2 & r3).mean() * 100))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / 'Table09_fish_modality_tests.csv', index=False)
    print('\n' + out.to_string(index=False))


if __name__ == '__main__':
    main()
