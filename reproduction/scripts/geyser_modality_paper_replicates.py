"""
Formal modality tests on the geyser bootstrap replicates used in the paper:
Table 12 (Section 5.3), written to tables/Table12_geyser_modality_tests.csv.

The geyser counterpart of fish_modality_paper_replicates.py (Table 9).
Hartigan's dip test and Silverman's critical-bandwidth test are applied to
each of the 1000 bootstrap replicates of the Old Faithful waiting times at the
manuscript's prescribed jitter of +/-0.5 min (half-width).

SAMPLE IDENTITY. The replicates are drawn with geyser_common.resample(x,
JITTER, b), which is the same call geyser_sweep.py's run_block makes.
Replicate b is therefore the same sample that backs Table 13. This is
asserted, not assumed: on startup every replicate_seed(
JITTER, b) is checked against the `seed` column recorded in
tables/geyser_sweep.csv, and the run aborts on any mismatch. (A same-seed
cross-check verifies stream consistency; checking against the seeds stored by
the run that produced the published tables verifies sample identity.)

GRID SIZE. apply_silverman_test's default grid of 2000 is used unchanged. The
fish script overrides it to 512 only because N = 3474 makes 2000 prohibitive;
its comment records that the default suits N ~ 200. Geyser's N = 299 is in
that range, so the default applies and this section uses the same grid as the
Monte Carlo (Table 3) and bootstrap (Table 5) sections.

LATTICE. The recorded waiting times are integers -- 52 distinct values in 299
observations, 82.6% ties -- but the critical bandwidths here (h_crit ~ 2.5 to
7.3 min on replicate 0) sit far above the 1-minute recording interval, so
unlike the fish data at jitter 0 this test is not measuring recording
precision.

REPORTING. The summary reports CUMULATIVE rejection rates on the common
denominator n_boot:

    reject <=1              : rejected H0: <=1 mode
    reject <=1 and <=2      : rejected <=1, then also rejected <=2
    reject <=1, <=2 and <=3 : rejected all three

These are monotone non-increasing by construction, which is what the
sequential procedure actually yields. The marginal per-k rates (every
replicate tested at every k independently) are NOT reported: they are not
stages of a procedure, they can invert, and 20-35% of replicates give
logically incompatible marginal verdicts (accepting <=1 while rejecting <=2)
because each k is tested against a different null built at its own h_crit.
The per-replicate p-values are kept in the partials file, so any other
summary can be derived without recomputing.

Rejection is at strict p < ALPHA, matching every other script in
reproduction/. With B = 199 the p-value grid has an atom exactly at 0.05, so
the strict comparison tests at an effective 0.045.
"""
import argparse, multiprocessing, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / 'code' / 'common'))

import numpy as np, pandas as pd

from geyser_common import load_geyser, resample, replicate_seed, JITTER, N_BOOT
from stats_tests import apply_silverman_test, apply_hartigan_dip_test, SILVERMAN_IMPL

OUT = REPO / 'reproduction' / 'tables'
ALPHA = 0.05
GRID_SIZE = 2000


def _chunk(task):
    x, b0, nb, n_inner, ks = task
    rows = []
    for b in range(b0, b0 + nb):
        s = np.sort(resample(x, JITTER, b))
        r = dict(jitter=JITTER, b=b, seed=replicate_seed(JITTER, b),
                 hartigan_p=apply_hartigan_dip_test(s))
        for k in ks:
            r[f'p{k}'] = apply_silverman_test(s, n_bootstrap=n_inner, k=k,
                                              grid_size=GRID_SIZE)
        rows.append(r)
    return rows


def _assert_sample_identity(nb):
    """Every replicate_seed(JITTER, b) must match the seed geyser_sweep.csv
    recorded for b. Guarantees these are the paper's samples."""
    ref = OUT / 'geyser_sweep.csv'
    if not ref.exists():
        sys.exit(f'cannot verify sample identity: {ref} is missing')
    d = pd.read_csv(ref, usecols=['b', 'seed']).drop_duplicates()
    if d.groupby('b').seed.nunique().max() != 1:
        sys.exit(f'{ref.name} records more than one seed for some b')
    recorded = dict(zip(d.b.astype(int), d.seed.astype(np.int64)))
    missing = [b for b in range(nb) if b not in recorded]
    if missing:
        sys.exit(f'{ref.name} has no seed for b = {missing[:5]}'
                 f'{"..." if len(missing) > 5 else ""}')
    bad = [(b, recorded[b], replicate_seed(JITTER, b)) for b in range(nb)
           if recorded[b] != replicate_seed(JITTER, b)]
    if bad:
        sys.exit('SAMPLE IDENTITY FAILURE -- these would not be the paper\'s '
                 f'samples.\n  first mismatches (b, in geyser_sweep.csv, here): '
                 f'{bad[:5]}')
    print(f'sample identity: {nb} seeds match tables/geyser_sweep.csv '
          f'[{replicate_seed(JITTER, 0)} .. {replicate_seed(JITTER, nb - 1)}]',
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    ap.add_argument('--n-inner', type=int, default=199)
    ap.add_argument('--k', default='1,2,3')
    ap.add_argument('--chunk-size', type=int, default=25)
    ap.add_argument('--n-workers', type=int,
                    default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    ks = [int(v) for v in a.k.split(',')]

    x, _ = load_geyser()
    x = np.asarray(x, float)
    print(f'N={len(x)} distinct={len(np.unique(x))} '
          f'ties={100 * (1 - len(np.unique(x)) / len(x)):.1f}% | '
          f'jitter=+/-{JITTER} min | k={ks} | n_boot={a.n_boot} '
          f'n_inner={a.n_inner} grid={GRID_SIZE} workers={a.n_workers}',
          flush=True)
    _assert_sample_identity(a.n_boot)

    OUT.mkdir(parents=True, exist_ok=True)
    part = OUT / 'geyser_modality_paper_replicates_partials.csv'
    done = set()
    if part.exists():
        d = pd.read_csv(part)
        stamps = sorted(set(d.impl.dropna().unique()))
        if stamps != [SILVERMAN_IMPL]:
            sys.exit('impl stamp mismatch in '
                     f'{part.name}\n  file: {stamps}\n  now : {SILVERMAN_IMPL}\n'
                     'Resuming would mix two tests in one result. '
                     'Move the file aside and rerun.')
        done = {int(r.b) for r in d.itertuples()}
        print(f'resuming: {len(done)} replicates already done', flush=True)

    tasks = []
    for b0 in range(0, a.n_boot, a.chunk_size):
        nb = min(a.chunk_size, a.n_boot - b0)
        if all(b in done for b in range(b0, b0 + nb)):
            continue
        tasks.append((x, b0, nb, a.n_inner, ks))
    print(f'{len(tasks)} chunks to run', flush=True)

    t0 = time.time()
    if tasks:
        with multiprocessing.Pool(a.n_workers) as pool:
            for i, rows in enumerate(pool.imap_unordered(_chunk, tasks, chunksize=1), 1):
                df = pd.DataFrame(rows)
                df['n_inner'] = a.n_inner
                df['grid_size'] = GRID_SIZE
                df['impl'] = SILVERMAN_IMPL
                df.to_csv(part, mode='a', header=not part.exists(), index=False)
                el = time.time() - t0
                print(f'  chunk {i}/{len(tasks)}  elapsed={el / 60:.1f}min  '
                      f'eta={(len(tasks) - i) * el / i / 60:.1f}min', flush=True)

    d = pd.read_csv(part).drop_duplicates(['b']).sort_values('b')
    d.to_csv(OUT / 'geyser_modality_paper_replicates_raw.csv', index=False)

    n = len(d)
    row = dict(jitter=JITTER, n=n, n_inner=a.n_inner, grid_size=GRID_SIZE,
               hartigan_reject=float((d.hartigan_p < ALPHA).mean() * 100))
    if set(ks) >= {1, 2, 3}:
        r1, r2, r3 = d.p1 < ALPHA, d.p2 < ALPHA, d.p3 < ALPHA
        c1, c2, c3 = r1, r1 & r2, r1 & r2 & r3
        row.update(
            # cumulative rejection, common denominator n -- monotone by construction
            sil_reject_le1=float(c1.mean() * 100),
            sil_reject_le1_le2=float(c2.mean() * 100),
            sil_reject_le1_le2_le3=float(c3.mean() * 100),
            n_reject_le1=int(c1.sum()), n_reject_le1_le2=int(c2.sum()),
            n_reject_le1_le2_le3=int(c3.sum()),
            # four-way classification: smallest k the procedure accepts
            seq_1mode=float((~r1).mean() * 100),
            seq_2modes=float((r1 & ~r2).mean() * 100),
            seq_3modes=float((r1 & r2 & ~r3).mean() * 100),
            seq_gt3=float(c3.mean() * 100),
            sil_p_median_k1=float(d.p1.median()),
            sil_p_median_k2=float(d.p2.median()),
            sil_p_median_k3=float(d.p3.median()))
        assert c1.sum() >= c2.sum() >= c3.sum(), 'cumulative counts not nested'
    else:
        for k in ks:
            row[f'sil_reject_le{k}_marginal'] = float((d[f'p{k}'] < ALPHA).mean() * 100)
            row[f'sil_p_median_k{k}'] = float(d[f'p{k}'].median())

    out = pd.DataFrame([row])
    out.to_csv(OUT / 'Table12_geyser_modality_tests.csv', index=False)
    print('\n' + out.to_string(index=False))


if __name__ == '__main__':
    main()
