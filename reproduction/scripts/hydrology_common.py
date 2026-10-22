"""
Canonical definitions for the hydrology (Williamson River) section,
Section 5.1 (Table 8, Figures 10-14).

Mirrors fish_common.py and geyser_common.py, so all three empirical sections
describe the same kind of experiment with the same W1.

DESIGN NOTES

1. PAIRED SEEDING. Each bootstrap replicate b has its own seed (below), so
   replicate b can be regenerated directly, without replaying a stream, and
   GEV and every QPD are fitted to the SAME replicate b: the comparison is
   paired, and the EQF band is computed from those same replicates.

2. W1 NORMALIZATION. Equation 6 divides by the TARGET's interdecile range --
   Q_T(0.9) - Q_T(0.1) from the observed sample -- not by the fit's own
   Q_fit(0.9) - Q_fit(0.1). Using the fit's own spread would make the
   denominator move with the model, so W1 would not be comparable across
   models.

3. W1 POINTS. W1 is taken at the N = 95 plotting positions themselves
   (eval_framework.w1) -- for every model, GEV included.

NO JITTER. Gauge heights are continuous measurements, unlike the fish weights
(1-lb heaping) and geyser waiting times (integer minutes). The manuscript does
not jitter this dataset and neither does this module; `JITTER = 0.0` is kept
explicit so the contrast with the other two sections is visible rather than
implied.

SEEDING SCHEME

    rng = default_rng(SEED + HYDRO_BASE + round(jitter*100)*10_000 + b)

HYDRO_BASE puts this dataset's replicates on a stream provably disjoint from
the fish and geyser sections'. The self-test at the bottom asserts it.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(REPO, 'data', 'Hydrology.xlsx')
COLUMN = 'Gauge Height (ft)'

SEED        = 42
HYDRO_BASE  = 40_000_000
STREAM_BASE = 45_000_000
N_BOOT      = 1000
JITTER      = 0.0             # continuous data: no jitter, by design
LOWER_BOUND = 0.0
FLOOR       = 1e-9
import sys as _sys
_sys.path.insert(0, os.path.join(REPO, 'code', 'common'))
import eval_framework as EF
P_GRID      = EF.GRID          # evaluation framework: 0.001..0.999

# The QPD specs the section compares against GEV (Table 8). QFlex here is
# A+ (unimodality-enforcing), not TA+ as in fish.
K_VALUES = [4, 7, 10, 13]    # the paper's standard order set
SPECS = ([('Log Metalog', 'metalog', K) for K in K_VALUES] +
         [('Log QFlex-A+', 'qflex_a', K) for K in K_VALUES])


def load_hydrology(sort=True):
    """Annual peak flood gauge heights, feet. Returns (x, p_weibull)."""
    x = pd.read_excel(DATA)[COLUMN].dropna().values.astype(float)
    if sort:
        x = np.sort(x)
    p = np.arange(1, len(x) + 1) / (len(x) + 1)
    return x, p


def data_fingerprint():
    """(filename, N, sha256) for provenance records."""
    import hashlib
    x, _ = load_hydrology()
    return (os.path.basename(DATA), int(len(x)),
            hashlib.sha256(open(DATA, 'rb').read()).hexdigest())


# ARGUMENT ORDER IS (jitter, b), matching fish_common and geyser_common.
# Both arguments are numbers, so a different order in one module would make
# a call like resample(x, 0.5, 3) silently mean something else there and
# return a different sample with no error. jitter keeps its 0.0 default, so
# hydrology call sites pass b alone -- as a KEYWORD.
def replicate_seed(jitter=JITTER, b=0):
    return SEED + HYDRO_BASE + int(round(jitter * 100)) * 10_000 + b


def resample(x_raw, jitter=JITTER, b=0):
    """Bootstrap replicate b. Every model must be given this same sample."""
    n = len(x_raw)
    rng = np.random.default_rng(replicate_seed(jitter, b))
    idx = rng.integers(0, n, size=n)
    s = x_raw[idx].astype(float)
    if jitter > 0:
        s = s + rng.uniform(-jitter, jitter, size=n)
    return np.clip(s, FLOOR, None)


def interdecile(x_sorted, p):
    """Q_T(0.9) - Q_T(0.1) of the OBSERVED sample: Equation 6's divisor."""
    return float(np.interp(0.9, p, x_sorted) - np.interp(0.1, p, x_sorted))


def empirical_qf(x_sorted, p, grid=P_GRID):
    return np.interp(grid, p, x_sorted)


def w1(model, x_ref):
    """Equation 6 at the sample points (eval_framework.w1): mean |Q(p_i) - x_(i)|
    over the reference sample's plotting positions p_i = i/(N+1), no
    interpolation; normalized by the reference sample's interdecile range.
    `model` is a fitted QPD or a callable Q(p). Returns (raw, normalized)."""
    return EF.w1(model, x_ref)


w1_points = EF.w1_points     # Q already evaluated at the plotting positions


if __name__ == '__main__':
    x, p = load_hydrology()
    fn, n, sha = data_fingerprint()
    print(f'{fn}: N = {n}, range [{x.min():.3f}, {x.max():.3f}] ft, '
          f'sha256 {sha[:16]}')
    print(f'interdecile divisor L = {interdecile(x, p):.4f} ft')
    print(f'evaluation grid: {P_GRID[0]:.3f} .. {P_GRID[-1]:.3f}, {len(P_GRID)} points')
    print(f'specs: {[f"{l} K={K}" for l, _k, K in SPECS]} + GEV')
    s = resample(x, b=0)
    print(f'replicate 0: seed {replicate_seed(b=0)}, range '
          f'[{s.min():.3f}, {s.max():.3f}], distinct {len(np.unique(s))} of {n}')
    assert np.array_equal(resample(x, b=7), resample(x, b=7))
    assert not np.array_equal(resample(x, b=7), resample(x, b=8))
    print('determinism check: OK')

    # all three empirical sections must sit on disjoint streams
    import importlib.util
    def _load(name):
        sp = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + '.py'))
        m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
    fc, gc = _load('fish_common'), _load('geyser_common')
    fj = [round(0.05 * i, 2) for i in range(21)]
    pools = {
      'fish replicates':      {fc.replicate_seed(j, b) for j in fj for b in range(fc.N_BOOT)},
      'fish jitter_original': {fc.SEED + fc.STREAM_BASE + int(round(j*100))*10_000 + d
                               for j in fj for d in range(100)},
      'geyser replicates':    {gc.replicate_seed(gc.JITTER, b) for b in range(gc.N_BOOT)},
      'geyser jitter_orig':   {gc.SEED + gc.STREAM_BASE + int(round(gc.JITTER*100))*10_000 + d
                               for d in range(100)},
      'hydrology replicates': {replicate_seed(b=b) for b in range(N_BOOT)},
    }
    names = list(pools)
    for i in range(len(names)):
        for j_ in range(i + 1, len(names)):
            a, b_ = names[i], names[j_]
            assert not (pools[a] & pools[b_]), f'seed collision: {a} vs {b_}'
    print('seed families disjoint across all three sections:')
    for k, v in pools.items():
        print(f'   {k:<22} [{min(v)}, {max(v)}]  n={len(v)}')
