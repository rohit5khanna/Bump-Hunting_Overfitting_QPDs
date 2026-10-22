"""
Canonical definitions for the bimodal-recovery section (Section 4.6:
Table 7, Figure 9).

THE DESIGN. 1000 random samples of size N = 200, each an INDEPENDENT draw from
the ANALYTIC mixture, one per replicate, seeded per replicate -- a Monte Carlo
of independent draws, not resamples of one fixed reference sample (bootstrap
resampling cannot produce observations the reference sample does not already
contain, so it understates true sampling variability).

THE REFERENCE DISTRIBUTION -- 60/40 mixture of two Johnson SU distributions,
the second shifted LEFT by 3.5 standard deviations of the first.

sigma IS ANALYTIC. BASE_SIGMA is the closed-form standard deviation of the
base Johnson SU (base_sigma below), so "3.5 standard deviations" holds exactly
and the population does not depend on any seed. LEGACY_SIGMA
(1.3792305294773437, 0.74 % above the true sigma) is the estimate the earlier
sampler used -- the std of 10,000 rvs draws under default_rng(9999) -- and is
kept only for the self-test below.

THE SAMPLER. sample_mixture reproduces the earlier sampler's ALGORITHM exactly
(binomial split on the mixture rng, then each component drawn by rvs under
seed_base+1000 / seed_base+2000); only sigma differs.

SELF-TEST AND check_against_source() NEED CODE THAT IS NOT IN THIS
REPOSITORY. Both load the earlier sampler source,
code/MC_Cloud_Bimodal/generate_bimodal_source.py, which is not in the
repository: check_against_source() asserts that its constants (parameters,
weights, separation factor, legacy sigma) equal the ones here, and the
self-test (`python3 bimodal_common.py`) asserts that sample_mixture with
sigma=LEGACY_SIGMA reproduces that sampler bit for bit. Without that file,
both stop with an error.

W1. Equation 6 at the replicate's own sample points (eval_framework.w1):
`w1_eq6_sample` (normalized) and `w1_raw_sample`. Every reported W1 is
measured against the sample the model was fitted to; no W1 against the true
mixture is computed.

SEEDING.  rng = default_rng(SEED + BIMODAL_BASE + replication)
BIMODAL_BASE keeps this section's stream disjoint from the three empirical
sections'; the self-test asserts it.
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
SRC = os.path.join(REPO, 'code', 'MC_Cloud_Bimodal')
sys.path.insert(0, os.path.join(REPO, 'code', 'common'))
import eval_framework as EF

SEED         = 42
BIMODAL_BASE = 60_000_000
N_REPLICATES = 1000
SAMPLE_SIZE  = 200
P_GRID       = EF.GRID          # evaluation framework: 0.001..0.999
K_VALUES     = list(range(4, 15))
SPECS        = [('Metalog', 'metalog'), ('QFlex-TA+', 'qflex_ta')]

DIST_PARAMS = dict(xi=0, lam=1, gamma=0.5, delta=1.2)
WEIGHT_1, WEIGHT_2 = 0.60, 0.40
MODE_SEPARATION_FACTOR = -3.5


def base_sigma(xi=0.0, lam=1.0, gamma=0.5, delta=1.2):
    """Closed-form standard deviation of the Johnson SU base distribution.

    Bickel's SU quantile function is Q(p) = xi + lam*sinh((z - gamma)/delta) with
    z = Phi^-1(p), so X = xi + lam*sinh(W) with W ~ Normal(mu, s^2),
    mu = -gamma/delta and s = 1/delta. Using E[e^W] = e^{mu + s^2/2},

        E[sinh W]   = e^{s^2/2} * sinh(mu)
        E[sinh^2 W] = (e^{2 s^2} * cosh(2 mu) - 1) / 2
        Var[X]      = lam^2 * (E[sinh^2 W] - E[sinh W]^2)

    xi drops out, as it must. This is exact -- no grid, no draws, no seed.
    """
    mu, s2 = -gamma / delta, 1.0 / delta ** 2
    m1 = np.exp(s2 / 2) * np.sinh(mu)
    m2 = (np.exp(2 * s2) * np.cosh(2 * mu) - 1) / 2
    return float(lam * np.sqrt(m2 - m1 ** 2))


BASE_SIGMA = base_sigma(**DIST_PARAMS)          # 1.3690932625389083
LEGACY_SIGMA = 1.3792305294773437               # earlier sampler's estimate; self-test only


def _src():
    import importlib.util, sys
    sys.path.insert(0, os.path.join(REPO, 'code', 'common'))
    sys.path.insert(0, SRC)
    sp = importlib.util.spec_from_file_location(
        'generate_bimodal_source', os.path.join(SRC, 'generate_bimodal_source.py'))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


def check_against_source():
    """Fail loudly if the earlier sampler source (code/MC_Cloud_Bimodal, not in
    this repository) and this module have diverged."""
    if not os.path.exists(os.path.join(SRC, 'generate_bimodal_source.py')):
        print('  (earlier sampler source not in this repository -- cross-check skipped)')
        return None
    m = _src()
    assert m.DIST_PARAMS == DIST_PARAMS, (m.DIST_PARAMS, DIST_PARAMS)
    assert abs(m.WEIGHT_1 - WEIGHT_1) < 1e-12, m.WEIGHT_1
    assert abs(m.MODE_SEPARATION_FACTOR - MODE_SEPARATION_FACTOR) < 1e-12
    # sigma is DELIBERATELY different -- assert the size of the known gap so an
    # unrelated change to that estimator cannot slip through unnoticed.
    assert abs(m.get_dist1_std() - LEGACY_SIGMA) < 1e-12, m.get_dist1_std()
    return True


def shift(sigma=None):
    """3.5 x the standard deviation of the base distribution, leftward."""
    return MODE_SEPARATION_FACTOR * (BASE_SIGMA if sigma is None else sigma)


def sample_mixture(n, seed_base, sigma=None):
    """The mixture sampler, with an explicit sigma.

    Same algorithm as the earlier sampler (generate_bimodal_source.
    sample_bimodal_mixture), so passing sigma=LEGACY_SIGMA reproduces that
    population bit for bit; the module self-test asserts exactly that. The
    default sigma is the analytic one.
    """
    from jpse import JohnsonSU
    n1 = np.random.default_rng(seed_base).binomial(n, WEIGHT_1)
    d = JohnsonSU(**DIST_PARAMS)
    a = d.rvs(size=n1, random_state=seed_base + 1000)
    b = d.rvs(size=n - n1, random_state=seed_base + 2000) + shift(sigma)
    return np.sort(np.concatenate([a, b]))


def draw(replication, n=SAMPLE_SIZE, sigma=None):
    """One INDEPENDENT sample of size n from the analytic mixture."""
    return sample_mixture(n, SEED + BIMODAL_BASE + replication, sigma)


def true_quantile(grid=P_GRID, n_grid=200_000, sigma=None):
    """Quantile function of the 60/40 mixture, by numerical inversion.

    jpse.JohnsonSU exposes `quantile` but no CDF, so each component's CDF is
    built by inverting its quantile function on a fine probability grid, the
    two are mixed on a common x-grid, and the result is inverted back.
    """
    from jpse import JohnsonSU
    d = JohnsonSU(**DIST_PARAMS)
    pf = np.linspace(1e-9, 1 - 1e-9, n_grid)
    x1 = np.asarray(d.quantile(pf), float)
    x2 = x1 + shift(sigma)
    xs = np.linspace(min(x1[0], x2[0]), max(x1[-1], x2[-1]), n_grid)
    F = WEIGHT_1 * np.interp(xs, x1, pf) + WEIGHT_2 * np.interp(xs, x2, pf)
    F = np.maximum.accumulate(F)
    return np.interp(grid, F, xs)


def interdecile_from_grid(qf_on_grid, grid=P_GRID):
    return float(np.interp(0.9, grid, qf_on_grid) - np.interp(0.1, grid, qf_on_grid))


def empirical_qf(x_sorted, grid=P_GRID):
    p = np.arange(1, len(x_sorted) + 1) / (len(x_sorted) + 1)
    return np.interp(grid, p, x_sorted)


def w1(model, x_ref):
    """Equation 6 at the sample points (eval_framework.w1): mean |Q(p_i) - x_(i)|
    over the reference sample's plotting positions p_i = i/(N+1), no
    interpolation; normalized by the reference sample's interdecile range.
    `model` is a fitted QPD or a callable Q(p). Returns (raw, normalized)."""
    return EF.w1(model, x_ref)


w1_points = EF.w1_points     # Q already evaluated at the plotting positions


if __name__ == '__main__':
    print('constants match generate_bimodal_source.py:', check_against_source())
    s = shift()
    print(f'mixture: {WEIGHT_1:.0%}/{WEIGHT_2:.0%} Johnson SU {DIST_PARAMS}, '
          f'second shifted by {s:.6f} (= {MODE_SEPARATION_FACTOR} x sigma '
          f'{BASE_SIGMA:.10f}, analytic)')
    print(f'legacy sigma {LEGACY_SIGMA:.10f} -> shift {MODE_SEPARATION_FACTOR*LEGACY_SIGMA:.6f} '
          f'({100*(LEGACY_SIGMA-BASE_SIGMA)/BASE_SIGMA:+.3f}% vs analytic)')

    # sample_mixture must reproduce the earlier sampler's population bit for
    # bit when handed LEGACY_SIGMA, so sigma is the only difference between
    # the two. Needs code/MC_Cloud_Bimodal, which is not in this repository.
    for r in (0, 7, 999):
        mine = draw(r, sigma=LEGACY_SIGMA)
        theirs, _, _ = _src().sample_bimodal_mixture(SAMPLE_SIZE, SEED + BIMODAL_BASE + r)
        assert np.array_equal(mine, np.sort(theirs)), f'sampler diverged at r={r}'
    print('sampler reproduces the earlier sampler exactly under LEGACY_SIGMA: OK')

    # Independent check on BASE_SIGMA by Gauss-Hermite quadrature over z.
    # (A Monte Carlo check is useless here: SU with d = 1.2 has a tail heavy
    # enough that a 4-million-draw sample std is still 0.14 % low.  Quadrature
    # agrees to 12 digits.  numpy's hermgauss overflows above ~300 nodes, so
    # 200 is both the practical and the sufficient choice.)
    _t, _w = np.polynomial.hermite.hermgauss(200)
    _x = np.sinh((np.sqrt(2) * _t - DIST_PARAMS['gamma']) / DIST_PARAMS['delta'])
    _wt = _w / np.sqrt(np.pi)
    _gh = DIST_PARAMS['lam'] * np.sqrt(np.sum(_wt * _x**2) - np.sum(_wt * _x)**2)
    print(f'BASE_SIGMA {BASE_SIGMA:.12f} vs Gauss-Hermite {_gh:.12f}')
    assert abs(_gh - BASE_SIGMA) < 1e-10
    tq = true_quantile()
    print(f'true QF on [0.01, 0.99]: [{tq[0]:.3f}, {tq[-1]:.3f}], '
          f'interdecile {interdecile_from_grid(tq):.4f}')
    x = draw(0)
    print(f'replicate 0: N={len(x)}, range [{x.min():.3f}, {x.max():.3f}], '
          f'seed {SEED + BIMODAL_BASE}')
    assert np.array_equal(draw(7), draw(7)), 'not deterministic'
    assert not np.array_equal(draw(7), draw(8)), 'replicates identical'
    print('determinism check: OK')

    import importlib.util
    def _load(n):
        sp = importlib.util.spec_from_file_location(n, os.path.join(HERE, n + '.py'))
        m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
    fc, gc, hc = _load('fish_common'), _load('geyser_common'), _load('hydrology_common')
    fj = [round(0.05*i, 2) for i in range(21)]
    pools = {
      'fish':      {fc.replicate_seed(j, b) for j in fj for b in range(fc.N_BOOT)} |
                   {fc.SEED + fc.STREAM_BASE + int(round(j*100))*10_000 + d
                    for j in fj for d in range(100)},
      'geyser':    {gc.replicate_seed(gc.JITTER, b) for b in range(gc.N_BOOT)} |
                   {gc.SEED + gc.STREAM_BASE + int(round(gc.JITTER*100))*10_000 + d
                    for d in range(100)},
      'hydrology': {hc.replicate_seed(b=b) for b in range(hc.N_BOOT)},
      'bimodal':   {SEED + BIMODAL_BASE + r for r in range(N_REPLICATES)},
    }
    ks = list(pools)
    for i in range(len(ks)):
        for j_ in range(i+1, len(ks)):
            assert not (pools[ks[i]] & pools[ks[j_]]), f'collision {ks[i]} vs {ks[j_]}'
    print('seed families disjoint across all four sections:')
    for k, v in pools.items():
        print(f'   {k:<10} [{min(v)}, {max(v)}]  n={len(v)}')
