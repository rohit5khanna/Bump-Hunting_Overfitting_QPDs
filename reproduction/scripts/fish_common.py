"""
Canonical definitions for the fish-biology section (Section 5.2).

Every fish script in `reproduction/` imports from here, so the section's
tables and figures describe the SAME bootstrap resamples and use the SAME W1.

RESAMPLING -- per-replicate seeding

    rng = default_rng(SEED + round(jitter*100)*10_000 + b)
    idx = rng.integers(0, n, size=n)
    s   = x_raw[idx] + rng.uniform(-jitter, +jitter, size=n)
    s   = clip(s, 0.01, None)

Seeding per replicate (rather than consuming one stream in a loop) means
replicate b at a given jitter is regenerable independently by any script, in
any order, without replaying anything. That is what makes the table and the
figures share draws.

`jitter` is the HALF-WIDTH: Uniform(-j, +j), consistent with the whole
project. The 0.01 lb floor is carried over from the original scripts for
exactness; it never binds here (min recorded weight 3.0 lb, max jitter 1.0).

The modality tests of Table 9 (fish_modality_paper_replicates.py) use these
same replicates as their outer sample; the Silverman test's own smoothed
bootstrap draws from a separate stream, so it cannot perturb them.

W1 -- Equation 6, matching the hydrology section exactly

    W1 = mean_i |Q_fit(p_i) - x_(i)| / (Q_T(0.9) - Q_T(0.1)),  p_i = i/(N+1)
         at the reference sample's own points (eval_framework.w1).
         Bootstrap scripts store it against the resample AND the original.

Conventions, as in hydrology:
  * the empirical reference uses Weibull plotting positions p_i = i/(N+1)
    (Equation 3), not np.quantile (whose default interpolation implies
    (i-1)/(N-1));
  * validity and modes are read on the evaluation framework's grid
    0.001..0.999 (999 points).
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(REPO, 'data', 'Fish Biology.xlsx')

SEED        = 42
N_BOOT      = 1000
LOWER_BOUND = 0.0
FLOOR       = 0.01
import sys as _sys
_sys.path.insert(0, os.path.join(REPO, 'code', 'common'))
import eval_framework as EF
P_GRID      = EF.GRID          # evaluation framework: 0.001..0.999
# The jitter widths of Table 9. The replicate stream is jitter-keyed --
# replicate_seed uses round(jitter*100) -- so each width has its own 1000
# replicates, not a rescaling of another width's.
PAPER_JITTERS = [0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]


def load_fish(sort=True):
    """Recorded weights. Returns (x, p_weibull)."""
    x = pd.read_excel(DATA)['Fish Weight (lbs)'].dropna().values.astype(float)
    if sort:
        x = np.sort(x)
    p = np.arange(1, len(x) + 1) / (len(x) + 1)
    return x, p


def replicate_seed(jitter, b):
    return SEED + int(round(jitter * 100)) * 10_000 + b


def resample(x_raw, jitter, b):
    """Bootstrap replicate b at this jitter. Independent of every other b."""
    n = len(x_raw)
    rng = np.random.default_rng(replicate_seed(jitter, b))
    idx = rng.integers(0, n, size=n)
    s = x_raw[idx].astype(float)
    if jitter > 0:
        s = s + rng.uniform(-jitter, jitter, size=n)
    return np.clip(s, FLOOR, None)


def interdecile(x_sorted, p):
    """Q_T(0.9) - Q_T(0.1) of the reference sample: Equation 6's divisor."""
    return float(np.interp(0.9, p, x_sorted) - np.interp(0.1, p, x_sorted))


def empirical_qf(x_sorted, p, grid=P_GRID):
    """Reference EQF at Weibull positions, on the grid (drawing only; W1 uses the points)."""
    return np.interp(grid, p, x_sorted)


def w1(model, x_ref):
    """Equation 6 at the sample points (eval_framework.w1): mean |Q(p_i) - x_(i)|
    over the reference sample's plotting positions p_i = i/(N+1), no
    interpolation; normalized by the reference sample's interdecile range.
    `model` is a fitted QPD or a callable Q(p). Returns (raw, normalized)."""
    return EF.w1(model, x_ref)


w1_points = EF.w1_points     # Q already evaluated at the plotting positions


def w1_vs_original(model, x_orig_sorted, p_orig=None):
    """Equation 6 W1 of a fitted model against the ORIGINAL sample (at its points)."""
    return EF.w1(model, x_orig_sorted)


# ---------------------------------------------------------------------------
# A jitter realization of the ORIGINAL sample: the point-estimate EQF of
# Figure 16 (fish_eqf_ci_figure.py) and the fitted data of Figure 17
# (fish_pdf_order_panels.py) and of fish_pdf_jitter_panels.py (not in the
# paper). Figure 15 needs no RNG -- its overlay is the raw-data fit.
#
# `resample` draws WITH replacement; these figures need the recorded weights
# themselves perturbed once, with no resampling -- a single plausible latent
# sample. It has no replicate index, so it needs its own stream: if it reused
# a replicate's seed it would silently be a duplicate of one band member.
#
# The offset must clear the WHOLE replicate pool, which runs to
#     max = SEED + 100*10_000 + 999 = 1_001_041   (jitter 1.0, b = 999)
# (a smaller offset such as 900_000 would collide: jitter_original(x, 0.1)
# would draw exactly what resample(x, 1.0, 0) draws). STREAM_BASE sits an
# order of magnitude above the pool, and the self-test below fails if that
# ever stops being true.
# ---------------------------------------------------------------------------
STREAM_BASE = 5_000_000


def jitter_original(x_raw, jitter, draw=0):
    """One jitter realization of the recorded weights. No resampling."""
    rng = np.random.default_rng(SEED + STREAM_BASE
                                + int(round(jitter * 100)) * 10_000 + draw)
    x = np.asarray(x_raw, float)
    if jitter > 0:
        x = x + rng.uniform(-jitter, jitter, size=len(x))
    return np.clip(x, FLOOR, None)


if __name__ == '__main__':
    x, p = load_fish()
    print(f'N = {len(x)}, distinct = {len(np.unique(x))}, '
          f'integer share = {100*np.mean(x == np.round(x)):.1f}%')
    print(f'interdecile divisor L = {interdecile(x, p):.4f} lb')
    print(f'evaluation grid: {P_GRID[0]:.3f} .. {P_GRID[-1]:.3f}, {len(P_GRID)} points')
    for j in (0.5, 1.0):
        s = resample(x, j, 0)
        print(f'jitter {j}: replicate 0 range [{s.min():.3f}, {s.max():.3f}], '
              f'seed {replicate_seed(j,0)}, floor binds: {bool((s<=FLOOR).any())}')
    # determinism
    assert np.array_equal(resample(x,0.5,7), resample(x,0.5,7)), 'not deterministic'
    assert not np.array_equal(resample(x,0.5,7), resample(x,0.5,8)), 'replicates identical'
    print('determinism check: OK')

    # the two seed families must never overlap
    jits = [round(0.05 * i, 2) for i in range(21)]
    pool = {replicate_seed(j, b) for j in jits for b in range(N_BOOT)}
    orig = {SEED + STREAM_BASE + int(round(j * 100)) * 10_000 + d
            for j in jits for d in range(100)}
    assert not (pool & orig), f'seed collision: {sorted(pool & orig)[:5]}'
    print(f'seed families disjoint: {len(pool)} replicate seeds '
          f'(max {max(pool)}), {len(orig)} jitter_original seeds '
          f'(min {min(orig)}) -- OK')
