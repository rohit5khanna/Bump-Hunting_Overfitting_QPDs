"""
Canonical definitions for the Old Faithful geyser section, Section 5.3
(Tables 12 and 13, Figures 18 and 19).

Mirrors `fish_common.py` exactly, so the geyser, fish and hydrology sections
describe the same kind of experiment with the same W1.

DESIGN NOTES

1. SEEDING. Each replicate's seed is a deterministic function of (jitter, b)
   (below). Seeds never involve hash() -- randomised per process for str in
   Python 3, so results would not regenerate -- and every model is fitted to
   the same replicate b.

2. JITTER. The recorded waiting times are integers, so the bootstrap analysis
   applies a uniform jitter of [-0.5 min, 0.5 min] (Section 5.3). With 52
   distinct values in 298 observations a plain bootstrap of the raw integers
   manufactures heavy ties, and ties are what drive spurious high-order
   modality.

3. W1. Equation 6 at the sample points (eval_framework.w1): Weibull plotting
   positions i/(N+1) (Equation 3), no interpolation, normalized by the
   reference sample's interdecile range -- the same W1 as the rest of the
   paper, so Table 13's W1 is comparable with the hydrology and fish tables
   (Tables 8 and 10).

4. MODE GRID. The evaluation framework's grid (eval_framework.GRID,
   0.001..0.999), as in hydrology and fish.

SEEDING SCHEME

    rng = default_rng(SEED + GEYSER_BASE + round(jitter*100)*10_000 + b)

GEYSER_BASE puts this dataset's replicates on a stream provably disjoint from
the fish section's (whose pool tops out at 1,001,041 and whose
jitter_original stream starts at 5,000,042), so the two empirical sections
can never accidentally share draws. The self-test at the bottom asserts it.

Every geyser bootstrap result -- Table 12, Table 13, and the median-bootstrap
fit figure (not in the paper) -- is built from these replicates, so replicate
b means the same sample everywhere. Figure 19 uses jitter_original (one
jitter realization of the full sample, no resampling).
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(REPO, 'data', 'geyser.txt')

SEED        = 42
GEYSER_BASE = 20_000_000
STREAM_BASE = 25_000_000      # for jitter_original, clear of the replicate pool
N_BOOT      = 1000
JITTER      = 0.5             # HALF-WIDTH: Uniform(-0.5, +0.5) minutes
LOWER_BOUND = 0.0
FLOOR       = 1e-9
import sys as _sys
_sys.path.insert(0, os.path.join(REPO, 'code', 'common'))
import eval_framework as EF
P_GRID      = EF.GRID          # evaluation framework: 0.001..0.999
K_VALUES    = list(range(4, 14))   # geyser lists every order 4-13
SPECS       = [('Log Metalog', 'metalog'), ('Log QFlex-U', 'qflex_u'),
               ('Log QFlex-TA+', 'qflex_ta')]   # paper order: Metalog, QFlex-U, QFlex-TA+
import tplus_mode as _TP                # QPD_TPLUS_ONLY=1: supplementary QFlex-T+ run
if _TP.TPLUS_ONLY:
    SPECS = [('Log ' + _TP.LABEL, 'qflex_tl')]


EXPECTED_N = 299        # Azzalini & Bowman (1990); MASS, sm and TH.data all ship 299


def geyser_file():
    """The data file actually used. A 299-point file, if present, wins.

    Drop the canonical series in as `data/geyser_299.txt` (or .csv) and every
    geyser script picks it up with no other change.
    """
    for name in ('geyser_299.csv', 'geyser_299.txt', 'geyser.txt'):
        f = os.path.join(REPO, 'data', name)
        if os.path.exists(f):
            return f
    raise FileNotFoundError('no geyser data file in data/')


def load_geyser(sort=True, verbose=False):
    """Waiting times in minutes. Returns (x, p_weibull).

    Accepts any of the forms this dataset circulates in:
      * the raw Azzalini-Bowman file -- 3 whitespace-separated columns
        (duration, waiting, indicator), durations left as S/M/L, CRLF endings;
      * the packaged 2-column form (duration, waiting) as in MASS / sm /
        TH.data, comma- or whitespace-separated, with or without a header.

    The waiting column is located by name when there is a header, else taken
    as the second column. Only waiting times are used anywhere in this
    section, so the duration coding and the indicator column are irrelevant.
    """
    f = geyser_file()
    head = open(f, 'rb').read(400).decode('utf-8', 'replace')
    sep = ',' if head.count(',') > head.count(' ') else r'\s+'
    first = head.splitlines()[0] if head.splitlines() else ''
    has_header = any(c.isalpha() for c in first.replace('M', '').replace('S', '')
                                              .replace('L', ''))
    df = pd.read_csv(f, sep=sep, header=0 if has_header else None,
                     engine='python')
    if has_header:
        col = next((c for c in df.columns if str(c).strip().lower().startswith('wait')),
                   df.columns[1])
    else:
        col = df.columns[1]
    x = pd.to_numeric(df[col], errors='coerce').dropna().values.astype(float)
    if sort:
        x = np.sort(x)
    p = np.arange(1, len(x) + 1) / (len(x) + 1)
    if verbose or len(x) != EXPECTED_N:
        import hashlib
        h = hashlib.sha256(open(f, 'rb').read()).hexdigest()[:16]
        msg = (f'geyser: N = {len(x)} from {os.path.basename(f)} (sha256 {h})')
        if len(x) != EXPECTED_N:
            msg += (f'  -- WARNING: Azzalini & Bowman (1990) report '
                    f'{EXPECTED_N} eruptions; this file has {len(x)}')
        print(msg)
    return x, p


def data_fingerprint():
    """(filename, N, sha256) for provenance records."""
    import hashlib
    f = geyser_file()
    x, _ = load_geyser()
    return (os.path.basename(f), int(len(x)),
            hashlib.sha256(open(f, 'rb').read()).hexdigest())


def replicate_seed(jitter, b):
    return SEED + GEYSER_BASE + int(round(jitter * 100)) * 10_000 + b


def resample(x_raw, jitter, b):
    """Bootstrap replicate b at this jitter. Independent of every other b."""
    n = len(x_raw)
    rng = np.random.default_rng(replicate_seed(jitter, b))
    idx = rng.integers(0, n, size=n)
    s = x_raw[idx].astype(float)
    if jitter > 0:
        s = s + rng.uniform(-jitter, jitter, size=n)
    return np.clip(s, FLOOR, None)


def jitter_original(x_raw, jitter, draw=0):
    """One jitter realization of the recorded waiting times. No resampling."""
    rng = np.random.default_rng(SEED + STREAM_BASE
                                + int(round(jitter * 100)) * 10_000 + draw)
    x = np.asarray(x_raw, float)
    if jitter > 0:
        x = x + rng.uniform(-jitter, jitter, size=len(x))
    return np.clip(x, FLOOR, None)


def interdecile(x_sorted, p):
    """Q_T(0.9) - Q_T(0.1): Equation 6's divisor."""
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
    x, p = load_geyser()
    print(f'N = {len(x)}, range [{x.min():.0f}, {x.max():.0f}] min, '
          f'distinct = {len(np.unique(x))}, integer share = '
          f'{100*np.mean(x == np.round(x)):.1f}%')
    print(f'interdecile divisor L = {interdecile(x, p):.4f} min')
    print(f'evaluation grid: {P_GRID[0]:.3f} .. {P_GRID[-1]:.3f}, {len(P_GRID)} points')
    s = resample(x, JITTER, 0)
    print(f'jitter {JITTER}: replicate 0 range [{s.min():.3f}, {s.max():.3f}], '
          f'seed {replicate_seed(JITTER, 0)}, distinct {len(np.unique(s))} '
          f'(raw sample has {len(np.unique(x))})')
    assert np.array_equal(resample(x, JITTER, 7), resample(x, JITTER, 7))
    assert not np.array_equal(resample(x, JITTER, 7), resample(x, JITTER, 8))
    print('determinism check: OK')

    # geyser streams must not overlap the fish section's
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'fish_common', os.path.join(HERE, 'fish_common.py'))
    fc = importlib.util.module_from_spec(spec); spec.loader.exec_module(fc)
    fish_jits = [round(0.05 * i, 2) for i in range(21)]
    fish_pool = {fc.replicate_seed(j, b) for j in fish_jits for b in range(fc.N_BOOT)}
    fish_orig = {fc.SEED + fc.STREAM_BASE + int(round(j * 100)) * 10_000 + d
                 for j in fish_jits for d in range(100)}
    gey_pool = {replicate_seed(JITTER, b) for b in range(N_BOOT)}
    gey_orig = {SEED + STREAM_BASE + int(round(JITTER * 100)) * 10_000 + d
                for d in range(100)}
    for a, b_, nm in ((gey_pool, fish_pool, 'geyser replicates vs fish replicates'),
                      (gey_pool, fish_orig, 'geyser replicates vs fish jitter_original'),
                      (gey_orig, fish_pool, 'geyser jitter_original vs fish replicates'),
                      (gey_orig, gey_pool, 'geyser jitter_original vs geyser replicates')):
        assert not (a & b_), f'seed collision: {nm}'
    print(f'seed families disjoint: geyser replicates '
          f'[{min(gey_pool)}, {max(gey_pool)}], fish pool max {max(fish_pool)}, '
          f'fish jitter_original [{min(fish_orig)}, {max(fish_orig)}] -- OK')
