"""
Bootstrap curve stores for the empirical examples
==================================================

Builds, for one empirical dataset and one list of QPD specifications, the
full store of bootstrap quantile/density curves needed by the representative-
curve processes in median_curve_processes.py.

Kept separate from the figure scripts so the (slow) resampling is written
once and every figure variant -- median-W1, Process A, Process B --
consumes the same curves rather than each re-deriving its own, which is how
the variants stay comparable.

Resampling:
  * hydrology / geyser: plain nonparametric bootstrap of the observed values
    (no jitter unless the caller passes one).
  * fish: bootstrap of the RAW recorded weights with a FRESH uniform jitter
    drawn per replicate (the paper uses +/-0.5 lb) -- the jitter is part of
    what the interval measures, so it must be redrawn each replicate rather
    than fixed once.

    JITTER CONVENTION: `jitter` is the AMPLITUDE, so the noise is
    Uniform(-jitter, +jitter) and jitter=0.5 means +/-0.5 lb -- the same
    convention as reproduction/scripts/fish_common.py
    (rng.uniform(-jitter, jitter)). The draws here are this module's own
    stream (default_rng(seed + b)), not fish_common's replicates.
Weibull plotting positions p_i = i/(n+1) throughout, matching Equation 3.
"""

import os
import numpy as np
import pandas as pd

from metalog.metalog_v2 import LogMetalog
from qflex.transforms import LogQFlex
from qflex.constraints import ConstraintType

# Evaluation grid. The linear part is the working range every W1 and
# representative-curve selection uses. The log-spaced ramp below it exists
# only so PLOTTED curves reach their lower bound: a Log Metalog / Log QFlex
# with lower bound L satisfies Q(p) -> L as p -> 0, but at p = 0.005 the
# curve still starts well above L (about 1.7 ft for the hydrology K=10 fit),
# so a density drawn on the linear grid alone appears to begin in mid-air
# instead of rising from the bound. The ramp is dense near zero because the
# approach to L happens almost entirely in the last decade of p.
#
# The ramp's point count is not free: the representative-curve density is
# obtained by differentiating a quantile curve, so a spacing that jumps
# discontinuously where the two pieces meet puts a visible kink in the
# density right at p = 0.005. The ramp is therefore given exactly as many
# points as make its spacing at the junction equal the core's, which makes
# dp/di continuous there; log spacing then thins it out smoothly toward 0,
# where the curve is nearly flat in p anyway.
P_CORE_LO, P_CORE_HI = 0.005, 0.995
P_TAIL_LO = 1e-6
_N_CORE = 1000
_DP_CORE = (P_CORE_HI - P_CORE_LO) / (_N_CORE - 1)
_N_RAMP = int(round(P_CORE_LO * np.log(P_CORE_LO / P_TAIL_LO) / _DP_CORE)) + 1

def _build_grid():
    ramp = np.logspace(np.log10(P_TAIL_LO), np.log10(P_CORE_LO), _N_RAMP)
    core = np.linspace(P_CORE_LO, P_CORE_HI, _N_CORE)
    # The logspace endpoint and the linspace start are both meant to be
    # P_CORE_LO but can differ in the last bit, which would leave a ~1e-18
    # gap that np.unique keeps and that blows up any 1/(dQ/dp).
    g = np.unique(np.concatenate([ramp[:-1], core]))
    return g[np.concatenate([[True], np.diff(g) > 1e-12])]


P_GRID_DEFAULT = _build_grid()


def core_mask(p_grid, lo=P_CORE_LO, hi=P_CORE_HI):
    """Boolean mask selecting the working range of a grid built as above.
    Use it for anything quantitative (W1, medoid selection) so the extreme
    tail added for drawing cannot influence a number."""
    p = np.asarray(p_grid, dtype=float)
    return (p >= lo) & (p <= hi)


# --- dataset loaders ---------------------------------------------------------

def load_dataset(name, data_dir):
    """Returns (x_sorted, jitter_width_or_None, value_label)."""
    if name == 'hydrology':
        df = pd.read_excel(os.path.join(data_dir, 'Hydrology.xlsx'))
        return np.sort(df['Gauge Height (ft)'].dropna().values.astype(float)), None, 'Gauge height (ft)'
    if name == 'fish':
        df = pd.read_excel(os.path.join(data_dir, 'Fish Biology.xlsx'))
        return np.sort(df['Fish Weight (lbs)'].dropna().values.astype(float)), None, 'Fish weight (lbs)'
    if name == 'geyser':
        df = pd.read_csv(os.path.join(data_dir, 'geyser.txt'), sep=r'\s+', header=None,
                         names=['eruption', 'waiting', 'indicator'])
        return np.sort(df['waiting'].dropna().values.astype(float)), None, 'Waiting time (min)'
    raise ValueError(f'unknown dataset {name!r}')


# --- fitting -----------------------------------------------------------------

def fit_spec(x_sorted, p_pos, spec, lower_bound=0.0):
    """spec = (key, label, K, family) with family in {'metalog', 'qflex_ta',
    'qflex_a'}. Returns the fitted model or None when infeasible/failed."""
    _key, _label, K, family = spec
    try:
        if family == 'metalog':
            m = LogMetalog(x_sorted, p_pos, lower_bound=lower_bound, terms=K)
        elif family == 'qflex_ta':
            m = LogQFlex(x_sorted, p_pos, lower_bound=lower_bound, terms=K,
                         constraint_type=ConstraintType.TA)
        elif family == 'qflex_a':
            m = LogQFlex(x_sorted, p_pos, lower_bound=lower_bound, terms=K,
                         constraint_type=ConstraintType.A)
        else:
            raise ValueError(f'unknown family {family!r}')
        return m if m.is_feasible else None
    except Exception:
        return None


def build_store(x_raw, specs, n_boot=1000, jitter=None, seed=42,
                p_grid=P_GRID_DEFAULT, lower_bound=0.0, progress_every=100,
                verbose=True):
    """Run the bootstrap and collect per-replicate QF/PDF curves per spec.

    Returns {key: {'qf': (B, n_p), 'pdf': (B, n_p), 'label':, 'K':,
                   'n_feasible':}} plus 'p_grid' and 'eqf' (the observed
    sample's own EQF on p_grid, for W1-vs-data if a caller wants it).
    Infeasible replicates are stored as all-NaN rows so row index still
    equals bootstrap id; the process functions skip them.
    """
    x_raw = np.asarray(x_raw, dtype=float)
    N = len(x_raw)
    p_pos = np.arange(1, N + 1) / (N + 1)
    store = {k: {'qf': np.full((n_boot, len(p_grid)), np.nan),
                 'pdf': np.full((n_boot, len(p_grid)), np.nan),
                 'label': lab, 'K': K, 'family': fam}
             for (k, lab, K, fam) in specs}

    for b in range(n_boot):
        rng = np.random.default_rng(seed + b)
        draw = rng.choice(x_raw, size=N, replace=True)
        if jitter:
            # fresh jitter per replicate, amplitude convention -- see docstring
            draw = draw + rng.uniform(-jitter, jitter, size=N)
            draw = np.clip(draw, lower_bound + 1e-6, None)
        x_boot = np.sort(draw)
        for spec in specs:
            m = fit_spec(x_boot, p_pos, spec, lower_bound=lower_bound)
            if m is None:
                continue
            try:
                qf = m.quantile(p_grid)
                pdf = np.clip(m.pdf(p_grid), 0, None)
            except Exception:
                continue
            if not (np.all(np.isfinite(qf)) and np.all(np.isfinite(pdf))):
                continue
            store[spec[0]]['qf'][b] = qf
            store[spec[0]]['pdf'][b] = pdf
        if verbose and progress_every and (b + 1) % progress_every == 0:
            print(f'    {b + 1}/{n_boot}', flush=True)

    for k in store:
        store[k]['n_feasible'] = int(np.isfinite(store[k]['qf']).all(axis=1).sum())
        if verbose:
            print(f'    {k}: {store[k]["n_feasible"]}/{n_boot} feasible', flush=True)

    store['p_grid'] = p_grid
    store['eqf'] = np.interp(p_grid, p_pos, x_raw)
    return store


def save_store(path, store):
    out = {'p_grid': store['p_grid'], 'eqf': store['eqf']}
    keys = [k for k in store if k not in ('p_grid', 'eqf')]
    for k in keys:
        out[f'{k}__qf'] = store[k]['qf']
        out[f'{k}__pdf'] = store[k]['pdf']
        out[f'{k}__meta'] = np.array([store[k]['label'], str(store[k]['K']),
                                      store[k]['family'], str(store[k]['n_feasible'])])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)


def load_store(path):
    d = np.load(path, allow_pickle=False)
    store = {'p_grid': d['p_grid'], 'eqf': d['eqf']}
    for name in d.files:
        if name.endswith('__qf'):
            k = name[:-4]
            meta = d[f'{k}__meta']
            store[k] = {'qf': d[f'{k}__qf'], 'pdf': d[f'{k}__pdf'],
                        'label': str(meta[0]), 'K': int(meta[1]),
                        'family': str(meta[2]), 'n_feasible': int(meta[3])}
    return store
