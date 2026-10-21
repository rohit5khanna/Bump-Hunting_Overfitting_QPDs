"""How much does ONE jitter realization decide the geyser answer?

Figure 19 (geyser_latent_fit_figure.py) fits a single jittered ("latent")
realization of the recorded waiting times, which is only
defensible if the conclusion does not depend on which realization was drawn.
This measures it: N_BOOT independent jitter draws of the FULL recorded sample
-- no bootstrap resampling -- at Figure 19's two orders, recording validity,
modality, Equation 6 W1 at that realization's own sample points, and the two
tallest mode locations when there are two. Writes
tables/geyser_latent_jitter_stability.csv (not a paper table).
"""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '1')
import sys, numpy as np, warnings, multiprocessing as mp
warnings.filterwarnings('ignore')
HERE = os.path.abspath(os.path.dirname(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'code', 'common')); sys.path.insert(0, HERE)
from geyser_common import (load_geyser, jitter_original, JITTER, N_BOOT, P_GRID,
                           LOWER_BOUND, empirical_qf, interdecile, w1)
from metalog.metalog_v2 import LogMetalog
from qflex.transforms import LogQFlex
from qflex.constraints import ConstraintType
import eval_framework as EF

# The lowest order at which each family is feasible AND bimodal on the data
# itself. Log Metalog is infeasible at K = 4..7 on the full sample (jittered
# or raw) and first becomes feasible, with two modes, at K = 8; Log QFlex-TA+
# is feasible from K = 4 but unimodal until K = 10. K = 7 Metalog exists only
# on bootstrap resamples (~40% of them), never on the data.
SPECS = (('Log Metalog', 8), ('Log QFlex-TA+', 10))
_X = None


def _init():
    global _X
    _X, _ = load_geyser()


def one(draw):
    if _X is None:
        _init()
    xs = np.sort(jitter_original(_X, JITTER, draw=draw))
    pp = np.arange(1, len(xs) + 1) / (len(xs) + 1)
    out = []
    for lab, K in SPECS:
        try:
            m = (LogMetalog(xs, pp, lower_bound=LOWER_BOUND, terms=K) if lab == 'Log Metalog'
                 else LogQFlex(xs, pp, lower_bound=LOWER_BOUND, terms=K,
                               constraint_type=ConstraintType.TA))
            if not EF.is_valid(m):
                out.append((draw, lab, K, False, -1, np.nan, np.nan, np.nan)); continue
            nm, locs, hts = EF.modes(m)
            _r, wn = w1(m, xs)          # Eq 6 at the realization's own sample points
            pri = sec = np.nan
            if locs is not None and len(locs) >= 1:
                order = np.argsort(np.asarray(hts, float))[::-1]
                pri = float(np.asarray(locs, float)[order[0]])
                if len(order) >= 2:
                    sec = float(np.asarray(locs, float)[order[1]])
            out.append((draw, lab, K, True, int(nm) if nm is not None else -1, wn, pri, sec))
        except Exception:
            out.append((draw, lab, K, False, -1, np.nan, np.nan, np.nan))
    return out


if __name__ == '__main__':
    import pandas as pd
    R = N_BOOT          # the section's declared replicate count, not a literal
    with mp.Pool(4, initializer=_init) as pool:
        res = [r for chunk in pool.map(one, range(R), chunksize=20) for r in chunk]
    df = pd.DataFrame(res, columns=['draw', 'model', 'K', 'valid', 'n_modes',
                                    'w1', 'primary_min', 'secondary_min'])
    df.to_csv(os.path.join(HERE, '..', 'tables',
                           'geyser_latent_jitter_stability.csv'), index=False)
    print(f'{R} jitter realizations, +-{JITTER} min\n')
    for (lab, K), g in df.groupby(['model', 'K']):
        v = g[g.valid]
        print(f'{lab:14s} K={K:2d}  valid {len(v)}/{len(g)}  '
              f'1-mode {100*(v.n_modes<=1).mean():5.1f}%  2-mode {100*(v.n_modes==2).mean():5.1f}%  '
              f'>2 {100*(v.n_modes>2).mean():5.1f}%  W1 med {v.w1.median():.4f} '
              f'[{v.w1.quantile(.25):.4f}, {v.w1.quantile(.75):.4f}]  '
              f'primary med {v.primary_min.median():.2f}  secondary med {v.secondary_min.median():.2f} min')
