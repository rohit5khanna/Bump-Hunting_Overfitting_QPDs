"""
Tables 4 and A1 (and, in the QFlex-T+ run, Table A2), fitted from scratch at
K = 4, 7, 10, 13.

Every sample is regenerated exactly from the paper's seed formula and every
order is refitted here; no stored fits are read.

    seed = 42 + sample_size*1000 + replication          (replication starts at 1)
    x    = dist.rvs(size=N, random_state=seed)          via jpse, NOT quantile(rng)

EVALUATION FRAMEWORK: code/common/eval_framework.py, for every number here.
  * grid p = 0.001..0.999 (999 points), nothing outside it;
  * validity = Q strictly increasing on the grid;
  * density = differences of Q between neighbouring grid points;
  * modes = interior peaks with relative prominence >= 1% (of the peak's own height);
    a fit with no peak counts as one mode; false modality = more than one;
  * W1 (Equation 6) = mean |Q_fit(p_i) - x_(i)| over the replicate's own sample
    at its plotting positions p_i = i/(N+1) (no interpolation, no grid),
    divided by EQF(0.9) - EQF(0.1) of that sample (the only interpolation).

OUTPUTS
  tables/mc_perrep_<dist>_n<N>.csv   one row per (replicate, model, K):
                                     valid, n_peaks, W1, tallest
                                     mode location / height
  tables/mc_rebuilt_<dist>_n<N>.csv  per-dataset Table 4 / A1 summary at K = 4, 7, 10, 13
  tables/Table04_TableA1_mc_summary_K4_7_10_13.csv   all six, with --all
      (QFlex-T+ run: tables/tplus/TableA2_QFlexTplus_mc_summary_K4_7_10_13.csv)
Orders K = 3..13 are fitted so the same run feeds Tables 4/A1/A2 (4, 7, 10, 13)
and Figures 3, 4, O1-O4 (3..10). The per-replicate files also feed Figure 5
(mc_valid_unimodal_figure.py).
"""
import os, sys, argparse, json, datetime, warnings
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '1')

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'code', 'common')); sys.path.insert(0, HERE)

from jpse import JohnsonSU, JohnsonSL, JohnsonSB
from metalog.metalog_v2 import Metalog, LogMetalog, LogitMetalog
from qflex.core import QFlex
from qflex.transforms import LogQFlex, LogitQFlex
from qflex.constraints import ConstraintType
import figure_style as FS
import eval_framework as EF
import multiprocessing as mp

TAB = os.path.join(HERE, '..', 'tables')
import tplus_mode as TP                 # QPD_TPLUS_ONLY=1: QFlex-T+ run (Table A2)
TAB = TP.table_dir(TAB)
P_GRID = EF.GRID
DIST_PARAMS = dict(xi=0, lam=1, gamma=0.5, delta=1.2)
#: short -> (display name, class, support). Support drives the transform.
DISTS = {'su': ('Johnson SU', JohnsonSU, 'unbounded'),
         'sl': ('Johnson SL', JohnsonSL, 'semi'),
         'sb': ('Johnson SB', JohnsonSB, 'bounded')}
KS = FS.K_STANDARD                      # 4, 7, 10, 13  (tables)
K_FIT = list(range(3, 14))              # 3..13: tables + figures
MODELS = ['Metalog', 'QFlex-U', 'QFlex-TA+', 'QFlex-A+']
if TP.TPLUS_ONLY:
    MODELS = [TP.LABEL]
CT = {'QFlex-U': ConstraintType.NONE, 'QFlex-A+': ConstraintType.A,
      'QFlex-TA+': ConstraintType.TA, 'QFlex-T+': ConstraintType.TL}
N_REPS = 1000


def mc_seed(n, rep):
    return 42 + n * 1000 + rep


def draw(dist, n, rep):
    return np.sort(np.asarray(dist.rvs(size=n, random_state=mc_seed(n, rep)), float))


def fit(model, x, p, K, support):
    """Bounds follow the support of the reference distribution (unbounded for
    SU, Log for SL, Logit on [0, 1] for SB). Fitting all three unbounded is wrong and
    collapses validity on SL and SB -- Metalog on Johnson SL at K=7 drops from
    96.1% to 46.5%, because an unbounded fit keeps crossing zero."""
    try:
        if support == 'unbounded':
            m = (Metalog(x, p, terms=K) if model == 'Metalog'
                 else QFlex(x, p, terms=K, constraint_type=CT[model]))
        elif support == 'semi':
            m = (LogMetalog(x, p, lower_bound=0, terms=K) if model == 'Metalog'
                 else LogQFlex(x, p, lower_bound=0, terms=K,
                               constraint_type=CT[model]))
        else:
            m = (LogitMetalog(x, p, lower_bound=0, upper_bound=1, terms=K)
                 if model == 'Metalog'
                 else LogitQFlex(x, p, lower_bound=0, upper_bound=1, terms=K,
                                 constraint_type=CT[model]))
        return m if EF.is_valid(m) else None
    except Exception:
        return None


def _rep_rows(args):
    short, n, rep = args
    name, cls, support = DISTS[short]
    d = cls(**DIST_PARAMS)
    x = draw(d, n, rep)
    p_pos = EF.plotting_positions(n)
    rows = []
    for model in MODELS:
        for K in K_FIT:
            m = fit(model, x, p_pos, K, support)
            if m is None:
                rows.append((rep, model, K, False, np.nan, np.nan, np.nan, np.nan))
                continue
            npk, locs, hgts = EF.modes(m)
            w1n = EF.w1(m, x)[1]
            if npk > 0:
                i = int(np.argmax(hgts)); ploc, phgt = float(locs[i]), float(hgts[i])
            else:
                ploc = phgt = np.nan
            rows.append((rep, model, K, True, float(npk), w1n, ploc, phgt))
    return rows


def run(short, n, n_reps=N_REPS, procs=4):
    name, cls, support = DISTS[short]
    with mp.Pool(procs) as pool:
        rows = [r for rr in pool.imap(_rep_rows, [(short, n, rep) for rep in range(1, n_reps + 1)],
                                      chunksize=4) for r in rr]
    df = pd.DataFrame(rows, columns=['rep', 'Model', 'K', 'valid', 'n_peaks', 'w1',
                                     'mode_loc', 'mode_height'])
    df.insert(0, 'N', n); df.insert(0, 'Distribution', name)
    out = []
    for model in MODELS:
        for K in KS:
            g = df[(df.Model == model) & (df.K == K)]
            f = g[g.valid]
            out.append(dict(
                Distribution=name, N=n, Model=model, K=K,
                Validity_pct=round(100 * len(f) / len(g), 1),
                n_valid=len(f),
                FalseModality_pct=round(100 * float((f.n_peaks > 1).mean()), 1) if len(f) else np.nan,
                MedianW1_Eq6=round(float(f.w1.median()), 4) if len(f) else np.nan,
            ))
    return pd.DataFrame(out), df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dist', choices=list(DISTS))
    ap.add_argument('--n', type=int)
    ap.add_argument('--all', action='store_true', help='all three distributions x N in {50, 200}')
    ap.add_argument('--n-reps', type=int, default=N_REPS)
    ap.add_argument('--procs', type=int, default=4)
    a = ap.parse_args()
    import time
    jobs = ([(d, n) for d in ('su', 'sl', 'sb') for n in (50, 200)] if a.all else [(a.dist, a.n)])
    tag = '' if a.n_reps == N_REPS else f'_reps{a.n_reps}'
    summaries = []
    for short, n in jobs:
        t0 = time.time()
        res, per = run(short, n, a.n_reps, a.procs)
        per.to_csv(os.path.join(TAB, f'mc_perrep_{short}_n{n}{tag}.csv'), index=False)
        dst = os.path.join(TAB, f'mc_rebuilt_{short}_n{n}{tag}.csv')
        res.to_csv(dst, index=False); summaries.append(res)
        print(res.to_string(index=False))
        print(f'\n{short} N={n}: {len(per)} fits in {time.time()-t0:.1f}s -> {dst}', flush=True)
    if a.all:
        order = {'Johnson SB': 0, 'Johnson SL': 1, 'Johnson SU': 2}
        allt = pd.concat(summaries)
        allt = allt.sort_values(['Distribution', 'N', 'Model', 'K'],
                                key=lambda c: c.map(order) if c.name == 'Distribution' else c)
        stem = ('TableA2_QFlexTplus_mc_summary_K4_7_10_13' if TP.TPLUS_ONLY
                else 'Table04_TableA1_mc_summary_K4_7_10_13')
        allt.to_csv(os.path.join(TAB, f'{stem}{tag}.csv'), index=False)
        print(f'-> {stem}{tag}.csv')
