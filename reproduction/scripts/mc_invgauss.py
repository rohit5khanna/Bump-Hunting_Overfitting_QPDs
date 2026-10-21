"""
Online Supplement Section O4: a NON-Johnson semi-bounded reference -- the
inverse Gaussian on (0, inf). The per-replicate output feeds Table O2
(invgauss_table_O2.py, N = 200); the reference density is Figure O8
(invgauss_reference_figure.py).

Same Monte Carlo design as Johnson SL (mc_tables_rebuild.py), so the two can be
compared row by row:
  * IG(mu = 0.919, lambda = 1.069), chosen to match Johnson SL's P10 / P50 / P90
    (SL: xi=0, lam=1, gamma=0.5, delta=1.2);
  * N in {50, 200}; 1000 replicates; orders K = 3..13 (tables use 4, 7, 10, 13,
    figures 3..10);
  * five QPDs: Metalog, QFlex-U, QFlex-T+, QFlex-TA+, QFlex-A+, all fitted with the
    Log transform and the TRUE lower bound 0 (exactly as Johnson SL);
  * the paper's evaluation framework (code/common/eval_framework.py) for validity,
    modes and W1 -- via mc_tables_rebuild.fit / EF, so nothing is re-implemented.

SAMPLES. Johnson SL replicate r is x = Q_SL(u) with u = 1000 uniforms from
np.random.seed(42 + N*1000 + r) (jpse.JohnsonBase.rvs). Here x = Q_IG(u) with the
SAME u: every IG replicate is the SL replicate pushed through the IG quantile
function instead, so SL and IG differ only in the shape of the distribution.

Outputs (all under tables/invgauss/; the Johnson outputs are not touched):
  tables/invgauss/mc_perrep_ig_n<N>.csv     one row per (replicate, model, K)
  tables/invgauss/mc_rebuilt_ig_n<N>.csv    summary at K = 4, 7, 10, 13
                                            (Table O2 itself is written by
                                            invgauss_table_O2.py)

usage:
  python3 mc_invgauss.py --n 200 --procs 4                 # everything for one N
  python3 mc_invgauss.py --n 200 --reps 1-250 --procs 4    # a chunk (partials)
  python3 mc_invgauss.py --n 200 --merge                   # combine the chunks
"""
import os, sys, argparse, time, warnings
import numpy as np, pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mc_tables_rebuild as MTR          # fit(), seed formula, framework imports
EF = MTR.EF
import multiprocessing as mp

MU, LAMBDA = 0.919, 1.069
IG = stats.invgauss(MU / LAMBDA, scale=LAMBDA)   # scipy: IG(mu, lambda) = invgauss(mu/lambda, scale=lambda)
MODELS = ['Metalog', 'QFlex-U', 'QFlex-T+', 'QFlex-TA+', 'QFlex-A+']
K_FIT, KS, N_REPS = MTR.K_FIT, MTR.KS, MTR.N_REPS
TAB = os.path.join(HERE, '..', 'tables', 'invgauss')
PART = os.path.join(TAB, 'partials')


def draw(n, rep):
    """The Johnson SL replicate's uniforms, through the IG quantile function."""
    np.random.seed(MTR.mc_seed(n, rep))
    u = np.random.uniform(0, 1, size=n)
    return np.sort(IG.ppf(u))


def true_mode():
    mu, lam = MU, LAMBDA
    m = mu * (np.sqrt(1 + (1.5 * mu / lam) ** 2) - 1.5 * mu / lam)
    return float(m), float(IG.pdf(m))


def _rep_rows(args):
    n, rep = args
    x = draw(n, rep)
    p = EF.plotting_positions(n)
    rows = []
    for model in MODELS:
        for K in K_FIT:
            m = MTR.fit(model, x, p, K, 'semi')
            if m is None:
                rows.append((rep, model, K, False, np.nan, np.nan, np.nan, np.nan)); continue
            npk, locs, hgts = EF.modes(m)
            w1n = EF.w1(m, x)[1]
            if npk > 0:
                i = int(np.argmax(hgts)); ploc, phgt = float(locs[i]), float(hgts[i])
            else:
                ploc = phgt = np.nan
            rows.append((rep, model, K, True, float(npk), w1n, ploc, phgt))
    return rows


COLS = ['rep', 'Model', 'K', 'valid', 'n_peaks', 'w1', 'mode_loc', 'mode_height']


def summarise(df, n):
    out = []
    for model in MODELS:
        for K in KS:
            g = df[(df.Model == model) & (df.K == K)]; f = g[g.valid]
            out.append(dict(Distribution='Inverse Gaussian', N=n, Model=model, K=K,
                            Validity_pct=round(100 * len(f) / len(g), 1), n_valid=len(f),
                            FalseModality_pct=round(100 * float((f.n_peaks > 1).mean()), 1) if len(f) else np.nan,
                            MedianW1_Eq6=round(float(f.w1.median()), 4) if len(f) else np.nan))
    return pd.DataFrame(out)


def finish(df, n):
    df = df.sort_values(['rep', 'Model', 'K'])
    df.insert(0, 'N', n); df.insert(0, 'Distribution', 'Inverse Gaussian')
    df.to_csv(os.path.join(TAB, f'mc_perrep_ig_n{n}.csv'), index=False)
    s = summarise(df, n); s.to_csv(os.path.join(TAB, f'mc_rebuilt_ig_n{n}.csv'), index=False)
    print(s.to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--reps', type=str, default='', help='chunk, e.g. 1-250')
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--procs', type=int, default=4)
    a = ap.parse_args()
    os.makedirs(PART, exist_ok=True)
    if a.merge:
        fs = sorted(f for f in os.listdir(PART) if f.startswith(f'ig_n{a.n}_'))
        df = pd.concat([pd.read_csv(os.path.join(PART, f)) for f in fs]).drop_duplicates(['rep', 'Model', 'K'])
        missing = sorted(set(range(1, N_REPS + 1)) - set(df.rep))
        if missing:
            raise SystemExit(f'missing {len(missing)} replicates, first {missing[:5]}')
        finish(df, a.n); raise SystemExit
    lo, hi = (map(int, a.reps.split('-')) if a.reps else (1, N_REPS))
    t0 = time.time()
    with mp.Pool(a.procs) as pool:
        rows = [r for rr in pool.imap(_rep_rows, [(a.n, r) for r in range(lo, hi + 1)], chunksize=2)
                for r in rr]
    df = pd.DataFrame(rows, columns=COLS)
    if a.reps:
        df.to_csv(os.path.join(PART, f'ig_n{a.n}_{lo:04d}_{hi:04d}.csv'), index=False)
        print(f'N={a.n} reps {lo}-{hi}: {len(df)} fits in {time.time() - t0:.1f}s')
    else:
        finish(df, a.n); print(f'{len(df)} fits in {time.time() - t0:.1f}s')
