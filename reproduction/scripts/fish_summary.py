"""
Fish biology -- per-model summary table: paper Table 10 (stability of model
features, 0.5 lb jitter), written as Table10_fish_summary_jitter0.5.csv.
Other jitters (--jitter 1.0) and the QFlex-T+ run write
fish_summary_jitter<j>.csv, which is not in the paper.

Table 9 reports only the Hartigan and Silverman results by jitter width; this
table states how often a fit is valid, how often it has more than one mode,
how dispersed its modes are, and how far it sits from the data.

Built on the section's canonical replicates -- `fish_common.resample` at the
paper's 0.5 lb jitter, the same 1000 draws Table 9 (the 0.5 lb column), Table
11 (fish_sample_size_sweep.py) and the Figure 16 band use -- so these
results describe the same samples.

Orders are the paper's standard set, K = 4, 7, 10, 13. fish_bimodal_vs_jitter.py
and fish_mode_scatter.py (not in the paper) use their own 9/10/11 and 10
windows, because the second mode appears there.

W1 is Equation 6 at the sample points (eval_framework.w1), stored BOTH ways:
w1_own vs the jittered resample each fit was made to, w1_orig vs the ORIGINAL
recorded sample; w1 / MedianW1_Eq6 = the one eval_framework.BOOTSTRAP_W1_REFERENCE
selects. Fish N = 3474 > 999, so the 3 lowest and 3 highest plotting positions
lie outside the 0.001..0.999 grid; W1 evaluates the fit there too.
"""
import os, sys, json, time, datetime, warnings, argparse
import multiprocessing as mp
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '1')

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'code', 'common')); sys.path.insert(0, HERE)

from fish_common import (load_fish, resample, P_GRID, N_BOOT, LOWER_BOUND,
                         interdecile, empirical_qf, w1)
from metalog.metalog_v2 import LogMetalog
from qflex.transforms import LogQFlex
from qflex.constraints import ConstraintType
import eval_framework as EF
import figure_style as FS

TAB = os.path.join(HERE, '..', 'tables')
import tplus_mode as TP                 # QPD_TPLUS_ONLY=1: supplementary QFlex-T+ run
TAB = TP.table_dir(TAB)
JITTER = 0.5      # overridden by --jitter; 0.5 lb is the paper's canonical width
KS = FS.K_STANDARD
# Model order follows the paper's convention (Metalog, QFlex-U, QFlex-TA+).
SPECS = [('Log Metalog', 'metalog'), ('Log QFlex-U', 'qflex_u'),
         ('Log QFlex-TA+', 'qflex_ta')]
CT = {'qflex_u': ConstraintType.NONE, 'qflex_ta': ConstraintType.TA, 'qflex_tl': ConstraintType.TL}
if TP.TPLUS_ONLY:
    SPECS = [('Log ' + TP.LABEL, 'qflex_tl')]


def fit(kind, x, p, K):
    try:
        m = (LogMetalog(x, p, lower_bound=LOWER_BOUND, terms=K) if kind == 'metalog'
             else LogQFlex(x, p, lower_bound=LOWER_BOUND, terms=K,
                           constraint_type=CT[kind]))
        return m if EF.is_valid(m) else None
    except Exception:
        return None


_X = _P = _EQF = _L = None


def _init(jitter=None):
    """Fitting at N=3474 is slow, so replicates run in a pool. Each worker
    rebuilds the shared reference once; the draws are seeded per replicate, so
    the result does not depend on how the work is split."""
    global _X, _P, _EQF, _L, JITTER
    # The jitter must be HANDED to each worker: on macOS, multiprocessing
    # starts workers with 'spawn', which re-imports this module and resets the
    # global JITTER to its default (0.5) -- so --jitter 1.0 silently ran 0.5.
    if jitter is not None:
        JITTER = jitter
    _X, _ = load_fish()
    n = len(_X)
    _P = np.arange(1, n + 1) / (n + 1)
    _EQF = empirical_qf(np.sort(_X), _P)
    _L = interdecile(np.sort(_X), _P)


def _rank_modes(locs, hgts):
    """(primary_loc, primary_height, secondary_loc, secondary_height).

    PRIMARY is the tallest peak of the fit, SECONDARY the next tallest -- the
    same rule fish_mode_scatter.py uses (there the secondary is the lower-height
    peak of a two-mode fit; ranking by height generalises it to fits with more
    than two).
    A fit with one mode has no secondary and contributes NaN, so the secondary
    columns are summarised over the multimodal fits only.
    """
    if locs is None or hgts is None or len(locs) == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    locs = np.asarray(locs, float); hgts = np.asarray(hgts, float)
    order = np.argsort(hgts)[::-1]                # tallest first
    i = order[0]
    if len(order) == 1:
        return (float(locs[i]), float(hgts[i]), np.nan, np.nan)
    j = order[1]
    return (float(locs[i]), float(hgts[i]), float(locs[j]), float(hgts[j]))


def _one(b):
    if _X is None:
        _init()
    s = np.sort(resample(_X, JITTER, b))
    out = []
    for label, kind in SPECS:
        for K in KS:
            m = fit(kind, s, _P, K)
            if m is None:
                out.append((label, K, False, np.nan, np.nan,
                            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)); continue
            nm, _l, _h = EF.modes(m)
            ww = EF.w1_both(m, s, _X)
            out.append((label, K, True, float(nm) if nm is not None else -1.0, ww['main'][1])
                       + _rank_modes(_l, _h) + (ww['own'][1], ww['orig'][1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jitter', type=float, default=0.5,
                    help='uniform jitter half-width in lb (0.5 = Table 10; '
                         'other widths are not in the paper)')
    a = ap.parse_args()
    global JITTER
    JITTER = a.jitter
    t0 = time.time()
    print(f'fish jitter +-{JITTER} lb, {N_BOOT} replicates, K={KS}', flush=True)
    rows = []
    with mp.Pool(4, initializer=_init, initargs=(JITTER,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_one, range(N_BOOT), chunksize=10), 1):
            rows.extend(r)
            if i % 250 == 0:
                print(f'  {i}/{N_BOOT}  {time.time()-t0:.0f}s', flush=True)

    df = pd.DataFrame(rows, columns=['Model', 'K', 'valid', 'n_modes', 'w1',
                                     'p_loc', 'p_ht', 's_loc', 's_ht',
                                     'w1_own', 'w1_orig'])
    df.to_csv(os.path.join(TAB, f'fish_summary_perreplicate_jitter{JITTER}.csv'),
              index=False)

    def _iqr(series):
        v = series.dropna()
        return round(float(v.quantile(.75) - v.quantile(.25)), 4) if len(v) else np.nan

    out = []
    for label, _k in SPECS:
        for K in KS:
            g = df[(df.Model == label) & (df.K == K)]
            v = g[g.valid]
            vm = v[v.n_modes >= 0]
            out.append(dict(
                Model=label, K=K, n_replicates=len(g),
                Validity_pct=round(100 * len(v) / len(g), 1), n_valid=len(v),
                Unimodal_pct=round(100 * float((vm.n_modes <= 1).mean()), 1) if len(vm) else np.nan,
                Bimodal_pct=round(100 * float((vm.n_modes == 2).mean()), 1) if len(vm) else np.nan,
                GT2_pct=round(100 * float((vm.n_modes > 2).mean()), 1) if len(vm) else np.nan,
                # Mode dispersion, raw (not normalised), in the data's own
                # units: lb for a location, 1/lb for a density height.
                # PRIMARY = tallest peak, over every fit that has one.
                # SECONDARY = next tallest, so only the multimodal fits
                # contribute; n_secondary says how many that is, because an
                # IQR over 3 fits is not the same claim as one over 900.
                n_with_mode=int(vm.p_loc.notna().sum()),
                PrimaryModeLocation_IQR_lb=_iqr(vm.p_loc),
                PrimaryModeHeight_IQR_perlb=_iqr(vm.p_ht),
                n_secondary=int(vm.s_loc.notna().sum()),
                SecondaryModeLocation_IQR_lb=_iqr(vm.s_loc),
                SecondaryModeHeight_IQR_perlb=_iqr(vm.s_ht),
                MedianW1_Eq6=round(float(v.w1.median()), 4) if len(v) else np.nan,
                W1_IQR=_iqr(v.w1),
                MedianW1_own_Eq6=round(float(v.w1_own.median()), 4) if len(v) else np.nan,
                MedianW1_orig_Eq6=round(float(v.w1_orig.median()), 4) if len(v) else np.nan,
            ))
    res = pd.DataFrame(out)
    res.insert(0, 'Jitter_lb', JITTER)
    dst = os.path.join(TAB, ('Table10_fish_summary_jitter0.5.csv' if JITTER == 0.5 and not TP.TPLUS_ONLY
                             else f'fish_summary_jitter{JITTER}.csv'))
    res.to_csv(dst, index=False)
    print('\n' + res.to_string(index=False))
    json.dump({'generated': datetime.datetime.now().isoformat(timespec='seconds'),
               'replicates': f'fish_common.resample, jitter {JITTER}, {N_BOOT} draws',
               'w1': 'Equation 6 at the sample points; _own vs the resample, _orig vs the '
                     'original recorded sample; main = eval_framework.BOOTSTRAP_W1_REFERENCE',
               'w1_main_reference': EF.BOOTSTRAP_W1_REFERENCE,
               'unimodal_convention': 'n_modes <= 1, so no-peak fits count as unimodal '
                                      '(same convention as Tables 7 and 8)',
               'mode_ranking': 'primary = tallest peak; secondary = next tallest '
                               '(the rule fish_mode_scatter.py uses). Secondary columns are '
                               'summarised over multimodal fits only; n_secondary '
                               'gives that denominator.',
               'mode_iqr_units': 'location lb, height 1/lb -- raw, not normalised',
               'per_replicate_dump': f'tables/fish_summary_perreplicate_jitter{JITTER}.csv',
               'jitter_lb': JITTER,
               'runtime_sec': round(time.time() - t0, 1)},
              open(os.path.join(TP.log_dir(os.path.join(HERE, '..', 'logs')), f'fish_summary_provenance_jitter{JITTER}.json'), 'w'),
              indent=2)
    print(f'\n-> {dst}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
