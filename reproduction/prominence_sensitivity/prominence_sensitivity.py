"""
Sensitivity of mode identification to the relative-prominence threshold
(Online Supplement Section O1).

The paper (eval_framework.RELATIVE_PROMINENCE = 0.01; Section 3.3) counts an
interior local maximum of the grid-difference density as a mode when its
prominence is at least 1% of its own height. Here every fit of every section
is redone with the section's OWN sampler and fitting function, and for each
valid fit we store the relative prominence r = prominence / height of EVERY
local maximum. The mode count at any threshold t is then
n_peaks(t) = #{r >= t}, n_modes = max(1, .), so one fit serves all thresholds
(0.5%, 1%, 2%, 5%) and the 1% column must reproduce the paper's numbers
exactly (checked by check_1pct.py).

summarize.py turns the partials into Table O1 (TableO1_prominence_JohnsonSU.csv,
Johnson SU only), the workbook excel/Prominence_Sensitivity.xlsx and
prominence_sensitivity_summary.csv (every section).

usage:  python3 prominence_sensitivity.py SECTION --lo 1 --hi 1000 [--procs 4]
SECTION in: mc_su_50 mc_su_200 mc_sl_50 mc_sl_200 mc_sb_50 mc_sb_200
            ig_50 ig_200 bimodal bootstrap hydrology geyser fish_0.5 fish_1.0
Writes partials/<SECTION>_<lo>_<hi>.csv (one row per fit).
"""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '1')
import sys, argparse, warnings, time
import multiprocessing as mp
import numpy as np, pandas as pd
from scipy.signal import find_peaks
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
S = os.path.abspath(os.path.join(HERE, '..', 'scripts'))
sys.path.insert(0, S); sys.path.insert(0, os.path.join(S, '..', '..', 'code', 'common'))
import eval_framework as EF
from qflex.constraints import ConstraintType as CTy
PART = os.path.join(HERE, 'partials'); os.makedirs(PART, exist_ok=True)


def ratios_from(q, f):
    """All interior local maxima: relative prominence, tallest-first order irrelevant.
    None when the density is unusable (same test as EF.modes_from_density)."""
    q, f = np.asarray(q, float), np.asarray(f, float)
    if f.shape != EF.GRID.shape or not np.all(np.isfinite(f)) or not np.all(f > 0):
        return None
    pk, props = find_peaks(f, prominence=0.0)
    return np.sort(props['prominences'] / f[pk])[::-1]


def enc(r):
    return 'NA' if r is None else ';'.join(f'{v:.6g}' for v in r)


# ----------------------------------------------------------------- sections
SEC = None


def setup(section):
    """Return (reps_fn(rep) -> list of (setting, model, K, ratios-or-None-or-'invalid'))."""
    global SEC
    if section.startswith('mc_') or section.startswith('ig_'):
        import mc_tables_rebuild as MTR
        models = ['Metalog', 'QFlex-U', 'QFlex-T+', 'QFlex-TA+', 'QFlex-A+']
        if section.startswith('mc_'):
            _, short, n = section.split('_'); n = int(n)
            name, cls, support = MTR.DISTS[short]
            d = cls(**MTR.DIST_PARAMS)
            draw = lambda rep: MTR.draw(d, n, rep)
            setting = f'{name}, N={n}'
        else:
            import mc_invgauss as IGR
            n = int(section.split('_')[1]); support = 'semi'
            draw = lambda rep: IGR.draw(n, rep)
            setting = f'Inverse Gaussian, N={n}'
        p = EF.plotting_positions(n)

        def one(rep):
            x = draw(rep); out = []
            for mo in models:
                for K in (4, 7, 10, 13):
                    m = MTR.fit(mo, x, p, K, support)
                    out.append((setting, mo, K, 'invalid' if m is None else ratios_from(*EF.density(m))))
            return out
        return one

    if section == 'bimodal':
        import bimodal_mc as BM, bimodal_common as BC
        specs = list(BC.SPECS) + [('QFlex-T+', 'qflex_tl')]

        def one(rep):
            x = BC.draw(rep, BC.SAMPLE_SIZE); p = np.arange(1, len(x) + 1) / (len(x) + 1)
            out = []
            for lab, key in specs:
                for K in BC.K_VALUES:
                    m = BM._fit(key, x, p, K)
                    out.append(('Bimodal Johnson SU mixture, N=200', lab, K,
                                'invalid' if m is None else ratios_from(*EF.density(m))))
            return out
        return one

    if section == 'bootstrap':
        import bootstrap_sweep as BS, bootstrap_section as BSec
        BS._init()
        specs = list(BS.SPECS) + [('QFlex-T+', 'qflex', CTy.TL)]

        def one(rep):
            b = rep - 1                               # sweep uses b = 0..999
            s = np.sort(BSec.bootstrap_resample(BS._XU, b)); out = []
            for lab, kind, ct in specs:
                for K in (4, 7, 10, 13):
                    m = BS._fit(kind, ct, s, K)
                    out.append(('Johnson SU bootstrap, N=200', lab, K,
                                'invalid' if m is None else ratios_from(*EF.density(m))))
            return out
        return one

    if section == 'hydrology':
        import hydrology_sweep as HS, hydrology_common as HC
        HS._init()
        specs = [('GEV', 'gev', 0)] + list(HS.SPECS) + [('Log QFlex-T+', 'qflex_tl', K) for K in (4, 7, 10)]

        def one(rep):
            b = rep - 1
            s = np.sort(HC.resample(HS._X, b=b)); out = []
            for lab, key, K in specs:
                o = HS.fit_curve(key, s, K)
                out.append(('River gauge (hydrology) bootstrap', lab, K,
                            'invalid' if o is None else ratios_from(o[0], o[1])))
            return out
        return one

    if section == 'geyser':
        import geyser_sweep as GS, geyser_common as GC
        GS._init()
        specs = list(GS.SPECS) + [('Log QFlex-T+', 'qflex_tl')]

        def one(rep):
            b = rep - 1
            s = np.sort(GC.resample(GS._X, GS.JITTER, b)); out = []
            for lab, key in specs:
                for K in GS.K_VALUES:
                    m = GS._fit(key, s, K)
                    out.append((f'Old Faithful geyser bootstrap, jitter {GS.JITTER}', lab, K,
                                'invalid' if m is None else ratios_from(*EF.density(m))))
            return out
        return one

    if section.startswith('fish_'):
        import fish_summary as FSm, fish_common as FC
        jit = float(section.split('_')[1])
        FSm._init(jit)
        specs = list(FSm.SPECS) + [('Log QFlex-T+', 'qflex_tl')]

        def one(rep):
            b = rep - 1
            s = np.sort(FC.resample(FSm._X, jit, b)); out = []
            for lab, kind in specs:
                for K in FSm.KS:
                    m = FSm.fit(kind, s, FSm._P, K)
                    out.append((f'Fish weights bootstrap, jitter {jit} lb', lab, K,
                                'invalid' if m is None else ratios_from(*EF.density(m))))
            return out
        return one
    raise SystemExit(f'unknown section {section}')


_ONE = None


def _work(args):
    global _ONE
    section, rep = args
    if _ONE is None:
        _ONE = setup(section)
    return [(rep,) + r for r in _ONE(rep)]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('section'); ap.add_argument('--lo', type=int, default=1)
    ap.add_argument('--hi', type=int, default=1000); ap.add_argument('--procs', type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    with mp.Pool(a.procs) as pool:
        rows = [r for rr in pool.imap(_work, [(a.section, r) for r in range(a.lo, a.hi + 1)], chunksize=2)
                for r in rr]
    df = pd.DataFrame([dict(section=a.section, rep=rep, setting=st, model=mo, K=K,
                            valid=not (isinstance(rt, str) and rt == 'invalid'),
                            ratios=('' if isinstance(rt, str) else enc(rt)))
                       for rep, st, mo, K, rt in rows])
    out = os.path.join(PART, f'{a.section}_{a.lo:04d}_{a.hi:04d}.csv')
    df.to_csv(out, index=False)
    print(f'{a.section} reps {a.lo}-{a.hi}: {len(df)} fits in {time.time() - t0:.1f}s -> {os.path.basename(out)}')
