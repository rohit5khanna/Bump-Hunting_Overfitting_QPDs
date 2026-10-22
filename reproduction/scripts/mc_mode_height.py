"""
Mode-height accuracy for the Monte Carlo section: the mode height
median/true column of Table 4 (N = 200, --n 200) and Table A1 (N = 50, --n 50),
and the same column of Table A2 (QFlex-T+, both N). One run fits all five QPDs.

WHAT THIS MEASURES. Validity, false modality and W1 all say whether a fit
invents structure or sits close to the sample. None of them says how far the
fit distorts the PEAK. This reports the median fitted mode height as a ratio to
the reference distribution's true mode height, so 1.0 is exact, above 1 means
the fit overstates peak density, below 1 understates.

THREE VARIANTS ARE REPORTED, because "the fitted mode height" is ambiguous the
moment a fit has more than one mode:

  unimodal   Only fits with EXACTLY one detected mode. The comparison is then
             like for like -- one peak against one peak -- but the denominator
             shrinks at high K, where most fits are multimodal, and the fits it
             keeps are a self-selected well-behaved subset.

  tallest    Every valid fit with at least one mode; the TALLEST peak. This is
             what mc_tables_rebuild.py records per replicate as `mode_height`,
             so it is the quantity the Mode Height panels of Figures 3, 4 and
             O1-O4 plot. Use this one when the table must agree with those
             figures.

  nearest    Every valid fit with at least one mode; the peak whose LOCATION is
             closest to the true mode's location is taken as the primary. Keeps
             the full denominator and asks "did the fit get the real peak
             right", ignoring whatever spurious peaks it added elsewhere.

None is obviously the right one, which is why all three are here.

TRUE MODE HEIGHT is measured exactly like a fit's (eval_framework): the
reference quantile on the 0.001..0.999 grid, its grid-difference density, the
same find_peaks rule; the tallest peak.

FRAMEWORK: validity, density and modes come from code/common/eval_framework.py
(grid 0.001..0.999, density from quantile differences, interior peaks with
relative prominence >= 1% (of the peak's own height)).

A RATIO OF MEDIANS EQUALS A MEDIAN OF RATIOS here, since the true height is one
constant per reference distribution. The header may say either.

Everything else -- seeds, bounds by support, the grid -- is imported from
mc_tables_rebuild so the two scripts cannot drift.
"""
import os, sys, argparse, json, datetime, warnings
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '1')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mc_tables_rebuild import (DISTS, KS, MODELS, TPLUS, DIST_PARAMS, P_GRID, N_REPS,
                               TAB, draw, fit)
import eval_framework as EF


def true_mode_height(cls):
    """Peak density of the reference distribution, measured like a fit."""
    d = cls(**DIST_PARAMS)
    qf = np.asarray(d.quantile(P_GRID), float)
    n, locs, hgts = EF.modes_from_density(qf, EF.density_from_quantile(qf))
    if hgts is None or not len(hgts):
        raise RuntimeError('reference distribution has no detectable mode')
    i = int(np.argmax(hgts))
    return float(hgts[i]), float(locs[i]), int(n)


def run(short, n=200, n_reps=N_REPS):
    name, cls, support = DISTS[short]
    h_true, x_true, n_true = true_mode_height(cls)
    d = cls(**DIST_PARAMS)
    p_pos = np.arange(1, n + 1) / (n + 1)

    rows = []
    for model in MODELS:
        for K in KS:
            uni, near, tall = [], [], []
            n_valid = n_usable = 0
            for rep in range(1, n_reps + 1):
                x = draw(d, n, rep)
                m = fit(model, x, p_pos, K, support)
                if m is None:
                    continue
                n_valid += 1
                nm, locs, hgts = EF.modes(m)
                if nm is None or nm < 0:           # unusable PDF: out of every denominator
                    continue
                n_usable += 1
                if nm < 1 or hgts is None or not len(hgts):
                    continue                        # flat fit: no height to contribute
                if nm == 1:
                    uni.append(float(hgts[0]))
                near.append(float(hgts[int(np.argmin(np.abs(np.asarray(locs, float) - x_true)))]))
                # TALLEST peak -- the rule mc_tables_rebuild.py uses for
                # `mode_height`, i.e. the quantity Figures 3, 4, O1-O4 plot.
                tall.append(float(np.max(hgts)))

            med = lambda v: float(np.median(v)) if v else np.nan
            rows.append(dict(
                Distribution=name, N=n, Model=model, K=K,
                n_valid=n_valid, n_usable=n_usable,
                true_mode_height=round(h_true, 6),
                n_unimodal=len(uni),
                MedianHeight_unimodal=round(med(uni), 4) if uni else np.nan,
                Ratio_unimodal=round(med(uni) / h_true, 3) if uni else np.nan,
                n_with_mode=len(near),
                MedianHeight_nearest=round(med(near), 4) if near else np.nan,
                Ratio_nearest=round(med(near) / h_true, 3) if near else np.nan,
                MedianHeight_tallest=round(med(tall), 4) if tall else np.nan,
                Ratio_tallest=round(med(tall) / h_true, 3) if tall else np.nan,
            ))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dists', default='su,sl,sb')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--n-reps', type=int, default=N_REPS)
    a = ap.parse_args()

    out = []
    for short in a.dists.split(','):
        df = run(short, a.n, a.n_reps)
        out.append(df)
        print(df.to_string(index=False), flush=True)
        print(flush=True)
    allm = pd.concat(out, ignore_index=True)
    stem4 = {200: 'Table04_mode_height_N200', 50: 'TableA1_mode_height_N50'}.get(a.n, f'mc_mode_height_N{a.n}')
    for stem, full in ((stem4, allm[allm.Model != TPLUS]),                    # Table 4 / A1
                       (f'TableA2_mode_height_N{a.n}', allm[allm.Model == TPLUS])):  # Table A2
        path = os.path.join(TAB, f'{stem}.csv')
        if os.path.exists(path) and a.dists != 'su,sl,sb':
            prev = pd.read_csv(path)
            prev = prev[~prev.Distribution.isin(full.Distribution.unique())]
            full = pd.concat([prev, full], ignore_index=True)
        full.to_csv(path, index=False)
        print(f'Saved -> {path}')
