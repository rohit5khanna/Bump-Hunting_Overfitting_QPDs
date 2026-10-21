"""
Hydrology -- the data-level modality tests quoted in the text of Section 5.1:

    "The Hartigan Dip and Silverman tests yield p-values of 0.55 and 0.31.
     Thus, neither test rejects unimodality at the 5% significance level."

This script produces those two numbers (tables/hydrology_modality_tests.csv;
not a paper table). Both tests are applied to the 95 observed gauge heights --
the raw sample, not bootstrap resamples.

IMPLEMENTATIONS. Both come from the repo's own common/stats_tests.py so the
section uses the same code as everywhere else:
  * Hartigan -- apply_hartigan_dip_test, a thin wrapper over the compiled
    `diptest` package.
  * Silverman -- apply_silverman_test, the critical-bandwidth test with
    k=1 and a smoothed bootstrap. It is NOT calibrated: the Hall & York
    (2001) adjustment available in the same module is not applied, matching
    what the manuscript reports. The calibrated value is printed alongside
    for reference so the difference is visible, not hidden.

SILVERMAN IS STOCHASTIC. Its p-value comes from a bootstrap, so it moves
between runs unless the seed is fixed. --seed pins it; --repeat reports the
spread across seeds, which is the honest way to quote a number that is not
deterministic.
"""
import os, sys, json, argparse, datetime, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'code', 'common'))
DATA = os.path.join(REPO, 'data', 'Hydrology.xlsx')
OUT  = os.path.join(HERE, '..', 'tables')
LOG  = os.path.join(HERE, '..', 'logs')

from stats_tests import apply_hartigan_dip_test, apply_silverman_test
try:
    from stats_tests import silverman_test_calibrated
except ImportError:
    silverman_test_calibrated = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-bootstrap', type=int, default=999)
    ap.add_argument('--repeat', type=int, default=5,
                    help='independent seeds, to show the spread of the stochastic test')
    a = ap.parse_args()

    x = np.sort(pd.read_excel(DATA)['Gauge Height (ft)'].dropna().values.astype(float))
    print(f'N = {len(x)}, range [{x.min():.3f}, {x.max():.3f}] ft')

    h = apply_hartigan_dip_test(x)
    print(f'\nHartigan dip test        p = {h:.4f}    (deterministic)')

    sil = []
    for i in range(a.repeat):
        np.random.seed(a.seed + i)
        sil.append(apply_silverman_test(x, n_bootstrap=a.n_bootstrap, k=1))
    sil = np.array([s for s in sil if s is not None], dtype=float)
    print(f'Silverman (k=1, B={a.n_bootstrap}, uncalibrated)')
    print(f'  seed {a.seed}            p = {sil[0]:.4f}')
    print(f'  across {len(sil)} seeds       min {sil.min():.4f}  '
          f'median {np.median(sil):.4f}  max {sil.max():.4f}')

    cal = None
    if silverman_test_calibrated is not None:
        try:
            np.random.seed(a.seed)
            raw = silverman_test_calibrated(x, n_bootstrap=a.n_bootstrap, k=1)
            # the helper's return type is not fixed: scalar, tuple, or a dict
            # keyed 'p_value'/'p'. Unwrap all three rather than indexing blind
            # (cal[0] on a dict raised KeyError and left `cal` holding the dict,
            # which then blew up in float() at the CSV step).
            if isinstance(raw, dict):
                # stats_tests returns {'p_raw': ..., 'p_hall_york': ...}
                for k_ in ('p_hall_york', 'p_value', 'p', 'pvalue'):
                    if k_ in raw:
                        raw = raw[k_]
                        break
            elif isinstance(raw, (tuple, list, np.ndarray)):
                raw = raw[0]
            cal = float(raw)
            print(f'Silverman, Hall-York calibrated   p = {cal:.4f}   '
                  f'(NOT what the manuscript reports)')
        except Exception as e:
            cal = None
            print(f'  (calibrated variant unavailable: {type(e).__name__}: {e})')

    print(f'\nmanuscript quotes: Hartigan 0.55, Silverman 0.31')
    print('neither rejects unimodality at the 5% level'
          if h > 0.05 and sil[0] > 0.05 else 'CHECK: a test rejects at 5%')

    os.makedirs(OUT, exist_ok=True); os.makedirs(LOG, exist_ok=True)
    pd.DataFrame([{'test': 'Hartigan dip', 'p_value': round(float(h), 4),
                   'deterministic': True, 'n_bootstrap': None},
                  {'test': 'Silverman (k=1, uncalibrated)', 'p_value': round(float(sil[0]), 4),
                   'deterministic': False, 'n_bootstrap': a.n_bootstrap}] +
                 ([{'test': 'Silverman (Hall-York calibrated)', 'p_value': round(float(cal), 4),
                    'deterministic': False, 'n_bootstrap': a.n_bootstrap}] if cal is not None else [])
                 ).to_csv(os.path.join(OUT, 'hydrology_modality_tests.csv'), index=False)
    json.dump({'generated': datetime.datetime.now().isoformat(timespec='seconds'),
               'N': int(len(x)), 'seed': a.seed, 'n_bootstrap': a.n_bootstrap,
               'hartigan_p': float(h),
               'silverman_p_seed': float(sil[0]),
               'silverman_p_across_seeds': {'min': float(sil.min()),
                                            'median': float(np.median(sil)),
                                            'max': float(sil.max()),
                                            'n_seeds': int(len(sil))},
               'silverman_calibrated_p': (float(cal) if cal is not None else None),
               'manuscript_quotes': {'hartigan': 0.55, 'silverman': 0.31},
               'note': 'Silverman is stochastic; quote it with its seed and B.'},
              open(os.path.join(LOG, 'hydrology_modality_tests.json'), 'w'), indent=2)
    print(f"\n-> {os.path.join(OUT, 'hydrology_modality_tests.csv')}")


if __name__ == '__main__':
    main()
