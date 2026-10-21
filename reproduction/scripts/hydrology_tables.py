"""
Table 8 -- stability of model features for bootstrap resamples of the river
gauge height data (Section 5.1), from tables/hydrology_sweep.csv (the same 1000
paired replicates as Figures 12 and 13: GEV and every QPD fitted to the same
resample).

IQRs are raw: mode location in feet, mode height in 1/ft; W1 in feet and in
Equation 6 units. The primary mode is the tallest peak of each fit.

`n_flat_no_peak_detected` counts valid fits whose density has no interior peak
on the 0.001..0.999 grid. The paper counts these as unimodal
(`modes_1_pct`); `modes_1_pct_strict` excludes them.

writes tables/Table08_hydrology.csv
       (QFlex-T+ run: tables/tplus/hydrology_summary_QFlexTplus.csv, not in the paper)
"""
import os, sys, json, datetime, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from hydrology_common import K_VALUES, load_hydrology, interdecile, N_BOOT, data_fingerprint

TAB = os.path.join(HERE, '..', 'tables'); LOG = os.path.join(HERE, '..', 'logs')
import tplus_mode as TP                 # QPD_TPLUS_ONLY=1: supplementary QFlex-T+ run
TAB, LOG = TP.table_dir(TAB), TP.log_dir(LOG)

ORDER = ([('GEV', 0)] +
         [('Log Metalog', K) for K in K_VALUES] +
         [('Log QFlex-A+', K) for K in K_VALUES])
if TP.TPLUS_ONLY:
    ORDER = [('Log ' + TP.LABEL, K) for K in K_VALUES]


def iqr(v):
    return float(np.subtract(*np.percentile(v, [75, 25]))) if len(v) else np.nan


def main():
    d = pd.read_csv(os.path.join(TAB, 'hydrology_sweep.csv'))
    x, p = load_hydrology(); L = interdecile(x, p)
    rows = []
    for model, K in ORDER:
        g = d[(d['model'] == model) & (d['K'] == K)]
        v = g[g['feasible'] & g['n_modes'].notna()]
        n, nv = len(g), len(v)
        m = v['n_modes'].values if nv else np.array([])
        locs, hts = [], []
        for _, r in v.iterrows():
            # A flat fit stores an empty cell, which pandas reads as NaN and
            # str() turns into 'nan' -- float('nan') would then succeed, and a
            # single no-peak fit would poison np.percentile and blank the whole
            # row's IQRs (only QFlex-A+ has flat fits). Skip those tokens.
            ll = [float(t) for t in str(r['mode_locs']).split(';')
                  if t and t.lower() != 'nan']
            hh = [float(t) for t in str(r['mode_heights']).split(';')
                  if t and t.lower() != 'nan']
            if ll and hh:
                i = int(np.argmax(hh)); locs.append(ll[i]); hts.append(hh[i])
        label = model if K == 0 else f'{model}-{K}'
        rows.append(dict(
            Model=label, n_total=n, n_valid=nv,
            ValidFits_pct=round(100 * nv / n, 1),
            ModeLocation_IQR_ft=round(iqr(np.array(locs)), 4),
            ModeHeight_IQR_perft=round(iqr(np.array(hts)), 4),
            MedianW1_Eq6=round(float(v['w1_eq6'].median()), 5) if nv else np.nan,
            MedianW1_unnormalized_ft=round(float(v['w1_ft'].median()), 4) if nv else np.nan,
            MedianW1_own_Eq6=round(float(v['w1_own_eq6'].median()), 5) if nv else np.nan,
            MedianW1_orig_Eq6=round(float(v['w1_orig_eq6'].median()), 5) if nv else np.nan,
            MedianW1_own_ft=round(float(v['w1_own_ft'].median()), 4) if nv else np.nan,
            MedianW1_orig_ft=round(float(v['w1_orig_ft'].median()), 4) if nv else np.nan,
            n_flat_no_peak_detected=int(np.sum(v['n_peaks'].values == 0)) if nv else 0,
            modes_1_pct_strict=round(100 * float(np.mean(v['n_peaks'].values == 1)), 1) if nv else np.nan,
            modes_1_pct=round(100 * np.mean(m <= 1), 1) if nv else np.nan,
            modes_2_pct=round(100 * np.mean(m == 2), 1) if nv else np.nan,
            modes_gt2_pct=round(100 * np.mean(m > 2), 1) if nv else np.nan))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TAB, TP.name('Table08_hydrology', 'hydrology_summary_QFlexTplus') + '.csv'), index=False)

    print(t[['Model', 'ValidFits_pct', 'modes_1_pct', 'n_flat_no_peak_detected', 'modes_2_pct',
             'modes_gt2_pct', 'ModeLocation_IQR_ft', 'ModeHeight_IQR_perft',
             'MedianW1_Eq6']].to_string(index=False))

    fn, nn, sha = data_fingerprint()
    json.dump({
        'generated': datetime.datetime.now().isoformat(timespec='seconds'),
        'data_file': fn, 'data_N': nn, 'data_sha256': sha,
        'source': 'tables/hydrology_sweep.csv -- the same 1000 paired replicates '
                  'as Figures 12 and 13',
        'iqr_convention': 'RAW: location in ft, height in 1/ft',
        'mode_convention': 'modes_1_pct counts no-peak fits as unimodal (as in the paper); '
                           'modes_1_pct_strict excludes them',
        'n_bootstrap': N_BOOT, 'interdecile_L_ft': L,
    }, open(os.path.join(LOG, TP.name('Table08_hydrology_provenance', 'hydrology_summary_QFlexTplus_provenance') + '.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
