"""
Table 7 -- Monte Carlo fit results for Metalog and QFlex-TA+ fits to the bimodal
reference distribution (Section 4.6), from the per-replicate output of
bimodal_mc.py (tables/bimodal_mc.csv): 1000 independent samples of N = 200.

A valid fit with no interior peak counts as unimodal (the paper's convention,
column Unimodal_pct); Unimodal_pct_strict and zero_mode_pct report those fits
separately.

W1 is Equation 6 at the replicate's own sample points (eval_framework.w1).

writes tables/Table07_bimodal_mc.csv
"""
import os, sys, json, datetime, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from bimodal_common import K_VALUES, SPECS, N_REPLICATES, SAMPLE_SIZE

TAB = os.path.join(HERE, '..', 'tables'); LOG = os.path.join(HERE, '..', 'logs')


def main():
    d = pd.read_csv(os.path.join(TAB, 'bimodal_mc.csv'))
    rows = []
    for lab, _k in SPECS:
        for K in K_VALUES:
            g = d[(d['model'] == lab) & (d['K'] == K)]
            v = g[g['feasible'] & g['n_modes'].notna()]
            n, nv = len(g), len(v)
            m = v['n_modes'].values if nv else np.array([])
            rows.append(dict(
                model=lab, K=K, n_replicates=n, n_valid=nv,
                Valid_pct=round(100 * nv / n, 1),
                MaxModes=int(m.max()) if nv else np.nan,
                zero_mode_pct=round(100 * float(np.mean(v['n_peaks'].values == 0)), 1) if nv else np.nan,   # no interior peak (counted as 1 mode)
                Unimodal_pct=round(100 * np.mean(m <= 1), 1) if nv else np.nan,
                Unimodal_pct_strict=round(100 * np.mean(m == 1), 1) if nv else np.nan,
                Bimodal_pct=round(100 * np.mean(m == 2), 1) if nv else np.nan,
                GT2_pct=round(100 * np.mean(m > 2), 1) if nv else np.nan,
                MedianW1_vs_sample=round(float(v['w1_eq6_sample'].median()), 4) if nv else np.nan,
                MedianW1_raw_sample=round(float(v['w1_raw_sample'].median()), 4) if nv else np.nan))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TAB, 'Table07_bimodal_mc.csv'), index=False)

    print(t[['model', 'K', 'Valid_pct', 'MaxModes', 'Unimodal_pct', 'zero_mode_pct',
             'Bimodal_pct', 'GT2_pct', 'MedianW1_vs_sample']].to_string(index=False))

    print('\nSummary:')
    for lab, _k in SPECS:
        s = t[t['model'] == lab]
        first_bi = s[s['Bimodal_pct'] >= 50]['K'].min()
        first_sp = s[s['GT2_pct'] >= 10]['K'].min()
        print(f'  {lab:<11} first K with >=50% bimodal: {first_bi}; '
              f'first K with >=10% spurious (>2 modes): {first_sp}; '
              f'W1 (Eq 6, vs sample) {s["MedianW1_vs_sample"].iloc[0]:.4f} -> '
              f'{s["MedianW1_vs_sample"].iloc[-1]:.4f}')

    json.dump({
        'generated': datetime.datetime.now().isoformat(timespec='seconds'),
        'source': 'tables/bimodal_mc.csv (1000 independent draws, N=200)',
        'mode_convention': 'Unimodal_pct counts fits with no interior peak as unimodal '
                           '(as in the paper); Unimodal_pct_strict and zero_mode_pct separate them',
        'w1_columns': {'MedianW1_raw_sample': 'unnormalized, at the replicate\'s own sample points',
                       'MedianW1_vs_sample': 'Equation 6 at the replicate\'s own sample points'},
        'n_replicates': N_REPLICATES, 'sample_size': SAMPLE_SIZE,
    }, open(os.path.join(LOG, 'Table07_bimodal_provenance.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
