"""
Online Supplement Table O2 -- QPD fit summary for the inverse Gaussian Monte Carlo
(Online Supplement Section O4, N = 200), from the per-replicate output of mc_invgauss.py.

For each QPD and K = 4, 7, 10, 13:
  Validity %             share of the 1000 fits that are valid on the grid
  False Modality %       share of VALID fits with more than one mode
  Mode Height: Median/True
                         median height of the tallest mode over valid fits,
                         divided by the true inverse Gaussian mode height
  Median W1 %            median Equation-6 W1 over valid fits, x 100

usage:  python3 invgauss_table_O2.py
writes: tables/invgauss/TableO2_invgauss_n200.csv
"""
import os, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mc_invgauss as IGR

TAB = os.path.join(HERE, '..', 'tables', 'invgauss')


def table(n):
    d = pd.read_csv(os.path.join(TAB, f'mc_perrep_ig_n{n}.csv'))
    true_h = IGR.true_mode()[1]
    rows = []
    for m in IGR.MODELS:
        for K in IGR.KS:
            g = d[(d.Model == m) & (d.K == K)]; v = g[g.valid == True]
            rows.append(dict(N=n, Model=m, K=K,
                             Validity_pct=round(100 * len(v) / len(g), 1),
                             FalseModality_pct=round(100 * float((v.n_peaks > 1).mean()), 1) if len(v) else float('nan'),
                             ModeHeight_median_over_true=round(float(v.mode_height.median()) / true_h, 2) if len(v) else float('nan'),
                             MedianW1_pct=round(100 * float(v.w1.median()), 2) if len(v) else float('nan')))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TAB, f'TableO2_invgauss_n{n}.csv'), index=False)
    return t


if __name__ == '__main__':
    print(table(200).to_string(index=False))
