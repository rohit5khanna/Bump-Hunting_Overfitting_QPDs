# Inverse Gaussian Monte Carlo (Online Supplement Section O4)

Non-Johnson semi-bounded reference: IG(mu = 0.919, lambda = 1.069) on (0, inf), matched to the
Johnson SL reference (xi = 0, lambda = 1, gamma = 0.5, delta = 1.2) at P10 / P50 / P90:

|            | P10    | P50    | P90    |
|---|---|---|---|
| Johnson SL | 0.2266 | 0.6592 | 1.9180 |
| IG         | 0.2407 | 0.6501 | 1.9184 |

IG mode: x = 0.3146, density 1.121 (Figure O8).

Design identical to the Johnson SL Monte Carlo: N = 50 and 200, 1000 replicates, K = 3..13,
Log transform with the true lower bound 0, five QPDs (Metalog, QFlex-U, QFlex-T+, QFlex-TA+,
QFlex-A+), the paper's evaluation framework. Replicate r uses the same uniforms as Johnson SL
replicate r (seed 42 + N*1000 + r), pushed through the IG quantile function instead.

Files
- `TableO2_invgauss_n200.csv` -- Table O2 (N = 200), from `scripts/invgauss_table_O2.py`
- `invgauss_summary_n50.csv` -- the same summary for N = 50 (not in the paper)
- `mc_perrep_ig_n{50,200}.csv` -- one row per replicate, model and K (`scripts/mc_invgauss.py`)
- `mc_rebuilt_ig_n{50,200}.csv` -- validity, false modality and median W1 at K = 4, 7, 10, 13
- `invgauss_vs_sl_K4_7_10_13.csv` -- inverse Gaussian next to Johnson SL (not in the paper)
