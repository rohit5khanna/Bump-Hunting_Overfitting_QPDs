"""Figure 5 -- percentage of ALL Monte Carlo fits that are valid AND unimodal,
by model (including QFlex-T+) and Johnson distribution, N = 200, K = 10.

    valid-and-unimodal % = Validity % x (1 - False Modality %)
                         = 100 * #(valid on G and n_peaks <= 1) / #replicates

Computed directly from the per-replicate MC files (1000 replicates per cell):
    tables/mc_perrep_<d>_n<N>.csv        Metalog, QFlex-U, QFlex-TA+, QFlex-A+
    tables/tplus/mc_perrep_<d>_n<N>.csv  QFlex-T+
A fit with no interior peak counts as unimodal (framework rule).

Style: figure_style.rcparams(). Bars are coloured by DISTRIBUTION, so they use
figure_style.REFERENCE (the Figure 1 colours for SU/SL/SB); the four QPD
colours stay reserved for models. Each distribution also gets a hatch
(SU solid, SL diagonal, SB dotted) so the figure reads in grayscale print.

Usage: python3 mc_valid_unimodal_figure.py [--N 200] [--K 10]
Writes figures/Figure05_valid_unimodal_N200_K10.{png,pdf} for the defaults
       (any other N, K: figures/other/mc_valid_unimodal_n<N>_K<K>_withTplus.*,
       not in the paper)
       tables/tplus/mc_valid_unimodal_n<N>_K<K>_withTplus.csv
"""
import os, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import figure_style as FS

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, '..', 'tables')
FIG = os.path.join(HERE, '..', 'figures')
MODELS = ['Metalog', 'QFlex-U', 'QFlex-T+', 'QFlex-TA+', 'QFlex-A+']
DISTS = [('su', 'Johnson SU'), ('sl', 'Johnson SL'), ('sb', 'Johnson SB')]
#: One fill pattern per distribution, so the bars stay distinguishable in grayscale print.
HATCH = ['', '////', '....']

ap = argparse.ArgumentParser()
ap.add_argument('--N', type=int, default=200)
ap.add_argument('--K', type=int, default=10)
a = ap.parse_args()

rows = []
for d, dname in DISTS:
    per = pd.concat([pd.read_csv(os.path.join(TAB, f'mc_perrep_{d}_n{a.N}.csv')),
                     pd.read_csv(os.path.join(TAB, 'tplus', f'mc_perrep_{d}_n{a.N}.csv'))])
    per = per[per.K == a.K]
    for m in MODELS:
        s = per[per.Model == m]
        assert len(s) == 1000 and s.rep.is_unique, (d, m, len(s))
        v = s.valid.astype(bool)
        uni = v & (s.n_peaks.fillna(0) <= 1)
        nv = int(v.sum())
        fm = 100 * (v & (s.n_peaks >= 2)).sum() / nv if nv else np.nan
        rows.append({'Distribution': dname, 'N': a.N, 'K': a.K, 'Model': m,
                     'replicates': len(s), 'valid': nv, 'valid_unimodal': int(uni.sum()),
                     'Validity_pct': 100 * nv / len(s), 'FalseModality_pct': fm,
                     'ValidUnimodal_pct': 100 * uni.sum() / len(s)})
t = pd.DataFrame(rows)
# identity check: valid-and-unimodal = validity x (1 - false modality)
chk = t.Validity_pct * (1 - t.FalseModality_pct.fillna(0) / 100)
assert np.allclose(chk, t.ValidUnimodal_pct), 'identity failed'

FS.apply()
plt.rcParams['hatch.linewidth'] = 0.7
fig, ax = plt.subplots(figsize=(8.5, 4.6))
x = np.arange(len(MODELS)); w = 0.26
for j, (d, dname) in enumerate(DISTS):
    y = [t[(t.Distribution == dname) & (t.Model == m)].ValidUnimodal_pct.iloc[0] for m in MODELS]
    bars = ax.bar(x + (j - 1) * w, y, w, color=FS.REFERENCE[j], label=dname, hatch=HATCH[j],
                  edgecolor='#2b2b2b', linewidth=0.7, zorder=3)
    for b, val in zip(bars, y):
        ax.text(b.get_x() + b.get_width() / 2, val + 1.2, f'{val:.1f}',
                ha='center', va='bottom', fontsize=7.5)
ax.set_xticks(x, MODELS)
ax.set_ylim(0, 108)
ax.set_yticks(range(0, 101, 25))
ax.set_ylabel('Valid and unimodal fits (% of all replicates)')
ax.set_title(f'Valid and unimodal fits, Monte Carlo (N = {a.N}, K = {a.K})')
ax.grid(axis='x', visible=False)
ax.legend(loc='upper left', ncol=3, handlelength=2.2, handleheight=1.2)
fig.tight_layout()
os.makedirs(FIG, exist_ok=True)
name = f'mc_valid_unimodal_n{a.N}_K{a.K}_withTplus'
fpath = FS.figure_path('Figure05_valid_unimodal_N200_K10' if (a.N, a.K) == (200, 10) else name)
fig.savefig(fpath + '.png', dpi=300, bbox_inches='tight')
fig.savefig(fpath + '.pdf', bbox_inches='tight')
os.makedirs(os.path.join(TAB, 'tplus'), exist_ok=True)
t.round(2).to_csv(os.path.join(TAB, 'tplus', name + '.csv'), index=False)
print(t[['Distribution', 'Model', 'Validity_pct', 'FalseModality_pct', 'ValidUnimodal_pct']].round(1).to_string(index=False))
print('saved', name)
