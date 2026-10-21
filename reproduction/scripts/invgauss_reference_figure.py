"""
Online Supplement Figure O8 (Section O4): the inverse Gaussian reference ("true
source") distribution, drawn in the style of a Figure 1 panel. IG(mu = 0.919,
lambda = 1.069) on (0, inf), the non-Johnson semi-bounded reference of
mc_invgauss.py. Deterministic (no sampling).

  figures/FigureO8_inverse_gaussian_reference.png / .pdf
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib as mpl
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import figure_style as FS
import mc_invgauss as IGR

mpl.rcParams.update({'font.family': 'sans-serif', 'font.size': 11, 'axes.titlesize': 12,
                     'axes.labelsize': 10, 'figure.dpi': 150, 'savefig.dpi': 300,
                     'axes.grid': True, 'grid.alpha': 0.3})
P = np.linspace(1e-4, 1 - 1e-4, 4000)          # same grid as Figure 1 ...
# ... plus a log-spaced lower tail, so the curve is drawn down to x ~ 0 where the IG density
# is 0, instead of starting in mid-air at the 0.01% quantile (x = 0.06); as for Figure 1's SB.
P = np.concatenate([np.geomspace(1e-14, 1e-4, 300, endpoint=False), P])
qf = IGR.IG.ppf(P); pdf = IGR.IG.pdf(qf)
fig, ax = plt.subplots(figsize=(4.9, 4.4))     # one Figure-1 panel
ax.plot(qf, pdf, color=FS.REFERENCE[1], linewidth=1.8, zorder=3)   # Figure 1's semi-bounded colour
ax.set_title(f'Inverse Gaussian  (semi-bounded)\nμ = {IGR.MU}, λ = {IGR.LAMBDA}', fontweight='bold', pad=8)
ax.set_xlabel('x'); ax.set_ylabel('Density'); ax.set_ylim(bottom=0)
ax.set_xlim(0, float(np.percentile(qf, 99.8))); ax.margins(y=0.06)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(FS.figure_path('FigureO8_inverse_gaussian_reference') + f'.{ext}', bbox_inches='tight')
i = int(np.argmax(pdf))
print(f'mode {qf[i]:.4f}  height {pdf[i]:.4f}  -> {FS.figure_path("FigureO8_inverse_gaussian_reference")}.png')
