"""
Fish -- paper Figure 15: histogram of recorded weights (0.5-lb bins) with a
10-term Log Metalog overlaid for reference.

The overlay is the K=10 Log Metalog fit to the RAW recorded weights, with no
jitter. Fitting the raw data needs no RNG,
so this figure is fully deterministic. The raw-data fit is feasible and
bimodal (modes near 4.86 and 10.05 lb).

Two figures are written: Figure15_fish_histogram (0.5-lb bins, the paper
figure) and fish_histogram_bins1.0 (1-lb bins, not in the paper; written to
figures/other/).
"""
import os, sys, json, datetime, warnings
import numpy as np
import figure_style as FS, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib as mpl
from matplotlib.ticker import MultipleLocator
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'code', 'common')); sys.path.insert(0, HERE)
from fish_common import load_fish, P_GRID, LOWER_BOUND
from metalog.metalog_v2 import LogMetalog
import eval_framework as EF

TAB = os.path.join(HERE, '..', 'tables')
FIG = os.path.join(HERE, '..', 'figures')
LOG = os.path.join(HERE, '..', 'logs')
for d in (TAB, FIG, LOG): os.makedirs(d, exist_ok=True)

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.titlesize': 12, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'legend.frameon': True, 'legend.framealpha': 0.85,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.color': '#cccccc',
    'lines.linewidth': 2.0,
})
HIST_COLOR, PDF_COLOR = FS.OTHER['data'], FS.QPD['Metalog']


def main():
    x, p = load_fish()
    N = len(x)
    print(f'N={N}, range [{x.min():.1f}, {x.max():.1f}] lb, '
          f'{len(np.unique(x))} distinct values')

    m = LogMetalog(x, p, lower_bound=LOWER_BOUND, terms=10)
    print(f'Log Metalog K=10 on RAW data: valid={EF.is_valid(m)}')
    qf = EF.quantile(m)
    n_modes, locs, hgts = EF.modes(m)

    # DRAWN curve: the evaluation grid only (figure_style.density_curve =
    # eval_framework) -- Q and the grid-difference density on 0.001..0.999.
    PLOT_GRID, qf_plot, pdf_plot = FS.density_curve(
        m.quantile, x_view=(2.5, 28.0), support=(LOWER_BOUND, np.inf))
    print(f'  drawn curve runs from {qf_plot[0]:.4f} lb to {qf_plot[-1]:.2f} lb '
          f'(lower bound {LOWER_BOUND:.1f})')
    print(f'  n_modes={n_modes}, locations={np.round(locs,3) if locs is not None else None}, '
          f'heights={np.round(hgts,5) if hgts is not None else None}')

    edges_half = np.arange(2.75, 33.51, 0.5)
    edges_one = np.arange(2.5, 34.01, 1.0)

    # One figure per bin width. Both use the same y-range and x-range, so the
    # pair stays directly comparable side by side.
    for edges, hlabel, title, stem in [
            (edges_half, 'Histogram (0.5 lb bins)',
             'Histogram of recorded fish weights with 0.5-lb bins',
             'Figure15_fish_histogram'),
            (edges_one, 'Histogram (1 lb bins)',
             'Histogram of recorded fish weights with 1-lb bins',
             'fish_histogram_bins1.0')]:
        fig, ax = plt.subplots(figsize=(10, 4.6))
        ax.hist(x, bins=edges, density=True, color=HIST_COLOR,
                edgecolor='white', linewidth=0.3, alpha=0.85, label=hlabel, zorder=1)
        ax.plot(qf_plot, pdf_plot, color=PDF_COLOR, linewidth=2,
                label='10-term Log Metalog fit (raw data)', zorder=3)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Weight (pounds)'); ax.set_ylabel('Density')
        ax.set_xlim(2.5, 28.0); ax.set_ylim(0, 0.37)
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.legend(loc='upper right')
        fig.tight_layout()
        out = FS.figure_path(stem)
        fig.savefig(out + '.png', dpi=300, bbox_inches='tight')
        fig.savefig(out + '.pdf', bbox_inches='tight'); plt.close(fig)
        print(f'Saved -> {out}.png')

    pd.DataFrame({'p': PLOT_GRID, 'qf_weight_lb': qf_plot, 'pdf': pdf_plot}).to_csv(
        os.path.join(TAB, 'Figure15_metalog_overlay_paneldata.csv'), index=False)
    for edges, name in [(edges_half, 'bins0.5'), (edges_one, 'bins1.0')]:
        cnt, e = np.histogram(x, bins=edges, density=True)
        pd.DataFrame({'bin_left': e[:-1], 'bin_right': e[1:], 'density': cnt}).to_csv(
            os.path.join(TAB, f'fish_histogram_{name}.csv'), index=False)

    json.dump({
        'generated': datetime.datetime.now().isoformat(timespec='seconds'),
        'paper_figure': 15,
        'overlay': 'LogMetalog K=10 fit to the RAW recorded weights (no jitter)',
        'overlay_valid': bool(EF.is_valid(m)),
        'overlay_n_modes': int(n_modes),
        'overlay_mode_locations': [float(v) for v in (locs if locs is not None else [])],
        'overlay_mode_heights': [float(v) for v in (hgts if hgts is not None else [])],
        'grid_modality': 'eval_framework.GRID = 0.001..0.999',
        'grid_drawn': 'linspace(1e-5, 0.995, 4000) -- density extended to the lower bound',
        'N': int(N),
    }, open(os.path.join(LOG, 'Figure15_fish_provenance.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
