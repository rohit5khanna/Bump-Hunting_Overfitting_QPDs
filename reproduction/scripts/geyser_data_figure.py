"""
Geyser -- paper Figure 18: the dataset, its empirical quantile function
(left) and its PDF, histogram + KDE (right).

Deterministic -- no resampling. The EQF uses Weibull plotting positions
p_i = i/(N+1) (Equation 3).

The manuscript refers to "the taller KDE implied mode ... ~79.8 min" and "the
shorter one ... ~53.5 min". Those two numbers are recomputed and printed here
so the text can be checked against the figure it cites.

Also recorded: the recorded waiting times are integers with only 52 distinct
values in 298 observations, which is why the bootstrap analysis jitters them.
One jittered realization is written to the panel-data CSV (not drawn) to show
what the bootstrap actually operates on.
"""
import os, sys, json, datetime, warnings
import numpy as np
import figure_style as FS, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib as mpl
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geyser_common import load_geyser, jitter_original, JITTER, data_fingerprint

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
    'lines.linewidth': 1.6,
})
EMP, KDEC = FS.OTHER['observed'], FS.OTHER['kde']


def main():
    x, p = load_geyser()
    N = len(x)
    print(f'N={N}, range [{x.min():.0f}, {x.max():.0f}] min, '
          f'{len(np.unique(x))} distinct values, '
          f'{100*np.mean(x==np.round(x)):.0f}% integers')

    xj = np.sort(jitter_original(x, JITTER))
    kde = gaussian_kde(x, bw_method='scott')
    grid = np.linspace(x.min() - 5, x.max() + 5, 1000)
    dens = kde(grid)
    pk = argrelextrema(dens, np.greater)[0]
    modes = sorted(((float(grid[i]), float(dens[i])) for i in pk),
                   key=lambda t: -t[1])
    print('KDE modes (tallest first):')
    for loc, h in modes:
        print(f'   loc {loc:6.2f} min   height {h:.5f}')
    print('   manuscript quotes: taller ~79.8 min, shorter ~53.5 min')

    fig, (ax_q, ax_d) = plt.subplots(1, 2, figsize=(12, 5))

    ax_q.scatter(p, x, s=9, color=EMP, alpha=0.85, zorder=3,
                 label='Recorded waiting times')
    ax_q.set_xlabel('Probability p')
    ax_q.set_ylabel('Waiting time (min)')
    ax_q.set_title('Empirical Quantile Function', fontweight='bold')
    ax_q.set_xlim(0, 1)
    ax_q.legend(loc='upper left')

    # Bin width = the full jitter window 2j = 1.0 min, the rule every section
    # follows. Waiting times are integers and the edges sit on the
    # half-integers, so a +-0.5 min jitter cannot move a value out of its own
    # bin: binning the raw record and binning a jittered realization give
    # identical counts here (checked over 50 draws).
    _bw = 2 * JITTER
    _edges = np.arange(np.floor(x.min()) - 0.5, x.max() + 1.5, _bw)
    ax_d.hist(x, bins=_edges, density=True, color=FS.OTHER['data'],
              alpha=0.75, edgecolor='white', linewidth=0.4, zorder=1,
              label=f'Histogram ({_bw:g} min bins)')
    ax_d.plot(grid, dens, color=KDEC, linewidth=1.8, zorder=3, label='KDE')
    # The two KDE modes: dotted drop-line to the axis plus the time above each
    # peak.
    for loc, h in modes[:2]:
        ax_d.plot([loc, loc], [0, h], color=KDEC, linestyle=':', linewidth=1.2,
                  zorder=2)
        ax_d.annotate(f'{loc:.1f} min', xy=(loc, h), xytext=(0, 7),
                      textcoords='offset points', ha='center', fontsize=9,
                      color=KDEC)
    ax_d.set_xlabel('Waiting time (min)')
    ax_d.set_ylabel('Density')
    ax_d.set_title('Probability Density', fontweight='bold')
    ax_d.legend(loc='upper left')

    fig.suptitle('Old Faithful Geyser Waiting Times '
                 f'(N = {N}, {len(np.unique(x))} distinct integer values)',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    out = FS.figure_path('Figure18_geyser_data')
    fig.savefig(out + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(out + '.pdf', bbox_inches='tight'); plt.close(fig)
    print(f'Saved -> {out}.png')

    pd.DataFrame({'p': p, 'waiting_min': x, 'waiting_jittered_min': xj}).to_csv(
        os.path.join(TAB, 'Figure18_geyser_eqf_paneldata.csv'), index=False)
    pd.DataFrame({'waiting_min': grid, 'kde_density': dens}).to_csv(
        os.path.join(TAB, 'Figure18_geyser_kde_paneldata.csv'), index=False)

    json.dump({
        'generated': datetime.datetime.now().isoformat(timespec='seconds'),
        'data_file': data_fingerprint()[0],
        'data_N': data_fingerprint()[1],
        'data_sha256': data_fingerprint()[2],
        'paper_figure': 18, 'N': int(N),
        'distinct_values': int(len(np.unique(x))),
        'integer_share_pct': float(100 * np.mean(x == np.round(x))),
        'kde_modes_tallest_first': [{'loc_min': l, 'height': h} for l, h in modes],
        'manuscript_quotes': {'taller_mode_min': 79.8, 'shorter_mode_min': 53.5},
        'plotting_positions': 'Weibull i/(N+1)',
        'deterministic': True,
        'jitter_overlay': 'not drawn; the jittered realization is written to '
                          'Figure18_geyser_eqf_paneldata.csv',
    }, open(os.path.join(LOG, 'Figure18_geyser_provenance.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
