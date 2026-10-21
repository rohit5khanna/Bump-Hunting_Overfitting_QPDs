"""
Fish -- paper Figure 16: EQF of the latent weights with a 95% bootstrap
interval, at +/-0.5 lb uniform jitter (Figure16_fish_EQF_bootstrap_interval).
The same figure at another jitter would be written as
fish_EQF_bootstrap_interval_jitter<j> (not in the paper).

  * The band comes from `fish_common.resample(x, 0.5, b)`, b = 0..999 --
    the SAME 1000 replicates as the 0.5 lb column of Table 9 and as
    Table 10 -- seeded per replicate, so it can be regenerated in any order.

  * The grid is the evaluation grid 0.001..0.999, so one grid is used
    throughout the section. With N = 3474 the sample covers
    p in [1/3475, 3474/3475] = [0.00029, 0.99971], so the grid sits inside
    the data range.

The point-estimate curve is one fixed jitter realization of the RECORDED
weights (`fish_common.jitter_original`), not a resample: it is "the latent
sample we would have seen", and the band is its sampling variability.

Legend wording follows the manuscript: "bootstrap interval", not "bootstrap
CI" -- these are pointwise percentile intervals, not a simultaneous band.
"""
import os, sys, json, datetime, warnings
import numpy as np
import figure_style as FS, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from fish_common import load_fish, resample, jitter_original, P_GRID, N_BOOT

TAB = os.path.join(HERE, '..', 'tables')
FIG = os.path.join(HERE, '..', 'figures')
LOG = os.path.join(HERE, '..', 'logs')
for d in (TAB, FIG, LOG): os.makedirs(d, exist_ok=True)

JITTER = 0.5
Y_MAX = 25.0
ZOOM_P_LO, ZOOM_P_HI = 0.93, 0.98
P_ANNOT = 0.96
CI_COLOR, LINE_COLOR, ZOOM_EDGE = FS.OTHER['band'], FS.OTHER['empirical'], FS.OTHER['zoom']

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.titlesize': 12, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'legend.frameon': True, 'legend.framealpha': 0.85,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.color': '#cccccc',
    'lines.linewidth': 1.8,
})


def main():
    x, _ = load_fish()
    N = len(x)
    p_emp = np.arange(1, N + 1) / (N + 1)
    print(f'N={N}, range [{x.min():.1f}, {x.max():.1f}] lb')

    M = np.empty((N_BOOT, len(P_GRID)))
    for b in range(N_BOOT):
        M[b] = np.interp(P_GRID, p_emp, np.sort(resample(x, JITTER, b)))
    q025 = np.percentile(M, 2.5, axis=0)
    q975 = np.percentile(M, 97.5, axis=0)

    x_ref = np.sort(jitter_original(x, JITTER))
    eqf_ref = np.interp(P_GRID, p_emp, x_ref)

    central = (P_GRID >= 0.25) & (P_GRID <= 0.75)
    print(f'mean central-region ({0.25}-{0.75}) interval width: '
          f'{np.mean(q975[central]-q025[central]):.3f} lb')

    zmask = (P_GRID >= ZOOM_P_LO) & (P_GRID <= ZOOM_P_HI)
    zy_lo = float(np.floor(q025[zmask].min()) - 0.5)
    zy_hi = float(np.ceil(q975[zmask].max()) + 0.5)
    i_an = int(np.argmin(np.abs(P_GRID - P_ANNOT)))
    w_an = float(q975[i_an] - q025[i_an])
    print(f'interval width at p={P_ANNOT}: {w_an:.2f} lb')

    fig, (ax, axz) = plt.subplots(1, 2, figsize=(13, 6),
                                  gridspec_kw={'width_ratios': [1.3, 1]})
    ax.fill_between(P_GRID, q025, q975, color=CI_COLOR, alpha=FS.BAND_ALPHA,
                    label='95% bootstrap interval', zorder=1)
    ax.plot(P_GRID, eqf_ref, color=LINE_COLOR, linewidth=1.4,
            label='Empirical quantile', zorder=3)
    ax.set_title(f'Latent weights (±{JITTER} lb jitter)', fontweight='bold')
    ax.set_xlabel('Probability p'); ax.set_ylabel('Fish weight (lbs)')
    ax.set_xlim(0, 1); ax.set_ylim(0, Y_MAX)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.legend(loc='upper left', fontsize=9)
    ax.add_patch(Rectangle((ZOOM_P_LO, zy_lo), ZOOM_P_HI - ZOOM_P_LO, zy_hi - zy_lo,
                           fill=False, edgecolor=ZOOM_EDGE, linewidth=1.4,
                           linestyle='--', zorder=4))

    axz.fill_between(P_GRID, q025, q975, color=CI_COLOR, alpha=FS.BAND_ALPHA, zorder=1)
    axz.plot(P_GRID, eqf_ref, color=LINE_COLOR, linewidth=1.6, zorder=3)
    axz.set_xlim(ZOOM_P_LO, ZOOM_P_HI); axz.set_ylim(zy_lo, zy_hi)
    axz.set_title(f'Zoomed: p ∈ [{ZOOM_P_LO:.2f}, {ZOOM_P_HI:.2f}]', fontweight='bold')
    axz.set_xlabel('Probability p'); axz.set_ylabel('Fish weight (lbs)')
    for s in axz.spines.values():
        s.set_edgecolor(ZOOM_EDGE); s.set_linewidth(1.4)
    axz.annotate('', xy=(P_ANNOT, q975[i_an]), xytext=(P_ANNOT, q025[i_an]),
                 arrowprops=dict(arrowstyle='<->', color='#333333', linewidth=1.3), zorder=5)
    axz.text(P_ANNOT + 0.003, (q025[i_an] + q975[i_an]) / 2,
             f'width ≈ {w_an:.2f} lb\n(p = {P_ANNOT:.2f})', fontsize=8.5,
             va='center', ha='left',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                       edgecolor='#cccccc', alpha=0.9))
    mark_inset(ax, axz, loc1=2, loc2=1, fc='none', ec=ZOOM_EDGE,
               linewidth=1.0, linestyle='--')
    fig.suptitle('Fish Weight Empirical Quantile Function of Latent Weights\n'
                 f'(±{JITTER} lb jitter, 95% Bootstrap Interval)',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = FS.figure_path('Figure16_fish_EQF_bootstrap_interval' if JITTER == 0.5
                         else f'fish_EQF_bootstrap_interval_jitter{JITTER}')
    fig.savefig(out + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(out + '.pdf', bbox_inches='tight'); plt.close(fig)
    print(f'Saved -> {out}.png')

    pd.DataFrame({'p': P_GRID, 'eqf_jittered': eqf_ref,
                  'q025': q025, 'q975': q975,
                  'width': q975 - q025}).to_csv(
        os.path.join(TAB, 'Figure16_fish_EQF_bootstrap_interval_paneldata.csv'), index=False)

    json.dump({
        'generated': datetime.datetime.now().isoformat(timespec='seconds'),
        'paper_figure': 16,
        'jitter': JITTER,
        'n_bootstrap': N_BOOT,
        'band': 'pointwise 2.5/97.5 percentiles of fish_common.resample replicates '
                '(shared with Tables 9 and 10)',
        'point_estimate': 'fish_common.jitter_original(x, 0.5) -- one fixed jitter '
                          'realization of the recorded weights, no resampling',
        'grid': 'eval_framework.GRID = 0.001..0.999',
        'mean_central_width_lb': float(np.mean(q975[central] - q025[central])),
        f'width_at_p{P_ANNOT}_lb': w_an,
    }, open(os.path.join(LOG, 'Figure16_fish_provenance.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
