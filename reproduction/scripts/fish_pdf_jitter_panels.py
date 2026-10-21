"""
Fish -- Log Metalog and Log QFlex densities, one panel per jitter width
(0, 0.5, 1.0 lb); the model and order per panel are set in PANEL below.
Not in the paper (written to figures/other/fish_pdf_jitter_panels). Its
conventions are imported by fish_pdf_order_panels.py, which makes Figure 17.

WHAT IS FITTED. Each panel fits ONE canonical jittered realization of the
recorded weights -- fish_common.jitter_original(x, j) -- and draws the
histogram of that SAME array beneath the curves. No bootstrap and no
replicate selection is involved, so the curve in a panel is the density of
the bars it sits on. A median-W1 bootstrap representative, by contrast, is
chosen by matching a scalar W1 distance, so it does not correspond to any
histogram. For a three-panel comparison the direct fit is the honest one,
and at j = 0 it degenerates cleanly to the recorded weights.

BIN WIDTH per panel:
  j = 0.5 -> 1.0 lb   the full jitter window 2j:
  j = 1.0 -> 2.0 lb   +-j asserts a weight is known only to within 2j, so 2j
                      is the finest resolution the histogram can claim.
  j = 0   -> 0.5 lb   2j degenerates at zero. 0.5 lb is the recording grid
                      itself (56 distinct values, minimum gap exactly 0.5),
                      i.e. the same "finest honest resolution" rule applied
                      to the data's own precision rather than to the jitter.

BIN EDGES are offset by half a bin so every value of the 0.5 lb recording
grid falls at a bin CENTRE. Anchored at a multiple of the bin width instead,
the j = 0 panel puts each recorded value exactly on an edge and the
histogram becomes an alternating filled/empty comb -- heaping, not
structure. The offset is applied in all three panels so the figure uses one
binning rule; for j = 0.5 and 1.0 the weights are continuous, so this only
shifts those bars half a bin and changes no shape.

Y AXIS is shared across panels. Density is normalized by bin width, so the
wider-bin panels are intrinsically shorter; that compression IS the result
being shown and a per-panel y-limit would hide it.

X AXIS is 0 to FISH_MAX = 28 lb, the same upper limit as Figure 15
(fish_histogram_figure.py): 10 of 3474 observations (0.29%) run to 33 lb and letting them set the range empties a
third of every panel. Data are fitted and binned in full; only the view is
clipped.

DRAWING AND MODALITY follow the evaluation framework (code/common/
eval_framework.py): every curve is drawn on p = 0.001..0.999 only, with the
density from differences of the quantile between neighbouring grid points --
the same density the modes are read from. Nothing beyond the grid is drawn.
"""
import os, sys, json, argparse, datetime, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib as mpl
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'code', 'common'))
import figure_style as FS
from fish_common import load_fish, jitter_original, P_GRID, LOWER_BOUND
from metalog.metalog_v2 import LogMetalog
from qflex.transforms import LogQFlex
from qflex.constraints import ConstraintType
import eval_framework as EF

TAB = os.path.join(HERE, '..', 'tables')
FIG = os.path.join(HERE, '..', 'figures')
LOG = os.path.join(HERE, '..', 'logs')
os.makedirs(FIG, exist_ok=True)

mpl.rcParams.update({'font.family': 'sans-serif', 'font.size': 11,
                     'axes.titlesize': 13, 'axes.labelsize': 12,
                     'figure.dpi': 150, 'savefig.dpi': 300,
                     'axes.grid': True, 'grid.alpha': 0.3})

COLORS = {'Log Metalog': FS.QPD['Metalog'], 'Log QFlex-TA+': FS.QPD['QFlex-TA+'],
          'Log QFlex-A+': FS.QPD['QFlex-A+']}
CT = {'Log QFlex-TA+': ConstraintType.TA, 'Log QFlex-A+': ConstraintType.A}
# Model and order are set PER PANEL. The 0 and 0.5 lb panels carry the two
# K=13 fits; the 1.0 lb panel carries K=7 and swaps TA+ for A+.
PANEL = {0.0: [('Log Metalog', 13), ('Log QFlex-TA+', 13)],
         0.5: [('Log Metalog', 13), ('Log QFlex-TA+', 13)],
         1.0: [('Log Metalog', 7),  ('Log QFlex-A+', 7)]}
JITTERS = [0.0, 0.5, 1.0]
HIST_BINWIDTH = {0.0: 0.5, 0.5: 1.0, 1.0: 2.0}
FISH_MAX = 28.0
TITLE = False        # no in-figure title: the manuscript caption does that job


def tail_start(qf, pdf, i0):
    """Index to start drawing at. Walk down from p = 0.01 while the quantile
    function still rises and the density still falls; stop where either turns."""
    i = i0
    while i > 0 and qf[i - 1] < qf[i] and pdf[i - 1] < pdf[i]:
        i -= 1
    return i


def fit(lab, K, xs, pp, grid):
    m = (LogMetalog(xs, pp, lower_bound=LOWER_BOUND, terms=K) if lab == 'Log Metalog'
         else LogQFlex(xs, pp, lower_bound=LOWER_BOUND, terms=K,
                       constraint_type=CT[lab]))
    if not EF.is_valid(m):
        return None
    q, f = EF.density(m)
    return q, f, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--title', action='store_true',
                    help='draw the in-figure title (off by default)')
    a = ap.parse_args()
    global TITLE
    TITLE = a.title

    x, _ = load_fish()
    grid = FS.plot_grid(P_GRID, len(x))   # see DRAWING GRID in the docstring
    i0 = int(np.searchsorted(grid, P_GRID[0]))   # where P_GRID begins
    rows = []

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), sharex=True, sharey=True)
    for ax, j in zip(axes, JITTERS):
        bw = HIST_BINWIDTH[j]
        xj = np.sort(jitter_original(x, j))
        pp = np.arange(1, len(xj) + 1) / (len(xj) + 1)

        # half-bin offset: recorded values land at bin centres, not on edges
        edges = np.arange(np.floor(xj.min() / bw) * bw - bw / 2,
                          xj.max() + bw, bw)
        ax.hist(xj, bins=edges, density=True, color=FS.OTHER['band'],
                alpha=0.75, edgecolor='white', linewidth=0.3, zorder=1,
                label='Histogram')

        for lab, K in PANEL[j]:
            out = fit(lab, K, xj, pp, grid)
            if out is None:
                print(f'  !! infeasible: {lab} K={K} jitter={j}', flush=True)
                continue
            qf, pdf, m = out
            # DRAWN curve: the exact density, run out to the ends by
            # FS.density_curve -- down the lower tail to the bound (or to where
            # the fitted quantile turns back) and up the unbounded upper tail
            # past the panel edge. Modality and the panel CSV below are still
            # read from the P_GRID arrays above, unchanged.
            _pd, qd, pd_ = FS.density_curve(m.quantile, x_view=(0.0, FISH_MAX),
                                             support=(LOWER_BOUND, np.inf))
            ax.plot(qd, pd_, color=COLORS[lab], linewidth=2, zorder=4,
                    label=f'{lab} K={K}')
            print(f'     drawn from {qd[0]:.3f} lb to {qd[-1]:.2f} lb', flush=True)
            # modality: eval_framework interior peaks -> (n, locs, heights)
            nm, mloc, mh = EF.modes_from_density(qf, pdf)
            print(f'  jitter {j}: {lab} K={K} -> {nm} mode(s) at '
                  + ', '.join(f'{L:.2f} lb (h={H:.4f})'
                              for L, H in zip(mloc, mh)), flush=True)
            rows += [{'jitter': j, 'model': lab, 'K': K, 'n_modes': nm,
                      'p': p, 'qf': q, 'pdf': d}
                     for p, q, d in zip(grid[i0:], qf[i0:], pdf[i0:])]

        lead = 'Jitter width = 0 lb' if j == 0 else f'Jitter width = ±{j} lb'
        ax.set_title(f'{lead}\n', fontweight='bold', loc='center')
        ax.text(0.5, 1.015, f'{bw:g} lb bins', transform=ax.transAxes,
                ha='center', va='bottom', fontsize=10, color='#555555')
        ax.set_xlim(0, FISH_MAX)
        ax.set_ylim(bottom=0)
        # one legend PER PANEL: the model/order pair differs between panels,
        # so a single shared key would mislabel two of the three.
        ax.legend(loc='upper right', fontsize=9)

    axes[0].set_ylabel('Density')
    fig.supxlabel('Fish weight (lbs)', fontsize=12)
    if TITLE:
        fig.suptitle('QPD densities fitted to fish weights',
                     fontsize=15, fontweight='bold')
    fig.tight_layout()

    out = FS.figure_path('fish_pdf_jitter_panels')
    fig.savefig(out + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(out + '.pdf', bbox_inches='tight')
    plt.close(fig)

    csv = os.path.join(TAB, 'fish_pdf_jitter_panels.csv')
    pd.DataFrame(rows).to_csv(csv, index=False)
    json.dump({
        'generated': datetime.datetime.now().isoformat(timespec='seconds'),
        'fit': 'direct fit to fish_common.jitter_original(x, j), draw=0 -- '
               'no bootstrap, no replicate selection; the curve is the density '
               'of the histogram drawn beneath it',
        'panel_models': {str(j): [f'{lab} K={K}' for lab, K in PANEL[j]]
                         for j in JITTERS},
        'lower_bound': LOWER_BOUND,
        'hist_binwidth_lb': HIST_BINWIDTH,
        'hist_binwidth_rule': 'full jitter window 2j; at j=0, the 0.5 lb '
                              'recording grid of the raw weights',
        'hist_edges': 'offset by half a bin so recording-grid values sit at '
                      'bin centres (avoids the j=0 comb artifact)',
        'y_axis': 'shared across panels',
        'x_axis': f'0 to {FISH_MAX} lb (view clipped; fits and bins use all data)',
        'panel_data': csv,
    }, open(os.path.join(LOG, 'fish_pdf_jitter_panels_provenance.json'), 'w'),
        indent=2)
    print(f'\nSaved -> {out}.png / .pdf\nSaved -> {csv}', flush=True)


if __name__ == '__main__':
    main()
