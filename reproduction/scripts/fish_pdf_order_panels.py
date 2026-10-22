"""
Fish -- Log Metalog and Log QFlex-U densities at K = 7, 10, 13, for the raw
weights and for the 0.5 lb and 1.0 lb jittered weights.

Writes four figures:
  Figure17_fish_fits_K7_10_13.png  1 x 3, +-0.5 lb jitter -- paper Figure 17
  fish_pdf_order_panels_j0.png     1 x 3, raw recorded weights   (not in the paper)
  fish_pdf_order_panels_j1.0.png   1 x 3, +-1.0 lb jitter        (not in the paper)
  fish_pdf_order_panels_3x3.png    all three, one row per jitter width,
                                   each row headed "Jitter = <w> lb"
                                   (not in the paper)
The paper figure goes to figures/, the others to figures/other/.

CONVENTIONS ARE IMPORTED, NOT RESTATED. Everything about how a fish density
panel is built comes from fish_pdf_jitter_panels.py (whose own figure is not in
the paper), which documents why each choice was made, so the two figure
families cannot drift apart:

  * the data: fish_common.jitter_original(x, j) -- ONE canonical realization
    per width, no bootstrap, no replicate selection. At j = 0 that is the
    recorded weights themselves. The histogram in a panel is the SAME array
    the curves were fitted to.
  * histogram bin width per row (HIST_BINWIDTH): 0.5 lb at j = 0 (the
    recording grid), else the full jitter window 2j; edges offset by half a
    bin so recorded values sit at bin centres.
  * x view 0..FISH_MAX lb (fits and bins use all the data).

DRAWING AND MODALITY follow the evaluation framework (code/common/
eval_framework.py): the 0.001..0.999 grid only, density from differences of
the quantile, validity = Q increasing on the grid, interior modes.

Model colours are the paper palette: Log Metalog = Metalog blue, Log QFlex-U =
QFlex-U purple. Log QFlex-U is ConstraintType.NONE, as in mc_tables_rebuild
and fish_summary.

Y AXIS is shared WITHIN a row (the three K panels of one jitter width) but
not across rows. One common scale was tried first, following
fish_pdf_jitter_panels: the raw weights' whole-pound heaping puts a 0.34 bar
at 10 lb, which squeezed the 0.5 and 1.0 lb rows -- whose curves peak at
~0.13, like the raw row's -- into their bottom third. The three single-row
figures are exactly the rows of the 3 x 3. The per-row caps are fixed values (YMAX:
0.35, 0.175, 0.14), chosen by the author; each sits just above that row's
tallest bar or curve (0.336, 0.171, 0.132).

Each legend entry carries the fit's W1 (Equation 6, to the sample it was fitted
to), as a percentage.

A fit that is not valid (Q not increasing on the grid) is still DRAWN and marked
"(invalid)" in the legend, the convention of the Johnson SB example fits
(Figure 2, mc_sb_pdf_fit_triptych.py), so a reader sees the curve the solver
produced rather than a gap.
Modality is read on the evaluation grid, as everywhere in the section.
"""
import os, sys, json, datetime, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'code', 'common'))
import figure_style as FS
import fish_pdf_jitter_panels as FPJ          # tail_start, HIST_BINWIDTH, FISH_MAX
from fish_common import load_fish, jitter_original, P_GRID, LOWER_BOUND
from metalog.metalog_v2 import LogMetalog
from qflex.transforms import LogQFlex
from qflex.constraints import ConstraintType
import eval_framework as EF

TAB = os.path.join(HERE, '..', 'tables')
FIG = os.path.join(HERE, '..', 'figures')
LOG = os.path.join(HERE, '..', 'logs')
for d in (TAB, FIG, LOG):
    os.makedirs(d, exist_ok=True)

KS = [7, 10, 13]
# Fixed y-axis caps per row (jitter width), set by the author. compute()
# reports the tallest bar/curve in each row against its cap, so a later data
# change that would clip the view is caught rather than silently cut off.
YMAX = {0.0: 0.35, 0.5: 0.175, 1.0: 0.14}
JITTERS = [0.0, 0.5, 1.0]
MODELS = [('Log Metalog', FS.QPD['Metalog']),
          ('Log QFlex-U', FS.QPD['QFlex-U'])]


def jlabel(j):
    return '0' if j == 0 else f'{j:.1f}'


def build(lab, K, xs, pp):
    if lab == 'Log Metalog':
        return LogMetalog(xs, pp, lower_bound=LOWER_BOUND, terms=K)
    return LogQFlex(xs, pp, lower_bound=LOWER_BOUND, terms=K,
                    constraint_type=ConstraintType.NONE)


def compute():
    x, _ = load_fish()
    grid = FS.plot_grid(P_GRID, len(x))
    i0 = int(np.searchsorted(grid, P_GRID[0]))
    rows, data = [], {}
    for j in JITTERS:
        xj = np.sort(jitter_original(x, j))
        pp = np.arange(1, len(xj) + 1) / (len(xj) + 1)
        bw = FPJ.HIST_BINWIDTH[j]
        edges = np.arange(np.floor(xj.min() / bw) * bw - bw / 2, xj.max() + bw, bw)
        h, _ = np.histogram(xj, bins=edges, density=True)
        fits = {}
        for lab, _c in MODELS:
            for K in KS:
                try:
                    m = build(lab, K, xj, pp)
                    feas = EF.is_valid(m)
                    if feas:
                        qf, pdf = EF.density(m)
                    else:
                        qf = EF.quantile(m); pdf = EF.density_from_quantile(qf)
                except Exception as e:
                    print(f'  !! fit failed: {lab} K={K} jitter={j}: {e}', flush=True)
                    fits[(lab, K)] = None
                    continue
                nm, mloc, mh = (EF.modes_from_density(qf, pdf) if feas else (-1, [], []))
                # drawn curve: the grid and its difference density (see docstring)
                _pd, qd, pdd = FS.density_curve(m.quantile, x_view=(0.0, FPJ.FISH_MAX),
                                                support=(LOWER_BOUND, np.inf))
                # W1 (Eq 6) against the sample this curve was fitted to -- the
                # jittered realization xj -- shown in the legend as a percentage,
                # like every other fit figure in the paper.
                w1n = EF.w1(m, xj)[1]
                fits[(lab, K)] = dict(qf=qf, pdf=pdf, qd=qd, pdd=pdd,
                                      feasible=feas, n_modes=nm, w1=w1n)
                print(f'  j={j:<3} {lab:12s} K={K:2d}  feasible={feas!s:5}  '
                      f'W1={100 * w1n:.2f}%  modes={nm}  ' + ', '.join(f'{L:.2f} lb' for L in mloc), flush=True)
                rows += [dict(jitter=j, model=lab, K=K, feasible=feas, n_modes=nm,
                              W1_Eq6=w1n, p=p, qf=q, pdf=d)
                         for p, q, d in zip(grid[i0:], qf[i0:], pdf[i0:])]
        data[j] = dict(xj=xj, edges=edges, hmax=float(h.max()), bw=bw, fits=fits)
    # one y scale PER ROW (per jitter width), shared by the three K panels of
    # that row: the fixed caps in YMAX. The tallest bar/curve is still computed
    # and reported against the cap so clipping can never happen silently.
    # A single scale across rows was tried first: the j = 0 histogram's
    # whole-pound heaping (a 0.34 bar at 10 lb, against curve peaks of ~0.13)
    # set it and squeezed the 0.5 and 1.0 lb rows into their bottom third.
    ymax = {}
    for j in JITTERS:
        tops = [data[j]['hmax']]
        for f in data[j]['fits'].values():
            if f is not None:
                tops.append(float(np.nanmax(f['pdd'][f['qd'] <= FPJ.FISH_MAX])))
        top = max(tops)
        ymax[j] = YMAX[j]
        clip = '  ** DATA EXCEEDS CAP -- would be clipped **' if top > YMAX[j] else ''
        print(f'  row {j} lb: tallest bar/curve {top:.4f}, y cap {YMAX[j]}{clip}', flush=True)
    return data, rows, ymax


def draw_row(axes, j, d, ymax, first_col_ylabel=True):
    for ax, K in zip(axes, KS):
        ax.hist(d['xj'], bins=d['edges'], density=True, color=FS.OTHER['band'],
                alpha=0.75, edgecolor='white', linewidth=0.3, zorder=1,
                label=f"Histogram ({d['bw']:.1f} lb bins)")
        for lab, col in MODELS:
            f = d['fits'][(lab, K)]
            if f is None:
                continue
            tag = '' if f['feasible'] else ' (invalid)'
            ax.plot(f['qd'], f['pdd'], color=col, linewidth=2, zorder=4,
                    label=f"{lab}  (W1 = {100 * f['w1']:.2f}%{tag})")
        ax.set_title(f'K = {K}', fontweight='bold')
        ax.set_xlim(0, FPJ.FISH_MAX)
        ax.set_ylim(0, ymax)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
    if first_col_ylabel:
        axes[0].set_ylabel('Density')
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)


def main():
    data, rows, ymax = compute()
    out_files = []

    for j in JITTERS:
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
        draw_row(axes, j, data[j], ymax[j])
        fig.supxlabel('Fish weight (lbs)', fontsize=12)
        fig.tight_layout()
        stem = FS.figure_path('Figure17_fish_fits_K7_10_13' if j == 0.5
                              else f'fish_pdf_order_panels_j{jlabel(j)}')
        fig.savefig(stem + '.png', dpi=300, bbox_inches='tight')
        fig.savefig(stem + '.pdf', bbox_inches='tight')
        plt.close(fig); out_files.append(stem + '.png')

    fig = plt.figure(figsize=(15.5, 14.2), layout='constrained')
    subs = fig.subfigures(3, 1, hspace=0.04)
    for sf, j in zip(subs, JITTERS):
        sf.suptitle(f'Jitter = {jlabel(j)} lb', fontsize=14, fontweight='bold')
        draw_row(sf.subplots(1, 3), j, data[j], ymax[j])
    fig.supxlabel('Fish weight (lbs)', fontsize=12)
    stem = FS.figure_path('fish_pdf_order_panels_3x3')
    fig.savefig(stem + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(stem + '.pdf', bbox_inches='tight')
    plt.close(fig); out_files.append(stem + '.png')

    csv = os.path.join(TAB, 'Figure17_fish_fits_K7_10_13_paneldata.csv')
    pd.DataFrame(rows).to_csv(csv, index=False)
    json.dump({
        'generated': datetime.datetime.now().isoformat(timespec='seconds'),
        'fit': 'direct fit to fish_common.jitter_original(x, j), draw=0',
        'models': [m for m, _ in MODELS], 'K': KS, 'jitters_lb': JITTERS,
        'qflex_u_constraint': 'ConstraintType.NONE',
        'hist_binwidth_lb': {str(k): v for k, v in FPJ.HIST_BINWIDTH.items()},
        'conventions_from': 'fish_pdf_jitter_panels.py',
        'y_axis': 'shared within each row (jitter width), not across rows: '
                  + ', '.join(f'{j} lb 0..{v:.4f}' for j, v in ymax.items()),
        'feasibility': {f'{j}|{lab}|K={K}': bool(f['feasible'])
                        for j in JITTERS for (lab, K), f in data[j]['fits'].items()
                        if f is not None},
        'w1_legend': 'Eq 6 to the jittered realization each curve was fitted to, shown x100 (%)',
        'w1_eq6': {f'{j}|{lab}|K={K}': float(f['w1'])
                   for j in JITTERS for (lab, K), f in data[j]['fits'].items() if f is not None},
        'panel_data': csv,
    }, open(os.path.join(LOG, 'fish_pdf_order_panels_provenance.json'), 'w'), indent=2)
    print('\nwrote:\n  ' + '\n  '.join(os.path.basename(p) for p in out_files))
    print('row ymax: ' + ', '.join(f'{j} lb -> {v:.4f}' for j, v in ymax.items()))


if __name__ == '__main__':
    main()
