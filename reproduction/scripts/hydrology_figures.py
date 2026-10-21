"""
Hydrology (Section 5.1) -- Figures 10-14, plus two median-bootstrap figures
that are not in the paper.

    Figure10_hydrology_EQF            EQF of the 95 annual peak gauge heights,
                                      with a zoomed window
    Figure11_hydrology_histogram_pdfs histogram + full-sample GEV and
                                      Log Metalog K=10
    Figure12_hydrology_EQF_bootstrap_interval
                                      EQF, pointwise 95% bootstrap interval,
                                      GEV + Log Metalog K=10
    Figure13_hydrology_mode_scatter   bootstrap mode height vs mode location
                                      (GEV, Log Metalog K=10, Log QFlex-A+ K=10)
    Figure14_hydrology_GEV_Metalog_QFlex
                                      QF and PDF: GEV, Log Metalog K=4/10,
                                      Log QFlex-A+ K=10, with Equation 6 W1
    hydrology_metalog_median_bootstrap    W1-median bootstrap fit, GEV + Log
                                      Metalog (figures/other/; not in the paper)
    hydrology_qflex_median_bootstrap      W1-median bootstrap fit, GEV + Log
                                      QFlex-A+ (figures/other/; not in the paper)

All bootstrap content comes from tables/hydrology_sweep.csv and the band in
Figure12_hydrology_EQF_bootstrap_interval_paneldata.csv, both written by
hydrology_sweep.py, so the figures and Table 8 describe the same 1000 paired
replicates: GEV and the QPDs in one picture are fitted to the same samples.

The median-fit figures use these steps: median the bootstrap SAMPLES' EQFs,
take that curve's Equation 6 distance from the observed EQF as W1*, then pick
the valid fit whose scalar W1 is closest to W1*.

Conventions:
  * mode-scatter axes capped at MODE_YMAX / MODE_XMAX (points beyond are
    clipped, not dropped);
  * the Log Metalog and Log QFlex-A+ median-fit figures share one PDF y-range,
    so the two are visually comparable;
  * EVALUATION FRAMEWORK (code/common/eval_framework.py): every curve is
    drawn on p = 0.001..0.999 only, with the density from differences of the
    quantile (GEV included); validity, modes and W1 are the framework's.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import sys, json, datetime, warnings
import numpy as np
import figure_style as FS, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy.stats import genextreme
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'code', 'common')); sys.path.insert(0, HERE)
from hydrology_common import (load_hydrology, resample, P_GRID, N_BOOT, K_VALUES,
                              LOWER_BOUND, empirical_qf, interdecile, w1, w1_points,
                              data_fingerprint)
from metalog.metalog_v2 import LogMetalog
from qflex.transforms import LogQFlex
from qflex.constraints import ConstraintType
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
    'lines.linewidth': 1.8,
})
C_GEV, C_MET, C_QFX = FS.OTHER['GEV'], FS.QPD['Metalog'], FS.QPD['QFlex-A+']
C_EMP, C_BAND = FS.OTHER['empirical'], FS.OTHER['band']
# Observed data drawn as MARKERS uses the 'observed' role (slate), the same
# as the geyser EQF scatter. The empirical QUANTILE CURVE stays C_EMP black
# -- they are different marks and keep different roles.
C_OBS = FS.OTHER['observed']
KSTYLE = FS.KSTYLE          # 4/7/10/13 -> solid/dashed/dotted/dash-dot
PLOT_GRID = None       # figure_style.plot_grid, built in main() (needs N)
MODE_YMAX = 6.0        # all but 4 Metalog K=10 points fall below this
                       # (those heights run to 24.3/ft; see the note at Figure 13)
MODE_XMAX = 6.0        # all but 1 Metalog K=10 point falls below this (7.31 ft)


def save(fig, name):
    fig.savefig(FS.figure_path(name) + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(FS.figure_path(name) + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {name}', flush=True)


def fit_curve(key, s, p, K, grid=P_GRID):
    try:
        if key == 'gev':
            c, loc, sc = genextreme.fit(s, method='MLE')
            qf = np.asarray(genextreme.ppf(EF.GRID, c, loc=loc, scale=sc), float)
            return qf, EF.density_from_quantile(qf)
        m = (LogMetalog(s, p, lower_bound=LOWER_BOUND, terms=K) if key == 'metalog'
             else LogQFlex(s, p, lower_bound=LOWER_BOUND, terms=K,
                           constraint_type=ConstraintType.A))
        if not EF.is_valid(m):
            return None
        return EF.density(m)
    except Exception:
        return None


KEY = {'GEV': 'gev', 'Log Metalog': 'metalog', 'Log QFlex-A+': 'qflex_a'}


def fit_qfun(key, s, p, K):
    """The Q-function of the same fit fit_curve() makes (GEV MLE and the QPD
    fits are deterministic), for W1 at the sample points."""
    if key == 'gev':
        c, loc, sc = genextreme.fit(s, method='MLE')
        return lambda pp, c=c, loc=loc, sc=sc: genextreme.ppf(pp, c, loc=loc, scale=sc)
    return (LogMetalog(s, p, lower_bound=LOWER_BOUND, terms=K) if key == 'metalog'
            else LogQFlex(s, p, lower_bound=LOWER_BOUND, terms=K,
                          constraint_type=ConstraintType.A))


def draw_curve(lab, s, p, K, x_view):
    """(x, density) to DRAW the same fit fit_curve() makes: figure_style.density_curve,
    i.e. the 0.001..0.999 grid and the grid-difference density (eval_framework).
    The GEV MLE is deterministic, so this is the same GEV as fit_curve's.
    Drawing only: modes, W1 and the CSVs use fit_curve's arrays."""
    key = KEY[lab]
    if key == 'gev':
        c, loc, sc = genextreme.fit(s, method='MLE')
        q = lambda pp, c=c, loc=loc, sc=sc: genextreme.ppf(pp, c, loc=loc, scale=sc)
        sup = genextreme.support(c, loc=loc, scale=sc)
    else:
        m = (LogMetalog(s, p, lower_bound=LOWER_BOUND, terms=K) if key == 'metalog'
             else LogQFlex(s, p, lower_bound=LOWER_BOUND, terms=K,
                           constraint_type=ConstraintType.A))
        q, sup = m.quantile, (LOWER_BOUND, np.inf)
    return FS.density_curve(q, x_view=x_view, support=sup)[1:]


def main():
    x, p = load_hydrology(); N = len(x)
    global PLOT_GRID
    PLOT_GRID = FS.plot_grid(P_GRID, N)
    pp = np.arange(1, N + 1) / (N + 1)
    eqf = empirical_qf(x, p); L = interdecile(x, p)
    sweep = pd.read_csv(os.path.join(TAB, 'hydrology_sweep.csv'))
    band = pd.read_csv(os.path.join(TAB, 'Figure12_hydrology_EQF_bootstrap_interval_paneldata.csv'))
    print(f'N={N}, range [{x.min():.3f}, {x.max():.3f}] ft, L={L:.4f} ft')

    # ---- full-sample reference fits -------------------------------------
    full = {}
    for lab, key, K in [('GEV', 'gev', 0)] + \
                       [('Log Metalog', 'metalog', K) for K in K_VALUES] + \
                       [('Log QFlex-A+', 'qflex_a', K) for K in K_VALUES]:
        out = fit_curve(key, x, pp, K, grid=PLOT_GRID)
        if out is not None:
            full[(lab, K)] = out
    print(f'full-sample fits feasible: {sorted((l,k) for l,k in full)}')

    # ===== Figure 10: EQF of the raw data, with a zoomed window ==========
    # An EQF, not an ECDF: the paper shows quantile functions everywhere else.
    #
    # REGION: the inset window is 3.88-4.02 ft, expressed in probability so
    # it works on an EQF. That window is not in
    # the tail: it sits near the first tercile and holds 8 observations
    # spanning just 3.91-3.94 ft -- four tied pairs, because the gauge is
    # recorded to 0.01 ft. On an ECDF that is a staircase; on an EQF it is a
    # near-horizontal run, which is the same fact drawn the other way round
    # and is precisely what makes the granularity visible.
    #
    # PRESENTATION: follows the fish Figure 16 template -- a dashed rectangle
    # drawn on the full panel and mark_inset connectors joining it to the
    # zoomed axes, whose spines carry the same colour. (A shaded vertical
    # band would mark only the p-range, not the gauge-height range, so it
    # would not show which part of the panel the inset actually magnifies.)
    ZOOM_X = (3.88, 4.02)
    _in = (x >= ZOOM_X[0]) & (x <= ZOOM_X[1])
    _pw = p[_in]
    _pad = 0.25 * (_pw.max() - _pw.min())
    ZOOM = (float(_pw.min() - _pad), float(_pw.max() + _pad))
    zmask = (p >= ZOOM[0]) & (p <= ZOOM[1])
    _zx = x[zmask]
    _zpad = 0.25 * (_zx.max() - _zx.min())
    zy_lo, zy_hi = float(_zx.min() - _zpad), float(_zx.max() + _zpad)
    print(f'  Figure 10 zoom {ZOOM_X[0]}-{ZOOM_X[1]} ft -> p in '
          f'[{ZOOM[0]:.4f}, {ZOOM[1]:.4f}]: {zmask.sum()} of {N} observations')
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(11.5, 5.4),
                                  gridspec_kw={'width_ratios': [1.35, 1]})
    # The observations themselves, as points. A connecting line here would
    # imply interpolation between order statistics that the data does not
    # support, and it would bury the markers underneath it.
    for a_, _ms in ((ax, 4.5), (axz, 8.0)):
        a_.plot(p, x, linestyle='none', marker='o', markersize=_ms,
                markerfacecolor='none', markeredgecolor=C_OBS,
                markeredgewidth=1.1, zorder=3,
                label=f'Observed quantiles (N = {N})')
    ax.set_xlabel('Probability p'); ax.set_ylabel('Gauge height (ft)')
    ax.set_title('EQF of annual peak flood gauge heights (1920\u20132014)',
                 fontweight='bold')
    ax.set_xlim(0, 1); ax.legend(loc='upper left')
    ax.add_patch(Rectangle((ZOOM[0], zy_lo), ZOOM[1] - ZOOM[0], zy_hi - zy_lo,
                           fill=False, edgecolor=FS.OTHER['zoom'],
                           linewidth=1.4, linestyle='--', zorder=4))
    axz.set_xlim(*ZOOM); axz.set_ylim(zy_lo, zy_hi)
    axz.set_xlabel('Probability p'); axz.set_ylabel('Gauge height (ft)')
    axz.set_title(f'Zoom: {ZOOM_X[0]:.2f}\u2013{ZOOM_X[1]:.2f} ft '
                  f'(p \u2208 [{ZOOM[0]:.2f}, {ZOOM[1]:.2f}])', fontweight='bold')
    for sp in axz.spines.values():
        sp.set_edgecolor(FS.OTHER['zoom']); sp.set_linewidth(1.4)
    # loc1/loc2 = 1/4: the connectors leave the box's RIGHT corners, toward the
    # inset sitting to the right. Leaving from the left corners (2/3) drags two
    # long diagonals back across the whole panel.
    mark_inset(ax, axz, loc1=1, loc2=4, fc='none', ec=FS.OTHER['zoom'],
               linewidth=1.0, linestyle='--')
    fig.tight_layout()
    # tight_layout sizes the gap for the widest tick labels and then leaves it;
    # here that is far more than the inset's y-axis actually needs, and the long
    # connectors made the two panels read as separate figures. Tighten it after.
    fig.subplots_adjust(wspace=0.13)
    save(fig, 'Figure10_hydrology_EQF')
    pd.DataFrame({'p': p, 'gauge_height_ft': x,
                  'in_zoom': zmask}).to_csv(
        os.path.join(TAB, 'Figure10_hydrology_EQF_paneldata.csv'), index=False)

    # ===== Figure 11: histogram + full-sample fits ========================
    # No KDE. It would be a third smoother of the same data competing with the
    # very QPD fits the panel exists to show, and its Scott bandwidth is an
    # arbitrary choice the paper never justifies -- the section's argument is
    # about the fitted densities against the histogram, not about a reference
    # smoother.
    xg = np.linspace(max(0.0, x.min() - 1), x.max() + 1, 1000)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.hist(x, bins=20, density=True, color=C_BAND, alpha=0.75, edgecolor='white',
            linewidth=0.4, zorder=1, label='Histogram')
    for (lab, K), col in ((('GEV', 0), C_GEV), (('Log Metalog', 10), C_MET)):
        if (lab, K) in full:
            ax.plot(*draw_curve(lab, x, pp, K, (0, 12)), color=col, linewidth=2, zorder=3,
                    label=f'{lab}' + (f' K={K}' if K else ''))
    ax.set_xlabel('Gauge height (ft)'); ax.set_ylabel('Density')
    ax.set_xlim(0, 12); ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MultipleLocator(1))   # 1 ft, not matplotlib's 2 ft default
    ax.set_title('Histogram of annual peak flood gauge heights with fitted densities',
                 fontweight='bold')
    ax.legend(loc='upper right')
    fig.tight_layout(); save(fig, 'Figure11_hydrology_histogram_pdfs')
    rows = [{'model': f'{l} K={k}' if k else l, 'gauge_height_ft': q, 'pdf': d}
            for (l, k), (qq, dd) in full.items() for q, d in zip(qq, dd)]
    pd.DataFrame(rows).to_csv(
        os.path.join(TAB, 'Figure11_hydrology_histogram_pdfs_paneldata.csv'), index=False)

    # ===== Figure 14: quantile function + density, four comparison fits ==
    # Two panels on the same set of full-sample fits: the GEV the hydrology
    # literature uses, a low-order Metalog (K=4), the Metalog order shown in
    # Figures 11-13 (K=10), and Log QFlex-A+ at that same K=10.
    #
    # Left  -- quantile functions against the observed EQF and its pointwise
    #          95% bootstrap interval (the same band as Figure 12, from
    #          Figure12_hydrology_EQF_bootstrap_interval_paneldata.csv). This
    #          is the space Equation 6 is defined in, so the W1 values live in
    #          this legend.
    # Right -- the matching densities against the histogram, where the
    #          high-order Metalog's spurious spike is visible.
    #
    # The W1 shown is Equation 6 for the FULL-SAMPLE fit actually drawn --
    # the distance between that curve and the observed EQF. It is NOT Table
    # 8's W1, which is the median over 1000 bootstrap replicates; labelling a
    # full-sample curve with a bootstrap median would attach a number to a
    # picture it was not computed from.
    #
    # QFlex-A+ is solid: its colour already separates it from Metalog, so the
    # dashing is only needed to tell Metalog K=4 from Metalog K=10.
    GH_LIM = (0, 12)            # one gauge-height range shared by both panels
    COMPARE = (('GEV', 0, C_GEV, '-'),
               ('Log Metalog', 4, C_MET, KSTYLE[4]),
               ('Log Metalog', 10, C_MET, KSTYLE[10]),
               ('Log QFlex-A+', 10, C_QFX, '-'))
    fig, (ax_q, ax_d) = plt.subplots(1, 2, figsize=(14, 5.8))
    ax_q.fill_between(band['p'], band['q025'], band['q975'], color=C_BAND,
                      alpha=FS.BAND_ALPHA, lw=0, zorder=1,
                      label='95% bootstrap interval')
    ax_q.plot(P_GRID, eqf, color=C_EMP, linewidth=1.5, zorder=5,
              label='Empirical quantile')
    h_d, _e, _pt = ax_d.hist(x, bins=20, density=True, color=C_BAND, alpha=0.75,
                             edgecolor='white', linewidth=0.4, zorder=1, label='Histogram')
    tops_d = [float(np.max(h_d))]
    cmp_rows = []
    for lab, K, col, ls in COMPARE:
        if (lab, K) not in full:
            print(f'  !! {lab} K={K} infeasible on the full sample -- omitted')
            continue
        qf, pdf = full[(lab, K)]
        tops_d.append(float(np.max(pdf[PLOT_GRID >= 0.01])))
        _raw, wn = w1(fit_qfun(KEY[lab], x, pp, K), x)     # Eq 6 at the sample points
        name = lab + (f' K={K}' if K else '')
        ax_q.plot(PLOT_GRID, qf, color=col, linewidth=2, linestyle=ls, zorder=4,
                  label=f'{name}  (W\u2081 = {100 * wn:.2f}%)')
        ax_d.plot(*draw_curve(lab, x, pp, K, GH_LIM), color=col, linewidth=2,
                  linestyle=ls, zorder=3, label=name)
        cmp_rows += [{'model': name, 'p': pv, 'gauge_height_ft': q, 'pdf': d,
                      'full_sample_W1_Eq6': wn}
                     for pv, q, d in zip(PLOT_GRID, qf, pdf)]
        print(f'  Figure 14 {name}: full-sample W1 (Eq 6) = {wn:.6f}')
    ax_q.set_xlabel('Probability p'); ax_q.set_ylabel('Gauge height (ft)')
    ax_q.set_xlim(0, 1); ax_q.set_ylim(*GH_LIM)
    ax_q.set_title('Quantile Function', fontweight='bold')
    ax_q.legend(loc='upper left', fontsize=8.5)
    ax_d.set_xlabel('Gauge height (ft)'); ax_d.set_ylabel('Density')
    # y range: the rule of the median-bootstrap figures below (1.15 x the
    # tallest fitted density on p >= 0.01), also covering the tallest
    # histogram bar. Autoscaling is not
    # usable: the drawn curves run to the lower bound, where some fits' exact
    # density diverges (e.g. Log QFlex-A+ K=10 at x < 1e-6 ft), which would
    # set the axis to ~1e15.
    ax_d.set_xlim(*GH_LIM); ax_d.set_ylim(0, 1.15 * max(tops_d))
    ax_d.set_title('Probability Density Function', fontweight='bold')
    ax_d.legend(loc='upper right', fontsize=8.5)
    fig.suptitle('Fitted quantile functions and densities\u2014annual peak '
                 'flood gauge heights (1920\u20132014)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(); save(fig, 'Figure14_hydrology_GEV_Metalog_QFlex')
    pd.DataFrame(cmp_rows).to_csv(
        os.path.join(TAB, 'Figure14_hydrology_GEV_Metalog_QFlex_paneldata.csv'),
        index=False)

    # ===== Figure 12: EQF + 95% bootstrap interval ========================
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.fill_between(band['p'], band['q025'], band['q975'], color=C_BAND, alpha=FS.BAND_ALPHA,
                    lw=0, zorder=1, label='95% bootstrap interval')
    ax.plot(P_GRID, eqf, color=C_EMP, linewidth=1.5, zorder=4, label='Empirical quantile')
    for (lab, K), col in ((('GEV', 0), C_GEV), (('Log Metalog', 10), C_MET)):
        if (lab, K) in full:
            qf, _ = full[(lab, K)]
            ax.plot(PLOT_GRID, qf, color=col, linewidth=1.8, linestyle='--', zorder=3,
                    label=f'{lab}' + (f' K={K}' if K else ''))
    ax.set_xlabel('Probability p'); ax.set_ylabel('Gauge height (ft)')
    ax.set_xlim(0, 1); ax.set_ylim(0, 12)
    ax.yaxis.set_major_locator(MultipleLocator(1))   # 1 ft, not matplotlib's 2 ft default
    ax.set_title('EQF of annual peak flood gauge heights\nwith pointwise 95% bootstrap intervals',
                 fontweight='bold')
    ax.legend(loc='upper left')
    fig.tight_layout(); save(fig, 'Figure12_hydrology_EQF_bootstrap_interval')

    # ===== Figure 13: bootstrap mode scatter, one panel per model =========
    # One panel per model: overlaid, the three clouds would hide one another.
    # Shared axes so the panels are directly comparable. GEV has no order.
    # y capped at MODE_YMAX: 4 of the 372 valid Log Metalog K=10 fits put the
    # dominant mode above it (up to 24.3/ft -- needle-thin spikes on a
    # near-degenerate density). Letting them set the range squashes the other
    # 2335 points into the bottom 2% of the panel. They are clipped, not
    # dropped, and the panel CSV keeps every point.
    SPECS = (('GEV', 0, C_GEV), ('Log Metalog', 10, C_MET),
             ('Log QFlex-A+', 10, C_QFX))
    pts = {}
    for lab, K, _c in SPECS:
        g = sweep[(sweep['model'] == lab) & (sweep['K'] == K) & sweep['feasible']]
        locs, hgts = [], []
        for _, r in g.iterrows():
            ll = [float(v) for v in str(r['mode_locs']).split(';')
                  if v and v.lower() != 'nan']
            hh = [float(v) for v in str(r['mode_heights']).split(';')
                  if v and v.lower() != 'nan']
            if not ll:
                continue
            i = int(np.argmax(hh))            # the dominant mode of that fit
            locs.append(ll[i]); hgts.append(hh[i])
        pts[lab] = (np.array(locs), np.array(hgts))
        print(f'  mode scatter {lab} K={K}: n={len(locs)}, '
              f'loc [{min(locs):.2f}, {max(locs):.2f}] ft')
    all_loc = np.concatenate([v[0] for v in pts.values()])
    # x capped at MODE_XMAX for the same reason as y: a single Log Metalog K=10
    # fit puts its dominant mode at 7.31 ft, 1.6 ft beyond the next-highest
    # point (5.68 ft), and letting it set the range leaves a quarter of every
    # panel empty. Clipped, not dropped; the panel CSV keeps it.
    xpad = 0.05 * (MODE_XMAX - all_loc.min())
    XL = (all_loc.min() - xpad, MODE_XMAX)
    print(f'  mode scatter x range {XL[0]:.2f}-{XL[1]:.2f} ft; '
          f'{int((all_loc > MODE_XMAX).sum())} of {all_loc.size} points clipped')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharex=True, sharey=True)
    for ax_, (lab, K, col) in zip(axes, SPECS):
        mloc, mht = pts[lab]          # not L/H: L is the interdecile range
        ax_.scatter(mloc, mht, s=18, alpha=0.55, marker='o', facecolors='none',
                    edgecolors=col, linewidths=0.9)
        ax_.set_title(lab + (f' (K = {K})' if K else ''), fontweight='bold')
        ax_.set_xlabel('Mode location (ft)')
        ax_.set_xlim(*XL); ax_.set_ylim(0, MODE_YMAX)
    axes[0].set_ylabel('Mode height (1/ft)')
    fig.suptitle('Bootstrap distribution of the dominant mode',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout(); save(fig, 'Figure13_hydrology_mode_scatter')
    sc = []
    for lab, K in (('GEV', 0), ('Log Metalog', 10), ('Log QFlex-A+', 10)):
        g = sweep[(sweep['model'] == lab) & (sweep['K'] == K) & sweep['feasible']]
        for _, r in g.iterrows():
            ll = [float(v) for v in str(r['mode_locs']).split(';')
                  if v and v.lower() != 'nan']
            hh = [float(v) for v in str(r['mode_heights']).split(';')
                  if v and v.lower() != 'nan']
            if ll:
                i = int(np.argmax(hh))
                sc.append({'model': lab, 'K': K, 'b': int(r['b']),
                           'mode_loc_ft': ll[i], 'mode_height_perft': hh[i]})
    pd.DataFrame(sc).to_csv(os.path.join(TAB, 'Figure13_hydrology_mode_scatter_paneldata.csv'),
                            index=False)

    # ===== W1-median bootstrap fits (figures/other/; not in the paper) ====
    # W1* at the sample points: the median of the resamples' i-th order
    # statistics against x_(i) (no interpolation)
    S = np.array([np.sort(resample(x, b=b)) for b in range(N_BOOT)])
    _r, W1_STAR = w1_points(np.median(S, axis=0), x)
    print(f'W1* (median bootstrap QF vs observed EQF, Eq 6) = {W1_STAR:.6f}')

    picks, curves = [], {}
    chosen = {}
    for lab, key, K in [('GEV', 'gev', 0)] + \
                       [('Log Metalog', 'metalog', K) for K in K_VALUES] + \
                       [('Log QFlex-A+', 'qflex_a', K) for K in K_VALUES]:
        g = sweep[(sweep['model'] == lab) & (sweep['K'] == K) &
                  sweep['feasible'] & sweep['w1_orig_eq6'].notna()]
        # W1* is a distance from the OBSERVED sample, so the match uses each
        # fit's W1 vs the observed sample (w1_orig_eq6), whatever
        # EF.BOOTSTRAP_W1_REFERENCE says
        j = (g['w1_orig_eq6'] - W1_STAR).abs().idxmin()
        b = int(g.loc[j, 'b'])
        out = fit_curve(key, np.sort(resample(x, b=b)), pp, K, grid=PLOT_GRID)
        qf, pdf = out
        curves[(lab, K)] = (qf, pdf)
        chosen[(lab, K)] = b
        nm, ll, hh = EF.modes_from_density(qf, pdf)
        _raw, wn = w1(fit_qfun(key, np.sort(resample(x, b=b)), pp, K), x)   # vs observed
        picks.append(dict(model=lab, K=K, n_valid=len(g), W1_star_Eq6=round(W1_STAR, 6),
                          selected_replicate=b, selected_W1_Eq6=round(wn, 6),
                          n_modes=EF.n_modes(nm) if nm is not None else np.nan,
                          mode_locations=';'.join(f'{v:.4f}' for v in (ll if ll is not None else []))))
        print(f'  {lab} K={K}: {len(g)} valid, b={b}, W1={wn:.6f}, modes={nm}')
    pd.DataFrame(picks).to_csv(os.path.join(TAB, 'hydrology_median_fit_selection.csv'),
                               index=False)

    pdf_max = max(float(np.max(pdf[PLOT_GRID >= 0.01])) for pdf in
                  (curves[k][1] for k in curves))
    ymax = 1.15 * pdf_max            # ONE shared y-range for both figures
    for fam, col, name in (('Log Metalog', C_MET, 'hydrology_metalog_median_bootstrap'),
                           ('Log QFlex-A+', C_QFX, 'hydrology_qflex_median_bootstrap')):
        fig, (ax_q, ax_p) = plt.subplots(1, 2, figsize=(13.5, 5.8))
        gq, gp = curves[('GEV', 0)]
        ax_q.plot(PLOT_GRID, gq, color=C_GEV, linewidth=2, label='GEV')
        ax_p.plot(*draw_curve('GEV', np.sort(resample(x, b=chosen[('GEV', 0)])), pp, 0,
                              (0, 14)), color=C_GEV, linewidth=2, label='GEV')
        for K in K_VALUES:
            qf, pdf = curves[(fam, K)]
            ax_q.plot(PLOT_GRID, qf, color=col, linewidth=1.8, linestyle=KSTYLE[K],
                      label=f'{fam} K={K}')
            ax_p.plot(*draw_curve(fam, np.sort(resample(x, b=chosen[(fam, K)])), pp, K,
                                  (0, 14)), color=col, linewidth=1.8, linestyle=KSTYLE[K],
                      label=f'{fam} K={K}')
        ax_q.plot(P_GRID, eqf, color=C_EMP, linewidth=1.2, alpha=0.8,
                  label='Empirical quantile')
        ax_q.set_xlabel('Probability p'); ax_q.set_ylabel('Gauge height (ft)')
        ax_q.set_xlim(0, 1); ax_q.set_ylim(0, 14); ax_q.set_title('Quantile Function')
        ax_q.legend(loc='upper left', fontsize=8.5)
        ax_p.set_xlabel('Gauge height (ft)'); ax_p.set_ylabel('Density')
        ax_p.set_xlim(0, 14); ax_p.set_ylim(0, ymax)
        ax_p.set_title('Probability Density Function')
        ax_p.legend(loc='upper right', fontsize=8.5)
        fig.suptitle(f'W1 Median Bootstrap Fit — GEV and {fam} (K = 4, 7, 10)',
                     fontsize=14, fontweight='bold')
        fig.tight_layout(); save(fig, name)

    rows = [{'model': l, 'K': k, 'p': pv, 'qf_ft': q, 'pdf': d}
            for (l, k), (qq, dd) in curves.items()
            for pv, q, d in zip(PLOT_GRID, qq, dd)]
    pd.DataFrame(rows).to_csv(
        os.path.join(TAB, 'hydrology_median_bootstrap_fits_paneldata.csv'), index=False)

    fn, n, sha = data_fingerprint()
    json.dump({
        'generated': datetime.datetime.now().isoformat(timespec='seconds'),
        'data_file': fn, 'data_N': n, 'data_sha256': sha,
        'paper_figures': {'Figure 10': 'EQF with zoom',
                          'Figure 11': 'histogram + GEV + Log Metalog K=10',
                          'Figure 12': 'EQF + 95% bootstrap interval',
                          'Figure 13': 'bootstrap mode scatter',
                          'Figure 14': 'QF + PDF (with histogram): GEV/Metalog-4/Metalog-10/'
                                       'QFlex-A+-10, full-sample W1 in legend',
                          'not in the paper': 'median bootstrap fits (Log Metalog, Log QFlex-A+)'},
        'source': 'tables/hydrology_sweep.csv -- same 1000 paired replicates as Table 8',
        'W1_star_Eq6': W1_STAR,
        'shared_pdf_ymax': ymax,
        'mode_scatter_ymax': MODE_YMAX,
        'density_drawn_from_p': float(PLOT_GRID[0]),
    }, open(os.path.join(LOG, 'hydrology_figures_provenance.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
