"""
Fish Biology — Secondary Mode Location/Height Scatter, K=10
================================================================
For Log Metalog K=10 and Log QFlex-TA+ K=10, at jitter widths ±0.5 lb
and ±1.0 lb: draw 1000 bootstrap resamples (resample with replacement,
fresh jitter each draw), fit the model, and for every fit that comes out
bimodal, record the location and height of the SECONDARY mode (the peak
with the smaller PDF height).

The scatter's axes are (weight, density) -- exactly the axes of a density
plot -- so the fitted PDF curves are overlaid on the same axes. Each
secondary-mode point then sits on its own curve's second peak, and the
height of that peak above the following antimode, plus the distance between
them, can be read straight off the figure instead of inferred from a cloud
of bare points. One reference curve is drawn per panel: the single raw-data Log Metalog
K=10 fit the peak/antimode marks come from, drawn in the Metalog colour
(dashed, so it does not read as a third point series); the peak/antimode
marks themselves stay neutral gray.

Not in the paper. It follows the fish section's one set of conventions:

  * resamples come from fish_common.resample, so the 1000 replicates per
    jitter are the SAME ones behind Tables 9 and 10 and the Figure 16 band,
    and both models see the same replicate b (seeded per replicate, so the
    figure is reproducible run to run);
  * the QF/PDF grid is fish_common.P_GRID = the evaluation grid 0.001..0.999,
    so modality here is judged on the same grid as everywhere else in the
    section.

Output:
  reproduction/tables/fish_mode_scatter_paneldata.csv
  reproduction/figures/other/fish_mode_scatter_k10.png  (1x2 figure,
    one panel per jitter width, secondary-mode location vs height, with the
    corresponding density curves overlaid)
"""

import os, sys, warnings, argparse, multiprocessing as mp
import numpy as np
import figure_style as FS
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DATA_PATH  = os.path.join(ROOT_DIR, 'data', 'Fish Biology.xlsx')
FIG_DIR    = os.path.join(SCRIPT_DIR, '..', 'figures')
TAB_DIR    = os.path.join(SCRIPT_DIR, '..', 'tables')
LOG_DIR    = os.path.join(SCRIPT_DIR, '..', 'logs')
for _d in (FIG_DIR, TAB_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT_DIR, 'code', 'common'))
sys.path.insert(0, SCRIPT_DIR)
from fish_common import load_fish, resample, P_GRID as FISH_P_GRID
from metalog.metalog_v2 import LogMetalog
from qflex.transforms import LogQFlex
from qflex.constraints import ConstraintType
import eval_framework as EF

P_GRID         = FISH_P_GRID          # evaluation grid 0.001..0.999 (eval_framework)
# Common weight grid for overlaying densities. Pointwise medians across fits
# need a shared x-axis, and each fit's own density is parameterised by p, not
# by weight, so every curve is interpolated onto this grid before combining.
W_GRID         = np.linspace(2.0, 14.0, 600)
JITTER_WIDTHS  = [0.5, 1.0]
K              = 10
N_BOOT         = 1000

x_full, _p_full = load_fish()
N      = len(x_full)
print(f'Loaded N={N} fish weights, range [{x_full[0]:.1f}, {x_full[-1]:.1f}] lbs')


def fit_and_get_secondary_mode(x_boot, p_boot, model_label):
    try:
        if model_label == 'Log Metalog':
            m = LogMetalog(x_boot, p_boot, lower_bound=0, terms=K)
        else:
            m = LogQFlex(x_boot, p_boot, lower_bound=0, terms=K, constraint_type=ConstraintType.TA)
        if not EF.is_valid(m):
            return None
        qf, pdf_vals = EF.density(m)
    except Exception:
        return None

    n_modes, locs, hgts = EF.modes_from_density(qf, pdf_vals)
    if n_modes != 2 or locs is None or len(locs) != 2:
        return None

    # secondary = lower-height peak
    order = np.argsort(hgts)
    sec_idx = order[0]
    # density resampled onto the shared weight grid, for the overlay
    pdf_w = np.interp(W_GRID, qf, pdf_vals, left=np.nan, right=np.nan)
    return float(locs[sec_idx]), float(hgts[sec_idx]), pdf_w


def run_one(args):
    jitter, model_label, boot_id = args
    # SEEDING: the section's canonical per-replicate scheme
    # (reproduction/scripts/fish_common.py):
    #     default_rng(SEED + round(jitter*100)*10_000 + b)
    # The seed does not depend on the model, so both models see the SAME
    # replicate b, which is what makes the two scatters comparable -- and the
    # same replicates Tables 9 and 10 use.
    x_boot = np.sort(resample(x_full, jitter, boot_id))
    p_boot = np.arange(1, N + 1) / (N + 1)

    result = fit_and_get_secondary_mode(x_boot, p_boot, model_label)
    if result is None:
        return None
    loc, hgt, pdf_w = result
    return {'jitter': jitter, 'model': model_label, 'loc': loc, 'height': hgt,
            'pdf_w': pdf_w}


CACHE_PATH = None   # set in __main__ once FIG_DIR is known

if __name__ == '__main__':
    _ap = argparse.ArgumentParser()
    _ap.add_argument('--procs', type=int, default=0,
                     help='worker processes (default: cpu_count()-1)')
    _ap.add_argument('--replot', action='store_true',
                     help='reuse the cached fits and only redraw the figure '
                          '(the 4000 bootstrap fits take several minutes; '
                          'axis/legend tweaks do not need them recomputed)')
    _args = _ap.parse_args()

    models = ['Log Metalog', 'Log QFlex-TA+']
    CACHE_PATH = os.path.join(TAB_DIR, 'fish_mode_scatter_cache.npz')
    csv_path = os.path.join(TAB_DIR, 'fish_mode_scatter_paneldata.csv')

    if _args.replot and os.path.exists(CACHE_PATH) and os.path.exists(csv_path):
        print(f'--replot: reusing cached fits from {CACHE_PATH}')
        df = pd.read_csv(csv_path)
        _c = np.load(CACHE_PATH)
        median_pdf = {(float(j), m): _c[f'median__{j}__{m}']
                      for j in JITTER_WIDTHS for m in models
                      if f'median__{j}__{m}' in _c.files}
    else:
        jobs = [
            (jitter, model, b)
            for jitter in JITTER_WIDTHS
            for model in models
            for b in range(N_BOOT)
        ]
        print(f'Running {len(jobs)} fits (K={K})...')

        n_workers = _args.procs if _args.procs > 0 else max(1, mp.cpu_count() - 1)
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(run_one, jobs, chunksize=50)

        rows = [r for r in results if r is not None]
        curves = {}      # (jitter, model) -> list of density curves on W_GRID
        for r in rows:
            curves.setdefault((r['jitter'], r['model']), []).append(r.pop('pdf_w'))
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        # pointwise median density per (jitter, model), over the bimodal fits only
        median_pdf = {}
        for k, lst in curves.items():
            arr = np.vstack(lst)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                median_pdf[k] = np.nanmedian(arr, axis=0)
        np.savez_compressed(CACHE_PATH, W_GRID=W_GRID,
                            **{f'median__{k[0]}__{k[1]}': v for k, v in median_pdf.items()})
        print(f'Cached fits -> {CACHE_PATH}')

    for jitter in JITTER_WIDTHS:
        for model in models:
            n = len(df[(df['jitter'] == jitter) & (df['model'] == model)])
            print(f'  jitter={jitter}, {model}: {n} bimodal fits (of {N_BOOT})')

    # ── reference: single Log Metalog K=10 fit on the raw (unjittered) data ──────
    p_emp = np.arange(1, N + 1) / (N + 1)
    m_ref = LogMetalog(x_full, p_emp, lower_bound=0, terms=K)
    qf_ref, pdf_ref = EF.density(m_ref)
    n_modes_ref, locs_ref, hgts_ref = EF.modes_from_density(qf_ref, pdf_ref)
    print(f'Reference fit (raw data, no jitter): n_modes={n_modes_ref}, '
          f'locs={locs_ref}, hgts={hgts_ref}')

    sec_i = np.argmin(hgts_ref)
    ref_sec_loc, ref_sec_hgt = locs_ref[sec_i], hgts_ref[sec_i]
    ref_pri_loc = locs_ref[1 - sec_i]

    i_sec = np.argmin(np.abs(qf_ref - ref_sec_loc))
    i_pri = np.argmin(np.abs(qf_ref - ref_pri_loc))
    lo, hi = sorted([i_sec, i_pri])
    valley_idx = lo + np.argmin(pdf_ref[lo:hi + 1])
    ref_anti_loc, ref_anti_hgt = qf_ref[valley_idx], pdf_ref[valley_idx]
    print(f'  secondary peak: loc={ref_sec_loc:.3f}, hgt={ref_sec_hgt:.5f}')
    print(f'  antimode:       loc={ref_anti_loc:.3f}, hgt={ref_anti_hgt:.5f}')

    # ── figure ────────────────────────────────────────────────────────────────
    mpl.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 11,
        'axes.titlesize': 12, 'axes.labelsize': 10,
        'xtick.labelsize': 9,  'ytick.labelsize': 9,
        'legend.fontsize': 9,  'legend.frameon': True, 'legend.framealpha': 0.85,
        'figure.dpi': 150, 'savefig.dpi': 300,
        'axes.grid': True, 'grid.alpha': 0.3, 'grid.color': '#cccccc',
    })
    # Model colours are the paper palette (figure_style.QPD), so a colour
    # means the same model across the fish figures. Shape carries the model as well, for black-and-white print. Both series
    # hollow: the two clouds overlap heavily here and filled markers turn the
    # overlap region into a solid blob.
    # The reference curve and its peak/antimode marks stay neutral gray --
    # they are an annotation, not a third series.
    MARKERS = {'Log Metalog': '^', 'Log QFlex-TA+': 's'}
    FACE = {
        'Log Metalog':   dict(facecolors='none', edgecolors=FS.QPD['Metalog'], linewidths=0.9),
        'Log QFlex-TA+': dict(facecolors='none', edgecolors=FS.QPD['QFlex-TA+'], linewidths=0.9),
    }
    # The raw-data reference density is a Log Metalog fit, so it carries the
    # Metalog colour -- a colour means the same model everywhere in the paper,
    # and the line style (dotted) already separates it from the point cloud.
    # The peak/antimode marks stay neutral: they are a measurement annotation,
    # not a model curve.
    REF_CURVE_COLOR = FS.QPD['Metalog']
    REF_COLOR = '#333333'

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)

    # Axis limits come from the SECONDARY-MODE points and the reference
    # peak/antimode marks only -- deliberately not from the overlaid density
    # curves. Those curves run up to the primary mode, an order of magnitude
    # taller than anything here, and letting them drive the range would
    # squash the secondary-mode cloud this figure exists to show. The curves
    # are drawn anyway and matplotlib clips them at the axes, so only the
    # portion crossing this window appears.
    y_lo = min(df['height'].min(), ref_anti_hgt)
    y_hi = max(df['height'].max(), ref_sec_hgt)
    y_pad = 0.15 * (y_hi - y_lo)

    for ax, jitter in zip(axes, JITTER_WIDTHS):
        for model in models:
            sub = df[(df['jitter'] == jitter) & (df['model'] == model)]
            ax.scatter(sub['loc'], sub['height'], s=20, alpha=0.6,
                       marker=MARKERS[model], **FACE[model],
                       label=model)

        # ONE reference curve: the single raw-data (no jitter) Log Metalog
        # K=10 density that the peak/antimode marks below are read off. The
        # scatter's axes are (weight, density), so this curve shares them --
        # the secondary peak, the antimode, and the drop between them can be
        # read directly instead of inferred from the point cloud. Drawn in
        # the Metalog colour, since that is the model it is a fit of; dashed,
        # so it does not compete with the two point clouds. Per-model median
        # bootstrap densities are deliberately NOT drawn -- they added two
        # more lines without adding a reference.
        ax.plot(qf_ref, pdf_ref, color=REF_CURVE_COLOR, linewidth=1.6, linestyle='--',
                alpha=0.95, zorder=4.5, label='Raw-data Metalog K=10 density')

        # reference lines from the single raw-data (no jitter) K=10 fit
        ax.plot([ref_anti_loc, ref_sec_loc], [ref_anti_hgt, ref_anti_hgt],
                color=REF_COLOR, linewidth=1.6, solid_capstyle='butt', zorder=5,
                label='Raw-data Metalog fit: peak-to-antimode width/height')
        ax.plot([ref_sec_loc, ref_sec_loc], [ref_anti_hgt, ref_sec_hgt],
                color=REF_COLOR, linewidth=1.6, solid_capstyle='butt', zorder=5)
        ax.scatter([ref_sec_loc, ref_anti_loc], [ref_sec_hgt, ref_anti_hgt],
                   marker='x', s=45, color=REF_COLOR, zorder=6)

        ax.set_title(f'Jitter width = ±{jitter:.1f} lb', fontweight='bold')
        ax.set_xlabel('Secondary mode location (lbs)')
        ax.set_xlim(4.0, 7.0)
        ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
        ax.legend(loc='upper right', fontsize=7.5, markerscale=1.4)

    axes[0].set_ylabel('Secondary mode height (density)')

    fig.suptitle(f'Secondary Mode Dispersion — Fish Biology, K={K}\n'
                 f'Log Metalog vs. Log QFlex-TA+ ({N_BOOT} bootstrap fits per panel)',
                 fontsize=13, fontweight='bold', y=1.03)
    fig.tight_layout()

    out_path = FS.figure_path('fish_mode_scatter_k10') + '.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    fig.savefig(FS.figure_path('fish_mode_scatter_k10') + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {out_path}')

    import json, datetime
    json.dump({
        'generated': datetime.datetime.now().isoformat(timespec='seconds'),
        'paper_figure': 'not in the paper',
        'K': K, 'n_bootstrap': N_BOOT, 'jitters': JITTER_WIDTHS,
        'resamples': 'fish_common.resample -- shared with Tables 9 and 10',
        'grid': 'eval_framework.GRID = 0.001..0.999',
        'reference_fit': 'LogMetalog K=10 on RAW recorded weights, no jitter '
                         '(same curve overlaid on Figure 15)',
        'reference_n_modes': int(n_modes_ref),
        'reference_secondary_mode': {'loc': float(ref_sec_loc), 'height': float(ref_sec_hgt)},
        'reference_antimode': {'loc': float(ref_anti_loc), 'height': float(ref_anti_hgt)},
        'bimodal_counts': {f'jitter{j}_{m}':
                           int(((df['jitter'] == j) & (df['model'] == m)).sum())
                           for j in JITTER_WIDTHS for m in models},
    }, open(os.path.join(LOG_DIR, 'fish_mode_scatter_provenance.json'), 'w'), indent=2)
