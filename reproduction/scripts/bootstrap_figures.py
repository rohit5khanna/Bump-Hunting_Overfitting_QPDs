"""
Bootstrap diagnostics (Section 4.5) -- Figures 7 and 8, the QFlex-U and
QFlex-TA+ versions in the Online Supplement (Figures O5 and O7), and one
figure that is not in the paper.

    Figure 7   Metalog mode count and dispersion under bootstrap resampling
               (Johnson SU, N = 200)          -> Figure07_Metalog_bootstrap_modes
    Figure 8   QFlex-A+ mode count and dispersion under bootstrap resampling
                                              -> Figure08_QFlexAplus_bootstrap_modes
    Figure O5  QFlex-U version                -> FigureO5_QFlexU_bootstrap_modes
    Figure O7  QFlex-TA+ version              -> FigureO7_QFlexTAplus_bootstrap_modes
    (not in the paper) valid unimodal Metalog and QFlex-A+ fits from the
               bootstrap samples at K = 10 with shifted modes
               -> figures/other/bootstrap_valid_unimodal_shifted_modes

(Figure O6, the QFlex-T+ version, is made by bootstrap_tplus_figure.py with
the same dispersion_figure function.)

All come from tables/bootstrap_sweep.csv -- the pinned realization
(Johnson SU, N=200, seed 200110) and its 1000 resamples.

Each dispersion figure has one row per order (K = 4, 7, 10, 13):
  left    distribution of mode counts across valid bootstrap fits, with the
          bar for the ORIGINAL realization's mode count in the model's colour
          and the others grey.
  right   primary-mode location vs height, against the population mode.

The population mode is computed, not hardcoded: jpse's pdf() is a function of
x (not of p), so it is evaluated at quantile(p). That gives -0.2505 / 0.4550,
the values the manuscript quotes as "(-0.250)" and "(0.455)".
"""
import os, sys, json, datetime, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib as mpl
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'code', 'common')); sys.path.insert(0, HERE)
from bootstrap_section import (load_original_sample, load_original_sample_unsorted,
                               bootstrap_resample, SEED, N, DIST_PARAMS)
from jpse import JohnsonSU
from metalog.metalog_v2 import Metalog
from qflex.core import QFlex
from qflex.constraints import ConstraintType
import eval_framework as EF
import figure_style as FS

TAB = os.path.join(HERE, '..', 'tables'); FIG = os.path.join(HERE, '..', 'figures')
LOG = os.path.join(HERE, '..', 'logs')
for d in (TAB, FIG, LOG): os.makedirs(d, exist_ok=True)

KS = FS.K_STANDARD          # 4, 7, 10, 13

# Shared axes, so the dispersion figures are directly comparable.
# Primary-mode LOCATION spans -1.96..0.98 over every model and order, so a
# common x-range costs nothing. HEIGHT is capped: 99.5% of points lie below
# 1.46, but two pathological fits (QFlex-TA+ K=10 at 18.56, Metalog K=10 at
# 7.54) would otherwise squash every panel into a sliver. Points outside the
# box fall outside the axes and are not drawn.
LOC_LIM = (-2.10, 1.10)
# The shifted-modes figure draws every curve -- fits and population -- on the
# evaluation grid 0.001..0.999 only (eval_framework.GRID); nothing is
# extrapolated beyond it.
HT_LIM  = (0.25, 2.05)
CT = {'Metalog': None, 'QFlex-U': ConstraintType.NONE,
      'QFlex-A+': ConstraintType.A, 'QFlex-TA+': ConstraintType.TA}
COL = dict(FS.QPD)
# The scatter carries the model's reserved colour; the histogram bars go
# neutral grey with only the original fit's bar highlighted. Bars and scatter
# stay distinguishable by form, not by hue.
mpl.rcParams.update({'font.family': 'sans-serif', 'font.size': 11,
                     'axes.titlesize': 11, 'axes.labelsize': 10,
                     'xtick.labelsize': 9, 'ytick.labelsize': 9,
                     'legend.fontsize': 8.5, 'figure.dpi': 150, 'savefig.dpi': 300,
                     'axes.grid': True, 'grid.alpha': 0.3})


def true_mode():
    d = JohnsonSU(**DIST_PARAMS)
    p = np.linspace(1e-6, 1 - 1e-6, 200001)
    q = np.asarray(d.quantile(p), float)
    pdf = np.asarray(d.pdf(q), float)        # pdf takes x, not p
    i = int(np.argmax(pdf))
    return float(q[i]), float(pdf[i])


def fit(label, x, p, K):
    try:
        m = (Metalog(x, p, terms=K) if label == 'Metalog'
             else QFlex(x, p, terms=K, constraint_type=CT[label]))
        return m if EF.is_valid(m) else None
    except Exception:
        return None


def primary_modes(sub):
    """(location, height) of the tallest mode of each valid fit."""
    L, H = [], []
    for _, r in sub.iterrows():
        ll = [float(t) for t in str(r['mode_locs']).split(';') if t]
        hh = [float(t) for t in str(r['mode_heights']).split(';') if t]
        if ll:
            i = int(np.argmax(hh)); L.append(ll[i]); H.append(hh[i])
    return np.array(L), np.array(H)


def dispersion_figure(df, orig, persist, label, tm, out_name, fignum):
    tl, th = tm
    # tick set is the union over ALL models, not this one, so the bar panels
    # line up across figures
    mode_ticks = sorted({int(v) for v in df['n_modes'].dropna().unique()})
    # 4 x 2: one row per order, histogram left, primary-mode scatter right.
    fig, axes = plt.subplots(len(KS), 2, figsize=(10.5, 3.3 * len(KS)))
    for j, K in enumerate(KS):
        g = df[(df['model'] == label) & (df['K'] == K) & df['feasible'] & df['n_modes'].notna()]
        o = orig.get((label, K), np.nan)
        pv = persist.get((label, K), np.nan)

        ax = axes[j, 0]
        if len(g):
            vc = g['n_modes'].value_counts().sort_index()
            ks = [int(v) for v in vc.index]
            cols = [COL[label] if k == o else FS.OTHER['data'] for k in ks]
            ax.bar(ks, 100 * vc.values / len(g), color=cols, edgecolor='white', width=0.7)
            for k, v in zip(ks, 100 * vc.values / len(g)):
                ax.text(k, v + 1.5, f'{v:.1f}', ha='center', fontsize=8)
        ax.set_title(f'K = {K}', fontweight='bold')
        ax.set_xlabel('Number of modes'); ax.set_ylabel('% of valid fits')
        ax.set_ylim(0, 108)
        ax.set_xticks(mode_ticks)
        ax.set_xlim(min(mode_ticks) - 0.6, max(mode_ticks) + 0.6)

        ax = axes[j, 1]
        L, H = primary_modes(g)
        if len(L):
            ax.scatter(L, H, s=16, alpha=0.5, facecolors='none',
                       edgecolors=COL[label], linewidths=0.8, label=label)
        ax.axvline(tl, color='k', linestyle='--', linewidth=1.1)
        ax.axhline(th, color='k', linestyle='--', linewidth=1.1)
        ax.plot([tl], [th], marker='X', markersize=10, color='k', linestyle='none',
                label='Population mode')
        ax.set_xlabel('Primary mode location'); ax.set_ylabel('Primary mode height')
        ax.set_xlim(*LOC_LIM); ax.set_ylim(*HT_LIM)
        ax.legend(loc='upper right', fontsize=7.5)
    fig.suptitle(f'{label} Mode Distribution and Dispersion under Bootstrap Resampling\n'
                 f'(Johnson SU, N = {N}, seed {SEED}, 1000 resamples)',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(FS.figure_path(out_name) + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(FS.figure_path(out_name) + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_name}')


def main():
    df = pd.read_csv(os.path.join(TAB, 'bootstrap_sweep.csv'))
    of = pd.read_csv(os.path.join(TAB, 'bootstrap_original_fits.csv'))
    pf = pd.read_csv(os.path.join(TAB, 'bootstrap_persist.csv'))
    orig = {(r['model'], r['K']): r['n_modes'] for _, r in of.iterrows()}
    persist = {(r['model'], r['K']): r['persist_pct'] for _, r in pf.iterrows()}
    tm = true_mode()
    print(f'population mode: location {tm[0]:.4f}, height {tm[1]:.4f}  '
          f'(manuscript quotes -0.250 and 0.455)')

    for label, out, num in (('Metalog', 'Figure07_Metalog_bootstrap_modes', '7'),
                            ('QFlex-A+', 'Figure08_QFlexAplus_bootstrap_modes', '8'),
                            ('QFlex-U', 'FigureO5_QFlexU_bootstrap_modes', 'O5'),
                            ('QFlex-TA+', 'FigureO7_QFlexTAplus_bootstrap_modes', 'O7')):
        dispersion_figure(df, orig, persist, label, tm, out, num)

    # ---- valid unimodal K=10 fits with shifted modes (not in the paper) ----
    xs = load_original_sample(); xu = load_original_sample_unsorted()
    p = np.arange(1, N + 1) / (N + 1)
    d = JohnsonSU(**DIST_PARAMS)
    true_qf = np.asarray(d.quantile(EF.GRID), float)     # population, same p range as the fits
    true_pdf = np.asarray(d.pdf(true_qf), float)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    rows = []
    for ax, label in zip(axes, ('Metalog', 'QFlex-A+')):
        g = df[(df['model'] == label) & (df['K'] == 10) & df['feasible'] &
               (df['n_modes'] == 1)]
        L, _H = primary_modes(g)
        bs = g['b'].values
        lo_b, hi_b = int(bs[int(np.argmin(L))]), int(bs[int(np.argmax(L))])
        ax.plot(true_qf, true_pdf, color='k', linestyle='--', linewidth=1.8,
                zorder=4, label='Population density')
        for b, style, tag in ((lo_b, '-', 'left-shifted'), (hi_b, ':', 'right-shifted')):
            m = fit(label, np.sort(bootstrap_resample(xu, b)), p, 10)
            # drawn on the 0.001..0.999 grid only, density from quantile differences
            qf_pl, pdf_pl = EF.density(m)
            nm, ml, mh = EF.modes(m)
            # W1 is read from the sweep, not recomputed, so figure and tables agree
            _sr = df[(df['model'] == label) & (df['K'] == 10) & (df['b'] == b)].iloc[0]
            w1n = float(_sr['w1_norm'])          # main column (EF.BOOTSTRAP_W1_REFERENCE)
            ax.plot(qf_pl, pdf_pl, color=COL[label], linestyle=style, linewidth=1.9,
                    zorder=3, label=f'{tag}: W1 = {100 * w1n:.2f}%')
            rows.append({'model': label, 'which': tag, 'b': b, 'n_modes': int(nm),
                         'mode_location': float(ml[0]), 'mode_height': float(mh[0]),
                         'w1_eq6': w1n, 'w1_own_eq6': float(_sr['w1_own_norm']),
                         'w1_orig_eq6': float(_sr['w1_orig_norm']),
                         'w1_legend_reference': EF.BOOTSTRAP_W1_REFERENCE})
            print(f'  {label} K=10 {tag:<14} b={b:<4} mode {ml[0]:+.4f} '
                  f'(population {tm[0]:+.4f})')
        ax.axvline(tm[0], color='k', linewidth=0.9, alpha=0.5)
        ax.set_title(f'{label} (K = 10), valid unimodal fits', fontweight='bold')
        ax.set_xlabel('x'); ax.set_ylabel('Density')
        ax.set_xlim(-6, 4); ax.set_ylim(bottom=0)
        ax.legend(loc='upper left')
    # both panels on one density scale
    y7 = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, y7)
    fig.suptitle('Valid unimodal fits from the bootstrap samples at K = 10 '
                 f'with shifted modes\n(Johnson SU, N = {N}, seed {SEED})',
                 fontsize=13, fontweight='bold', y=1.03)
    fig.tight_layout()
    fig.savefig(FS.figure_path('bootstrap_valid_unimodal_shifted_modes') + '.png',
                dpi=300, bbox_inches='tight')
    fig.savefig(FS.figure_path('bootstrap_valid_unimodal_shifted_modes') + '.pdf',
                bbox_inches='tight')
    plt.close(fig)
    print('  saved bootstrap_valid_unimodal_shifted_modes  (not in the paper)')
    pd.DataFrame(rows).to_csv(os.path.join(TAB, 'bootstrap_valid_unimodal_shifted_modes.csv'), index=False)

    json.dump({
        'generated': datetime.datetime.now().isoformat(timespec='seconds'),
        'paper_figures': {'Figure 7': 'Metalog mode dispersion', 'Figure 8': 'QFlex-A+ mode dispersion',
                          'Figures O5, O7': 'QFlex-U and QFlex-TA+ versions'},
        'source': 'tables/bootstrap_sweep.csv -- pinned realization, 1000 resamples',
        'population_mode': {'location': tm[0], 'height': tm[1],
                            'manuscript_quotes': [-0.250, 0.455]},
        'persist_metalog': {str(K): persist.get(('Metalog', K)) for K in KS},
    }, open(os.path.join(LOG, 'bootstrap_figures_provenance.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
