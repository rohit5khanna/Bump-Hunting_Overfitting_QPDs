"""
Bimodal recovery (Section 4.6) -- Figure 9: the bimodal reference
distribution (60/40 mixture of two Johnson SU) with the Metalog and QFlex-TA+
fits at K = 12 to Monte Carlo replicate 436 (N = 200). Left panel: true
mixture quantile function, sample points and the two fitted QFs; right panel:
sample histogram, true mixture density and the two fitted PDFs.

Default (K = 12) writes figures/Figure09_bimodal_reference.{png,pdf} plus
panel data tables/Figure09_bimodal_{true_qf,paneldata,sample}.csv. `--K`
with any other order writes figures/other/bimodal_K<K>_reference and
tables/bimodal_K<K>_* (not in the paper).

The replicate is one of the Table 7 Monte Carlo draws (bimodal_common.draw),
so the population shown is the analytic mixture and the fits shown are fits
the experiment actually produces. It is a FIXED illustrative choice
(FIXED_REP = 436), not rule-selected. The legend W1 is Equation 6 at the
sample's own points (eval_framework.w1), as in Table 7. No W1 against the
true mixture is computed.

ON jpse VERSUS scipy. The sampler uses `jpse.JohnsonSU`; the analytic density
here uses `scipy.stats.johnsonsu`, because it exposes a pdf/cdf in x. The two
differ in their random STREAM (same seed, different draws) but not in the
distribution: their quantile functions agree to 0.0 on these parameters, and
both match a 200,000-point draw to 0.02 (Monte Carlo noise). The agreement is
asserted at run time.
"""
import os, sys, json, datetime, warnings
import numpy as np
import figure_style as FS, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib as mpl
from scipy.stats import johnsonsu
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'code', 'common')); sys.path.insert(0, HERE)
from bimodal_common import (DIST_PARAMS, WEIGHT_1, WEIGHT_2, shift, draw,
                            true_quantile, empirical_qf, interdecile_from_grid,
                            w1, P_GRID, SAMPLE_SIZE, MODE_SEPARATION_FACTOR)
from metalog.metalog_v2 import Metalog
from qflex.core import QFlex
from qflex.constraints import ConstraintType
import eval_framework as EF
from jpse import JohnsonSU

TAB = os.path.join(HERE, '..', 'tables')
FIG = os.path.join(HERE, '..', 'figures')
LOG = os.path.join(HERE, '..', 'logs')
for d in (TAB, FIG, LOG): os.makedirs(d, exist_ok=True)

K = 12
FIXED_REP = 436       # illustrative replicate shown in Figure 9
# Lower limit -10 trims empty margin on the left. The fitted PDFs still run
# to the edge of the axis.
XLIM = (-10, 5)

# Fits, the true mixture and modes are evaluated and DRAWN on the evaluation
# grid p = 0.001..0.999 only (eval_framework.GRID). W1 uses the sample points.
PLOT_GRID = P_GRID
COLORS = {'Metalog': FS.QPD['Metalog'], 'QFlex-TA+': FS.QPD['QFlex-TA+']}
#: QFlex dashed so the two fits stay distinguishable in grayscale print.
LS = {'Metalog': '-', 'QFlex-TA+': '--'}
mpl.rcParams.update({'font.family': 'sans-serif', 'font.size': 12,
                     'axes.titlesize': 14, 'axes.labelsize': 13,
                     'xtick.labelsize': 11, 'ytick.labelsize': 11,
                     'figure.dpi': 150, 'savefig.dpi': 300,
                     'axes.grid': True, 'grid.alpha': 0.3})


def main():
    C, D = DIST_PARAMS['gamma'], DIST_PARAMS['delta']     # jpse names; scipy johnsonsu(a=gamma, b=delta)
    ETA, KAPPA = DIST_PARAMS['xi'], DIST_PARAMS['lam']
    s = shift()

    # assert scipy and jpse describe the same distribution before using scipy
    pc = np.linspace(0.001, 0.999, 501)
    dj = float(np.max(np.abs(np.asarray(JohnsonSU(**DIST_PARAMS).quantile(pc), float)
                             - johnsonsu.ppf(pc, C, D, loc=ETA, scale=KAPPA))))
    assert dj < 1e-8, f'jpse and scipy disagree by {dj}'
    print(f'jpse vs scipy quantile agreement: max|diff| = {dj:.2e}  OK')

    def true_pdf(xa):
        return (WEIGHT_1 * johnsonsu.pdf(xa, C, D, loc=ETA, scale=KAPPA) +
                WEIGHT_2 * johnsonsu.pdf(xa, C, D, loc=ETA + s, scale=KAPPA))

    tq = true_quantile()                 # P_GRID: x-values for the true density
    tq_plot = true_quantile(PLOT_GRID)   # PLOT_GRID (= P_GRID): the drawn true QF

    # ---- representative replicate ---------------------------------------
    mc = pd.read_csv(os.path.join(TAB, 'bimodal_mc.csv'))
    g = mc[(mc['K'] == K) & mc['feasible']]
    both = (set(g[g['model'] == 'Metalog']['replication']) &
            set(g[g['model'] == 'QFlex-TA+']['replication']))
    met = g[(g['model'] == 'Metalog') & g['replication'].isin(both)]
    counts = met['n_modes'].value_counts()
    share = {int(k): round(100 * v / len(met), 1) for k, v in counts.sort_index().items()}
    # FIGURE SAMPLE: FIXED illustrative replicate (FIXED_REP). Not rule-selected.
    rep = FIXED_REP
    print(f'{len(both)} replicates feasible for BOTH models at K={K}; Metalog mode-count '
          f'shares {share}; figure uses FIXED replicate {rep}')

    x = draw(rep, SAMPLE_SIZE)
    p = np.arange(1, len(x) + 1) / (len(x) + 1)
    fits = {}; w1_sample = {}
    for lab, mk in (('Metalog', lambda: Metalog(x, p, terms=K)),
                    ('QFlex-TA+', lambda: QFlex(x, p, terms=K,
                                                constraint_type=ConstraintType.TA))):
        m = mk()
        qf, pdf = EF.density(m)
        nm, locs, _h = EF.modes(m)
        _rs, wn_s = w1(m, x)          # Eq 6 at the sample points, as in Table 7
        qf_pl, pdf_pl = qf, pdf
        fits[lab] = (qf, pdf, nm, locs, wn_s, qf_pl, pdf_pl)
        w1_sample[lab] = wn_s
        print(f'  {lab} K={K}: valid={EF.is_valid(m)}, '
              f'modes={nm}, locs={np.round(locs,3) if locs is not None else None}, '
              f'W1 (Eq 6, vs sample)={wn_s:.4f}')

    # ---- figure ----------------------------------------------------------
    fig, (ax_q, ax_d) = plt.subplots(1, 2, figsize=(13, 5.5))
    # NO SUPTITLE BY DEFAULT. The figure as placed in the paper carries no
    # overall title -- the caption of Figure 9 does that job, and a title
    # inside the image duplicates it. `--title` restores it for standalone use.
    if TITLE:
        fig.suptitle('Bimodal Reference Distribution — 60/40 Mixture of Two Johnson SU '
                     f'Distributions\n(separation {MODE_SEPARATION_FACTOR}σ, fits at K = {K} '
                     f'to a representative N = {SAMPLE_SIZE} draw)',
                     fontsize=13, fontweight='bold', y=1.03)

    ax_q.plot(PLOT_GRID, tq_plot, color='black', linewidth=1.8, linestyle='--',
              zorder=3, label='True mixture quantile function')
    ax_q.plot(p, x, color=FS.OTHER['data'], linewidth=0, marker='.', markersize=3.5,
              zorder=2, label=f'Sample points (N = {SAMPLE_SIZE})')
    for lab in ('Metalog', 'QFlex-TA+'):
        ax_q.plot(PLOT_GRID, fits[lab][5], color=COLORS[lab], linewidth=2, zorder=4,
                  linestyle=LS[lab],
                  label=f'{lab} (K={K})  (W\u2081 = {100 * w1_sample[lab]:.2f}%)')
    ax_q.set_xlabel('p'); ax_q.set_ylabel('x'); ax_q.set_title('Quantile Functions', fontweight='bold')
    ax_q.set_xlim(0, 1); ax_q.set_ylim(*XLIM)
    ax_q.legend(loc='upper left', fontsize=9)

    xg = tq                                   # the true density over the same p range
    ax_d.hist(x, bins=25, density=True, color=FS.OTHER['data'], alpha=0.6,
              edgecolor='white', linewidth=0.4, zorder=1, label='Sample histogram')
    ax_d.plot(xg, true_pdf(xg), color='black', linewidth=1.8, linestyle='--',
              zorder=3, label='True mixture density')
    for lab in ('Metalog', 'QFlex-TA+'):
        _q, _p, _nm, _l, _w, qf_pl, pdf_pl = fits[lab]
        ax_d.plot(qf_pl, pdf_pl, color=COLORS[lab], linewidth=2, zorder=4,
                  linestyle=LS[lab],
                  label=f'{lab} (K={K})')
    ax_d.set_xlabel('x'); ax_d.set_ylabel('Density')
    ax_d.set_title('Population Density and Fitted PDFs', fontweight='bold')
    ax_d.set_xlim(*XLIM); ax_d.set_ylim(bottom=0)
    ax_d.legend(loc='upper right', fontsize=9)

    fig.tight_layout()
    out = FS.figure_path(f'{BASE}_reference')
    fig.savefig(out + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(out + '.pdf', bbox_inches='tight'); plt.close(fig)
    print(f'Saved -> {out}.png')

    rows = [{'p': pv, 'true_qf': t} for pv, t in zip(PLOT_GRID, tq_plot)]
    pd.DataFrame(rows).to_csv(os.path.join(TAB, f'{BASE}_true_qf.csv'), index=False)
    pd.DataFrame({'model': np.repeat(list(fits), len(PLOT_GRID)),
                  'p': np.tile(PLOT_GRID, len(fits)),
                  'qf': np.concatenate([fits[l][5] for l in fits]),
                  'pdf': np.concatenate([fits[l][6] for l in fits])}).to_csv(
        os.path.join(TAB, f'{BASE}_paneldata.csv'), index=False)
    pd.DataFrame({'p_weibull': p, 'x': x}).to_csv(
        os.path.join(TAB, f'{BASE}_sample.csv'), index=False)

    json.dump({
        'generated': datetime.datetime.now().isoformat(timespec='seconds'),
        'K': K, 'representative_replicate': rep, 'sample_size': SAMPLE_SIZE,
        'selection': 'FIXED illustrative replicate (FIXED_REP). Not rule-selected.',
        'metalog_mode_count_shares_pct': share,
        'w1_vs_sample_in_legend': {l: float(w1_sample[l]) for l in w1_sample},
        'n_replicates_both_feasible': len(both),
        'modes': {l: (int(fits[l][2]) if fits[l][2] is not None else None) for l in fits},
        'mixture': {'weights': [WEIGHT_1, WEIGHT_2], 'params': DIST_PARAMS,
                    'separation_factor': MODE_SEPARATION_FACTOR, 'shift': float(s)},
        'jpse_vs_scipy_quantile_maxdiff': dj,
    }, open(os.path.join(LOG, f'{BASE}_provenance.json'), 'w'), indent=2)


TITLE = False          # overall title off: the manuscript caption does that job
BASE = 'Figure09_bimodal'   # K = 12 is the paper's Figure 9; other K go to figures/other/bimodal_K<K>_*

if __name__ == '__main__':
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument('--title', action='store_true',
                     help='draw the overall title (for standalone use, not the paper)')
    _ap.add_argument('--K', type=int, default=12,
                     help='model order; 12 (the paper) writes Figure09_*, any other K writes bimodal_K<K>_*')
    _a = _ap.parse_args()
    TITLE = _a.title
    K = _a.K
    BASE = 'Figure09_bimodal' if K == 12 else f'bimodal_K{K}'
    main()
