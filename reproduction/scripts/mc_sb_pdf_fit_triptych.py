"""
Figure 2 -- Johnson SB, MC replicate #99 at N = 200: Logit Metalog and Logit
QFlex-A+ density fits at K = 4, 7, 10 (three panels).

The sample and the fits are produced from mc_tables_rebuild's OWN draw() and
the same Logit-on-[0,1] construction fit() uses, so this figure shows exactly
the sample and the fitted objects that back the Monte Carlo tables -- same
seed (mc_seed(200, 99) = 200141), same transform, same feasibility rule.

SAMPLE: replicate 99, chosen for illustration; it is not rule-selected.

Each panel is one model order; colour carries the MODEL (Metalog blue,
QFlex-A+ green, from the paper palette) and every curve is drawn SOLID -- the
order is already named in the panel title, so no line-style coding is used.
The legend reports each fit's W1 distance (Equation 6, to the sample):

    W1 = (1/N) sum_i |Q_F(p_i) - x_(i)|,  p_i = i/(N+1)  (the sample points;
         no interpolation), divided by Q_hat(0.9) - Q_hat(0.1)

the same per-sample quantity whose median across replicates is the paper's
MedianW1_Eq6 column.

EVALUATION FRAMEWORK (code/common/eval_framework.py): every curve is drawn on
p = 0.001..0.999 only, density from differences of the quantile; validity = Q
increasing on that grid; W1 = Equation 6 at the sample points. An invalid fit is
still drawn and marked "invalid" in the legend; where its quantile decreases
the difference density is negative and that stretch is left blank. The y-axis
is FIXED at 3.5 (YMAX below). The script warns if any histogram bar or curve
on p in [0.01, 0.99] would exceed it.

Two versions are written: one WITH the sample histogram (the data being fit;
Figure 2, figures/Figure02_SB_example_fits), one WITHOUT (curves only; not in
the paper, figures/other/). The true SB population density is not drawn.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', 'code', 'common'))

from mc_tables_rebuild import DIST_PARAMS, P_GRID, mc_seed, draw, DISTS
from metalog.metalog_v2 import LogitMetalog
from qflex.transforms import LogitQFlex
from qflex.constraints import ConstraintType
import figure_style as FS
import eval_framework as EF

N = 200
REP = 99
YMAX = 3.5          # fixed density-axis cap for the figure
KS = [4, 7, 10]
# (palette key, display name, builder) -- builder returns the fitted object
# WITHOUT the feasibility gate, so an infeasible fit can still be shown.
MODELS = [
    ('Metalog', 'Logit Metalog',
     lambda x, p, K: LogitMetalog(x, p, lower_bound=0, upper_bound=1, terms=K)),
    ('QFlex-A+', 'Logit QFlex-A+',
     lambda x, p, K: LogitQFlex(x, p, lower_bound=0, upper_bound=1, terms=K,
                                constraint_type=ConstraintType.A)),
]
FIG_DIR = os.path.join(HERE, '..', 'figures')
TAB_DIR = os.path.join(HERE, '..', 'tables')
STEM = 'Figure02_SB_example_fits'


def compute():
    """Fit both models at every K once; return sample + per-fit curves/W1."""
    name, cls, support = DISTS['sb']
    d = cls(**DIST_PARAMS)
    x = draw(d, N, REP)                       # seed mc_seed(200, 99) = 200141
    p_pos = np.arange(1, x.size + 1) / (x.size + 1)
    idr = EF.interdecile(x)
    fits = {}
    rows = []
    for key, disp, build in MODELS:
        for K in KS:
            try:
                m = build(x, p_pos, K)
                feas = EF.is_valid(m)
                if feas:
                    qf, pdf = EF.density(m)
                else:
                    qf = EF.quantile(m); pdf = EF.density_from_quantile(qf)
                w1 = EF.w1(m, x)[1]
                qd, pdd = qf, np.where(pdf > 0, pdf, np.nan)
                print(f'  {disp:15s} K={K:2d}  valid={feas}  drawn x [{qd[0]:.3g}, {qd[-1]:.6g}]',
                      flush=True)
            except Exception:
                qf = pdf = qd = pdd = None; w1 = np.nan; feas = False
            fits[(key, K)] = dict(disp=disp, qf=qf, pdf=pdf, qd=qd, pdd=pdd,
                                  w1=w1, feasible=feas)
            rows.append(dict(K=K, model=disp, feasible=feas,
                             W1=round(w1, 4) if np.isfinite(w1) else np.nan))
    return x, fits, pd.DataFrame(rows), idr


def make_figure(x, fits, show_hist):
    plt.rcParams.update(FS.rcparams())
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), sharey=True)
    # headroom: find the tallest thing drawn, then leave a clear band on top
    # for the legend so it never sits over a curve or the histogram.
    win = (EF.GRID >= 0.01) & (EF.GRID <= 0.99)
    curve_max = max(np.nanmax(np.where(f['pdf'][win] > 0, f['pdf'][win], np.nan))
                    for f in fits.values() if f['pdf'] is not None)
    data_max = curve_max
    if show_hist:
        h, _ = np.histogram(x, bins=24, range=(0, 1), density=True)
        data_max = max(data_max, float(h.max()))
    ymax = YMAX
    if data_max > YMAX:
        print(f'  WARNING: tallest drawn element {data_max:.3f} exceeds YMAX = {YMAX}', flush=True)
    for ax, K in zip(axes, KS):
        if show_hist:
            ax.hist(x, bins=24, density=True, range=(0, 1),
                    color=FS.OTHER['data'], alpha=0.45, edgecolor='white',
                    linewidth=0.3, zorder=1, label='Sample (N=200)')
        for key, disp, _ in MODELS:
            f = fits[(key, K)]
            if f['qf'] is None:
                continue
            tag = '' if f['feasible'] else ', invalid'
            ax.plot(f['qd'], f['pdd'], color=FS.color(key), lw=2.1,
                    ls=('--' if key.startswith('QFlex') else '-'),   # dashed QFlex: readable in B&W print
                    zorder=3, label=f"{disp}  (W1 = {100 * f['w1']:.2f}%{tag})")
        ax.set_title(f'K = {K}', fontweight='bold')
        ax.set_xlabel('x')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, ymax)
        ax.legend(loc='upper right', fontsize=8.5, framealpha=0.95,
                  handlelength=1.6, borderpad=0.5, labelspacing=0.35)
    axes[0].set_ylabel('Density')
    fig.suptitle(f'Johnson SB  |  MC replicate #{REP}, N = 200  |  Logit Metalog vs '
                 'Logit QFlex-A+ density fits', y=1.02, fontsize=12,
                 fontweight='bold')
    fig.tight_layout()
    suffix = 'hist' if show_hist else 'nohist'
    path = FS.figure_path(STEM if show_hist else f'sb_example_fits_rep{REP}_nohist')
    fig.savefig(path + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(path + '.pdf', bbox_inches='tight')
    plt.close(fig)
    return path + '.png'


def main():
    x, fits, table, idr = compute()
    os.makedirs(TAB_DIR, exist_ok=True)
    table.to_csv(os.path.join(TAB_DIR, f'{STEM}_W1.csv'), index=False)
    p1 = make_figure(x, fits, show_hist=True)
    p2 = make_figure(x, fits, show_hist=False)
    print(f'seed={mc_seed(N, REP)}  N={x.size}  idr={idr:.4f}')
    print(table.to_string(index=False))
    print('wrote:', os.path.basename(p1), '|', os.path.basename(p2))


if __name__ == '__main__':
    main()
