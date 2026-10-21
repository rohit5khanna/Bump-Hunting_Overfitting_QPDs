"""
Figure 1 -- the three Johnson reference distributions used to test QPDs
(Section 4.1): unbounded (SU), semi-bounded (SL) and bounded (SB).

All three use the SAME parameters -- xi=0, lam=1, c=0.5, d=1.2 -- the values
the Monte Carlo samples from (mc_tables_rebuild.DIST_PARAMS). They are printed
by this script and recorded in its provenance.

Deterministic: densities and quantile functions are evaluated analytically, no
sampling.
"""
import os, sys, json, datetime, warnings
import numpy as np
import figure_style as FS, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib as mpl
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'code', 'common'))
from jpse import JohnsonSU, JohnsonSL, JohnsonSB

TAB = os.path.join(HERE, '..', 'tables'); FIG = os.path.join(HERE, '..', 'figures')
LOG = os.path.join(HERE, '..', 'logs')
for d in (TAB, FIG, LOG): os.makedirs(d, exist_ok=True)

DIST_PARAMS = dict(xi=0, lam=1, gamma=0.5, delta=1.2)
DISTS = [('Johnson SU', 'unbounded', JohnsonSU, FS.REFERENCE[0]),
         ('Johnson SL', 'semi-bounded', JohnsonSL, FS.REFERENCE[1]),
         ('Johnson SB', 'bounded', JohnsonSB, FS.REFERENCE[2])]
P = np.linspace(1e-4, 1 - 1e-4, 4000)
# Johnson SB only: the bounded panel shows the WHOLE support [0, 1]. The common
# grid above stops at p = 1e-4 / 1 - 1e-4 (x ~ 0.02 / 0.95 for these
# parameters), so on its own the density would start and end in mid-air.
# Log-spaced tails carry it to the bounds, where the SB density is 0. SU and SL
# use the common grid.
_T = np.geomspace(1e-14, 1e-4, 300, endpoint=False)
P_SB = np.concatenate([_T, P, (1 - _T)[::-1]])

mpl.rcParams.update({'font.family': 'sans-serif', 'font.size': 11,
                     'axes.titlesize': 12, 'axes.labelsize': 10,
                     'figure.dpi': 150, 'savefig.dpi': 300,
                     'axes.grid': True, 'grid.alpha': 0.3})


def main():
    print(f'parameters (all three): {DIST_PARAMS}')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    rows, meta = [], {}
    for ax, (name, support, cls, col) in zip(axes, DISTS):
        d = cls(**DIST_PARAMS)
        pg = P_SB if support == 'bounded' else P
        qf = np.asarray(d.quantile(pg), float)
        # jpse's pdf() is a function of x, NOT of p (passing the probability
        # grid returns densities at the wrong points). Evaluate it at the
        # quantiles, which pairs each density with its own x exactly (no
        # numerical differentiation).
        pdf = np.clip(np.asarray(d.pdf(qf), float), 0, None)
        if support == 'bounded':
            # jpse's SB pdf() returns 0 below x ~ 0.05 and above x ~ 0.9 (an
            # internal cut-off), which would draw the density dropping to zero
            # in mid-air. Use the closed form instead:
            #   f(x) = delta / (sqrt(2 pi) x (1 - x)) * phi(gamma + delta logit x)
            g_, d_ = DIST_PARAMS['gamma'], DIST_PARAMS['delta']
            u = (qf - DIST_PARAMS['xi']) / DIST_PARAMS['lam']
            pdf = (d_ / (np.sqrt(2 * np.pi) * DIST_PARAMS['lam'] * u * (1 - u))
                   * np.exp(-0.5 * (g_ + d_ * np.log(u / (1 - u))) ** 2))
        i = int(np.argmax(pdf))
        # No mode marker: the figure shows the three SHAPES; the mode location
        # and height are printed below and kept in the provenance record.
        ax.plot(qf, pdf, color=col, linewidth=1.8, zorder=3)
        ax.set_title(f'{name}  ({support})', fontweight='bold', pad=8)
        ax.set_xlabel('x'); ax.set_ylabel('Density')
        ax.set_ylim(bottom=0)
        if support == 'bounded':
            ax.set_xlim(0.0, 1.0)              # the full support
        else:
            lo, hi = float(np.percentile(qf, 0.2)), float(np.percentile(qf, 99.8))
            ax.set_xlim(lo, hi)
        ax.margins(y=0.06)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        print(f'  {name:<11} support {support:<13} mode {qf[i]:+.4f}  '
              f'height {pdf[i]:.4f}  range [{qf[0]:.3f}, {qf[-1]:.3f}]')
        rows += [{'distribution': name, 'p': pv, 'x': q, 'pdf': d_}
                 for pv, q, d_ in zip(pg, qf, pdf)]
        meta[name] = dict(support=support, mode_location=float(qf[i]),
                          mode_height=float(pdf[i]),
                          x_min=float(qf[0]), x_max=float(qf[-1]))
    fig.suptitle('Reference distributions used to test QPDs\n'
                 f'Johnson system, η={DIST_PARAMS["xi"]}, '
                 f'κ={DIST_PARAMS["lam"]}, c={DIST_PARAMS["gamma"]}, '
                 f'd={DIST_PARAMS["delta"]}', fontsize=13, fontweight='bold', y=1.04)
    fig.tight_layout(w_pad=2.2)
    out = FS.figure_path('Figure01_reference_distributions')
    fig.savefig(out + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(out + '.pdf', bbox_inches='tight'); plt.close(fig)
    print(f'Saved -> {out}.png')

    pd.DataFrame(rows).to_csv(os.path.join(TAB, 'Figure01_reference_distributions_paneldata.csv'),
                              index=False)
    json.dump({'generated': datetime.datetime.now().isoformat(timespec='seconds'),
               'paper_figure': 1, 'deterministic': True,
               'params_all_three': DIST_PARAMS,
               'jpse_api_note': 'jpse .pdf() takes x, not p; densities are '
                                'evaluated at quantile(p)',
               'note': "the parameter values the Monte Carlo used "
                       "(mc_tables_rebuild.DIST_PARAMS)",
               'distributions': meta},
              open(os.path.join(LOG, 'Figure01_provenance.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
