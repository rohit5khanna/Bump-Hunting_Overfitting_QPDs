"""Geyser -- paper Figure 19: Log Metalog (K = 8) and Log QFlex-TA+ (K = 10)
fitted to a LATENT (jittered) realization of the recorded waiting times.

With --raw it instead fits the RECORDED values with no jitter and writes
geyser_raw_fit (figures/other/; not in the paper).

Rules: no bootstrap anywhere, W1 in the quantile legend, histogram behind the
density, curves drawn on the evaluation grid 0.001..0.999 only
(eval_framework).

WHY THESE ORDERS.  They are the lowest order at which each family is feasible
AND bimodal on the data itself. On the full sample -- jittered or raw -- Log
Metalog is INFEASIBLE at K = 4, 5, 6 and 7 and first becomes feasible, with
two modes, at K = 8. Log QFlex-TA+ is feasible from K = 4 but unimodal until
K = 10. Metalog K = 7, which Table 13 reports, exists only on bootstrap
resamples (about 40% of them, matching Table 13's 36.1% feasibility); it never
occurs on the recorded data.

WHY ONE REALIZATION IS ENOUGH.  geyser_latent_stability.py refits over N_BOOT
independent jitter draws of the full sample. Both specs are valid in 1000/1000
and bimodal in 100% of them, with W1 IQRs of ~0.0004, so the drawn realization
is typical rather than lucky.

HISTOGRAM.  Bin width = the full jitter window 2j = 1.0 min, the same rule the
fish figures use (--raw: the 1-min recording step). Waiting times are
recorded as integers, so with edges on the half-integers a +-0.5 min jitter
can never move a value out of its own bin: the jittered and raw histograms
are identical here (verified over 50 draws).

W1 is Equation 6 at the sample points of the realization being fitted.
"""
import os, sys, json, datetime, warnings, argparse
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
import figure_style as FS, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib as mpl
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'code', 'common')); sys.path.insert(0, HERE)
from geyser_common import (load_geyser, jitter_original, JITTER, N_BOOT, P_GRID,
                           LOWER_BOUND, empirical_qf, interdecile, w1)
from metalog.metalog_v2 import LogMetalog
from qflex.transforms import LogQFlex
from qflex.constraints import ConstraintType
import eval_framework as EF

TAB = os.path.join(HERE, '..', 'tables'); FIG = os.path.join(HERE, '..', 'figures')
LOG = os.path.join(HERE, '..', 'logs')
for _d in (TAB, FIG, LOG): os.makedirs(_d, exist_ok=True)

mpl.rcParams.update({'font.family': 'sans-serif', 'font.size': 11,
                     'axes.titlesize': 13, 'axes.labelsize': 12,
                     'figure.dpi': 150, 'savefig.dpi': 300,
                     'axes.grid': True, 'grid.alpha': 0.3})
COLORS = {'Log Metalog': FS.QPD['Metalog'], 'Log QFlex-TA+': FS.QPD['QFlex-TA+']}
SPECS = (('Log Metalog', 8), ('Log QFlex-TA+', 10))
DRAW = 0                 # the realization drawn, seeded in geyser_common
N_ENVELOPE = N_BOOT      # the section's declared replicate count, not a literal
HIST_BINWIDTH = 2 * JITTER         # the full jitter window
RAW = False                        # set by --raw


def recording_step(x):
    """Smallest gap between distinct recorded values -- 1 min here. Used as the
    bin width in --raw mode, where 2j is undefined and each bin then holds
    exactly one value the clock could report."""
    return float(np.min(np.diff(np.unique(np.asarray(x, float)))))
WAIT_LIM = (35.0, 115.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', action='store_true',
                    help='fit the RECORDED values with no jitter (no envelope band; '
                         'histogram binned at the recording step)')
    a = ap.parse_args()
    global RAW
    RAW = a.raw
    x, _ = load_geyser()
    N = len(x); pp = np.arange(1, N + 1) / (N + 1)
    PLOT_GRID = FS.plot_grid(P_GRID, N)
    print(f'N = {N} recorded waiting times, {len(np.unique(x))} distinct integer values')

    xs = (np.sort(np.asarray(x, float)) if RAW
          else np.sort(jitter_original(x, JITTER, draw=DRAW)))
    eqf = empirical_qf(xs, pp); L = interdecile(xs, pp)

    if RAW:
        lo = hi = None
    else:
        env = np.empty((N_ENVELOPE, len(P_GRID)))
        for r in range(N_ENVELOPE):
            env[r] = np.interp(P_GRID, pp, np.sort(jitter_original(x, JITTER, draw=r)))
        lo, hi = np.percentile(env, 2.5, axis=0), np.percentile(env, 97.5, axis=0)

    fig, (ax_q, ax_d) = plt.subplots(1, 2, figsize=(14, 6.2))
    fig.suptitle('Log Metalog (K = 8) and Log QFlex-TA+ (K = 10) fitted to the '
                 + ('RECORDED geyser waiting times\n(no jitter)' if RAW else
                    f'latent geyser waiting times\n(uniform jitter width = ±{JITTER} min)'),
                 fontsize=14, fontweight='bold')
    if not RAW:
        ax_q.fill_between(P_GRID, lo, hi, color=FS.OTHER['band'], alpha=FS.BAND_ALPHA,
                          lw=0, zorder=1,
                          label=f'95% jitter envelope ({N_ENVELOPE} realizations)')
    ax_q.plot(P_GRID, eqf, color=FS.OTHER['empirical'], linewidth=1.3, zorder=3,
              label='EQF of the recorded times' if RAW
                    else 'EQF of the drawn realization')

    bw = recording_step(x) if RAW else HIST_BINWIDTH
    edges = np.arange(np.floor(xs.min()) - 0.5, xs.max() + 1.5, bw)
    ax_d.hist(xs, bins=edges, density=True, color=FS.OTHER['band'], alpha=0.75,
              edgecolor='white', linewidth=0.3, zorder=1,
              label=f'Histogram ({bw:g} min bins)')

    rows, prov = [], {}
    for lab, K in SPECS:
        m = (LogMetalog(xs, pp, lower_bound=LOWER_BOUND, terms=K) if lab == 'Log Metalog'
             else LogQFlex(xs, pp, lower_bound=LOWER_BOUND, terms=K,
                           constraint_type=ConstraintType.TA))
        assert EF.is_valid(m), f'{lab} K={K} infeasible'
        qf = EF.quantile(m)
        nm, locs, hts = EF.modes(m)
        _raw, wn = w1(m, xs)        # Eq 6 at the realization's own sample points
        # both panels: the evaluation grid only (figure_style.density_curve =
        # eval_framework), Q and the grid-difference density on 0.001..0.999.
        p_d, qf_d, pdf_d = FS.density_curve(m.quantile, x_view=WAIT_LIM,
                                            support=(LOWER_BOUND, np.inf))
        ax_q.plot(p_d, qf_d, color=COLORS[lab], linewidth=2, zorder=4,
                  label=f'{lab} K={K}  (W₁ = {100 * wn:.2f}%)')
        ax_d.plot(qf_d, pdf_d, color=COLORS[lab], linewidth=2, zorder=4,
                  label=f'{lab} K={K}')
        print(f'  {lab:14s} K={K:2d}  W1={wn:.4f}  modes={nm}  '
              f'locs={np.round(locs,2) if locs is not None else None}  '
              f'density from {qf_d[0]:.2f} min')
        prov[f'{lab}_K{K}'] = {'W1_Eq6': round(float(wn), 6),
                               'n_modes': int(nm) if nm is not None else -1,
                               'mode_locations_min': [round(float(v), 3) for v in (locs if locs is not None else [])],
                               'mode_heights': [round(float(v), 6) for v in (hts if hts is not None else [])]}
        rows += [{'model': lab, 'K': K, 'p': p_, 'waiting_min': q, 'pdf': d}
                 for p_, q, d in zip(p_d, qf_d, pdf_d)]

    ax_q.set_title('Quantile Function', fontweight='bold'); ax_q.set_xlabel('p')
    ax_q.set_ylabel('Waiting time (min)')
    ax_q.set_xlim(0, 1); ax_q.set_ylim(*WAIT_LIM)
    ax_q.legend(loc='upper left', fontsize=9)
    ax_d.set_title('Probability Density Function', fontweight='bold')
    ax_d.set_xlabel('Waiting time (min)'); ax_d.set_ylabel('Density')
    ax_d.set_xlim(*WAIT_LIM); ax_d.set_ylim(bottom=0)
    ax_d.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    out = FS.figure_path('geyser_raw_fit' if RAW else 'Figure19_geyser_fits')
    fig.savefig(out + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(out + '.pdf', bbox_inches='tight'); plt.close(fig)
    print(f'saved {os.path.basename(out)}.png', flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(TAB, ('geyser_raw_fit_paneldata.csv' if RAW
                          else 'Figure19_geyser_fits_paneldata.csv')),
                              index=False)
    if not RAW:
        pd.DataFrame({'p': P_GRID, 'eqf_min': eqf,
                      'env_q025': lo, 'env_q975': hi}).to_csv(
            os.path.join(TAB, 'Figure19_geyser_latent_envelope.csv'), index=False)
    json.dump({'generated': datetime.datetime.now().isoformat(timespec='seconds'),
               'fit_to': f'one jitter realization (draw={DRAW}) of all {N} recorded waiting times',
               'jitter_half_width_min': JITTER,
               'bootstrap': 'none -- no resampling anywhere in this figure',
               'orders': {lab: K for lab, K in SPECS},
               'why_K8_not_K7': 'Log Metalog is infeasible at K=4..7 on the full sample '
                                '(0/1000 jitter realizations); K=7 occurs only on resamples',
               'w1': "Equation 6 at the drawn realization's own sample points (no interpolation)",
               'band': f'pointwise 2.5-97.5% of the EQF over {N_ENVELOPE} jitter realizations',
               'histogram': f'bin width = 2j = {HIST_BINWIDTH:g} min, edges on the half-integers',
               'stability': 'tables/geyser_latent_jitter_stability.csv',
               'fits': prov},
              open(os.path.join(LOG, 'geyser_raw_fit_provenance.json' if RAW else 'Figure19_geyser_provenance.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
