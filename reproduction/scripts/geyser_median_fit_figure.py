"""
Geyser -- the W1 median bootstrap fit for Log Metalog (K = 7, 8) and
Log QFlex-TA+ (K = 10). Not in the paper; written to
figures/other/geyser_median_bootstrap_fit.

PROCEDURE:

  1. fixed probability grid            (geyser_common.P_GRID)
  2. B bootstrap resamples             (geyser_common.resample, 0.5-min jitter)
  3. median quantile at each p across the B SAMPLES' EQFs -- for W1, at the
     plotting positions p_i: the median of the B resamples' i-th order
     statistics (no interpolation)
  4. W1* = that median QF's Equation 6 distance from the original sample, at
     its sample points (eval_framework.w1_points)
  5. fit the QPD to each resample      (already done: tables/geyser_sweep.csv)
  6. pick the valid fit whose W1 is CLOSEST to W1*

Step 6 matches a SCALAR, not curve shape -- it selects a typical distance, not
a typical curve (readers tend to assume a curve median).

The W1 values come straight from the sweep, so the fit chosen here is the same
fit Table 13 counted.
"""
import os, sys, json, datetime, warnings
import numpy as np
import figure_style as FS, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib as mpl
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'code', 'common')); sys.path.insert(0, HERE)
from geyser_common import (load_geyser, resample, P_GRID, N_BOOT, JITTER,
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

SPECS = [('Log Metalog', 'metalog', 7), ('Log Metalog', 'metalog', 8),
         ('Log QFlex-TA+', 'qflex_ta', 10)]
# One colour per MODEL; order is carried by line style (FS.KSTYLE).
COLORS = {('Log Metalog', 7): FS.QPD['Metalog'], ('Log Metalog', 8): FS.QPD['Metalog'],
          ('Log QFlex-TA+', 10): FS.QPD['QFlex-TA+']}
STYLES = {7: '--', 8: ':', 10: '-'}

mpl.rcParams.update({'font.family': 'sans-serif', 'font.size': 11,
                     'axes.titlesize': 13, 'axes.labelsize': 11,
                     'figure.dpi': 150, 'savefig.dpi': 300,
                     'axes.grid': True, 'grid.alpha': 0.3})


def fit(key, s, p, K):
    try:
        m = (LogMetalog(s, p, lower_bound=LOWER_BOUND, terms=K) if key == 'metalog'
             else LogQFlex(s, p, lower_bound=LOWER_BOUND, terms=K,
                           constraint_type=ConstraintType.TA))
        return m if EF.is_valid(m) else None
    except Exception:
        return None


def main():
    x, p = load_geyser()
    N = len(x); p_pos = np.arange(1, N + 1) / (N + 1)
    eqf = empirical_qf(x, p); L = interdecile(x, p)
    sweep = pd.read_csv(os.path.join(TAB, 'geyser_sweep.csv'))

    # steps 2-4
    M = np.empty((N_BOOT, len(P_GRID)))          # drawing only (band on the grid)
    S = np.empty((N_BOOT, N))                     # resample order statistics
    for b in range(N_BOOT):
        S[b] = np.sort(resample(x, JITTER, b))
        M[b] = np.interp(P_GRID, p_pos, S[b])
    median_qf = np.median(M, axis=0)
    lo, hi = np.percentile(M, 2.5, axis=0), np.percentile(M, 97.5, axis=0)
    # W1* at the sample points: median i-th order statistic vs x_(i)
    _raw, W1_STAR = w1_points(np.median(S, axis=0), x)
    print(f'W1* (median bootstrap QF vs original EQF, Eq 6) = {W1_STAR:.6f}')

    rows, curves, draws = [], {}, {}
    for lab, key, K in SPECS:
        g = sweep[(sweep['model'] == lab) & (sweep['K'] == K) &
                  sweep['feasible'] & sweep['w1_orig_eq6'].notna()]
        # step 6 compares with W1*, a distance from the ORIGINAL sample, so it
        # uses each fit's W1 vs the original (w1_orig_eq6) whatever
        # EF.BOOTSTRAP_W1_REFERENCE says
        j = (g['w1_orig_eq6'] - W1_STAR).abs().idxmin()     # step 6
        b = int(g.loc[j, 'b'])
        s = np.sort(resample(x, JITTER, b))
        m = fit(key, s, p_pos, K)
        qf, pdf = EF.density(m)
        nm, locs, hgts = EF.modes(m)
        _r, wn = w1(m, x)                                   # vs the original sample
        curves[(lab, K)] = (qf, pdf)
        # both panels draw the evaluation grid only (figure_style.density_curve
        # = eval_framework): Q and the grid-difference density on 0.001..0.999.
        draws[(lab, K)] = FS.density_curve(m.quantile, x_view=(35.0, 115.0),
                                           support=(LOWER_BOUND, np.inf))
        order = np.argsort(locs)[::-1] if locs is not None and len(locs) >= 2 else None
        rows.append(dict(model=lab, K=K, n_valid=len(g), W1_star_Eq6=round(W1_STAR, 6),
                         selected_replicate=b, selected_W1_Eq6=round(wn, 6),
                         W1_gap=round(abs(wn - W1_STAR), 6), n_modes=int(nm),
                         mode_locations=';'.join(f'{v:.3f}' for v in (locs if locs is not None else [])),
                         mode_heights=';'.join(f'{v:.5f}' for v in (hgts if hgts is not None else [])),
                         rank_flipped=(bool(hgts[order[0]] < hgts[order[1]])
                                       if order is not None and nm == 2 else None)))
        print(f'  {lab} K={K}: {len(g)} valid, picked b={b}, W1={wn:.6f}, '
              f'modes={nm}, locs={np.round(locs,2) if locs is not None else None}')

    sel = pd.DataFrame(rows)
    sel.to_csv(os.path.join(TAB, 'geyser_median_fit_selection.csv'), index=False)

    fig, (ax_q, ax_p) = plt.subplots(1, 2, figsize=(14, 6.2))
    fig.suptitle('Median-W1 Bootstrap Fits — Old Faithful Waiting Times '
                 f'(jitter width = ±{JITTER} min)', fontsize=15, fontweight='bold')
    ax_q.fill_between(P_GRID, lo, hi, color=FS.OTHER['band'], alpha=FS.BAND_ALPHA, lw=0, zorder=1,
                      label='95% bootstrap interval')
    ax_q.plot(P_GRID, eqf, color='k', linewidth=1.3, zorder=3,
              label='Empirical quantile (recorded data)')
    panel = []
    for lab, _key, K in SPECS:
        qf, pdf = curves[(lab, K)]
        c, ls = COLORS[(lab, K)], STYLES[K]
        p_d, q_d, f_d = draws[(lab, K)]
        ax_q.plot(p_d, q_d, color=c, linewidth=2, linestyle=ls, zorder=4,
                  label=f'{lab} (K={K})')
        ax_p.plot(q_d, f_d, color=c, linewidth=2, linestyle=ls, zorder=4,
                  label=f'{lab} (K={K})')
        for pp, q, pv in zip(P_GRID, qf, pdf):
            panel.append({'model': lab, 'K': K, 'p': pp, 'qf_min': q, 'pdf': pv})
    ax_q.set_title('Quantile Function'); ax_q.set_xlabel('p')
    ax_q.set_ylabel('Waiting time (min)'); ax_q.set_xlim(0, 1); ax_q.set_ylim(35, 115)
    ax_q.legend(loc='upper left', fontsize=8.5)
    ax_p.set_title('Probability Density Function')
    ax_p.set_xlabel('Waiting time (min)'); ax_p.set_ylabel('Density')
    ax_p.set_xlim(35, 115)

    # The selection rule can land on a fit with one very narrow, very tall
    # spike (step 6 matches a SCALAR W1, which is blind to shape). Letting that
    # spike set the y-range flattens every other feature into the axis. Cap the
    # view at 1.25x the MEDIAN curve maximum and label whatever is clipped, so
    # the spike is still reported -- its full height is in the panel-data CSV.
    peaks = [float(np.max(pdf)) for _lab, _k, _K in SPECS
             for pdf in [curves[(_lab, _K)][1]]]
    cap = 1.25 * float(np.median(peaks))
    clipped = []
    for lab, _key, K in SPECS:
        qf, pdf = curves[(lab, K)]
        m_ = float(np.max(pdf))
        if m_ > cap:
            loc_ = float(qf[int(np.argmax(pdf))])
            clipped.append((lab, K, loc_, m_))
    if clipped:
        ax_p.set_ylim(0, cap)
        for lab, K, loc_, m_ in clipped:
            ax_p.annotate(f'{lab} K={K} peaks at {m_:.3f}\n(clipped; {loc_:.1f} min)',
                          xy=(loc_, cap), xytext=(loc_ + 9, cap * 0.93),
                          fontsize=8, color=COLORS[(lab, K)],
                          arrowprops=dict(arrowstyle='->', color=COLORS[(lab, K)],
                                          linewidth=1.1))
            print(f'  NOTE {lab} K={K} density peaks at {m_:.4f} ({loc_:.1f} min); '
                  f'PDF panel clipped at {cap:.4f}', flush=True)
    else:
        ax_p.set_ylim(bottom=0)
    ax_p.legend(loc='upper right', fontsize=8.5)
    fig.tight_layout()
    out = FS.figure_path('geyser_median_bootstrap_fit')
    fig.savefig(out + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(out + '.pdf', bbox_inches='tight'); plt.close(fig)
    print(f'Saved -> {out}.png')

    pd.DataFrame(panel).to_csv(os.path.join(TAB, 'geyser_median_bootstrap_fit_paneldata.csv'), index=False)
    pd.DataFrame({'p': P_GRID, 'median_bootstrap_qf': median_qf,
                  'eqf_original': eqf, 'q025': lo, 'q975': hi}).to_csv(
        os.path.join(TAB, 'geyser_median_bootstrap_fit_eqf_band.csv'), index=False)
    json.dump({
        'generated': datetime.datetime.now().isoformat(timespec='seconds'),
        'data_file': data_fingerprint()[0],
        'data_N': data_fingerprint()[1],
        'data_sha256': data_fingerprint()[2],
        'paper_figure': 'not in the paper',
        'procedure': "steps 1-6 of the script docstring; step 6 matches the SCALAR W1",
        'W1_star_Eq6': W1_STAR,
        'w1': 'Equation 6 at the ORIGINAL sample\'s points (eval_framework.w1), no interpolation; '
              'W1* = median resample order statistic vs x_(i); step 6 uses w1_orig_eq6',
        'source': 'tables/geyser_sweep.csv -- same fits Table 13 counted',
        'jitter_half_width_min': JITTER, 'n_bootstrap': N_BOOT,
        'pdf_panel_clipped': [{'model': l, 'K': int(k), 'peak_density': pk,
                               'peak_location_min': lo}
                              for l, k, lo, pk in clipped],
        'note': 'step 6 matches a scalar W1 and is blind to shape; on this data '
                'the nearest bimodal fit differs from the selected one by ~1e-6 in W1',
    }, open(os.path.join(LOG, 'geyser_median_bootstrap_fit_provenance.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
