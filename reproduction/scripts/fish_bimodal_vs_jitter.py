"""
Fish -- bimodality persistence vs jitter width (not in the paper)
================================================================

Share of bootstrap fits of Log Metalog and Log QFlex-TA+ at K = 9, 10 and 11
that are bimodal, with uniform jitter width varying from 0 to 1.0 lb.
Written to figures/other/fish_bimodal_vs_jitter and
tables/fish_bimodal_vs_jitter.csv.

All six curves come from ONE sweep, on the section's canonical replicates
(`fish_common.resample`) -- the same draws behind Tables 9 and 10 at the
widths they share -- over the full 0 to 1.0 lb range in steps of 0.05.

CONVENTIONS
  * jitter j means Uniform(-j, +j); j = 0 is a plain bootstrap (no jitter).
  * replicate b at a given jitter is identical across all six model specs,
    so the curves differ only by model, never by draw.
  * modality is detected on P_GRID = the evaluation grid 0.001..0.999, the same
    grid used for validity -- consistent across the whole fish section.
  * "bimodal" means the mode detector returns n_modes >= 2.

Both denominators are written out: share of FEASIBLE fits and share of ALL
B resamples; the plot draws the share of ALL resamples (pct_bimodal_of_all).
"""
import os
# MUST precede the numpy import: BLAS reads these at load time. With them set
# afterwards each worker spawned one thread per core, so N workers fought over
# the same cores -- 4-way parallelism ran SLOWER than 2-way.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import sys, re, json, time, datetime, warnings, argparse
import multiprocessing as mp
import numpy as np
import figure_style as FS
import pandas as pd

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'code', 'common'))
sys.path.insert(0, HERE)

from fish_common import load_fish, resample, replicate_seed, P_GRID, N_BOOT, LOWER_BOUND
from metalog.metalog_v2 import LogMetalog
from qflex.transforms import LogQFlex
from qflex.constraints import ConstraintType
import eval_framework as EF

TAB = os.path.join(HERE, '..', 'tables')
FIG = os.path.join(HERE, '..', 'figures')
LOG = os.path.join(HERE, '..', 'logs')
for d in (TAB, FIG, LOG):
    os.makedirs(d, exist_ok=True)

JITTERS = [float(j) for j in np.round(np.arange(0.0, 1.0001, 0.05), 2)]
SPECS = [('Log Metalog', 'metalog', K) for K in (9, 10, 11)] + \
        [('Log QFlex-TA+', 'qflex_ta', K) for K in (9, 10, 11)]

_X = None
_P = None


def _init():
    global _X, _P
    _X, _P = load_fish()
    _P = np.arange(1, len(_X) + 1) / (len(_X) + 1)


def _fit(key, x_sorted, p, K):
    try:
        m = (LogMetalog(x_sorted, p, lower_bound=LOWER_BOUND, terms=K) if key == 'metalog'
             else LogQFlex(x_sorted, p, lower_bound=LOWER_BOUND, terms=K,
                           constraint_type=ConstraintType.TA))
        return m if EF.is_valid(m) else None
    except Exception:
        return None


def sweep_one_jitter(args):
    jitter, n_boot = args
    if _X is None:
        _init()
    counts = {(lab, K): {'feasible': 0, 'bimodal': 0, 'modes_sum': 0}
              for lab, _k, K in SPECS}
    for b in range(n_boot):
        s = np.sort(resample(_X, jitter, b))
        for lab, key, K in SPECS:
            m = _fit(key, s, _P, K)
            if m is None:
                continue
            try:
                qf = EF.quantile(m)
                n_modes, _locs, _h = EF.modes(m)
            except Exception:
                continue
            if n_modes is None or n_modes < 0:
                continue
            c = counts[(lab, K)]
            c['feasible'] += 1
            c['modes_sum'] += int(n_modes)
            if n_modes >= 2:
                c['bimodal'] += 1
    rows = []
    for lab, _key, K in SPECS:
        c = counts[(lab, K)]
        rows.append(dict(
            jitter=jitter, model=lab, K=K, n_bootstrap=n_boot,
            n_feasible=c['feasible'], n_bimodal=c['bimodal'],
            pct_bimodal_of_feasible=(100.0 * c['bimodal'] / c['feasible']
                                     if c['feasible'] else np.nan),
            pct_bimodal_of_all=100.0 * c['bimodal'] / n_boot,
            mean_n_modes=(c['modes_sum'] / c['feasible'] if c['feasible'] else np.nan),
        ))
    return rows


# NO TITLE BY DEFAULT, as for the paper figures, whose captions state what they
# show. `--title` restores it for standalone use.
TITLE = False


def plot(df, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 11,
        'axes.titlesize': 13, 'axes.labelsize': 11,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 9, 'legend.frameon': True, 'legend.framealpha': 0.85,
        'figure.dpi': 150, 'savefig.dpi': 300,
        'axes.grid': True, 'grid.alpha': 0.3, 'grid.color': '#cccccc',
        'lines.linewidth': 2.0,
    })
    METALOG_COLOR, QFLEX_COLOR = FS.QPD['Metalog'], FS.QPD['QFlex-TA+']
    style = {9: ('-', 'o'), 10: ('--', 's'), 11: (':', '^')}
    fig, ax = plt.subplots(figsize=(13.4, 6.5))   # width incl. the legend strip
    notes = []
    for lab, color in (('Log Metalog', METALOG_COLOR), ('Log QFlex-TA+', QFLEX_COLOR)):
        series = {K: df[(df['model'] == lab) & (df['K'] == K)].sort_values('jitter')
                  for K in (9, 10, 11)}
        # Two orders of a constrained family can be the SAME fit when the extra
        # coefficients are driven to zero -- for Log QFlex-TA+, K=11 returns the
        # K=10 coefficient vector with a trailing 0.0, so the two quantile
        # functions agree to the last bit. Plotting them as two thin lines hides
        # one completely, which reads as a missing series. Draw the coincident
        # higher order as a wide translucent band underneath and say so in its
        # legend entry.
        coincident = set()
        for Ka, Kb in ((10, 11), (9, 10), (9, 11)):
            if np.array_equal(series[Ka]['n_bimodal'].values,
                              series[Kb]['n_bimodal'].values):
                coincident.add((Ka, Kb))
                notes.append(f'{lab}: K={Ka} and K={Kb} are identical at every '
                             f'jitter (the TA+ constraint drives the extra '
                             f'coefficients to zero)')
        drawn_wide = {Kb: Ka for Ka, Kb in coincident}
        for K in (11, 10, 9):
            d = series[K]
            ls, mk = style[K]
            if K in drawn_wide:
                ax.plot(d['jitter'], d['pct_bimodal_of_all'], color=color,
                        linestyle='-', linewidth=6, alpha=0.30,
                        solid_capstyle='round', zorder=2,
                        label=f'{lab} K={K}\nK={K} fit identical to K={drawn_wide[K]}')
            else:
                ax.plot(d['jitter'], d['pct_bimodal_of_all'], color=color,
                        linestyle=ls, marker=mk, markersize=5, linewidth=2,
                        zorder=3, label=f'{lab} K={K}')
    for n in notes:
        print('  NOTE ' + n, flush=True)
    ax.set_xlabel('Jitter width h (lbs), Uniform(-h, +h)')
    ax.set_ylabel('% of bootstrap fits that are bimodal')
    if TITLE:
        ax.set_title('Bimodality Persistence vs. Jitter Width\n'
                     'Fish Biology, Log Metalog vs. Log QFlex-TA+ (K=9-11)',
                     fontweight='bold')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 100)
    # The legend sits OUTSIDE the axes, to the right. Six entries in a box large
    # enough to read covered the curves wherever it was placed inside the panel;
    # moving it out costs a little width and nothing else.
    h, l = ax.get_legend_handles_labels()
    # sort by model, then by K NUMERICALLY -- a string sort puts K=10 and K=11
    # ahead of K=9
    order = sorted(range(len(l)),
                   key=lambda i: (l[i].split(' K=')[0],
                                  int(re.search(r'K=(\d+)', l[i]).group(1))))
    ax.legend([h[i] for i in order], [l[i] for i in order],
              loc='upper left', bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0.0, fontsize=9, frameon=True)
    fig.tight_layout(rect=[0, 0, 0.73, 1])
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
    print(f'Saved -> {out_png}', flush=True)


PART_DIR = os.path.join(TAB, 'fish_bimodal_vs_jitter_partials')   # this sweep's partials only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    ap.add_argument('--procs', type=int, default=3)
    ap.add_argument('--title', action='store_true',
                    help='draw the in-figure title (off by default)')
    ap.add_argument('--jitters', type=str, default='',
                    help='comma-separated subset to run, e.g. "0.0,0.05,0.1,0.15". '
                         'Results are written to tables/fish_bimodal_vs_jitter_partials/ and combined '
                         'by --merge. The sweep is chunked because a shell on the '
                         'user\'s machine is capped at 180 s per call.')
    ap.add_argument('--merge', action='store_true',
                    help='combine the partials into the final table, figure and '
                         'provenance record; runs no fits')
    a = ap.parse_args()
    global TITLE
    TITLE = a.title
    os.makedirs(PART_DIR, exist_ok=True)

    t0 = time.time()
    if a.merge:
        parts = sorted(f for f in os.listdir(PART_DIR) if f.endswith('.csv'))
        if not parts:
            raise SystemExit('no partials to merge')
        df = pd.concat([pd.read_csv(os.path.join(PART_DIR, f)) for f in parts],
                       ignore_index=True)
        df = df.drop_duplicates(subset=['jitter', 'model', 'K'], keep='last')
        missing = sorted(set(JITTERS) - set(df['jitter'].round(2)))
        if missing:
            raise SystemExit(f'incomplete: missing jitters {missing}')
        finish(df, a, t0)
        return

    js = ([round(float(v), 2) for v in a.jitters.split(',')] if a.jitters else JITTERS)
    unknown = [j for j in js if j not in JITTERS]
    if unknown:
        raise SystemExit(f'jitters not on the grid: {unknown}')
    print(f'Bimodality-vs-jitter sweep: {len(JITTERS)} jitters x {a.n_boot} resamples x '
          f'{len(SPECS)} specs = {len(JITTERS)*a.n_boot*len(SPECS):,} fits; '
          f'this call runs {js}', flush=True)
    tasks = [(j, a.n_boot) for j in js]
    rows = []
    with mp.Pool(min(a.procs, len(tasks)), initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(sweep_one_jitter, tasks), 1):
            rows.extend(r)
            print(f'  [{i}/{len(tasks)}] {time.time()-t0:7.1f}s  '
                  f'jitter {r[0]["jitter"]:.2f} done', flush=True)
    part = pd.DataFrame(rows).sort_values(['model', 'K', 'jitter'])
    tag = '_'.join(f'{j:.2f}' for j in js)
    part.to_csv(os.path.join(PART_DIR, f'part_{tag}.csv'), index=False)
    print(f'partial saved: {len(part)} rows for jitters {js}', flush=True)

    have = set()
    for f in os.listdir(PART_DIR):
        if f.endswith('.csv'):
            have |= set(pd.read_csv(os.path.join(PART_DIR, f))['jitter'].round(2))
    left = sorted(set(JITTERS) - have)
    if left:
        print(f'still to run ({len(left)}): {left}', flush=True)
        print(f'\nTotal {time.time()-t0:.1f}s', flush=True)
        return
    df = pd.concat([pd.read_csv(os.path.join(PART_DIR, f))
                    for f in sorted(os.listdir(PART_DIR)) if f.endswith('.csv')],
                   ignore_index=True)
    df = df.drop_duplicates(subset=['jitter', 'model', 'K'], keep='last')
    finish(df, a, t0)


def finish(df, a, t0):
    df = df.sort_values(['model', 'K', 'jitter']).reset_index(drop=True)
    csv = os.path.join(TAB, 'fish_bimodal_vs_jitter.csv')
    df.to_csv(csv, index=False)
    print(f'\nSaved -> {csv}', flush=True)

    print('\n% bimodal (of all 1000 resamples)')
    print(df.pivot_table(index='jitter', columns=['model', 'K'],
                         values='pct_bimodal_of_all').round(1).to_string())
    print('\nfeasible fits (of 1000)')
    print(df.pivot_table(index='jitter', columns=['model', 'K'],
                         values='n_feasible').to_string())

    plot(df, FS.figure_path('fish_bimodal_vs_jitter') + '.png')

    with open(os.path.join(LOG, 'fish_bimodal_vs_jitter_provenance.json'), 'w') as f:
        json.dump({
            'generated': datetime.datetime.now().isoformat(timespec='seconds'),
            'paper_figure': 'not in the paper',
            'jitters': JITTERS,
            'n_bootstrap': a.n_boot,
            'specs': [f'{lab} K={K}' for lab, _k, K in SPECS],
            'resamples': 'fish_common.resample -- shared with Tables 9 and 10',
            'mode_grid': 'eval_framework.GRID = 0.001..0.999',
            'plotted_series': 'pct_bimodal_of_all',
            'runtime_sec': round(time.time() - t0, 1),
        }, f, indent=2)
    print(f'\nTotal {time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
