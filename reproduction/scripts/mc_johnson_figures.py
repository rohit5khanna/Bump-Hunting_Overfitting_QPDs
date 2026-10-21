"""
Johnson Monte Carlo figures (paper Figures 3, 4 and O1-O4), from the per-
replicate output of mc_tables_rebuild.py (tables/mc_perrep_<dist>_n<N>.csv),
i.e. on the paper's evaluation framework (code/common/eval_framework.py):
validity = Q increasing on the 0.001..0.999 grid; modes = interior peaks of the
grid-difference density (relative prominence >= 1% of the peak's own height); W1 = Equation 6 at
each replicate's own sample points (no interpolation).

The paper's figures include QFlex-T+ and are written by the QFlex-T+ run
(QPD_TPLUS_ONLY=1, run_tplus_supplement.sh); see the QPD_TPLUS_ONLY note below.

  Row 1: Validity Rate    | Median W1 Distance      | False Multimodality Rate
  Row 2: Mean Mode Count  | Mode Location Median    | Mode Height Median

Mode count of a fit = max(1, interior peaks). Mode location / height = the
tallest interior peak, over valid fits that have one. Orders K = 3..10.

Truth lines: mode location/height for xi=0, lam=1, c=0.5, d=1.2, obtained
by numerically maximising the Johnson density.

Y-axis ranges are shared across the N=50 / N=200 pair for each distribution so
the two figures stay visually comparable.

Usage:
    python3 mc_johnson_figures.py johnson_su 200
    python3 mc_johnson_figures.py --all          # all 3 dists x N in {50,200}
"""
import os, sys, argparse, json, datetime
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib as mpl
from scipy.optimize import minimize_scalar

HERE     = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(HERE, '..', 'figures')
TAB_DIR  = os.path.join(HERE, '..', 'tables')
LOG_DIR  = os.path.join(HERE, '..', 'logs')

DIST_LABELS = {'johnson_su': 'Johnson SU', 'johnson_sl': 'Johnson SL',
               'johnson_sb': 'Johnson SB'}
PAIR_NS = [50, 200]
MODEL_ORDER = ['Metalog', 'QFlex-U', 'QFlex-TA+', 'QFlex-A+']
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import QPD as _QPD
import figure_style as FS
COLORS  = dict(_QPD)
MARKERS = {'Metalog': 'o', 'QFlex-U': 's', 'QFlex-TA+': 'D', 'QFlex-A+': '^'}

# QPD_TPLUS_ONLY=1: the paper's Figures 3, 4 and O1-O4, which show all five
# models. Reads the four-model per-replicate files plus tables/tplus/; the
# figures go to figures/, their panel data and provenance to tables/tplus/ and
# logs/tplus/. Without it, the four-model versions (no QFlex-T+; not in the
# paper) are written to figures/other/.
import tplus_mode as TP
SUFFIX = ''
READ_TAB = TAB_DIR
if TP.TPLUS_ONLY:
    MODEL_ORDER = ['Metalog', 'QFlex-U', TP.LABEL, 'QFlex-TA+', 'QFlex-A+']   # legend order
    COLORS[TP.LABEL] = '#e377c2'     # pink: not one of the four reserved QPD colours
    MARKERS[TP.LABEL] = 'v'
    TAB_DIR, LOG_DIR = TP.table_dir(TAB_DIR), TP.log_dir(LOG_DIR)
    SUFFIX = '_withTplus'

K_FIG = list(range(3, 11))

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11, 'axes.titlesize': 11,
    'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'legend.frameon': True, 'legend.framealpha': 0.8,
    'figure.dpi': 150, 'savefig.dpi': 300, 'axes.grid': True,
    'grid.alpha': 0.3, 'grid.color': '#cccccc',
    'lines.linewidth': 1.8, 'lines.markersize': 6,
})

xi, lam, c, d = 0.0, 1.0, 0.5, 1.2

def _su_pdf(x):
    z = (x - xi) / lam
    return (d / (lam*np.sqrt(2*np.pi)*np.sqrt(1+z**2))) * np.exp(-0.5*(c + d*np.arcsinh(z))**2)

def _sl_pdf(x):
    if x <= xi: return 0.0
    z = c + d*np.log((x-xi)/lam)
    return (d/(np.sqrt(2*np.pi)*(x-xi))) * np.exp(-0.5*z**2)

def _sb_pdf(x):
    if x <= xi or x >= xi+lam: return 0.0
    u = (x-xi)/lam
    z = c + d*np.log(u/(1-u))
    return (d/(np.sqrt(2*np.pi)*lam*u*(1-u))) * np.exp(-0.5*z**2)

_PDF_FN = {'johnson_su': (_su_pdf, (-5, 5)),
           'johnson_sl': (_sl_pdf, (0.001, 20)),
           'johnson_sb': (_sb_pdf, (0.001, 0.999))}


def truth(dist):
    fn, bounds = _PDF_FN[dist]
    r = minimize_scalar(lambda x: -fn(x), bounds=bounds, method='bounded')
    return float(r.x), float(fn(r.x))


def load_and_aggregate(dist, n):
    short = dist.replace('johnson_', '')
    path = os.path.join(READ_TAB, f'mc_perrep_{short}_n{n}.csv')
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    if TP.TPLUS_ONLY:
        tpath = os.path.join(READ_TAB, 'tplus', f'mc_perrep_{short}_n{n}.csv')
        df = pd.concat([df, pd.read_csv(tpath)], ignore_index=True)
        path = [path, tpath]
    df = df[df.K.isin(K_FIG)]
    prov = {'file': path, 'rows': len(df),
            'mtime': [datetime.datetime.fromtimestamp(os.stat(p).st_mtime).isoformat(timespec='seconds')
              for p in (path if isinstance(path, list) else [path])]}
    k_vals = K_FIG
    v = df[df.valid == True].copy()
    v['n_modes'] = v.n_peaks.clip(lower=1)
    m = {}
    for model in MODEL_ORDER:
        a = df[df.Model == model]; f = v[v.Model == model]
        g = lambda col, fn: [fn(f[f.K == k][col].dropna()) for k in k_vals]
        m[model] = {
            'feasibility':        [a[a.K == k].valid.mean()*100 for k in k_vals],
            'false_multi':        [f[f.K == k].n_modes.gt(1).mean()*100 for k in k_vals],
            'w1_median':          g('w1', lambda s: s.median()),
            'mean_mode_count':    g('n_modes', lambda s: s.mean()),
            'mode_loc_median':    g('mode_loc', lambda s: s.median()),
            'mode_height_median': g('mode_height', lambda s: s.median()),
            'n_feasible':         [int(len(f[f.K == k])) for k in k_vals],
        }
    return {'k_vals': k_vals, 'metrics': m}, prov


def make_figure(dist, n, data, pair, prov):
    K = data['k_vals']; M = data['metrics']
    tl, th = truth(dist)

    def shared_ylim(key, floor_zero=False, pad=0.06):
        vals = [v for dd in pair.values() if dd for mo in MODEL_ORDER
                for v in dd['metrics'][mo][key] if v is not None and not np.isnan(v)]
        if not vals: return None
        lo, hi = min(vals), max(vals)
        p = (hi-lo)*pad if hi != lo else (abs(hi)*pad or 1.0)
        lo, hi = lo-p, hi+p
        return (max(0.0, lo) if floor_zero else lo, hi)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f'{DIST_LABELS[dist]}  |  N = {n}', fontsize=13, fontweight='bold', y=0.98)
    fig.subplots_adjust(hspace=0.38, wspace=0.32)
    cfgs = [
        dict(ax=axes[0,0], key='feasibility', title='Validity Rate vs Model Order',
             ylabel='Validity (%)', ylim=(0,105), truth=None),
        # W1 is DISPLAYED as a percentage (x100); the panel data CSV keeps
        # the unrounded fraction.
        dict(ax=axes[0,1], key='w1_median', title='Median W1 Distance vs Model Order',
             ylabel='Median W1 (%)', scale=100.0,
             ylim=tuple(100.0 * v for v in shared_ylim('w1_median', True)), truth=None),
        dict(ax=axes[0,2], key='false_multi', title='False Multimodality Rate vs Model Order',
             ylabel='False Multimodality (%)', ylim=(0, 100), truth=None),
        dict(ax=axes[1,0], key='mean_mode_count', title='Mean Mode Count vs Model Order',
             ylabel='Mean Mode Count', ylim=shared_ylim('mean_mode_count', True), truth=1.0),
        dict(ax=axes[1,1], key='mode_loc_median', title='Mode Location Median vs Model Order',
             ylabel='Mode Location', ylim=shared_ylim('mode_loc_median'), truth=tl),
        dict(ax=axes[1,2], key='mode_height_median', title='Mode Height Median vs Model Order',
             ylabel='Mode Height', ylim=shared_ylim('mode_height_median', True), truth=th),
    ]
    for i, cfg in enumerate(cfgs):
        ax = cfg['ax']
        for mo in MODEL_ORDER:
            ax.plot(K, cfg.get('scale', 1.0) * np.asarray(M[mo][cfg['key']], float),
                    color=COLORS[mo], marker=MARKERS[mo],
                    label=mo if i == 0 else '_nolegend_')
        if cfg['truth'] is not None:
            ax.axhline(cfg['truth'], color='black', linewidth=2, linestyle='--', zorder=1)
        ax.set_title(cfg['title']); ax.set_xlabel('Model Order (K)')
        ax.set_ylabel(cfg['ylabel']); ax.set_xticks(K)
        if cfg['ylim']: ax.set_ylim(cfg['ylim'])
    axes[0,0].legend(loc='lower left')

    short = dist.replace('johnson_', '')
    # Output names follow the paper's figure numbers: Johnson SU is Figures 3
    # and 4; SL and SB are the Online Supplement Figures O1-O4. The paper's
    # figures include QFlex-T+ (QPD_TPLUS_ONLY=1 run); without it the same
    # figure, minus QFlex-T+, is written to figures/other/ (not in the paper).
    PAPER_FIG = {('su', 50): 'Figure03_MC_JohnsonSU_N50', ('su', 200): 'Figure04_MC_JohnsonSU_N200',
                 ('sl', 50): 'FigureO1_MC_JohnsonSL_N50', ('sl', 200): 'FigureO2_MC_JohnsonSL_N200',
                 ('sb', 50): 'FigureO3_MC_JohnsonSB_N50', ('sb', 200): 'FigureO4_MC_JohnsonSB_N200'}
    DIST = {'su': 'johnson_su', 'sl': 'johnson_sl', 'sb': 'johnson_sb'}
    name = PAPER_FIG[(short, n)] if TP.TPLUS_ONLY else f'mc_{DIST[short]}_n{n}_without_Tplus'
    panel_name = f'{DIST[short]}_n{n}{SUFFIX}'          # panel data is named by dataset
    path = FS.figure_path(name)
    fig.savefig(path + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(path + '.pdf', bbox_inches='tight')
    plt.close(fig)

    rows = []
    for mo in MODEL_ORDER:
        for j, k in enumerate(K):
            rows.append({'Distribution': DIST_LABELS[dist], 'N': n, 'Model': mo, 'K': k,
                         'Validity_pct': round(M[mo]['feasibility'][j], 2),
                         'FalseMultimodality_pct': round(M[mo]['false_multi'][j], 2),
                         'MedianW1_Eq6': round(M[mo]['w1_median'][j], 5),
                         'MeanModeCount': round(M[mo]['mean_mode_count'][j], 4),
                         'ModeLocation_median': round(M[mo]['mode_loc_median'][j], 5),
                         'ModeHeight_median': round(M[mo]['mode_height_median'][j], 5),
                         'TrueModeLocation': round(tl, 5), 'TrueModeHeight': round(th, 5)})
    os.makedirs(TAB_DIR, exist_ok=True)
    csv = os.path.join(TAB_DIR, f'{panel_name}_paneldata.csv')
    pd.DataFrame(rows).to_csv(csv, index=False)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, f'{panel_name}_provenance.json'), 'w') as fh:
        json.dump({'generated': datetime.datetime.now().isoformat(timespec='seconds'),
                   'framework': 'code/common/eval_framework.py (grid 0.001..0.999)',
                   'seed_formula': 'seed = 42 + sample_size*1000 + replication',
                   'true_mode_location': tl, 'true_mode_height': th,
                   'source': prov}, fh, indent=2)
    print(f'  saved {name}.png/.pdf  +  {os.path.basename(csv)}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dist', nargs='?', default=None)
    ap.add_argument('n', nargs='?', type=int, default=None)
    ap.add_argument('--all', action='store_true')
    a = ap.parse_args()
    jobs = ([(d, n) for d in DIST_LABELS for n in PAIR_NS] if a.all
            else [(a.dist, a.n)])
    for dist in sorted({d for d, _ in jobs}):
        pair, provs = {}, {}
        for n in PAIR_NS:
            pair[n], provs[n] = load_and_aggregate(dist, n)
        for d2, n in [j for j in jobs if j[0] == dist]:
            if pair.get(n) is None:
                print(f'  ! no mc_perrep file for {dist} N={n} -- run mc_tables_rebuild.py --all'); continue
            print(f'{dist} N={n}:', flush=True)
            make_figure(dist, n, pair[n], pair, provs[n])


if __name__ == '__main__':
    main()
