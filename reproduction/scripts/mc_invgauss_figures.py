"""
Supplementary figures and comparison table for the inverse Gaussian run of Online
Supplement Section O4 (mc_invgauss.py). None of these outputs is in the paper (the
section's figure is Figure O8, invgauss_reference_figure.py; its table is Table O2,
invgauss_table_O2.py).

  figures/other/invgauss_vs_sl_reference.png      the two reference densities, P10/P50/P90 marked
  figures/other/invgauss_mc_n50.png, _n200.png    same six panels as Figures O1/O2 (Johnson SL), five QPDs
  tables/invgauss/invgauss_vs_sl_K4_7_10_13.csv   IG next to Johnson SL, both N, all five QPDs

Panels, aggregation and colours follow mc_johnson_figures.py exactly (mode count = max(1, peaks);
location / height of the tallest interior peak over valid fits; K = 3..10; y-ranges shared by
the N = 50 / 200 pair; W1 shown x100). QFlex-T+ uses the same colour/marker as in Figures 3, 4
and O1-O4 (the QFlex-T+ run of mc_johnson_figures.py). Truth lines: the IG mode and the density
at the mode (closed form).
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib as mpl

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mc_invgauss as IGR
import figure_style as FS
from jpse import JohnsonSL

TAB = os.path.join(HERE, '..', 'tables'); FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(FIG, exist_ok=True)
MODELS = IGR.MODELS
COLORS = dict(FS.QPD); COLORS['QFlex-T+'] = '#e377c2'
MARKERS = {'Metalog': 'o', 'QFlex-U': 's', 'QFlex-T+': 'v', 'QFlex-TA+': 'D', 'QFlex-A+': '^'}
K_FIG = list(range(3, 11))
mpl.rcParams.update({'font.family': 'sans-serif', 'font.size': 11, 'axes.titlesize': 11,
                     'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                     'legend.fontsize': 9, 'legend.frameon': True, 'legend.framealpha': 0.8,
                     'figure.dpi': 150, 'savefig.dpi': 300, 'axes.grid': True, 'grid.alpha': 0.3,
                     'grid.color': '#cccccc', 'lines.linewidth': 1.8, 'lines.markersize': 6})
SL = JohnsonSL(xi=0, lam=1, gamma=0.5, delta=1.2)


def aggregate(n):
    df = pd.read_csv(os.path.join(TAB, 'invgauss', f'mc_perrep_ig_n{n}.csv'))
    df = df[df.K.isin(K_FIG)]
    v = df[df.valid == True].copy(); v['n_modes'] = v.n_peaks.clip(lower=1)
    M = {}
    for mo in MODELS:
        a = df[df.Model == mo]; f = v[v.Model == mo]
        g = lambda col, fn: [fn(f[f.K == k][col].dropna()) for k in K_FIG]
        M[mo] = {'feasibility': [a[a.K == k].valid.mean() * 100 for k in K_FIG],
                 'false_multi': [f[f.K == k].n_modes.gt(1).mean() * 100 for k in K_FIG],
                 'w1_median': g('w1', lambda s: s.median()),
                 'mean_mode_count': g('n_modes', lambda s: s.mean()),
                 'mode_loc_median': g('mode_loc', lambda s: s.median()),
                 'mode_height_median': g('mode_height', lambda s: s.median())}
    return M


def mc_figure(n, M, pair):
    tl, th = IGR.true_mode()

    def ylim(key, floor0=False, pad=0.06):
        vals = [x for P in pair.values() for mo in MODELS for x in P[mo][key] if x is not None and not np.isnan(x)]
        lo, hi = min(vals), max(vals); p = (hi - lo) * pad if hi != lo else 1.0
        return (max(0.0, lo - p) if floor0 else lo - p, hi + p)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f'Inverse Gaussian (μ = {IGR.MU}, λ = {IGR.LAMBDA})  |  N = {n}',
                 fontsize=13, fontweight='bold', y=0.98)
    fig.subplots_adjust(hspace=0.38, wspace=0.32)
    cfgs = [(axes[0, 0], 'feasibility', 'Validity Rate vs Model Order', 'Validity (%)', (0, 105), None, 1),
            (axes[0, 1], 'w1_median', 'Median W1 Distance vs Model Order', 'Median W1 (%)',
             tuple(100 * x for x in ylim('w1_median', True)), None, 100.0),
            (axes[0, 2], 'false_multi', 'False Multimodality Rate vs Model Order', 'False Multimodality (%)', (0, 100), None, 1),
            (axes[1, 0], 'mean_mode_count', 'Mean Mode Count vs Model Order', 'Mean Mode Count', ylim('mean_mode_count', True), 1.0, 1),
            (axes[1, 1], 'mode_loc_median', 'Mode Location Median vs Model Order', 'Mode Location', ylim('mode_loc_median'), tl, 1),
            (axes[1, 2], 'mode_height_median', 'Mode Height Median vs Model Order', 'Mode Height', ylim('mode_height_median', True), th, 1)]
    for i, (ax, key, title, yl, lim, truth, sc) in enumerate(cfgs):
        for mo in MODELS:
            ax.plot(K_FIG, sc * np.asarray(M[mo][key], float), color=COLORS[mo], marker=MARKERS[mo],
                    label=mo if i == 0 else '_nolegend_')
        if truth is not None:
            ax.axhline(truth, color='black', linewidth=2, linestyle='--', zorder=1)
        ax.set_title(title); ax.set_xlabel('Model Order (K)'); ax.set_ylabel(yl)
        ax.set_xticks(K_FIG); ax.set_ylim(lim)
    axes[0, 0].legend(loc='lower left')
    for ext in ('png', 'pdf'):
        fig.savefig(FS.figure_path(f'invgauss_mc_n{n}') + f'.{ext}', bbox_inches='tight')
    plt.close(fig)


def reference_figure():
    x = np.linspace(1e-4, 4, 2000)
    zsl = 0.5 + 1.2 * np.log(x)
    f_sl = 1.2 / (np.sqrt(2 * np.pi) * x) * np.exp(-0.5 * zsl ** 2)
    f_ig = IGR.IG.pdf(x)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(x, f_sl, color='black', lw=2, label='Johnson SL (ξ=0, λ=1, γ=0.5, δ=1.2)')
    ax.plot(x, f_ig, color='#8c564b', lw=2, ls='--', label=f'Inverse Gaussian (μ={IGR.MU}, λ={IGR.LAMBDA})')
    p = np.array([0.1, 0.5, 0.9])
    for q_sl, q_ig, pp in zip(SL.quantile(p), IGR.IG.ppf(p), p):
        ax.axvline(q_sl, color='black', lw=0.8, alpha=0.5)
        ax.axvline(q_ig, color='#8c564b', lw=0.8, alpha=0.6, ls='--')
        ax.text(q_sl, 1.08, f'P{int(pp*100)}', ha='center', fontsize=9)
    ax.set_xlim(0, 4); ax.set_ylim(0, 1.15); ax.set_xlabel('x'); ax.set_ylabel('Density')
    ax.set_title('Reference distributions: Johnson SL and the inverse Gaussian matched to its P10 / P50 / P90')
    ax.legend(loc='upper right', fontsize=9)
    for ext in ('png', 'pdf'):
        fig.savefig(FS.figure_path('invgauss_vs_sl_reference') + f'.{ext}', bbox_inches='tight')
    plt.close(fig)


def comparison_table():
    rows = []
    for n in (50, 200):
        ig = pd.read_csv(os.path.join(TAB, 'invgauss', f'mc_rebuilt_ig_n{n}.csv'))
        sl = pd.concat([pd.read_csv(os.path.join(TAB, f'mc_rebuilt_sl_n{n}.csv')),
                        pd.read_csv(os.path.join(TAB, 'tplus', f'mc_rebuilt_sl_n{n}.csv'))])
        for mo in MODELS:
            for K in IGR.KS:
                a = sl[(sl.Model == mo) & (sl.K == K)].iloc[0]; b = ig[(ig.Model == mo) & (ig.K == K)].iloc[0]
                rows.append(dict(N=n, Model=mo, K=K,
                                 SL_Validity_pct=a.Validity_pct, IG_Validity_pct=b.Validity_pct,
                                 SL_FalseModality_pct=a.FalseModality_pct, IG_FalseModality_pct=b.FalseModality_pct,
                                 SL_MedianW1_pct=round(100 * a.MedianW1_Eq6, 2), IG_MedianW1_pct=round(100 * b.MedianW1_Eq6, 2)))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TAB, 'invgauss', 'invgauss_vs_sl_K4_7_10_13.csv'), index=False)
    return t


if __name__ == '__main__':
    pair = {n: aggregate(n) for n in (50, 200)}
    for n in (50, 200):
        mc_figure(n, pair[n], pair)
    reference_figure()
    p = np.array([0.1, 0.5, 0.9])
    print('quantiles  P10 P50 P90')
    print('  Johnson SL      ', np.round(SL.quantile(p), 4))
    print('  Inverse Gaussian', np.round(IGR.IG.ppf(p), 4))
    print('IG mode (location, height):', np.round(IGR.true_mode(), 4))
    pd.set_option('display.width', 200)
    print(comparison_table().to_string(index=False))
