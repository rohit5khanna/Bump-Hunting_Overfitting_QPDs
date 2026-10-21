"""
Geyser -- Table 13 (Section 5.3), from tables/geyser_sweep.csv.

writes
  tables/Table13_geyser_summary.csv   per model and K: % valid, and of the valid
                                      fits the unimodal / bimodal / 3+-mode shares,
                                      plus the median and IQR of W1. The primary-
                                      and secondary-mode IQR columns of Table 13
                                      come from geyser_mode_iqr.py.
  tables/geyser_mode_dispersion.csv   mode dispersion for Log Metalog K = 7, 8 and
                                      Log QFlex-TA+ K = 10 over the bimodal fits
                                      (supporting detail; not a table in the paper)

IQRs are raw: the interval [q25, q75] and its width in natural units (minutes
for locations, density for heights, Equation 6 units for W1); IQR / median is
given alongside in geyser_mode_dispersion.csv.

Primary mode = the mode at the longer waiting time. A rank flip is a bimodal
fit whose primary mode is lower than its secondary mode.

(QFlex-T+ run: tables/tplus/geyser_summary_QFlexTplus.csv, not in the paper.)
"""
import os, sys, json, datetime, warnings
import numpy as np, pandas as pd
from table_rounding import split_pct   # rows sum to exactly 100.0
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geyser_common import load_geyser, interdecile, JITTER, N_BOOT, data_fingerprint

TAB = os.path.join(HERE, '..', 'tables')
LOG = os.path.join(HERE, '..', 'logs')
import tplus_mode as TP                 # QPD_TPLUS_ONLY=1: supplementary QFlex-T+ run
TAB, LOG = TP.table_dir(TAB), TP.log_dir(LOG)

T9_SPECS = [('Log Metalog', 7), ('Log Metalog', 8), ('Log QFlex-TA+', 10)]
if TP.TPLUS_ONLY:
    T9_SPECS = [('Log ' + TP.LABEL, 10)]


def iqr(v):
    q1, q3 = np.percentile(v, [25, 75])
    return float(q1), float(q3), float(q3 - q1)


def main():
    df = pd.read_csv(os.path.join(TAB, 'geyser_sweep.csv'))
    x, p = load_geyser()
    L = interdecile(x, p)
    n_boot = int(df.groupby(['model', 'K']).size().max())

    # ---------------- Table 13 summary ----------------
    rows = []
    for (model, K), g in df.groupby(['model', 'K'], sort=False):
        f = g[g['feasible'] & g['n_modes'].notna()]
        nf = len(f)
        w = f['w1_eq6'].dropna().values
        wu = f['w1_raw'].dropna().values
        lo6, hi6, w6 = iqr(w) if len(w) else (np.nan,) * 3
        lou, hiu, wu_ = iqr(wu) if len(wu) else (np.nan,) * 3
        rows.append(dict(
            model=model, K=K, n_bootstrap=len(g), n_feasible=nf,
            pct_feasible=round(100 * nf / len(g), 1),
            # n_modes <= 1: a fit with no interior peak counts as unimodal, as
            # everywhere in the paper.
            **dict(zip(['pct_unimodal', 'pct_bimodal', 'pct_3plus'],
                       split_pct([(f['n_modes'] <= 1).sum(), (f['n_modes'] == 2).sum(),
                                  (f['n_modes'] >= 3).sum()]))),
            W1_Eq6_median=round(float(np.median(w)), 4) if len(w) else np.nan,
            W1_Eq6_q25=round(lo6, 4), W1_Eq6_q75=round(hi6, 4), W1_Eq6_IQR=round(w6, 4),
            W1_min_median=round(float(np.median(wu)), 4) if len(wu) else np.nan,
            W1_own_Eq6_median=round(float(f['w1_own_eq6'].median()), 4) if len(f) else np.nan,
            W1_orig_Eq6_median=round(float(f['w1_orig_eq6'].median()), 4) if len(f) else np.nan,
            W1_own_min_median=round(float(f['w1_own_raw'].median()), 4) if len(f) else np.nan,
            W1_orig_min_median=round(float(f['w1_orig_raw'].median()), 4) if len(f) else np.nan,
            W1_min_q25=round(lou, 4), W1_min_q75=round(hiu, 4), W1_min_IQR=round(wu_, 4),
        ))
    t8 = pd.DataFrame(rows)
    t8.to_csv(os.path.join(TAB, TP.name('Table13_geyser_summary', 'geyser_summary_QFlexTplus') + '.csv'), index=False)

    print('=' * 100)
    print('TABLE 13 -- geyser bootstrap summary (jitter applied, Equation 6 W1)')
    print('=' * 100)
    show = ['model', 'K', 'pct_feasible', 'pct_unimodal', 'pct_bimodal', 'pct_3plus',
            'W1_Eq6_median', 'W1_Eq6_IQR']
    print(t8[show].to_string(index=False))
    if TP.TPLUS_ONLY:       # the mode-dispersion slice is only for the paper's models
        return

    # ---------------- mode dispersion over the bimodal fits ----------------
    rows9 = []
    for model, K in T9_SPECS:
        g = df[(df['model'] == model) & (df['K'] == K)]
        f = g[g['feasible'] & g['n_modes'].notna()]
        bim = f[f['n_modes'] == 2]
        pl, ph, sl, sh = [], [], [], []
        for _, r in bim.iterrows():
            locs = np.array([float(v) for v in str(r['mode_locs']).split(';') if v])
            hgts = np.array([float(v) for v in str(r['mode_heights']).split(';') if v])
            if len(locs) != 2 or len(hgts) != 2:
                continue
            hi = int(np.argmax(locs)); lo = 1 - hi      # primary = longer waiting time
            pl.append(locs[hi]); ph.append(hgts[hi])
            sl.append(locs[lo]); sh.append(hgts[lo])
        pl, ph, sl, sh = map(np.asarray, (pl, ph, sl, sh))
        flip = 100 * float(np.mean(ph < sh)) if len(ph) else np.nan
        d = dict(model=model, K=K, n_valid=len(f), n_bimodal=len(pl))
        for nm, v in (('primary_loc', pl), ('primary_height', ph),
                      ('secondary_loc', sl), ('secondary_height', sh)):
            q1, q3, w_ = iqr(v)
            med = float(np.median(v))
            d[f'{nm}_median'] = round(med, 4)
            d[f'{nm}_q25'] = round(q1, 4)
            d[f'{nm}_q75'] = round(q3, 4)
            d[f'{nm}_IQR_raw'] = round(w_, 4)
            d[f'{nm}_IQR_relative'] = round(w_ / med, 4) if med else np.nan
        d.update(rank_flip_pct=round(flip, 1))
        rows9.append(d)
    t9 = pd.DataFrame(rows9)
    t9.to_csv(os.path.join(TAB, 'geyser_mode_dispersion.csv'), index=False)

    print()
    print('=' * 100)
    print('Mode dispersion over the bimodal fits: raw IQR (IQR / median)')
    print('=' * 100)
    print(t9[['model', 'K', 'n_valid', 'n_bimodal', 'rank_flip_pct']].to_string(index=False))
    print()
    print(f'{"model":<15}{"K":>3}{"":>3}{"primary loc":>22}{"primary ht":>22}'
          f'{"secondary loc":>22}{"secondary ht":>22}')
    for _, r in t9.iterrows():
        cells = ''
        for nm in ('primary_loc', 'primary_height', 'secondary_loc', 'secondary_height'):
            cells += f'{r[nm+"_IQR_raw"]:>10.4f} ({r[nm+"_IQR_relative"]:.3f})'.rjust(22)
        print(f'{r["model"]:<15}{int(r["K"]):>3}{"":>3}{cells}')

    # ---------------- consistency ----------------
    print()
    print('=' * 100)
    print('CONSISTENCY: the mode-dispersion slice must match the summary table')
    print('=' * 100)
    ok = True
    for _, r in t9.iterrows():
        t = t8[(t8['model'] == r['model']) & (t8['K'] == r['K'])].iloc[0]
        exp_bim = round(t['n_feasible'] * t['pct_bimodal'] / 100)
        a = int(r['n_valid']) == int(t['n_feasible'])
        b = abs(int(r['n_bimodal']) - exp_bim) <= 1
        ok &= a and b
        print(f'  {r["model"]} K={int(r["K"])}: valid {int(r["n_valid"])} vs summary '
              f'{int(t["n_feasible"])} [{"OK" if a else "MISMATCH"}]; bimodal '
              f'{int(r["n_bimodal"])} vs {exp_bim} [{"OK" if b else "MISMATCH"}]')
    print('  ALL CONSISTENT' if ok else '  INCONSISTENT')

    json.dump({
        'generated': datetime.datetime.now().isoformat(timespec='seconds'),
        'data_file': data_fingerprint()[0],
        'data_N': data_fingerprint()[1],
        'data_sha256': data_fingerprint()[2],
        'source': 'tables/geyser_sweep.csv (one sweep, both outputs)',
        'n_bootstrap': n_boot, 'jitter_half_width_min': JITTER,
        'interdecile_divisor_min': L,
        'iqr_convention': 'raw [q25, q75] and width; IQR / median alongside',
        'rank_flip': 'within-fit: primary = longer waiting time; flip if its height '
                     '< secondary height',
        'tables_consistent': bool(ok),
    }, open(os.path.join(LOG, 'geyser_tables_provenance.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
