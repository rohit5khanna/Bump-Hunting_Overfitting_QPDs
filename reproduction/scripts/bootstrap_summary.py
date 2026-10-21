"""
Table 6 -- summary statistics per QPD across the 1000 bootstrap resamples of
the Johnson SU sample (Section 4.5).

Source: tables/bootstrap_sweep.csv -- the pinned realization (Johnson SU,
N = 200, seed 200110) resampled 1000 times, all four QPDs, K = 3..13.
Reported at the paper's standard order set, K = 4, 7, 10, 13.

W1: Equation 6 at the sample points (eval_framework.w1), reported BOTH ways:
MedianW1_own_Eq6 against the resample each fit was made to, MedianW1_orig_Eq6
against the ORIGINAL realization. MedianW1_Eq6 / W1_IQR repeat whichever of
the two eval_framework.BOOTSTRAP_W1_REFERENCE selects.

W1, validity and modes come from bootstrap_sweep.py, which uses the paper's
evaluation framework (code/common/eval_framework.py).

COLUMNS mirror the hydrology table (Table 8), so the two read the same way:
validity, then the full modality breakdown (1 / 2 / 3+ modes), then the raw
IQRs of mode location and mode height, then median W1.

MODALITY DENOMINATOR: valid fits whose PDF was usable (n_modes >= 0). The
"1 mode" column follows the paper's convention and counts a fit with no
detected peak as unimodal, as the bimodal, hydrology and fish tables
(Tables 7, 8 and 10) do; `n_no_peak` records how many those were so the
convention is auditable rather than silent.

IQRs are RAW, as in Tables 8 and 13 -- mode location in the standardized
units of the Johnson SU realization, mode height as a density (1/x). They are
taken over the DOMINANT mode of each fit (the tallest), the same rule the
hydrology table uses.

PERSIST: the share of valid bootstrap fits whose mode count matches the
original realization's (from tables/bootstrap_persist.csv).

writes tables/Table06_bootstrap_summary.csv
       (QFlex-T+ run: tables/tplus/bootstrap_summary_QFlexTplus.csv, not in the paper)
"""
import os, sys, json, datetime
import numpy as np, pandas as pd
from table_rounding import split_pct   # rows sum to exactly 100.0

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figure_style as FS

TAB = os.path.join(HERE, '..', 'tables')
import tplus_mode as TP                 # QPD_TPLUS_ONLY=1: supplementary QFlex-T+ run
TAB = TP.table_dir(TAB)
KS = FS.K_STANDARD
MODELS = ['Metalog', 'QFlex-U', 'QFlex-A+', 'QFlex-TA+']
if TP.TPLUS_ONLY:
    MODELS = [TP.LABEL]
from bootstrap_section import SEED, N


def main():
    sw = pd.read_csv(os.path.join(TAB, 'bootstrap_sweep.csv'))
    pf = pd.read_csv(os.path.join(TAB, 'bootstrap_persist.csv'))
    of = pd.read_csv(os.path.join(TAB, 'bootstrap_original_fits.csv'))
    persist = {(r['model'], r['K']): r['persist_pct'] for _, r in pf.iterrows()}
    orig = {(r['model'], r['K']): r['n_modes'] for _, r in of.iterrows()}

    def _dominant(sub):
        """(location, height) of the tallest mode of each fit.

        A fit with no detected peak writes an empty cell, which pandas reads
        back as NaN and str() turns into the literal 'nan' -- float('nan')
        then succeeds, so one such fit would poison the whole row's IQR. The
        guard below is the same one hydrology needed.
        """
        L, H = [], []
        for _, r in sub.iterrows():
            ll = [float(t) for t in str(r['mode_locs']).split(';')
                  if t and t.lower() != 'nan']
            hh = [float(t) for t in str(r['mode_heights']).split(';')
                  if t and t.lower() != 'nan']
            if ll and hh:
                i = int(np.argmax(hh)); L.append(ll[i]); H.append(hh[i])
        return np.array(L), np.array(H)

    def _iqr(a):
        return (round(float(np.subtract(*np.percentile(a, [75, 25]))), 4)
                if len(a) >= 4 else np.nan)

    rows = []
    for m in MODELS:
        for K in KS:
            g = sw[(sw.model == m) & (sw.K == K)]
            v = g[g.feasible == True]
            vm = v[v.n_modes.notna() & (v.n_modes >= 0)]
            w1 = v.w1_norm.dropna()
            loc, hgt = _dominant(v)
            n = len(vm)
            rows.append(dict(
                Model=m, K=K, n_resamples=len(g),
                Validity_pct=round(100 * len(v) / len(g), 1) if len(g) else np.nan,
                n_valid=len(v),
                **dict(zip(['modes_1_pct', 'modes_2_pct', 'modes_3plus_pct'],
                           split_pct([(vm.n_modes <= 1).sum(), (vm.n_modes == 2).sum(),
                                      (vm.n_modes >= 3).sum()]))),
                n_no_peak=int((vm.n_peaks == 0).sum()) if n else 0,
                ModeLocation_IQR=_iqr(loc),
                ModeHeight_IQR=_iqr(hgt),
                MedianW1_Eq6=round(float(w1.median()), 4) if len(w1) else np.nan,
                W1_IQR=round(float(w1.quantile(.75) - w1.quantile(.25)), 4) if len(w1) else np.nan,
                MedianW1_own_Eq6=round(float(v.w1_own_norm.median()), 4) if len(v) else np.nan,
                MedianW1_orig_Eq6=round(float(v.w1_orig_norm.median()), 4) if len(v) else np.nan,
                OriginalFit_modes=(int(orig[(m, K)]) if (m, K) in orig
                                   and orig[(m, K)] == orig[(m, K)] else np.nan),
                Persist_pct=(round(float(persist[(m, K)]), 1)
                             if (m, K) in persist and persist[(m, K)] == persist[(m, K)] else np.nan),
            ))
    res = pd.DataFrame(rows)
    dst = os.path.join(TAB, TP.name('Table06_bootstrap_summary', 'bootstrap_summary_QFlexTplus') + '.csv')
    res.to_csv(dst, index=False)
    print(res.to_string(index=False))

    chk = res[(res.Model == 'Metalog') & (res.K.isin([4, 7, 10]))].Persist_pct.tolist()
    print(f'\nPersist, Metalog K=4/7/10: {chk}')
    json.dump({'generated': datetime.datetime.now().isoformat(timespec='seconds'),
               # seed read from bootstrap_section.SEED, never hardcoded here
               'source': f'tables/bootstrap_sweep.csv (Johnson SU, N={N}, seed {SEED}, 1000 resamples)',
               'w1': 'Equation 6 at the sample points (eval_framework.w1); _own = vs the '
                     'resample, _orig = vs the original realization; MedianW1_Eq6 = the one '
                     'eval_framework.BOOTSTRAP_W1_REFERENCE selects',
               'persist': 'share of valid fits whose mode count equals the original fit\'s',
               'modality_denominator': 'valid fits with a usable PDF (n_modes >= 0); '
                                       'the 1-mode column counts no-peak fits as unimodal, '
                                       'as Tables 7, 8 and 10 do (n_no_peak records how many)',
               'iqr': 'RAW, over the dominant (tallest) mode of each fit -- location in the '
                      'realization\'s units, height as a density'},
              open(os.path.join(TP.log_dir(os.path.join(HERE, '..', 'logs')), 'bootstrap_summary_provenance.json'), 'w'),
              indent=2)
    print(f'-> {dst}')


if __name__ == '__main__':
    main()
