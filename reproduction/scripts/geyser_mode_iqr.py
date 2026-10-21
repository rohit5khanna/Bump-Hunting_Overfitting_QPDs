"""
Geyser -- mode-dispersion IQRs for Table 13 (Section 5.3): its "Primary Mode
IQR" and "Secondary Mode IQR" columns.

Reads tables/geyser_sweep.csv -- the paper's 1000 jittered bootstrap
replicates (+-0.5 min, geyser_common.resample) -- and writes
tables/Table13_geyser_mode_iqr.csv, one row per (model, K).

The definition below reproduces all 20 (Log Metalog, Log QFlex-TA+) x K rows
of these columns exactly -- every count and every IQR. Primary = tallest /
secondary = next tallest (the fish section's rule) does NOT reproduce them.

DEFINITION (geyser):
  PRIMARY   = the RIGHTMOST mode of a fit (the long-wait cluster, ~80 min),
              over every feasible fit with at least one detected mode.
  SECONDARY = the LEFTMOST mode, over fits with at least two modes.
  IQR       = q75 - q25 (pandas linear interpolation), RAW units:
              location in min, height in 1/min.

NOTE: this differs from the fish section (fish_summary.py), which ranks modes
by HEIGHT (primary = tallest, secondary = next tallest). For geyser the two
clusters are physically identified by location, which is presumably why the
rule differs; for two-mode fits the difference only matters when the left
mode is the taller one.
"""
import os, json, datetime
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, '..', 'tables')
LOG = os.path.join(HERE, '..', 'logs')
import tplus_mode as TP                 # QPD_TPLUS_ONLY=1: supplementary QFlex-T+ run
TAB, LOG = TP.table_dir(TAB), TP.log_dir(LOG)


def _parse(locs, hts):
    if not isinstance(locs, str) or not locs:
        return None
    return (np.array([float(v) for v in locs.split(';')]),
            np.array([float(v) for v in hts.split(';')]))


def _iqr(v):
    s = pd.Series(v, dtype=float).dropna()
    return float(s.quantile(.75) - s.quantile(.25)) if len(s) else np.nan


def main():
    df = pd.read_csv(os.path.join(TAB, 'geyser_sweep.csv'))
    rows = []
    for (model, K), g in df.groupby(['model', 'K'], sort=False):
        P, S = [], []
        for a, b in zip(g[g.feasible].mode_locs, g[g.feasible].mode_heights):
            t = _parse(a, b)
            if t is None:
                continue
            L, H = t
            p = int(np.argmax(L)); P.append((L[p], H[p]))          # rightmost
            if len(L) > 1:
                s = int(np.argmin(L)); S.append((L[s], H[s]))      # leftmost
        P = np.array(P) if P else np.empty((0, 2))
        S = np.array(S) if S else np.empty((0, 2))
        rows.append(dict(model=model, K=int(K),
                         n_primary=len(P), prim_loc=_iqr(P[:, 0]), prim_ht=_iqr(P[:, 1]),
                         n_secondary=len(S), sec_loc=_iqr(S[:, 0]), sec_ht=_iqr(S[:, 1])))
    out = pd.DataFrame(rows)
    dst = os.path.join(TAB, TP.name('Table13_geyser_mode_iqr', 'geyser_mode_iqr_QFlexTplus') + '.csv')
    out.to_csv(dst, index=False)
    os.makedirs(LOG, exist_ok=True)
    json.dump({'generated': datetime.datetime.now().isoformat(timespec='seconds'),
               'source': 'tables/geyser_sweep.csv',
               'primary': 'rightmost mode, feasible fits with >= 1 mode',
               'secondary': 'leftmost mode, fits with >= 2 modes',
               'iqr': 'q75 - q25, raw units (min, 1/min)',
               'models': sorted(out.model.unique().tolist())},
              open(os.path.join(LOG, 'geyser_mode_iqr_provenance.json'), 'w'), indent=2)
    print(out.to_string(index=False))
    print(f'\n-> {dst}')


if __name__ == '__main__':
    main()
