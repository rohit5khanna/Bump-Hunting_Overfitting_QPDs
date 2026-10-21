"""
Figure O6 (Online Supplement O3): QFlex-T+ mode count and dispersion under
bootstrap resampling, Johnson SU, N = 200 -- the QFlex-T+ counterpart of
Figures 7, 8, O5 and O7.

Uses bootstrap_figures.dispersion_figure unchanged, on the main sweep plus
the T+ sweep from run_tplus_supplement.sh (tables/tplus/), so axes, mode ticks
and styling match the other dispersion figures. Writes
figures/FigureO6_QFlexTplus_bootstrap_modes.{png,pdf}.
"""
import os, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import bootstrap_figures as BF
import tplus_mode as TP

BF.COL[TP.LABEL] = '#e377c2'          # pink, as in the notebook and the MC figures
TT = os.path.join(BF.TAB, 'tplus')

df = pd.concat([pd.read_csv(os.path.join(BF.TAB, 'bootstrap_sweep.csv')),
                pd.read_csv(os.path.join(TT, 'bootstrap_sweep.csv'))], ignore_index=True)
of = pd.read_csv(os.path.join(TT, 'bootstrap_original_fits.csv'))
pf = pd.read_csv(os.path.join(TT, 'bootstrap_persist.csv'))
orig = {(r['model'], r['K']): r['n_modes'] for _, r in of.iterrows()}
persist = {(r['model'], r['K']): r['persist_pct'] for _, r in pf.iterrows()}
BF.dispersion_figure(df, orig, persist, TP.LABEL, BF.true_mode(),
                     'FigureO6_QFlexTplus_bootstrap_modes', 'O6')
