"""
QFlex-T+ run.

QFlex-T+ = the library's TL+ constraint: only the LEADING (highest-order) left
and right tail coefficients are constrained, strictly positive (imposed as
>= 1e-6 x data range; qflex/constraints.LEADING_TAIL_FLOOR_REL).

Set the environment variable QPD_TPLUS_ONLY=1 and the section scripts that
honour it fit ONLY QFlex-T+ (same samples, seeds, framework and summaries as
the other four models) and write every table/log to tables/tplus/ and
logs/tplus/, so the four-model outputs are never overwritten. Run it with
scripts/run_tplus_supplement.sh. In the paper this run produces Table A2, the
QFlex-T+ lines of Figures 3, 4 and O1-O4 (mc_johnson_figures.py, which also
writes those figures in this run) and Figure O6 (bootstrap_tplus_figure.py);
its bimodal, hydrology, fish and geyser sweeps are not in the paper but are
the QFlex-T+ half of the reference that the prominence check
(prominence_sensitivity/check_1pct.py) compares against. An environment variable (not a module
global) is used because worker processes on macOS start with 'spawn' and
re-import the module; they inherit the environment.
"""
import os

TPLUS_ONLY = os.environ.get('QPD_TPLUS_ONLY') == '1'
LABEL = 'QFlex-T+'


def table_dir(tab):
    if not TPLUS_ONLY:
        return tab
    d = os.path.join(tab, 'tplus'); os.makedirs(d, exist_ok=True); return d


def log_dir(log):
    if not TPLUS_ONLY:
        return log
    d = os.path.join(log, 'tplus'); os.makedirs(d, exist_ok=True); return d


def name(paper, supplementary):
    """Output file stem: the paper's name in the normal run; `supplementary`
    in the QFlex-T+ run, whose tables are not the paper's (except Table A2)."""
    return supplementary if TPLUS_ONLY else paper
