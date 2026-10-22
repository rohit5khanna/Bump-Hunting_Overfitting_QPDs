#!/usr/bin/env bash
# Regenerate every framework-dependent table and figure (eval_framework.py).
# Stops at the first failing step; full output goes to logs/run_all_framework.log.
# Usage (from reproduction/):  bash scripts/run_all_framework.sh
# The Monte Carlo and SU bootstrap steps fit all five QPDs (Metalog, QFlex-U,
# QFlex-T+, QFlex-TA+, QFlex-A+); the other sections fit the QPDs the paper
# reports for them. Comments give the paper table/figure each step produces.
# Tables 3, 5, 9, 12, O1, O2 and Figures 1, 18, O8 have their own commands
# (see README.md).
set -euo pipefail
PY=${PY:-python}
cd "$(dirname "$0")/.."
LOG=logs/run_all_framework.log
mkdir -p logs; : > "$LOG"
step() { echo "=== $(date +%H:%M:%S)  $*" | tee -a "$LOG"; "$PY" "$@" >> "$LOG" 2>&1 || { echo "FAILED: $* (see $LOG)"; exit 1; }; }
step ../code/common/eval_framework.py
# Monte Carlo (Section 4)
step scripts/mc_tables_rebuild.py --all --procs 4    # Tables 4, A1, A2 (validity, false modality, W1)
step scripts/mc_mode_height.py --n 200               # mode height column of Table 4 and Table A2 (N = 200)
step scripts/mc_mode_height.py --n 50                # mode height column of Table A1 and Table A2 (N = 50)
step scripts/mc_johnson_figures.py --all             # Figures 3, 4, O1-O4
step scripts/mc_valid_unimodal_figure.py             # Figure 5
step scripts/mc_sb_pdf_fit_triptych.py               # Figure 2
# SU bootstrap (Section 4.5)
step scripts/bootstrap_sweep.py --procs 4
step scripts/bootstrap_summary.py                    # Table 6
step scripts/bootstrap_figures.py                    # Figures 7, 8, O5, O6, O7
step scripts/bootstrap_qf_figure.py                  # Figure 6
# Bimodal (Section 4.6)
step scripts/bimodal_mc.py --procs 4
step scripts/bimodal_summary.py                      # Table 7
step scripts/bimodal_figure.py   # Figure 9; uses FIXED_REP = 436 (illustrative; not rule-selected)
# Hydrology (Section 5.1)
step scripts/hydrology_sweep.py --procs 4
step scripts/hydrology_tables.py                     # Table 8
step scripts/hydrology_figures.py                    # Figures 10-14
# Fish (Section 5.2)
step scripts/fish_summary.py --jitter 0.5            # Table 10
step scripts/fish_summary.py --jitter 1.0            # not in the paper; input to the prominence check (check_1pct.py)
step scripts/fish_sample_size_sweep.py               # Table 11
step scripts/fish_histogram_figure.py                # Figure 15
step scripts/fish_eqf_ci_figure.py                   # Figure 16
step scripts/fish_pdf_order_panels.py                # Figure 17
# Geyser (Section 5.3)
step scripts/geyser_sweep.py --procs 4
step scripts/geyser_tables.py                        # Table 13
step scripts/geyser_mode_iqr.py                      # Table 13 (primary/secondary mode IQRs)
step scripts/geyser_latent_fit_figure.py             # Figure 19
step scripts/geyser_latent_stability.py              # not in the paper; supports Figure 19 (one jitter realization suffices)
echo "ALL DONE  $(date +%H:%M:%S)" | tee -a "$LOG"
