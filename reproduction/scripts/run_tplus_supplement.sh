#!/usr/bin/env bash
# QFlex-T+ run: fits ONLY QFlex-T+ with the same samples, seeds, framework and
# summaries as the other four models, writing tables to tables/tplus/ and logs
# to logs/tplus/; the four-model tables are not overwritten. Run it after
# run_all_framework.sh. In the paper it produces Table A2, the QFlex-T+ lines of
# Figures 3, 4 and O1-O4, and Figure O6; its bimodal, hydrology, fish, geyser
# and bootstrap-summary outputs are not in the paper. See scripts/tplus_mode.py.
# Usage (from reproduction/):  bash scripts/run_tplus_supplement.sh
set -euo pipefail
PY=${PY:-python}
cd "$(dirname "$0")/.."
export QPD_TPLUS_ONLY=1
mkdir -p logs/tplus tables/tplus
LOG=logs/tplus/run_tplus_supplement.log; : > "$LOG"
step() { echo "=== $(date +%H:%M:%S)  $*" | tee -a "$LOG"; "$PY" "$@" >> "$LOG" 2>&1 || { echo "FAILED: $* (see $LOG)"; exit 1; }; }
step scripts/mc_tables_rebuild.py --all --procs 4   # Table A2
step scripts/mc_mode_height.py --n 200              # Table A2 mode height, N = 200
step scripts/mc_mode_height.py --n 50               # Table A2 mode height, N = 50
step scripts/mc_johnson_figures.py --all            # Figures 3, 4, O1-O4 (with QFlex-T+) -> figures/
step scripts/bootstrap_sweep.py --procs 4
step scripts/bootstrap_summary.py                   # not in the paper
step scripts/bootstrap_tplus_figure.py              # Figure O6 -> figures/
step scripts/bimodal_mc.py --procs 4
step scripts/bimodal_summary.py
step scripts/hydrology_sweep.py --procs 4
step scripts/hydrology_tables.py
step scripts/fish_summary.py --jitter 0.5
step scripts/fish_summary.py --jitter 1.0
step scripts/geyser_sweep.py --procs 4
step scripts/geyser_tables.py
step scripts/geyser_mode_iqr.py
echo "ALL DONE  $(date +%H:%M:%S)" | tee -a "$LOG"
