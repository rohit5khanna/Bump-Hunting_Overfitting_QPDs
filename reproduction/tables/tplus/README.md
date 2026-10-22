# QFlex-T+ run

The files here come from a separate run that fits **only QFlex-T+**, with the same samples, seeds
and evaluation framework as the other four models (`scripts/run_tplus_supplement.sh`, which sets
`QPD_TPLUS_ONLY=1`; see `scripts/tplus_mode.py`).

**Several file names repeat those in `tables/`, but the files here contain only the QFlex-T+
rows.** For example, `bootstrap_sweep.csv` in `tables/` has Metalog, QFlex-U, QFlex-TA+ and
QFlex-A+; `tplus/bootstrap_sweep.csv` has QFlex-T+. The scripts that need all five models read both
files and stack them.

| File | Used for |
|---|---|
| `TableA2_QFlexTplus_mc_summary_K4_7_10_13.csv`, `TableA2_mode_height_N50.csv`, `TableA2_mode_height_N200.csv` | Table A2 |
| `johnson_su_n50_withTplus_paneldata.csv`, `johnson_su_n200_withTplus_paneldata.csv` | Data behind Figures 3 and 4 (all five models) |
| `johnson_sl_n{50,200}_withTplus_paneldata.csv`, `johnson_sb_n{50,200}_withTplus_paneldata.csv` | Data behind Figures O1–O4 (all five models) |
| `mc_valid_unimodal_n200_K10_withTplus.csv` | Data behind Figure 5 (written by `mc_valid_unimodal_figure.py`) |
| `mc_perrep_{su,sl,sb}_n{50,200}.csv`, `mc_rebuilt_*` | QFlex-T+ Monte Carlo results, one row per replicate and K; input to the rows above |
| `bootstrap_sweep.csv`, `bootstrap_original_fits.csv`, `bootstrap_persist.csv` | Figure O6 |
| `bimodal_mc.csv`, `hydrology_sweep.csv`, `geyser_sweep.csv`, `fish_summary_perreplicate_jitter{0.5,1.0}.csv` | Not in the paper. QFlex-T+ half of the reference that the prominence check (`prominence_sensitivity/check_1pct.py`) compares against |
| `fish_summary_jitter0.5.csv`, `fish_summary_jitter1.0.csv` | Not in the paper; written alongside the fish per-replicate files |
