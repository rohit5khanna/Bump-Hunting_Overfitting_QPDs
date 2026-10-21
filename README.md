# Bump Hunting, Structural Overfitting, and Quantile-Parameterized Distributions

Code, data, results and figures for

> Rohit Khanna and J. Eric Bickel, *Bump Hunting, Structural Overfitting, and
> Quantile-Parameterized Distributions*. The University of Texas at Austin. Under review.

The paper, including its Online Supplement, is available on SSRN: <https://ssrn.com/abstract=7578879>. The interactive online
companion (cited in the paper as "Anonymous (2026), Bump Hunting with QPDs, Version v2.1") is
live at <https://qpd-bump-hunting.netlify.app/>; its source is in `companion/`.

Every table and figure in the paper comes from a script in this repository, and every
experiment is seeded, so a rerun reproduces the stored results. All file names use the
paper's final numbering.

## Layout

| Folder | Contents |
|---|---|
| `data/` | The three empirical datasets: `Fish Biology.xlsx` (Babine River steelhead trout weights, Keelin 2016), `Hydrology.xlsx` (Williamson River annual peak gauge heights, 1920–2014), `geyser_299.csv` (Old Faithful waiting times, Azzalini and Bowman 1990; 299 observations). |
| `code/common/` | Shared library code: `metalog` and `qflex` (QPD fitting), `jpse` (Johnson SU/SL/SB), `eval_framework.py` (validity, densities, modes, W1), `stats_tests.py` (Hartigan dip and Silverman tests), `mode_utils.py`. |
| `reproduction/scripts/` | The analysis scripts. |
| `reproduction/figures/` | Every figure in the paper and the Online Supplement, `Figure01_…` to `Figure19_…` and `FigureO1_…` to `FigureO8_…` (PNG and PDF), exactly as they appear in the paper. Figures a script draws that are not in the paper are written to `figures/other/` when it is rerun. |
| `reproduction/tables/` | Every table in the paper (`Table03_…` to `Table13_…`, `TableA1_…`, `TableO2_…`), plus the per-replicate results and figure panel data they come from. `tables/tplus/` holds the QFlex-T+ run (Table A2); `tables/invgauss/` the inverse Gaussian run. |
| `reproduction/data/` | The fixed Johnson SU sample (N = 200, seed 200110) used in Section 4.5. |
| `reproduction/logs/` | Provenance files (inputs, seeds, checksums) written by the scripts when they run. |
| `supplement/prominence_sensitivity/` | Online Supplement Section O1 (Table O1). |
| `excel/` | Excel workbooks: the paper's tables (`QPD_Paper_Tables.xlsx`), inverse Gaussian vs Johnson SL, prominence sensitivity, and the worked Johnson SL/SB validity example for Appendix A3. |
| `companion/` | The marimo notebook behind the online companion. |

## Setup

Tested with Python 3.10.

```bash
pip install -r requirements.txt
```

Run every analysis script from `reproduction/`. The scripts find `code/common/` and `data/` by
relative path, so keep the folder layout as it is.

## Conventions used throughout

* **Evaluation grid.** Every fitted QPD is assessed on p = 0.001, 0.002, …, 0.999 only.
* **Validity.** The fitted quantile function is strictly increasing on the grid.
* **Density.** f(p_j) = (p_{j+1} − p_{j−1}) / (Q(p_{j+1}) − Q(p_{j−1})), one-sided at the two ends.
* **Modes** (Section 3.3). Interior local maxima of that density whose prominence is at least 1%
  of the peak's own height; the grid end points are never modes; a valid fit with no such peak
  counts as unimodal.
* **W1** (Equation 6). mean_i |Q_fit(p_i) − x_(i)| at the plotting positions p_i = i/(N+1),
  divided by EQF(0.9) − EQF(0.1) of the same sample and reported in %. The reference sample is
  always the one the model was fitted to (in a bootstrap, the resample).
* **Seeding.** Each replicate has its own seed, so any replicate can be regenerated on its own.
* **Jitter j** means Uniform(−j, +j) noise (j is the half-width).

## Figures

All files are in `reproduction/figures/`; all scripts in `reproduction/scripts/`.

| Paper | Content | Script | File |
|---|---|---|---|
| Figure 1 | Johnson SU, SL and SB reference distributions | `reference_distributions_figure.py` | `Figure01_reference_distributions` |
| Figure 2 | Logit Metalog and Logit QFlex-A+ fits to one Johnson SB sample, K = 4, 7, 10 | `mc_sb_pdf_fit_triptych.py` | `Figure02_SB_example_fits` |
| Figure 3 | Monte Carlo fit performance, Johnson SU, N = 50 | `mc_johnson_figures.py` (QFlex-T+ run) | `Figure03_MC_JohnsonSU_N50` |
| Figure 4 | Monte Carlo fit performance, Johnson SU, N = 200 | `mc_johnson_figures.py` (QFlex-T+ run) | `Figure04_MC_JohnsonSU_N200` |
| Figure 5 | Valid and unimodal fits, N = 200, K = 10 | `mc_valid_unimodal_figure.py` | `Figure05_valid_unimodal_N200_K10` |
| Figure 6 | Empirical and fitted Metalog QF with 95% bootstrap interval | `bootstrap_qf_figure.py` | `Figure06_SU_bootstrap_QF_interval` |
| Figure 7 | Metalog mode count and dispersion under bootstrap | `bootstrap_figures.py` | `Figure07_Metalog_bootstrap_modes` |
| Figure 8 | QFlex-A+ mode count and dispersion under bootstrap | `bootstrap_figures.py` | `Figure08_QFlexAplus_bootstrap_modes` |
| Figure 9 | Bimodal reference distribution and K = 12 fits | `bimodal_figure.py` | `Figure09_bimodal_reference` |
| Figure 10 | River gauge heights: EQF | `hydrology_figures.py` | `Figure10_hydrology_EQF` |
| Figure 11 | River gauge heights: histogram with GEV and Log Metalog K = 10 | `hydrology_figures.py` | `Figure11_hydrology_histogram_pdfs` |
| Figure 12 | River gauge heights: EQF with 95% bootstrap intervals | `hydrology_figures.py` | `Figure12_hydrology_EQF_bootstrap_interval` |
| Figure 13 | River gauge heights: bootstrap mode height vs location | `hydrology_figures.py` | `Figure13_hydrology_mode_scatter` |
| Figure 14 | GEV, Log Metalog-4 and Log QFlex-A+-10 fits to the flood data | `hydrology_figures.py` | `Figure14_hydrology_GEV_Metalog_QFlex` |
| Figure 15 | Fish weights: histogram with 10-term Log Metalog | `fish_histogram_figure.py` | `Figure15_fish_histogram` |
| Figure 16 | Fish weights: EQF with 95% bootstrap intervals, 0.5 lb jitter | `fish_eqf_ci_figure.py` | `Figure16_fish_EQF_bootstrap_interval` |
| Figure 17 | Fish weights: Log Metalog and Log QFlex-U fits, K = 7, 10, 13 | `fish_pdf_order_panels.py` | `Figure17_fish_fits_K7_10_13` |
| Figure 18 | Old Faithful data: EQF and density | `geyser_data_figure.py` | `Figure18_geyser_data` |
| Figure 19 | Old Faithful: Log Metalog (K = 8) and Log QFlex-TA+ (K = 10) fits | `geyser_latent_fit_figure.py` | `Figure19_geyser_fits` |
| Figures O1–O4 | Monte Carlo fit performance, Johnson SL and SB, N = 50 and 200 | `mc_johnson_figures.py` (QFlex-T+ run) | `FigureO1_MC_JohnsonSL_N50` … `FigureO4_MC_JohnsonSB_N200` |
| Figure O5 | QFlex-U mode count and dispersion under bootstrap | `bootstrap_figures.py` | `FigureO5_QFlexU_bootstrap_modes` |
| Figure O6 | QFlex-T+ mode count and dispersion under bootstrap | `bootstrap_tplus_figure.py` | `FigureO6_QFlexTplus_bootstrap_modes` |
| Figure O7 | QFlex-TA+ mode count and dispersion under bootstrap | `bootstrap_figures.py` | `FigureO7_QFlexTAplus_bootstrap_modes` |
| Figure O8 | Inverse Gaussian reference distribution | `invgauss_reference_figure.py` | `FigureO8_inverse_gaussian_reference` |

## Tables

Tables 1 and 2 are analytical (constraint systems and structural mode bounds) and need no code.
All files are in `reproduction/tables/` unless stated otherwise.

| Paper | Content | Script(s) | File |
|---|---|---|---|
| Table 3 | Hartigan and Silverman tests on the Monte Carlo samples | `mc_modality_tests.py` | `Table03_mc_modality_tests.csv` |
| Table 4 | QPD fit summary, Monte Carlo, N = 200 | `mc_tables_rebuild.py`, `mc_mode_height.py` | `Table04_TableA1_mc_summary_K4_7_10_13.csv` (N = 200 rows), `Table04_mode_height_N200.csv` |
| Table 5 | Hartigan and Silverman tests on the bootstrap resamples of the Johnson SU sample | `bootstrap_hartigan.py`, `bootstrap_silverman.py` | `Table05_hartigan_pvals_seed200110.csv`, `Table05_silverman_seed200110_summary.csv` |
| Table 6 | QPD summary across the 1000 bootstrap resamples | `bootstrap_sweep.py`, `bootstrap_summary.py` | `Table06_bootstrap_summary.csv` |
| Table 7 | Bimodal Monte Carlo, Metalog and QFlex-TA+ | `bimodal_mc.py`, `bimodal_summary.py` | `Table07_bimodal_mc.csv` |
| Table 8 | River gauge heights: stability of model features | `hydrology_sweep.py`, `hydrology_tables.py` | `Table08_hydrology.csv` |
| Table 9 | Fish weights: modality tests and sequential Silverman | `fish_modality_paper_replicates.py --k 1,2,3` | `Table09_fish_modality_tests.csv` |
| Table 10 | Fish weights: stability of model features, 0.5 lb jitter | `fish_summary.py --jitter 0.5` | `Table10_fish_summary_jitter0.5.csv` |
| Table 11 | Fish weights: sensitivity to sample size, K = 10 and 13 | `fish_sample_size_sweep.py` | `Table11_fish_sample_size_sweep.csv` |
| Table 12 | Old Faithful: modality tests and sequential Silverman | `geyser_modality_paper_replicates.py` | `Table12_geyser_modality_tests.csv` |
| Table 13 | Old Faithful: Log Metalog and Log QFlex-TA+ bootstrap results | `geyser_sweep.py`, `geyser_tables.py`, `geyser_mode_iqr.py` | `Table13_geyser_summary.csv`, `Table13_geyser_mode_iqr.csv` |
| Table A1 | QPD fit summary, Monte Carlo, N = 50 | `mc_tables_rebuild.py`, `mc_mode_height.py` | `Table04_TableA1_mc_summary_K4_7_10_13.csv` (N = 50 rows), `TableA1_mode_height_N50.csv` |
| Table A2 | QFlex-T+ fit summary, Monte Carlo, N = 50 and 200 | `run_tplus_supplement.sh` | `tplus/TableA2_QFlexTplus_mc_summary_K4_7_10_13.csv`, `tplus/TableA2_mode_height_N50.csv`, `tplus/TableA2_mode_height_N200.csv` |
| Table O1 | False modality for Johnson SU at three prominence thresholds | `supplement/prominence_sensitivity/` | `supplement/prominence_sensitivity/TableO1_prominence_JohnsonSU.csv` |
| Table O2 | Inverse Gaussian Monte Carlo fit summary, N = 200 | `mc_invgauss.py`, `invgauss_table_O2.py` | `invgauss/TableO2_invgauss_n200.csv` |

The Hartigan and Silverman results quoted in the text of Section 5.1 are in
`hydrology_modality_tests.csv` (`hydrology_modality_tests.py`). `excel/QPD_Paper_Tables.xlsx`
collects the tables in the paper's layout.

## How to run

All commands run from `reproduction/`. Run times are for 4 worker processes on a laptop; the
Monte Carlo and fish-jitter steps take the longest.

```bash
cd reproduction

# Tables 4, 6, 7, 8, 10, 11, 13, A1 and Figures 2, 6-17, 19
bash scripts/run_all_framework.sh

# QFlex-T+: Table A2, Figures 3, 4, O1-O4 (which include QFlex-T+) and Figure O6
bash scripts/run_tplus_supplement.sh
python3 scripts/mc_valid_unimodal_figure.py                 # Figure 5

python3 scripts/reference_distributions_figure.py           # Figure 1
python3 scripts/geyser_data_figure.py                        # Figure 18
python3 scripts/mc_modality_tests.py                         # Table 3
python3 scripts/bootstrap_hartigan.py                        # Table 5 (Hartigan)
python3 scripts/bootstrap_silverman.py                       # Table 5 (Silverman)
python3 scripts/fish_modality_paper_replicates.py --k 1,2,3  # Table 9
python3 scripts/geyser_modality_paper_replicates.py          # Table 12
python3 scripts/hydrology_modality_tests.py --repeat 3       # tests quoted in Section 5.1

# Online Supplement
python3 scripts/mc_invgauss.py --n 200 --procs 4             # Section O4
python3 scripts/invgauss_table_O2.py                         # Table O2
python3 scripts/invgauss_reference_figure.py                 # Figure O8
```

Table O1 has its own instructions at the top of
`supplement/prominence_sensitivity/prominence_sensitivity.py`; `summarize.py` there writes
`TableO1_prominence_JohnsonSU.csv`. Each script's docstring lists its options.

The self-tests in `bootstrap_section.py --verify` and `bimodal_common.py` compare against
simulation checkpoints and a sampler from an earlier version of the code that are not part of
this repository; the analyses themselves do not need them.

## Online companion

`companion/app.py` is a [marimo](https://marimo.io) notebook. It carries its own copy of the
library code (`companion/common/`) so that it can be bundled for the browser.

```bash
cd companion
pip install -r requirements.txt
marimo edit app.py                                     # interactive
marimo export html-wasm app.py -o _site --mode run     # the static site deployed to Netlify
```

## Citation

Khanna R, Bickel JE. Bump Hunting, Structural Overfitting, and Quantile-Parameterized
Distributions. Under review.
