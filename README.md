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
| `reproduction/figures/` | Every figure in the paper and the Online Supplement, `Figure01_…` to `Figure19_…` and `FigureO1_…` to `FigureO8_…` (PNG and PDF). The plotted content is the paper's; the stored files also carry a title line that the paper's versions crop out. Figures a script draws that are not in the paper are written to `figures/other/` when it is rerun. |
| `reproduction/tables/` | Every table in the paper (`Table03_…` to `Table13_…`, `TableA1_…`), the data behind each figure (see the Figures table below), and the per-replicate results they come from. Subfolder `invgauss/` holds the inverse Gaussian run (Table O2; see its README). |
| `reproduction/data/` | The fixed Johnson SU sample (N = 200, seed 200110) used in Section 4.5, with a manifest holding its checksum. |
| `reproduction/logs/` | Provenance files (inputs, seeds, checksums). Scripts write or overwrite one each time they run; those stored here come from the runs that produced the stored outputs. |
| `reproduction/prominence_sensitivity/` | Online Supplement Section O1: mode counts at prominence thresholds of 0.5%, 1%, 2% and 5% (Table O1 and `excel/Prominence_Sensitivity.xlsx`). `partials/` holds the per-fit results. |
| `excel/` | Excel workbooks: `QPD_Paper_Tables.xlsx`, Tables 3–13, A1 and O2 in the paper's layout, one sheet per section (Table A2's numbers are the QFlex-T+ columns next to Tables 4 and A1; the one table marked "Not in the paper", fish weights at 1.0 lb jitter, is an extra); `InverseGaussian_vs_JohnsonSL_MC.xlsx`, the inverse Gaussian results next to Johnson SL (Section O4; a supporting comparison, not a paper table; from `mc_invgauss_figures.py`); prominence sensitivity (`Prominence_Sensitivity.xlsx`, Section O1, from `prominence_sensitivity/summarize.py`); the worked Johnson SL/SB validity example for Appendix A3 (`SL_SB_Validity_Example.xlsx`). |
| `companion/` | The marimo notebook behind the online companion (see Online companion below). |

## Setup

Tested with Python 3.10.

```bash
pip install -r requirements.txt
```

Run every analysis script from `reproduction/`. The scripts find `code/common/` and `data/` by
relative path, so keep the folder layout as it is.

## Conventions used throughout

* **QPDs.** The paper's five variants: Metalog, QFlex-U, QFlex-T+, QFlex-TA+ and QFlex-A+.
  The Monte Carlo (Section 4.4, Appendix A4, Online Supplement O2), the Johnson SU bootstrap
  (Section 4.5, Online Supplement O3) and the inverse Gaussian run (O4) fit all five. The bimodal
  (4.6), river gauge (5.1), fish (5.2) and geyser (5.3) analyses fit the QPDs the paper reports
  there. Where the paper reports QFlex-T+ separately (Table A2, Figure O6), it comes from the same
  run as the other four.
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

Figure files are in `reproduction/figures/` (PNG and PDF); scripts in `reproduction/scripts/`; the data behind each figure in `reproduction/tables/`.

| Paper | Content | Script | Figure file | Data behind it (in `reproduction/tables/`) |
|---|---|---|---|---|
| Figure 1 | Johnson SU, SL and SB reference distributions | `reference_distributions_figure.py` | `Figure01_reference_distributions` | `Figure01_reference_distributions_paneldata.csv` |
| Figure 2 | Logit Metalog and Logit QFlex-A+ fits to one Johnson SB sample, K = 4, 7, 10 | `mc_sb_pdf_fit_triptych.py` | `Figure02_SB_example_fits` | `Figure02_SB_example_fits_W1.csv` |
| Figure 3 | Monte Carlo fit performance, Johnson SU, N = 50 | `mc_johnson_figures.py` | `Figure03_MC_JohnsonSU_N50` | `Figure03_MC_JohnsonSU_N50_paneldata.csv` |
| Figure 4 | Monte Carlo fit performance, Johnson SU, N = 200 | `mc_johnson_figures.py` | `Figure04_MC_JohnsonSU_N200` | `Figure04_MC_JohnsonSU_N200_paneldata.csv` |
| Figure 5 | Valid and unimodal fits, N = 200, K = 10 | `mc_valid_unimodal_figure.py` | `Figure05_valid_unimodal_N200_K10` | `Figure05_valid_unimodal_N200_K10_paneldata.csv` |
| Figure 6 | Empirical and fitted Metalog QF with 95% bootstrap interval | `bootstrap_qf_figure.py` | `Figure06_SU_bootstrap_QF_interval` | `Figure06_SU_bootstrap_QF_interval_paneldata.csv`, `…_empiricalqf.csv` |
| Figure 7 | Metalog mode count and dispersion under bootstrap | `bootstrap_figures.py` | `Figure07_Metalog_bootstrap_modes` | `bootstrap_sweep.csv`, `bootstrap_original_fits.csv`, `bootstrap_persist.csv` |
| Figure 8 | QFlex-A+ mode count and dispersion under bootstrap | `bootstrap_figures.py` | `Figure08_QFlexAplus_bootstrap_modes` | `bootstrap_sweep.csv`, `bootstrap_original_fits.csv`, `bootstrap_persist.csv` |
| Figure 9 | Bimodal reference distribution and K = 12 fits | `bimodal_figure.py` | `Figure09_bimodal_reference` | `Figure09_bimodal_paneldata.csv`, `…_sample.csv`, `…_true_qf.csv` |
| Figure 10 | River gauge heights: EQF | `hydrology_figures.py` | `Figure10_hydrology_EQF` | `Figure10_hydrology_EQF_paneldata.csv` |
| Figure 11 | River gauge heights: histogram with GEV and Log Metalog K = 10 | `hydrology_figures.py` | `Figure11_hydrology_histogram_pdfs` | `Figure11_hydrology_histogram_pdfs_paneldata.csv` |
| Figure 12 | River gauge heights: EQF with 95% bootstrap intervals | `hydrology_figures.py` | `Figure12_hydrology_EQF_bootstrap_interval` | `Figure12_hydrology_EQF_bootstrap_interval_paneldata.csv` |
| Figure 13 | River gauge heights: bootstrap mode height vs location | `hydrology_figures.py` | `Figure13_hydrology_mode_scatter` | `Figure13_hydrology_mode_scatter_paneldata.csv` |
| Figure 14 | GEV, Log Metalog-4 and Log QFlex-A+-10 fits to the flood data | `hydrology_figures.py` | `Figure14_hydrology_GEV_Metalog_QFlex` | `Figure14_hydrology_GEV_Metalog_QFlex_paneldata.csv` |
| Figure 15 | Fish weights: histogram with 10-term Log Metalog | `fish_histogram_figure.py` | `Figure15_fish_histogram` | `Figure15_fish_histogram_paneldata.csv` (the Log Metalog curve), `fish_histogram_bins0.5.csv` (the bars) |
| Figure 16 | Fish weights: EQF with 95% bootstrap intervals, 0.5 lb jitter | `fish_eqf_ci_figure.py` | `Figure16_fish_EQF_bootstrap_interval` | `Figure16_fish_EQF_bootstrap_interval_paneldata.csv` |
| Figure 17 | Fish weights: Log Metalog and Log QFlex-U fits, K = 7, 10, 13 | `fish_pdf_order_panels.py` | `Figure17_fish_fits_K7_10_13` | `Figure17_fish_fits_K7_10_13_paneldata.csv` |
| Figure 18 | Old Faithful data: EQF and density | `geyser_data_figure.py` | `Figure18_geyser_data` | `Figure18_geyser_eqf_paneldata.csv`, `Figure18_geyser_kde_paneldata.csv` |
| Figure 19 | Old Faithful: Log Metalog (K = 8) and Log QFlex-TA+ (K = 10) fits | `geyser_latent_fit_figure.py` | `Figure19_geyser_fits` | `Figure19_geyser_fits_paneldata.csv`, `Figure19_geyser_latent_envelope.csv` |
| Figures O1–O4 | Monte Carlo fit performance, Johnson SL and SB, N = 50 and 200 | `mc_johnson_figures.py` | `FigureO1_MC_JohnsonSL_N50` … `FigureO4_MC_JohnsonSB_N200` | `FigureO1_MC_JohnsonSL_N50_paneldata.csv` … `FigureO4_MC_JohnsonSB_N200_paneldata.csv` |
| Figure O5 | QFlex-U mode count and dispersion under bootstrap | `bootstrap_figures.py` | `FigureO5_QFlexU_bootstrap_modes` | `bootstrap_sweep.csv`, `bootstrap_original_fits.csv`, `bootstrap_persist.csv` |
| Figure O6 | QFlex-T+ mode count and dispersion under bootstrap | `bootstrap_figures.py` | `FigureO6_QFlexTplus_bootstrap_modes` | `bootstrap_sweep.csv`, `bootstrap_original_fits.csv`, `bootstrap_persist.csv` |
| Figure O7 | QFlex-TA+ mode count and dispersion under bootstrap | `bootstrap_figures.py` | `FigureO7_QFlexTAplus_bootstrap_modes` | `bootstrap_sweep.csv`, `bootstrap_original_fits.csv`, `bootstrap_persist.csv` |
| Figure O8 | Inverse Gaussian reference distribution | `invgauss_reference_figure.py` | `FigureO8_inverse_gaussian_reference` | — (closed-form densities) |

## Tables

Tables 1 and 2 are analytical (constraint systems and structural mode bounds) and need no code.
All files are in `reproduction/tables/` unless stated otherwise.
Tables 4, A1 and A2 come from one Monte Carlo run of all five QPDs. Table 4 and Table A1 share one
file, `Table04_TableA1_mc_summary_K4_7_10_13.csv`: its N = 200 rows are Table 4 and its N = 50 rows
are Table A1. QFlex-T+ is written to the Table A2 files.

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
| Table A2 | QFlex-T+ fit summary, Monte Carlo, N = 50 and 200 | `mc_tables_rebuild.py`, `mc_mode_height.py` | `TableA2_QFlexTplus_mc_summary_K4_7_10_13.csv`, `TableA2_mode_height_N50.csv`, `TableA2_mode_height_N200.csv` |
| Table O1 | False modality for Johnson SU at three prominence thresholds | `prominence_sensitivity/prominence_sensitivity.py`, `summarize.py` | `reproduction/prominence_sensitivity/TableO1_prominence_JohnsonSU.csv` |
| Table O2 | Inverse Gaussian Monte Carlo fit summary, N = 200 | `mc_invgauss.py`, `invgauss_table_O2.py` | `invgauss/TableO2_invgauss_n200.csv` |

The Hartigan and Silverman results quoted in the text of Section 5.1 are in
`hydrology_modality_tests.csv` (`hydrology_modality_tests.py`). `excel/QPD_Paper_Tables.xlsx`
holds the same tables in the paper's layout (Table O1 is in `excel/Prominence_Sensitivity.xlsx`).

## Other files

Everything not named in the Figures and Tables sections above falls into one of these groups.

**Per-replicate results behind the paper's tables** (in `reproduction/tables/`): `mc_perrep_*`,
`mc_rebuilt_*`, `mc_modality_tests_raw.csv`, `bootstrap_sweep.csv`, `bimodal_mc.csv`,
`hydrology_sweep.csv`, `hydrology_modality_tests.csv` (the tests quoted in Section 5.1),
`fish_summary_perreplicate_jitter0.5.csv`, `fish_sample_size_sweep_perreplicate.csv`,
`fish_modality_paper_replicates_raw.csv`, `geyser_sweep.csv`,
`geyser_modality_paper_replicates_raw.csv`, `Table05_silverman_seed200110_raw.csv`.

**Written by a paper script but not in the paper** (they are regenerated whenever that script runs):

| File (in `reproduction/tables/`) | Written by | What it is |
|---|---|---|
| `bootstrap_valid_unimodal_shifted_modes.csv` | `bootstrap_figures.py` | Valid unimodal Metalog and QFlex-A+ bootstrap fits at K = 10 whose mode is shifted (its figure goes to `figures/other/`) |
| `hydrology_median_bootstrap_fits_paneldata.csv`, `hydrology_median_fit_selection.csv` | `hydrology_figures.py` | Two median-bootstrap-fit figures drawn alongside Figures 10–14 |
| `geyser_mode_dispersion.csv` | `geyser_tables.py` | Mode dispersion for Log Metalog K = 7, 8 and Log QFlex-TA+ K = 10 over the bimodal fits |
| `fish_histogram_bins1.0.csv` | `fish_histogram_figure.py` | The Figure 15 histogram with 1-lb bins |
| `fish_summary_jitter1.0.csv`, `fish_summary_perreplicate_jitter1.0.csv` | `fish_summary.py --jitter 1.0` | Fish results at 1.0 lb jitter; input to the prominence check |
| `geyser_latent_jitter_stability.csv` | `geyser_latent_stability.py` | Checks that Figure 19's conclusion does not depend on which jitter realization was drawn |

**Supporting scripts** (in `reproduction/scripts/`):

* `*_common.py`, `figure_style.py`, `table_rounding.py`: helpers imported by the analysis scripts.
* `fish_pdf_jitter_panels.py`: imported by `fish_pdf_order_panels.py` (Figure 17) for shared
  constants; run on its own it draws a figure that is not in the paper.
* `bootstrap_section.py`: defines the Section 4.5 sample (seed 200110, stored in `reproduction/data/`)
  and the resampling scheme; the bootstrap scripts import it.
* `mc_invgauss_figures.py`: the inverse Gaussian vs Johnson SL comparison behind
  `excel/InverseGaussian_vs_JohnsonSL_MC.xlsx` (Section O4); its figures are not in the paper.
* `export_samples.py`: optional utility that writes every sample the tables were computed from
  to Excel, one workbook per section (outside the repository by default; see `--out`).
* `validation/`: cross-check of our Silverman test (`code/common/stats_tests.py`) against R's
  `multimode::modetest`; steps are in the docstring of `silverman_vs_multimode.py`.

**Library code** (`code/common/`): `metalog/` contains two versions; every script uses version 2
(`metalog_v2.py`, the package default), and `metalog_v1.py` is kept only as an explicitly
requested alternative. `jpse/` is the Johnson SU/SL/SB code from Bickel (2026, *Communications in
Statistics – Theory and Methods*); its `replicate_figure1.py`, `test_johnson.py` and
`IMPLEMENTATION_SUMMARY.md` belong to that package, and "Figure 1" there means Figure 1 of
Bickel (2026), not of this paper.

## How to run

All commands run from `reproduction/`, in the order shown (later steps read the outputs of
earlier ones). The Monte Carlo, bootstrap and fish-jitter steps are the slow ones; `--procs`
sets the number of worker processes.

```bash
cd reproduction

# Tables 4, 6, 7, 8, 10, 11, 13, A1, A2 and Figures 2-17, 19, O1-O7
bash scripts/run_all_framework.sh

python3 scripts/reference_distributions_figure.py           # Figure 1
python3 scripts/geyser_data_figure.py                        # Figure 18
python3 scripts/mc_modality_tests.py                         # Table 3
python3 scripts/bootstrap_hartigan.py                        # Table 5 (Hartigan)
python3 scripts/bootstrap_silverman.py                       # Table 5 (Silverman)
python3 scripts/fish_modality_paper_replicates.py --k 1,2,3  # Table 9
python3 scripts/geyser_modality_paper_replicates.py          # Table 12
python3 scripts/hydrology_modality_tests.py --repeat 3       # tests quoted in Section 5.1

# Online Supplement
python3 scripts/mc_invgauss.py --n 200 --procs 4             # Section O4 (N = 200)
python3 scripts/invgauss_table_O2.py                         # Table O2
python3 scripts/mc_invgauss_figures.py                       # data for excel/InverseGaussian_vs_JohnsonSL_MC.xlsx
python3 scripts/invgauss_reference_figure.py                 # Figure O8
```

Table O1 has its own instructions at the top of
`reproduction/prominence_sensitivity/prominence_sensitivity.py`; `summarize.py` there writes
`TableO1_prominence_JohnsonSU.csv` and `excel/Prominence_Sensitivity.xlsx`, and `check_1pct.py`
confirms that the 1% threshold reproduces the stored per-fit results of every section. Each script's docstring lists its options.

Two optional self-tests, `bootstrap_section.py --verify` and the check in `bimodal_common.py`,
compare against files from an earlier version of the code that are not included here. Nothing
above needs them.

## Online companion

`companion/app.py` is a [marimo](https://marimo.io) notebook. It carries its own copy of the
library code (`companion/common/`) so that it can be bundled for the browser; that copy differs
from `code/common/` only in comments. `companion/public/` holds the paper's three datasets plus
`Returns-Stocks_Bonds_Bills.xlsx` (annual US returns on stocks, bonds, bills, real estate and gold
since 1928), which feeds an extra
exploratory section of the companion that is not part of the paper.

```bash
cd companion
pip install -r requirements.txt
marimo edit app.py                                     # interactive
marimo export html-wasm app.py -o _site --mode run     # the static site deployed to Netlify
```

## Citation

Khanna R, Bickel JE. Bump Hunting, Structural Overfitting, and Quantile-Parameterized
Distributions. Under review. SSRN: <https://ssrn.com/abstract=7578879>.
