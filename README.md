# Replication package: Network Structure and Nonlinear Price Adjustment in India's Wholesale Onion Market

Author: Parnika Sabharwal (Dr. B. R. Ambedkar School of Economics University, Bengaluru)

## Overview
The code builds the analysis sample from AGMARKNET daily onion records for the 240 mandis with the largest cumulative arrivals, 2021–2024. It then produces:

1. A planar maximally filtered graph (PMFG) of daily price co-movement, with centrality, communities, algebraic connectivity, a permutation benchmark and alternative correlation estimators.
2. Daily community price indices, unit-root and cointegration tests.
3. Two-regime threshold VECMs for all 28 community pairs, with bootstrap threshold tests, Newey–West tests of equal regime speeds, band-TAR models and robustness variants.
4. A noise-robust (instrumental-variable) check of gap-dependent adjustment between community pairs.
5. Rainfall deficits against 1991–2020 normals at the mandis' own grid cells: their effect on price gaps, on connectivity and on price levels.
6. A price-trigger accounting of buffer-stock releases.

Every table and figure in the paper is produced by the scripts below. Every number quoted in the text is read from `output/paper_numbers.json`, which `code/12_paper_numbers.py` writes from `output/results.json`.

## Data availability and provenance
| File | Source | Provided |
|---|---|---|
| `data/raw/onion_prices_master_2021_2024_corrected.xlsx` | Daily onion modal/min/max prices (Rs/quintal) and arrivals (t) by APMC, Jan 2021–Dec 2024, compiled from AGMARKNET (agmarknet.gov.in, Directorate of Marketing & Inspection, Government of India). Carries IMD 0.25° gridded daily rainfall (RF25) at each record's location and the IMD subdivision. | Included (60 MB) |
| `data/raw/geonames_cities1000_india.csv` | The 3,435 Indian places (country code IN) in the GeoNames list of places with population ≥ 1,000 (CC-BY 4.0), as bundled in `reverse_geocoder` 1.5.1; used to verify mandi coordinates against place and district names | Included |
| `data/raw/rf25_cell_month_normals_1991_2020.xlsx` | 1991–2020 monthly rainfall normals at the 822 RF25 grid cells of verified mandi locations, computed from IMD RF25 daily NetCDF files (imdpune.gov.in) by `notebooks/RF25_normals_1991_2020_mandi_cells.ipynb` | Included. The NetCDF inputs (~25 MB/year) are public; download them from IMD to re-run that notebook. |
| `data/raw/lpa_normals_2010_2020.csv` | Superseded subdivision normals, used only for a documented comparison | Included |

AGMARKNET and IMD data are public and may be redistributed with attribution. GeoNames data are CC-BY 4.0.

## Computational requirements
- Python 3.11+ (the results were produced with Python 3.13), with the packages in `requirements.txt` (pinned versions).
- Run time: about 20 minutes on 2 cores and 8 GB of RAM. The log of a complete run from scratch is in `output/logs/run_all_verification.log`.
- Random seeds are fixed (`SEED = 42` in `code/common.py`; Louvain seeds 42, 123 and 999).

## Instructions
```
pip install -r requirements.txt
cd code
python run_all.py                  # all results, tables (output/tables) and figures (output/figures)
```
**Alternative: one notebook.** `notebooks/Network_paper_replication.ipynb` runs the same steps in Jupyter, shows each step's purpose and the main tables and figures. Open it from the `notebooks/` folder and run all cells. It runs on Windows, macOS and Linux.

## Tables and figures
| Exhibit | Script(s) |
|---|---|
| Table 2 (data and sample), Appendix C (locations) | `01_clean_data.py`, `00_geocode_check.py` |
| Table 2, Figure 1, Online Appendix Tables A1–A2 (communities and network); missing-data check in Section 4.1 | `02_network.py` (with the `--shrinkage` and `--pairwise` variants), `02b_missing_data_check.py`, `11_figures.py` |
| Appendix E (reproduction of the uncleaned-data network) | `verify_original.py` |
| Appendix Tables B1–B2 (cointegration, unit roots) | `04_indices_cointegration.py` |
| Table 5, Figure 2 (threshold VECMs) | `05_tvecm.py`, `11_figures.py` |
| Table 6 (robustness), Appendix Table B3 (band-TAR), SIMAM test | `05_tvecm.py`, `10_asymmetry_simam_partitions.py` |
| Section VI.C (noise-robust check) | `14_community_iv.py` |
| Table 7, Figure 3, Appendix Figure B1 (rainfall) | `03_rainfall.py`, `06_rainfall_moderation.py`, `07_fiedler_rolling.py`, `08_rain_price_timing.py`, `11_figures.py` |
| Table 8 (price-trigger accounting) | `09_policy_bounds.py` |
| Appendix Tables B4–B5 (bands and distance; production-belt rainfall) | `13_economic_interpretation.py` |
| Every number in the text | `12_paper_numbers.py` → `output/paper_numbers.json` |

## Scripts
| Script | What it does |
|---|---|
| `common.py` | Paths, seeds and shared helpers |
| `01_clean_data.py` | Mandi identifiers by (state, market), outlier and duplicate cleaning, sample of 240 mandis |
| `00_geocode_check.py` | Verifies mandi locations against GeoNames; flags mis-located records for the rainfall analysis |
| `verify_original.py` | Re-runs the network stage on the uncleaned file |
| `02_network.py` | Ledoit–Wolf correlations, PMFG, centrality, Louvain communities, Fiedler value and permutation benchmark |
| `02b_missing_data_check.py` | Pairwise-complete (no imputation) and one-day-change correlations: PMFG centrality compared with the baseline |
| `03_rainfall.py` | Rainfall, deficits and anomalies relative to 1991–2020 normals at mandi grid cells |
| `04_indices_cointegration.py` | Community price indices; ADF/KPSS, Johansen and Engle–Granger tests |
| `04b_johansen_cases.py` | Johansen trace statistics under restricted/unrestricted constant and trend |
| `05_tvecm.py` | Threshold VECMs, sup-LR bootstrap, tests of equal regime speeds, robustness variants |
| `06_rainfall_moderation.py` | Rainfall shocks and inter-community price gaps (pair-month panel) |
| `07_fiedler_rolling.py` | Rolling algebraic connectivity and rainfall |
| `08_rain_price_timing.py` | Local projections of community price levels on rainfall deficits |
| `09_policy_bounds.py` | Price-trigger accounting |
| `10_asymmetry_simam_partitions.py` | Band-TAR, SIMAM test, alternative partitions and state indices |
| `13_economic_interpretation.py` | Bands in rupees and distance; production-belt rainfall and the national price level |
| `14_community_iv.py` | Conventional and instrumented gap-dependence across community pairs |
| `15_monte_carlo.py` | Monte Carlo: conventional and instrumented estimators under linear, band and fragmentation dynamics with transitory noise (about 15 minutes) |
| `11_figures.py` | Figures 1–3 and Appendix Figure B1 |
| `12_paper_numbers.py` | Formats every number and table in the paper |
