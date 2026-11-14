"""Builds notebooks/Network_paper_replication.ipynb: one notebook that runs the whole pipeline of the network paper
in the order of code/run_all.py, explains each step with the script's own description, and shows the main tables
and figures. Each step runs the script itself (%run), so the notebook and the scripts cannot give different results."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": t.strip("\n").splitlines(keepends=True)})


def doc(script):
    d = ast.get_docstring(ast.parse((CODE / script).read_text(encoding="utf8"))) or ""
    return d.split(" - ", 1)[-1]


STEPS = [
    ("01_clean_data.py", "", "Section IV.A; Table 2", ["table1_data_summary"]),
    ("00_geocode_check.py", "", "Section IV.A; Appendix C; Figure 1", []),
    ("verify_original.py", "", "Appendix E", []),
    ("02_network.py", "", "Section V.A; Tables 3–4", ["table_communities"]),
    ("03_rainfall.py", "", "Section IV.A.2", []),
    ("04_indices_cointegration.py", "", "Section V.B; Appendix Tables B1–B2", ["table_unit_roots", "table_engle_granger"]),
    ("04b_johansen_cases.py", "", "Section 4.2 (Johansen deterministic cases)", []),
    ("05_tvecm.py", "", "Section V.C–D; Tables 5–6", ["table_tvecm_main"]),
    ("06_rainfall_moderation.py", "", "Section V.E; Table 7", ["table_rain_moderation"]),
    ("07_fiedler_rolling.py", "", "Section V.E; Appendix Figure B1", ["table_fiedler_rain"]),
    ("08_rain_price_timing.py", "", "Section V.E; Figure 3", ["table_rain_price_timing"]),
    ("09_policy_bounds.py", "", "Section VIII.C; Table 8", ["table_policy_bounds"]),
    ("02_network.py", "--shrinkage 0.15 --tag alpha015", "Section V.A (partition robustness)", []),
    ("02_network.py", "--shrinkage 0.50 --tag alpha050", "Section V.A (partition robustness)", []),
    ("02_network.py", "--pairwise --tag pairwise", "Section V.A (partition robustness)", []),
    ("02b_missing_data_check.py", "", "Section 4.1 (missing-data check)", []),
    ("10_asymmetry_simam_partitions.py", "", "Sections V.C–D; Table 6; Appendix Table B3", ["table_bandtar"]),
    ("13_economic_interpretation.py", "", "Section VII; Appendix Tables B4–B5", ["table_thresholds_distance", "table_production_rain_lp"]),
    ("14_community_iv.py", "", "Section VI.C", []),
    ("15_monte_carlo.py", "", "Section 5.3 (Monte Carlo of the estimators)", ["table_monte_carlo"]),
    ("11_figures.py", "", "Figures 1–3; Appendix Figure B1", []),
    ("12_paper_numbers.py", "", "Every number quoted in the text", []),
]

md("""
# Do Large Price Gaps Close Faster? Spatial Arbitrage, Market Networks and Weather Shocks in India’s Onion Markets
## Replication notebook

This notebook runs every computation in the paper, from the raw AGMARKNET file to the numbers quoted in the text,
in the order of `code/run_all.py`. Each step runs one script of the replication package and shows its output.
Run all cells from top to bottom (about 20–40 minutes on a laptop). Place
`onion_prices_master_2021_2024_corrected.xlsx` in `data/raw/` first.
""")
code("""
import os, pandas as pd
from IPython.display import Image, Markdown, display
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir(os.path.join("..", "code"))
print("working directory:", os.getcwd())
def show(name):
    p = os.path.join("..", "output", "tables", name + ".md")
    if os.path.exists(p):
        display(Markdown(open(p, encoding="utf8").read()))
""")
for k, (script, args, where, tabs) in enumerate(STEPS, 1):
    md(f"### Step {k}. `{script}` {args}\n\n*Feeds:* {where}\n\n" + doc(script).replace("\n", "  \n"))
    code(f"%run {script} {args}".strip() + "".join(f"\nshow('{t}')" for t in tabs))
md("### Figures")
code("""
for f in ["fig1_communities_map.png", "fig2_tvecm_regimes.png", "fig3_rain_local_projections.png", "fig4_rolling_fiedler.png"]:
    display(Image(filename=os.path.join("..", "output", "figures", f), width=700))
""")
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(ROOT / "notebooks" / "Network_paper_replication.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf8")
print("wrote notebook with", len(cells), "cells")
