"""03 - Monthly rainfall anomalies relative to 1991-2020 normals, at mandi grid cells; community exposure.

Observed rainfall. The AGMARKNET file carries IMD RF25 (0.25 deg) daily rainfall at each record's
coordinates on trading dates. Records whose coordinates are out of state or > 50 km from the mandi's
verified location (00_geocode_check.py) are dropped. Each remaining record is assigned to its RF25
grid cell. For each cell and month, observed rainfall = mean daily value over observed dates x days
in month; cell-months with fewer than half of days observed are not used.

Normals. 1991-2020 (WMO/IMD standard 30-year period) monthly mean and SD at each of the same 822
grid cells, computed from the RF25 NetCDF files by notebooks/RF25_normals_1991_2020_mandi_cells.ipynb
(data/raw/rf25_cell_month_normals_1991_2020.xlsx).

Subdivision-month measures average the cell-level quantities over the cells observed that month, so
the observations and the baseline always refer to the same locations:
  rain_mm, normal_mm   mean over cells
  deficit_mm           max(0, normal_mm - rain_mm)
  excess_mm            max(0, rain_mm - normal_mm)
  pdef                 max(0, -(rain_mm/normal_mm - 1))  (IMD-style percentage departure; normal >= 10 mm)
  z, zdef, absz        mean over cells of (R - mean)/SD, its negative part, its absolute value
Robustness baseline that uses no external normals: 2021-24 subdivision-month mean and SD (deficit_ins, zdef_ins).
Previous versions used lpa_normals_2010_2020.csv, which the author's notebook shows was built from 1990-1999
files at one hand-placed grid cell per subdivision; it is no longer used (comparison saved for the log).
Requires 02_network.py (community labels).
"""
import numpy as np
import pandas as pd

from common import DER, LPA_CSV, RAW, save_results

GRID_LAT0, GRID_LON0, STEP = 6.5, 66.5, 0.25
m = pd.read_parquet(DER / "master_clean.parquet")
normals = pd.read_excel(RAW / "rf25_cell_month_normals_1991_2020.xlsx")
normals = normals.rename(columns={"cell_lat": "clat", "cell_lon": "clon"})
assert normals["n_years"].eq(30).all()

raw = pd.read_parquet(DER / "master_raw.parquet")
raw["mandi"] = raw["State"].str.strip() + "|" + raw["Market"].str.strip()
raw["vlat"], raw["vlon"] = raw["Latitude"].round(2), raw["Longitude"].round(2)
flags = pd.read_parquet(DER / "record_location_flags.parquet").drop_duplicates(["mandi", "vlat", "vlon"])
raw = raw.merge(flags[["mandi", "vlat", "vlon", "flag"]], on=["mandi", "vlat", "vlon"], how="left")
n_flag_rain = int(raw["flag"].fillna(True).sum())
use = raw[~raw["flag"].fillna(True)].copy()
use["clat"] = (np.round((use["vlat"] - GRID_LAT0) / STEP) * STEP + GRID_LAT0).round(2)
use["clon"] = (np.round((use["vlon"] - GRID_LON0) / STEP) * STEP + GRID_LON0).round(2)
sub_mode = m.drop_duplicates("mandi").set_index("mandi")["IMD_Subdivision"]
use["IMD_Subdivision"] = use["mandi"].map(sub_mode).fillna(use["IMD_Subdivision"])
# a cell belongs to the subdivision of most of its records
cell_sub = use.groupby(["clat", "clon"])["IMD_Subdivision"].agg(lambda s: s.mode().iloc[0])

# ---- cell-month observed rainfall
day = use.groupby(["clat", "clon", "Date"])["Rainfall_mm"].mean().reset_index()
day["ym"] = day["Date"].dt.to_period("M")
cm = day.groupby(["clat", "clon", "ym"]).agg(days_obs=("Date", "nunique"), mean_daily=("Rainfall_mm", "mean")).reset_index()
cm["dim"] = cm["ym"].dt.days_in_month
cm["coverage"] = cm["days_obs"] / cm["dim"]
cm["rain_mm"] = cm["mean_daily"] * cm["dim"]
cm["month"] = cm["ym"].dt.month
cm = cm.merge(normals, on=["clat", "clon", "month"], how="left")
n_cellmonth_no_normal = int(cm["normal_mean_mm"].isna().sum())
cm = cm[(cm["coverage"] >= 0.5) & cm["normal_mean_mm"].notna()].copy()
sd = cm["normal_sd_mm"].where(cm["normal_sd_mm"] > 0)
cm["z"] = (cm["rain_mm"] - cm["normal_mean_mm"]) / sd
cm["IMD_Subdivision"] = [cell_sub.loc[(a, b)] for a, b in zip(cm["clat"], cm["clon"])]
cm.to_parquet(DER / "rain_cell_month.parquet")

# ---- subdivision-month
mon = cm.groupby(["IMD_Subdivision", "ym"]).agg(rain_mm=("rain_mm", "mean"), normal_mm=("normal_mean_mm", "mean"),
                                                z=("z", "mean"), n_cells=("rain_mm", "size"),
                                                coverage=("coverage", "mean")).reset_index()
mon["month"] = mon["ym"].dt.month
mon["deficit_mm"] = np.maximum(0, mon["normal_mm"] - mon["rain_mm"])
mon["excess_mm"] = np.maximum(0, mon["rain_mm"] - mon["normal_mm"])
mon["pdef"] = np.where(mon["normal_mm"] >= 10, np.maximum(0, 1 - mon["rain_mm"] / mon["normal_mm"]), np.nan)
mon["zdef"] = np.maximum(0, -mon["z"])
mon["absz"] = mon["z"].abs()
clim = mon.groupby(["IMD_Subdivision", "month"])["rain_mm"].agg(["mean", "std"]).rename(columns={"mean": "ins_mean", "std": "ins_sd"})
mon = mon.join(clim, on=["IMD_Subdivision", "month"])
mon["deficit_ins"] = np.maximum(0, mon["ins_mean"] - mon["rain_mm"])
mon["zdef_ins"] = np.maximum(0, -(mon["rain_mm"] - mon["ins_mean"]) / mon["ins_sd"].where(mon["ins_sd"] > 0))
# IMD rainfall categories on monthly departure (normal >= 10 mm)
dep = np.where(mon["normal_mm"] >= 10, mon["rain_mm"] / mon["normal_mm"] - 1, np.nan)
mon["imd_category"] = pd.cut(dep, [-np.inf, -0.59, -0.19, 0.19, 0.59, np.inf],
                             labels=["large deficient", "deficient", "normal", "excess", "large excess"])
mon.to_parquet(DER / "rain_subdiv_month.parquet")

# ---- comparison with the superseded normals file (documentation only)
old = pd.read_csv(LPA_CSV)
old.columns = old.columns.str.strip()
cmp_ = mon.groupby(["IMD_Subdivision", "month"])["normal_mm"].mean().rename("new").reset_index().merge(
    old[["IMD_Subdivision", "month", "lpa_mean_mm"]], on=["IMD_Subdivision", "month"])
old_vs_new = {"corr": float(cmp_[["new", "lpa_mean_mm"]].corr().iloc[0, 1]),
              "median_abs_pct_diff": float((cmp_["lpa_mean_mm"] / cmp_["new"] - 1).abs().replace(np.inf, np.nan).median()),
              "annual_ratio_old_to_new_by_subdiv": (cmp_.groupby("IMD_Subdivision")["lpa_mean_mm"].sum() /
                                                    cmp_.groupby("IMD_Subdivision")["new"].sum()).round(2).to_dict()}

# ---- original (erroneous) summed construction, for documentation only
orig = m.groupby(["IMD_Subdivision", m["Date"].dt.to_period("M")])["Rainfall_mm"].sum().reset_index()
orig.columns = ["IMD_Subdivision", "ym", "rain_sum"]
infl = orig.merge(mon[["IMD_Subdivision", "ym", "rain_mm"]], on=["IMD_Subdivision", "ym"])
orig_infl = float((infl["rain_sum"] / infl["rain_mm"]).replace(np.inf, np.nan).median())

# ---- community exposure (arrival-weighted over member mandis)
cc = pd.read_csv(DER / "centrality_communities.csv", index_col=0)
meta = cc[["IMD_Subdivision", "arrivals_total_MT", "community"]]
vars_ = ["deficit_mm", "excess_mm", "pdef", "z", "zdef", "absz", "rain_mm", "normal_mm", "deficit_ins", "zdef_ins"]
rows = []
for mandi, r in meta.iterrows():
    sub = mon[mon["IMD_Subdivision"] == r["IMD_Subdivision"]][["ym"] + vars_].copy()
    sub["mandi"] = mandi; sub["w"] = r["arrivals_total_MT"]; sub["community"] = int(r["community"])
    rows.append(sub)
mx = pd.concat(rows)
n_mandis_no_rain_subdiv = int(len(set(meta["IMD_Subdivision"]) - set(mon["IMD_Subdivision"])))


def wavg(g):
    out = {}
    for v in vars_:
        ok = g[v].notna()
        out[v] = np.average(g.loc[ok, v], weights=g.loc[ok, "w"]) if ok.any() else np.nan
    return pd.Series(out)


comm_m = mx.groupby(["community", "ym"])[vars_ + ["w"]].apply(wavg).reset_index()
comm_m.to_parquet(DER / "rain_community_month.parquet")
sys_m = mx.groupby("ym")[vars_ + ["w"]].apply(wavg).reset_index()
sys_m.to_parquet(DER / "rain_system_month.parquet")

monsoon = mon["month"].isin([6, 7, 8, 9])
ms = mon[mon["IMD_Subdivision"].isin(meta["IMD_Subdivision"].unique())]
cat = ms["imd_category"].value_counts(normalize=True).round(3).to_dict()
sys_jjas = sys_m[sys_m["ym"].dt.month.isin([6, 7, 8, 9])].groupby(sys_m["ym"].dt.year)[["rain_mm", "normal_mm"]].sum()
out = {
    "normal_period": "1991-2020", "n_cells": int(cm[["clat", "clon"]].drop_duplicates().shape[0]),
    "n_cell_months_used": len(cm), "n_cellmonth_no_normal": n_cellmonth_no_normal,
    "n_subdiv_months": len(mon), "n_subdivisions": int(mon["IMD_Subdivision"].nunique()),
    "mean_cells_per_subdiv_month": float(mon["n_cells"].mean()),
    "n_records_rain_excluded_location": n_flag_rain, "n_sample_subdivs_without_rain": n_mandis_no_rain_subdiv,
    "monsoon_rain_to_normal_ratio": float(mon.loc[monsoon, "rain_mm"].sum() / mon.loc[monsoon, "normal_mm"].sum()),
    "corr_cell_obs_vs_normal": float(cm[["rain_mm", "normal_mean_mm"]].corr().iloc[0, 1]),
    "deficit_share_positive": float((ms["deficit_mm"] > 0).mean()),
    "imd_category_shares": {str(k): v for k, v in cat.items()},
    "system_jjas_departure_by_year": (sys_jjas["rain_mm"] / sys_jjas["normal_mm"] - 1).round(3).to_dict(),
    "community_mean_deficit": comm_m.groupby("community")["deficit_mm"].mean().to_dict(),
    "community_mean_deficit_monsoon": comm_m[comm_m["ym"].dt.month.isin([6, 7, 8, 9])].groupby("community")["deficit_mm"].mean().to_dict(),
    "community_mean_pdef": comm_m.groupby("community")["pdef"].mean().to_dict(),
    "orig_rain_inflation_median": orig_infl,
    "old_normals_vs_new": old_vs_new,
}
save_results("rainfall", out)
for k, v in out.items():
    print(k, v)
