"""01 - Load AGMARKNET master file, clean, and select the 240-mandi analysis sample.

Corrections relative to the original notebooks (v5-v8):
  * Mandis are identified by (State, Market). `Market` alone is not unique: nine market
    names occur in two states (e.g. Sirsa, Siwan, Pratapgarh) and were previously merged
    into a single price series.
  * Records with a modal price above 5x or below 1/5 of the same-day national median are
    dropped (unit errors such as Rs 0.8 or Rs 460,000 per quintal).
  * Duplicate mandi-day records are collapsed (arrivals summed, log modal price
    arrival-weighted).
Outputs: data/derived/master_clean.parquet, data/derived/panel240.parquet, results['data'].
"""
import numpy as np
import pandas as pd

from common import (DER, MASTER_XLSX, MIN_REPORT_DAYS, N_MANDIS, OUTLIER_FACTOR,
                    save_results, write_table)

cache = DER / "master_raw.parquet"
if cache.exists():
    raw = pd.read_parquet(cache)
else:
    raw = pd.read_excel(MASTER_XLSX, engine="calamine")
    raw.to_parquet(cache)

raw["Date"] = pd.to_datetime(raw["Date"])
n_raw = len(raw)
n_names = raw["Market"].nunique()
raw["mandi"] = raw["State"].str.strip() + "|" + raw["Market"].str.strip()
n_keys = raw["mandi"].nunique()
homonyms = raw.groupby("Market")["State"].nunique()
homonyms = sorted(homonyms[homonyms > 1].index.tolist())
n_dates = raw["Date"].nunique()
all_days = pd.date_range(raw["Date"].min(), raw["Date"].max())
missing_dates = sorted(set(all_days) - set(raw["Date"]))

# ------------------------------------------------------------------ outliers
nat_med = raw.groupby("Date")["Modal_Price_Rs_Quintal"].transform("median")
ratio = raw["Modal_Price_Rs_Quintal"] / nat_med
bad = (ratio > OUTLIER_FACTOR) | (ratio < 1 / OUTLIER_FACTOR) | (raw["Modal_Price_Rs_Quintal"] <= 0)
n_outliers = int(bad.sum())
df = raw.loc[~bad].copy()

# ------------------------------------------------------------------ duplicates
n_dup_rows = int(df.duplicated(["mandi", "Date"], keep=False).sum())
n_dup_extra = int(df.duplicated(["mandi", "Date"]).sum())
df["logp"] = np.log(df["Modal_Price_Rs_Quintal"])
df["w"] = df["Arrivals_MT"].clip(lower=0.01)
df["wlogp"] = df["w"] * df["logp"]
agg = df.groupby(["mandi", "Date"], sort=True).agg(
    State=("State", "first"), Market=("Market", "first"), District=("District", "first"),
    IMD_Subdivision=("IMD_Subdivision", "first"), Latitude=("Latitude", "first"),
    Longitude=("Longitude", "first"), Rainfall_mm=("Rainfall_mm", "mean"),
    Arrivals_MT=("Arrivals_MT", "sum"), wlogp=("wlogp", "sum"), w=("w", "sum"),
    Min_Price=("Min_Price_Rs_Quintal", "min"), Max_Price=("Max_Price_Rs_Quintal", "max"),
).reset_index()
agg["logp"] = agg["wlogp"] / agg["w"]
agg["Modal_Price"] = np.exp(agg["logp"])
agg = agg.drop(columns=["wlogp", "w"])
# a mandi keeps one subdivision (modal assignment) so that rainfall exposure is well defined
sub_mode = agg.groupby("mandi")["IMD_Subdivision"].agg(lambda s: s.mode().iloc[0])
agg["IMD_Subdivision"] = agg["mandi"].map(sub_mode)
agg.to_parquet(DER / "master_clean.parquet")

# ------------------------------------------------------------------ sample selection
rep_days = agg.groupby("mandi")["Date"].nunique()
cum_arr = agg.groupby("mandi")["Arrivals_MT"].sum()
eligible = rep_days[rep_days >= MIN_REPORT_DAYS].index
top = cum_arr.loc[eligible].sort_values(ascending=False).head(N_MANDIS).index.tolist()
panel = agg[agg["mandi"].isin(top)].copy()
panel = panel.sort_values(["mandi", "Date"]).reset_index(drop=True)
# daily log return between consecutive reports of the same mandi (as in the original design)
panel["dP"] = panel.groupby("mandi")["logp"].diff()
panel["gap_days"] = panel.groupby("mandi")["Date"].diff().dt.days
panel.to_parquet(DER / "panel240.parquet")

meta = panel.drop_duplicates("mandi").set_index("mandi")[
    ["State", "Market", "District", "IMD_Subdivision", "Latitude", "Longitude"]]
meta["arrivals_total_MT"] = cum_arr.loc[meta.index]
meta["report_days"] = rep_days.loc[meta.index]
meta.to_csv(DER / "meta240.csv")

rd = rep_days.loc[top]
state_counts = meta["State"].value_counts()
nat_daily = panel.groupby("Date")
out = {
    "n_raw_records": n_raw,
    "n_market_names": n_names,
    "n_mandis_state_market": n_keys,
    "homonym_market_names": homonyms,
    "n_trading_dates": n_dates,
    "missing_calendar_dates": [str(d.date()) for d in missing_dates],
    "coverage_raw_share": n_raw / (n_keys * len(all_days)),
    "n_price_outliers_dropped": n_outliers,
    "n_duplicate_rows": n_dup_rows,
    "n_duplicate_extra_rows_collapsed": n_dup_extra,
    "n_clean_mandi_days": len(agg),
    "n_eligible_mandis": int(len(eligible)),
    "n_sample_mandis": len(top),
    "n_sample_obs": len(panel),
    "sample_share_of_ceiling": len(panel) / (len(top) * n_dates),
    "arrival_coverage_share": float(cum_arr.loc[top].sum() / cum_arr.sum()),
    "report_days_min": int(rd.min()), "report_days_max": int(rd.max()),
    "report_days_mean": float(rd.mean()), "report_days_median": float(rd.median()),
    "n_states_sample": int(meta["State"].nunique()),
    "n_states_raw": int(raw["State"].nunique()),
    "n_subdivisions_raw": int(raw["IMD_Subdivision"].nunique()),
    "n_subdivisions_sample": int(meta["IMD_Subdivision"].nunique()),
    "state_counts_sample": state_counts.to_dict(),
    "median_gap_days": float(panel["gap_days"].median()),
    "share_returns_gap_gt3": float((panel["gap_days"] > 3).mean()),
    "mean_modal_price_sample": float(panel["Modal_Price"].mean()),
}
save_results("data", out)

tbl = pd.DataFrame([
    ["Raw AGMARKNET records (Jan 2021-Dec 2024)", f"{n_raw:,}"],
    ["Distinct market names / distinct (state, market) mandis", f"{n_names:,} / {n_keys:,}"],
    ["Trading dates with any record (of 1,461 calendar days)", f"{n_dates:,}"],
    ["Records dropped as price outliers (>5x or <1/5 of daily national median)", f"{n_outliers:,}"],
    ["Duplicate mandi-day records collapsed", f"{n_dup_extra:,}"],
    ["Clean mandi-day observations", f"{len(agg):,}"],
    ["Mandis with >= 200 reporting days", f"{len(eligible):,}"],
    ["Sample mandis (top 240 by cumulative arrivals)", f"{len(top)}"],
    ["Sample mandi-day observations", f"{len(panel):,}"],
    ["Share of cumulative arrivals covered by sample", f"{out['arrival_coverage_share']:.1%}"],
    ["Reporting days per sample mandi: min / median / mean / max",
     f"{rd.min()} / {rd.median():.0f} / {rd.mean():.0f} / {rd.max()}"],
    ["States / IMD subdivisions represented in sample", f"{out['n_states_sample']} / {out['n_subdivisions_sample']}"],
], columns=["Item", "Value"])
write_table(tbl, "table1_data_summary")
print(tbl.to_string(index=False))
print("homonyms:", homonyms)
