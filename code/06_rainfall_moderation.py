"""06 - Does rainfall stress widen inter-community price disequilibrium?

Pair-month panel: |ECT_ij| averaged within month (28 pairs x 48 months).
Exposure of a pair: mean of the two communities' arrival-weighted deficits (mm), or the
absolute DIFFERENCE in the two communities' standardised anomalies (a common shock should
not move relative prices; a differential shock should).
Specifications: (1) pair FE (as in the original notebook, HC3);
(2) pair FE + calendar-month FE; (3) pair FE + year-month FE. (2)-(3) cluster by year-month.
"""
import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import DER, save_results, write_table

ects = pd.read_parquet(DER / "ects.parquet")
rain = pd.read_parquet(DER / "rain_community_month.parquet")
rain["ym"] = rain["ym"].astype(str)
cols = sorted({p.split("-")[0] for p in ects.columns} | {p.split("-")[1] for p in ects.columns}, key=lambda s: int(s[1:]))

ym = ects.index.to_period("M").astype(str)
abs_m = ects.abs().groupby(ym).mean()
rows = []
R = rain.set_index(["community", "ym"])
for pair in ects.columns:
    ci, cj = pair.split("-"); i, j = int(ci[1:]), int(cj[1:])
    for m in abs_m.index:
        ri, rj = R.loc[(i, m)], R.loc[(j, m)]
        rows.append({"pair": pair, "ym": m, "abs_ect": abs_m.loc[m, pair],
                     "deficit": (ri["deficit_mm"] + rj["deficit_mm"]) / 2, "deficit_dest": rj["deficit_mm"],
                     "zdef": (ri["zdef"] + rj["zdef"]) / 2, "excess": (ri["excess_mm"] + rj["excess_mm"]) / 2,
                     "absz": (ri["absz"] + rj["absz"]) / 2,
                     "dz_gap": abs(ri["z"] - rj["z"]), "ddef_gap": abs(ri["deficit_mm"] - rj["deficit_mm"]),
                     "deficit_ins": (ri["deficit_ins"] + rj["deficit_ins"]) / 2, "pdef": (ri["pdef"] + rj["pdef"]) / 2, "zdef_ins": (ri["zdef_ins"] + rj["zdef_ins"]) / 2})
P = pd.DataFrame(rows).dropna()
P["month"] = P["ym"].str[-2:].astype(int)
P["kharif"] = P["month"].isin([7, 8, 9])
P.to_parquet(DER / "moderation_panel.parquet")


def fe_reg(df, x, fe="pair", cluster=True):
    D = [pd.get_dummies(df["pair"], prefix="p", drop_first=True, dtype=float)]
    if fe in ("pair+cal", "pair+cal_k"):
        D.append(pd.get_dummies(df["month"], prefix="m", drop_first=True, dtype=float))
    if fe == "pair+ym":
        D.append(pd.get_dummies(df["ym"], prefix="t", drop_first=True, dtype=float))
    X = pd.concat([sm.add_constant(df[[x]].astype(float))] + D, axis=1)
    y = df["abs_ect"].astype(float)
    if cluster:
        m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df["ym"])[0]})
    else:
        m = sm.OLS(y, X).fit(cov_type="HC3")
    sd_x = df[x].std()
    return {"x": x, "fe": fe, "beta": m.params[x], "se": m.bse[x], "t": m.tvalues[x], "p": m.pvalues[x],
            "N": int(m.nobs), "r2": m.rsquared, "beta_per_sd": m.params[x] * sd_x,
            "mean_dep": float(y.mean()), "sd_x": float(sd_x)}


res = []
for x in ["deficit", "deficit_dest", "pdef", "zdef", "excess", "absz", "dz_gap", "ddef_gap", "deficit_ins", "zdef_ins"]:
    res.append(fe_reg(P, x, "pair", cluster=False))
    res.append(fe_reg(P, x, "pair+cal"))
    res.append(fe_reg(P, x, "pair+ym"))
K = P[P["kharif"]]
for x in ["deficit", "pdef", "zdef", "dz_gap"]:
    r = fe_reg(K, x, "pair+ym"); r["fe"] = "pair+ym (Jul-Sep only)"; res.append(r)
res = pd.DataFrame(res)
write_table(res.round(5), "table_rain_moderation")
save_results("moderation", {"N": len(P), "n_pairs": P["pair"].nunique(), "n_months": P["ym"].nunique(),
                            "results": res.to_dict("records"),
                            "deficit_mean": float(P["deficit"].mean()), "deficit_sd": float(P["deficit"].std()),
                            "abs_ect_mean": float(P["abs_ect"].mean())})
print(res.round(5).to_string())
