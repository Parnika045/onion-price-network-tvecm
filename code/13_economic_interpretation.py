"""13 - Economic interpretation: thresholds in rupees, thresholds and distance, production-belt rainfall.

(a) Implied transaction-cost band in Rs/quintal: (exp(gamma) - 1) x mean price of the pair's two indices.
(b) Spatial arbitrage predicts wider no-trade bands between more distant markets. We regress gamma (and
    theta_2 - theta_1) on log great-circle distance between the arrival-weighted centroids of the two
    communities (verified coordinates), and on the number of direct PMFG edges between them.
    N = 28 pairs that share communities, so inference uses the exact community-permutation test
    (relabelling communities' locations), as for SIMAMs.
(c) Supply-side rainfall: rainfall stress in the Nashik production belt (community C0 exposure) is a
    national supply shock. Monthly local projections of the system price index (arrival-weighted,
    240 mandis) on C0 deficit measures with calendar-month effects and Newey-West (3) errors.
"""
import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import DER, load_results, save_results, write_table

R = load_results()
tv = pd.DataFrame(R["tvecm"]["pairs"])
idx = pd.read_parquet(DER / "community_index.parquet")
cc = pd.read_csv(DER / "centrality_communities.csv", index_col=0)
coords = pd.read_csv(DER / "mandi_coordinates.csv").set_index("mandi")
cc["lat"], cc["lon"] = coords.loc[cc.index, "lat"].values, coords.loc[cc.index, "lon"].values


def hav(a, b, c, d):
    p = np.pi / 180
    x = np.sin((c - a) * p / 2) ** 2 + np.cos(a * p) * np.cos(c * p) * np.sin((d - b) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(x))


cent = cc.groupby("community").apply(lambda g: pd.Series({
    "lat": np.average(g["lat"], weights=g["arrivals_total_MT"]), "lon": np.average(g["lon"], weights=g["arrivals_total_MT"])}))
inter = pd.read_csv(DER / "intercommunity_edges.csv")
edges = inter.groupby("corridor").size()

rows = []
for _, r in tv.iterrows():
    i, j = [int(x[1:]) for x in r["pair"].split("-")]
    pbar = float(np.exp(idx[[f"C{i}", f"C{j}"]]).mean().mean())
    rows.append({"pair": r["pair"], "i": i, "j": j, "gamma": r["gamma"], "gamma_pct": 100 * (np.exp(r["gamma"]) - 1),
                 "gamma_rs": (np.exp(r["gamma"]) - 1) * pbar, "mean_price": pbar,
                 "dist_km": hav(cent.loc[i, "lat"], cent.loc[i, "lon"], cent.loc[j, "lat"], cent.loc[j, "lon"]),
                 "pmfg_edges": int(edges.get(f"C{i}-C{j}", 0)), "dtheta": r["theta_2"] - r["theta_1"],
                 "hl1": r["hl_1"], "hl2": r["hl_2"]})
D = pd.DataFrame(rows)
D["log_dist"] = np.log(D["dist_km"])


def ols_b(y, x):
    return sm.OLS(y, sm.add_constant(x)).fit()


def perm_p(yname, xfun):
    """exact permutation over community labels (8! relabellings of community locations)."""
    obs = ols_b(D[yname], xfun(cent)).params.iloc[1]
    ks = list(cent.index); stats_ = []
    for perm in itertools.permutations(ks):
        c2 = cent.copy(); c2.index = list(perm); c2 = c2.loc[ks]
        stats_.append(ols_b(D[yname], xfun(c2)).params.iloc[1])
    stats_ = np.array(stats_)
    return float(obs), float((np.abs(stats_) >= abs(obs) - 1e-12).mean())


def logd(c):
    return np.log([hav(c.loc[i, "lat"], c.loc[i, "lon"], c.loc[j, "lat"], c.loc[j, "lon"]) for i, j in zip(D["i"], D["j"])])


b_g, p_g = perm_p("gamma", logd)
b_dt, p_dt = perm_p("dtheta", logd)
m_g = ols_b(D["gamma"], D["log_dist"]); m_e = ols_b(D["gamma"], D["pmfg_edges"])
corr_g_d = float(np.corrcoef(D["gamma"], D["log_dist"])[0, 1])
write_table(D.round(3), "table_thresholds_distance")

# ---- (c) production-belt rainfall and the national price
rain = pd.read_parquet(DER / "rain_community_month.parquet")
c0 = rain[rain["community"] == 0].set_index("ym")
panel = pd.read_parquet(DER / "panel240.parquet")
panel["w"] = panel["Arrivals_MT"].clip(lower=0.01); panel["wl"] = panel["w"] * panel["logp"]
g = panel.groupby("Date"); sysp = (g["wl"].sum() / g["w"].sum())
sysm = sysp.groupby(sysp.index.to_period("M")).mean()
lp = []
for x in ("deficit_mm", "pdef"):
    for h in range(0, 7):
        d = pd.DataFrame({"dy": sysm.shift(-h) - sysm.shift(1), "x": c0[x]}).dropna()
        X = pd.concat([sm.add_constant(d[["x"]]),
                       pd.get_dummies(d.index.month, prefix="m", drop_first=True, dtype=float).set_index(d.index)], axis=1)
        m = sm.OLS(d["dy"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        lp.append({"shock": x, "h": h, "beta": m.params["x"], "se": m.bse["x"], "p": m.pvalues["x"], "N": int(m.nobs),
                   "effect_1sd_pct": 100 * m.params["x"] * d["x"].std()})
LP = pd.DataFrame(lp)
write_table(LP.round(4), "table_production_rain_lp")

save_results("econ", {
    "gamma_pct_median": float(D["gamma_pct"].median()), "gamma_pct_min": float(D["gamma_pct"].min()), "gamma_pct_max": float(D["gamma_pct"].max()),
    "gamma_rs_median": float(D["gamma_rs"].median()), "gamma_rs_min": float(D["gamma_rs"].min()), "gamma_rs_max": float(D["gamma_rs"].max()),
    "dist_median_km": float(D["dist_km"].median()), "dist_min_km": float(D["dist_km"].min()), "dist_max_km": float(D["dist_km"].max()),
    "gamma_on_logdist_beta": b_g, "gamma_on_logdist_perm_p": p_g, "gamma_on_logdist_ols_p": float(m_g.pvalues.iloc[1]),
    "corr_gamma_logdist": corr_g_d,
    "dtheta_on_logdist_beta": b_dt, "dtheta_on_logdist_perm_p": p_dt,
    "gamma_on_edges_beta": float(m_e.params.iloc[1]), "gamma_on_edges_p": float(m_e.pvalues.iloc[1]),
    "prod_rain_lp": LP.to_dict("records"),
    "community_centroids": cent.round(2).to_dict("index"),
})
print(D.round(3).to_string())
print("gamma~logdist", b_g, p_g, m_g.pvalues.iloc[1], "corr", corr_g_d, "| dtheta~logdist", b_dt, p_dt, "| gamma~edges", m_e.params.iloc[1], m_e.pvalues.iloc[1])
print(LP.round(4).to_string())
