"""07 - Time-varying algebraic connectivity and rainfall.

Rolling 90-trading-day Ledoit-Wolf correlations re-weight the edges of the pooled PMFG
(weight = max(rho_t, 0.001)), and the Fiedler value of the weighted Laplacian is recorded every
7 trading days. Main version uses all 240 mandis; the original notebook used the top-60 mandis by
arrivals (reported for comparison). Monthly averages are regressed on the arrival-weighted system
deficit with Newey-West (3 lags) standard errors, with and without calendar-month dummies.
Note: overlapping 90-day windows mean that each monthly Fiedler value reflects about three months
of price data, so the regression is a reduced-form association, not a causal estimate.
"""
import pickle

import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.covariance import LedoitWolf

from common import DER, NW_LAGS_MONTHLY, ROLL_N, ROLL_STEP, ROLL_WINDOW, save_results, write_table

with open(DER / "network_baseline.pkl", "rb") as fh:
    net = pickle.load(fh)
G = net["G"]
panel = pd.read_parquet(DER / "panel240.parquet")
meta = pd.read_csv(DER / "meta240.csv", index_col=0)
dp = panel.pivot_table(index="Date", columns="mandi", values="dP")


def rolling_fiedler(nodes):
    R = dp[nodes]; R = R.fillna(R.mean())
    H = G.subgraph(nodes)
    edges = list(H.edges()); pos = {n: i for i, n in enumerate(nodes)}
    vals, dates = [], []
    for s in range(0, len(R) - ROLL_WINDOW, ROLL_STEP):
        W = R.iloc[s:s + ROLL_WINDOW].values
        cov = LedoitWolf().fit(W).covariance_; sd = np.sqrt(np.diag(cov)); C = cov / np.outer(sd, sd)
        A = np.zeros((len(nodes), len(nodes)))
        for u, v in edges:
            w = max(C[pos[u], pos[v]], 0.001); A[pos[u], pos[v]] = A[pos[v], pos[u]] = w
        comp = max(nx.connected_components(nx.from_numpy_array(A)), key=len)
        ix = sorted(comp); A = A[np.ix_(ix, ix)]
        L = np.diag(A.sum(1)) - A
        vals.append(np.sort(np.linalg.eigvalsh(L))[1]); dates.append(R.index[s + ROLL_WINDOW // 2])
    return pd.Series(vals, index=pd.DatetimeIndex(dates))


f_all = rolling_fiedler(list(G.nodes()))
top60 = meta.loc[list(G.nodes())].sort_values("arrivals_total_MT", ascending=False).head(ROLL_N).index.tolist()
f_60 = rolling_fiedler(top60)
pd.DataFrame({"fiedler_240": f_all, "fiedler_top60": f_60}).to_csv(DER / "fiedler_rolling.csv")

sysr = pd.read_parquet(DER / "rain_system_month.parquet")
sysr.index = sysr["ym"].dt.to_timestamp()
out, rows = {}, []
for name, f in (("all240", f_all), ("top60", f_60)):
    fm = f.resample("MS").mean().dropna()
    d = pd.concat([fm.rename("f"), sysr[["deficit_mm", "pdef", "zdef", "deficit_ins"]]], axis=1).dropna()
    for x in ("deficit_mm", "pdef", "zdef", "deficit_ins"):
        for cal in (False, True):
            X = sm.add_constant(d[[x]])
            if cal:
                X = pd.concat([X, pd.get_dummies(d.index.month, prefix="m", drop_first=True, dtype=float).set_index(d.index)], axis=1)
            m = sm.OLS(d["f"], X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS_MONTHLY})
            rows.append({"network": name, "x": x, "calendar_fe": cal, "beta": m.params[x], "se": m.bse[x],
                         "p": m.pvalues[x], "N": int(m.nobs), "r2": m.rsquared,
                         "beta_x_sd_over_mean_f": m.params[x] * d[x].std() / d["f"].mean()})
    out[name] = {"mean": float(f.mean()), "min": float(f.min()), "max": float(f.max()), "n_windows": len(f),
                 "n_months": int(len(fm))}
tab = pd.DataFrame(rows)
write_table(tab.round(5), "table_fiedler_rain")
save_results("fiedler_rolling", {"series": out, "regressions": tab.to_dict("records")})
print(out); print(tab.round(5).to_string())
