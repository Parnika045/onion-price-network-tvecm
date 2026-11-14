"""04 - Community price indices, unit roots, VAR lag order, Johansen and Engle-Granger tests.

Main index: arrival-weighted mean of log modal prices across the community's mandis that
report on a date (as in the original code; the paper text previously said "equal-weighted").
Robustness index: two-way fixed-effects (time-dummy) index that removes mandi level
differences, so that day-to-day changes in which mandis report do not move the index.
Engle-Granger direction: for pair (i, j) with i < j, C_j is regressed on C_i, i.e. the
community with the smaller arrival share on the one with the larger share (C0 has the
largest share). P-values use MacKinnon residual-based critical values (statsmodels.coint).
"""
import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint, kpss
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from common import DER, VAR_MAXLAGS, save_results, write_table

warnings.filterwarnings("ignore")
panel = pd.read_parquet(DER / "panel240.parquet")
cc = pd.read_csv(DER / "centrality_communities.csv", index_col=0)
panel["community"] = panel["mandi"].map(cc["community"])
panel["w"] = panel["Arrivals_MT"].clip(lower=0.01)
K = int(cc["community"].max()) + 1
cols = [f"C{k}" for k in range(K)]

# ---------------------------------------------------------------- arrival-weighted index
panel["wl"] = panel["w"] * panel["logp"]
g = panel.groupby(["Date", "community"])
idx = (g["wl"].sum() / g["w"].sum()).unstack("community").sort_index()
idx.columns = [f"C{c}" for c in idx.columns]
n_missing_before = int(idx.isna().sum().sum())
idx = idx.ffill(limit=3).dropna()
n_report = g["mandi"].nunique().unstack("community").reindex(idx.index)
n_report.columns = cols


# ---------------------------------------------------------------- fixed-effects index
def fe_index(sub: pd.DataFrame, iters: int = 200) -> pd.Series:
    y = sub["logp"].values; w = sub["w"].values
    di = pd.factorize(sub["Date"])[0]; mi = pd.factorize(sub["mandi"])[0]
    a = np.zeros(mi.max() + 1); d = np.zeros(di.max() + 1)
    for _ in range(iters):
        d_new = np.bincount(di, w * (y - a[mi])) / np.bincount(di, w)
        a = np.bincount(mi, w * (y - d_new[di])) / np.bincount(mi, w)
        a -= np.average(a, weights=np.bincount(mi, w))
        if np.max(np.abs(d_new - d)) < 1e-10:
            d = d_new; break
        d = d_new
    dates = pd.factorize(sub["Date"])[1]
    return pd.Series(d, index=dates).sort_index()


fe = pd.concat({f"C{k}": fe_index(panel[panel["community"] == k]) for k in range(K)}, axis=1).sort_index()
fe = fe.ffill(limit=3).reindex(idx.index).ffill()
idx.to_parquet(DER / "community_index.parquet")
fe.to_parquet(DER / "community_index_fe.parquet")
n_report.to_parquet(DER / "community_nreport.parquet")

# within-community dispersion: average over dates of cross-mandi SD of log price
disp = panel.groupby(["Date", "community"])["logp"].std().groupby("community").mean()
disp_fe = None

# ---------------------------------------------------------------- unit roots
ur = []
for c in cols:
    lv = idx[c]; df_ = lv.diff().dropna()
    a_l = adfuller(lv, maxlag=10, autolag="AIC", regression="ct")
    k_l = kpss(lv, regression="ct", nlags="auto")
    a_d = adfuller(df_, maxlag=10, autolag="AIC", regression="c")
    k_d = kpss(df_, regression="c", nlags="auto")
    i1 = (a_l[1] > 0.05) and (k_l[0] > k_l[3]["5%"]) and (a_d[1] < 0.05)
    ur.append({"series": c, "adf_level": a_l[0], "p_level": a_l[1], "kpss_level": k_l[0], "kpss_cv5_ct": k_l[3]["5%"],
               "adf_diff": a_d[0], "p_diff": a_d[1], "kpss_diff": k_d[0], "kpss_cv5_c": k_d[3]["5%"],
               "kpss_diff_reject": bool(k_d[0] > k_d[3]["5%"]), "I1": bool(i1)})
ur = pd.DataFrame(ur)

# ---------------------------------------------------------------- VAR lag and Johansen
sel = VAR(idx.values).select_order(maxlags=VAR_MAXLAGS)
p_bic = int(sel.selected_orders["bic"]); p_aic = int(sel.selected_orders["aic"])
k_ar_diff = max(1, p_bic - 1)
joh = {}
for det in (0, 1):
    j = coint_johansen(idx.values, det_order=det, k_ar_diff=k_ar_diff)
    rank_tr = int(sum(j.lr1 > j.cvt[:, 1])) if all(j.lr1[:r + 1].min() > j.cvt[r, 1] for r in range(len(j.lr1)) if j.lr1[r] > j.cvt[r, 1]) else None
    # sequential rank: first r at which H0 (rank <= r) is not rejected
    seq_tr = next((r for r in range(len(j.lr1)) if j.lr1[r] <= j.cvt[r, 1]), len(j.lr1))
    seq_me = next((r for r in range(len(j.lr2)) if j.lr2[r] <= j.cvm[r, 1]), len(j.lr2))
    joh[f"det{det}"] = {"trace": j.lr1.tolist(), "trace_cv5": j.cvt[:, 1].tolist(), "maxeig": j.lr2.tolist(),
                        "maxeig_cv5": j.cvm[:, 1].tolist(), "eig": j.eig.tolist(), "rank_trace": seq_tr, "rank_maxeig": seq_me}

# ---------------------------------------------------------------- Engle-Granger pairs
pairs = list(itertools.combinations(cols, 2))
eg_rows = []; ects = {}
for ci, cj in pairs:
    y = idx[cj].values; x = idx[ci].values
    X = np.column_stack([np.ones_like(x), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    t, p, _ = coint(y, x, trend="c", autolag="AIC", maxlag=10)
    t_r, p_r, _ = coint(x, y, trend="c", autolag="AIC", maxlag=10)
    ects[f"{ci}-{cj}"] = pd.Series(e, index=idx.index)
    eg_rows.append({"pair": f"{ci}-{cj}", "alpha": b[0], "beta": b[1], "eg_t": t, "eg_p": p,
                    "eg_t_reverse": t_r, "eg_p_reverse": p_r, "ect_sd": e.std(),
                    "beta_class": "dampening (<0.95)" if b[1] < 0.95 else ("amplifying (>1.05)" if b[1] > 1.05 else "near one-to-one")})
eg = pd.DataFrame(eg_rows)
pd.DataFrame(ects).to_parquet(DER / "ects.parquet")

# FE-index ECTs
ects_fe = {}
for ci, cj in pairs:
    y = fe[cj].values; x = fe[ci].values
    X = np.column_stack([np.ones_like(x), x]); b, *_ = np.linalg.lstsq(X, y, rcond=None)
    ects_fe[f"{ci}-{cj}"] = pd.Series(y - X @ b, index=fe.index)
pd.DataFrame(ects_fe).to_parquet(DER / "ects_fe.parquet")

desc = pd.DataFrame({"mandis": cc.groupby("community").size().values,
                     "mean_log_price": idx.mean().values, "sd_log_price": idx.std().values,
                     "mean_price_rs": np.exp(idx).mean().values,
                     "mean_reporting_mandis_per_day": n_report.mean().values,
                     "within_sd_logp": disp.values,
                     "sd_daily_change_aw": idx.diff().std().values,
                     "sd_daily_change_fe": fe.diff().std().values,
                     "first_order_autocorr_dlogp_aw": [idx[c].diff().autocorr() for c in cols],
                     "first_order_autocorr_dlogp_fe": [fe[c].diff().autocorr() for c in cols]},
                    index=cols).reset_index().rename(columns={"index": "community"})

save_results("cointegration", {
    "T": len(idx), "K": K, "n_index_missing_before_ffill": n_missing_before,
    "index_start": str(idx.index.min().date()), "index_end": str(idx.index.max().date()),
    "unit_roots": ur.to_dict("records"), "all_I1": bool(ur["I1"].all()),
    "var_bic": p_bic, "var_aic": p_aic, "k_ar_diff": k_ar_diff, "johansen": joh,
    "eg": eg.to_dict("records"), "n_eg_coint_5pct": int((eg["eg_p"] < 0.05).sum()),
    "n_eg_coint_5pct_reverse": int((eg["eg_p_reverse"] < 0.05).sum()),
    "eg_t_range": [float(eg["eg_t"].min()), float(eg["eg_t"].max())],
    "beta_counts": eg["beta_class"].value_counts().to_dict(),
    "desc": desc.to_dict("records"),
    "corr_aw_fe_levels": {c: float(np.corrcoef(idx[c], fe[c])[0, 1]) for c in cols},
})
write_table(ur.round(3), "table_unit_roots")
write_table(eg.round(3), "table_engle_granger")
write_table(desc.round(3), "table_community_desc")
print(ur.round(3).to_string()); print("VAR bic", p_bic, "aic", p_aic)
for k, v in joh.items():
    print(k, "rank tr", v["rank_trace"], "rank me", v["rank_maxeig"], np.round(v["trace"], 1), np.round(v["trace_cv5"], 1))
print(eg.round(3).to_string()); print(desc.round(3).to_string())
