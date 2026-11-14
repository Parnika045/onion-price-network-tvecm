"""10 - (a) Band-TAR asymmetry, (b) SIMAM density and regime differences, (c) robustness of the
adjustment pattern to the market partition (alternative network partitions and state indices).

(a) Three regimes on the signed ECT_{t-1}: below (ECT <= g_L < 0), band, above (ECT > g_U > 0).
    g_L and g_U are searched over quantiles of the negative and positive ECT values; each outer
    regime holds >= 10% and the band >= 10% of observations. The criterion is the SSR of the dECT
    equation (same regressors as the TVECM). The original notebook did not force the band to
    contain zero and its upper regime always sat at the 15% trimming bound (219 obs for all pairs).
    H0: theta_below = theta_above, Newey-West (10 lags).
    Sign convention: ECT = C_j - a - b C_i. "Above" = destination community j is expensive
    relative to the long-run relation with i.
(b) Pair outcome: theta_2 - theta_1 (negative = faster adjustment in the outer regime).
    Regressor: mean SIMAM share of the two communities. Inference: exact permutation test that
    permutes community-level SIMAM shares across communities (all K! permutations).
(c) The TVECM (point estimates and HAC regime-equality tests, no bootstrap) is re-estimated on
    indices built from alternative partitions.
"""
import importlib.util
import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from common import DER, NW_LAGS_DAILY, half_life, load_results, nw_cov, save_results, write_table

spec = importlib.util.spec_from_file_location("tv", str(DER.parents[1] / "code" / "05_tvecm.py"))
tv = importlib.util.module_from_spec(spec); spec.loader.exec_module(tv)
res0 = load_results()
idx = tv.idx_aw
pairs = tv.pairs

# ------------------------------------------------------------------ (a) Band-TAR
bt_rows = []
for ci, cj in pairs:
    Y, X, q, e, beta, _ = tv.design(idx, ci, cj, L=tv.L_MAIN)
    y = Y[:, 1] - beta * Y[:, 0]
    T = len(y); best = (np.inf, None)
    neg, pos = np.sort(e[e < 0]), np.sort(e[e > 0])
    gl_grid = np.quantile(neg, np.linspace(0.05, 0.95, 25)); gu_grid = np.quantile(pos, np.linspace(0.05, 0.95, 25))
    for gl in gl_grid:
        for gu in gu_grid:
            lo, hi = e <= gl, e > gu; band = ~(lo | hi)
            if min(lo.sum(), hi.sum(), band.sum()) < 0.10 * T:
                continue
            Z = np.column_stack([X * lo[:, None], X * band[:, None], X * hi[:, None]])
            b, *_ = np.linalg.lstsq(Z, y, rcond=None)
            ssr = np.sum((y - Z @ b) ** 2)
            if ssr < best[0]:
                best = (ssr, (gl, gu))
    gl, gu = best[1]
    lo, hi = e <= gl, e > gu; band = ~(lo | hi)
    Z = np.column_stack([X * lo[:, None], X * band[:, None], X * hi[:, None]])
    b, *_ = np.linalg.lstsq(Z, y, rcond=None); u = y - Z @ b
    V = nw_cov(Z, u, NW_LAGS_DAILY); k = X.shape[1]
    tb, ta = b[1], b[2 * k + 1]
    vd = V[1, 1] + V[2 * k + 1, 2 * k + 1] - 2 * V[1, 2 * k + 1]
    bt_rows.append({"pair": f"{ci}-{cj}", "gamma_L": gl, "gamma_U": gu, "n_below": int(lo.sum()), "n_band": int(band.sum()),
                    "n_above": int(hi.sum()), "theta_below": tb, "theta_band": b[k + 1], "theta_above": ta,
                    "hl_below": half_life(tb), "hl_above": half_life(ta), "diff_above_minus_below": ta - tb,
                    "p_equal": 2 * (1 - stats.norm.cdf(abs((ta - tb) / np.sqrt(vd))))})
bt = pd.DataFrame(bt_rows)
bt_sum = {"n_above_faster": int((bt["theta_above"] < bt["theta_below"]).sum()),
          "n_below_faster": int((bt["theta_below"] < bt["theta_above"]).sum()),
          "n_above_faster_sig": int(((bt["theta_above"] < bt["theta_below"]) & (bt["p_equal"] < 0.05)).sum()),
          "n_below_faster_sig": int(((bt["theta_below"] < bt["theta_above"]) & (bt["p_equal"] < 0.05)).sum()),
          "median_hl_below": float(bt["hl_below"].median()), "median_hl_above": float(bt["hl_above"].median())}
write_table(bt.round(3), "table_bandtar")
print(bt.round(3).to_string()); print(bt_sum)

# ------------------------------------------------------------------ (b) SIMAM
cc = pd.read_csv(DER / "centrality_communities.csv", index_col=0)
share = cc.groupby("community")["SIMAM"].mean()
tm = pd.DataFrame(res0["tvecm"]["pairs"]) if "tvecm" in res0 else pd.read_csv(DER / "tvecm_main.csv")
tm["i"] = tm["pair"].str.split("-").str[0].str[1:].astype(int)
tm["j"] = tm["pair"].str.split("-").str[1].str[1:].astype(int)
tm["dtheta"] = tm["theta_2"] - tm["theta_1"]


def simam_beta(sh):
    x = (sh[tm["i"]].values + sh[tm["j"]].values) / 2
    X = sm.add_constant(x)
    return sm.OLS(tm["dtheta"].values, X).fit().params[1], x


b_obs, xs = simam_beta(share)
perm = [simam_beta(pd.Series(p, index=share.index))[0] for p in itertools.permutations(share.values)]
perm = np.array(perm)
p_perm = float((np.abs(perm) >= abs(b_obs) - 1e-12).mean())
ols = sm.OLS(tm["dtheta"].values, sm.add_constant(xs)).fit()
simam = {"beta": float(b_obs), "ols_p": float(ols.pvalues[1]), "perm_p_exact": p_perm, "n_perm": len(perm), "N_pairs": len(tm),
         "community_simam_share": share.to_dict()}
print("SIMAM", simam)

# ------------------------------------------------------------------ (c) partitions
panel = pd.read_parquet(DER / "panel240.parquet")
panel["w"] = panel["Arrivals_MT"].clip(lower=0.01); panel["wl"] = panel["w"] * panel["logp"]


def group_index(labels: pd.Series):
    p = panel.assign(grp=panel["mandi"].map(labels)).dropna(subset=["grp"])
    arr = p.groupby("grp")["Arrivals_MT"].sum().sort_values(ascending=False)
    rename = {g: f"G{k}" for k, g in enumerate(arr.index)}
    p["grp"] = p["grp"].map(rename)
    gg = p.groupby(["Date", "grp"])
    ix = (gg["wl"].sum() / gg["w"].sum()).unstack("grp")
    ix = ix[[f"G{k}" for k in range(len(arr))]]
    return ix.ffill(limit=3).dropna(), {v: k for k, v in rename.items()}


def run_partition(ix):
    rows = []
    for ci, cj in itertools.combinations(ix.columns, 2):
        Y, X, q, e, beta, _ = tv.design(ix, ci, cj, L=tv.L_MAIN)
        g, *_ = tv.grid_search(Y, X, q)
        r = tv.fit_regimes(Y, X, q, g, e, beta); r["pair"] = f"{ci}-{cj}"; r["gamma_at_grid_edge"] = False
        rows.append(r)
    return pd.DataFrame(rows)


part = {}
for tag in ("alpha015", "alpha050", "pairwise"):
    f = DER / f"communities_{tag}.csv"
    if not f.exists():
        print("missing", f); continue
    lab = pd.read_csv(f, index_col=0)["community"]
    ix, _ = group_index(lab)
    df = run_partition(ix)
    part[tag] = {**tv.summarise(df), "K": ix.shape[1], "T": len(ix)}
    print(tag, part[tag])
meta = pd.read_csv(DER / "meta240.csv", index_col=0)
big = meta["State"].value_counts(); big = big[big >= 9].index
lab = meta.loc[meta["State"].isin(big), "State"]
ix, names = group_index(lab)
df = run_partition(ix)
part["states"] = {**tv.summarise(df), "K": ix.shape[1], "T": len(ix), "groups": names}
print("states", part["states"])

save_results("asymmetry_simam_partitions", {"bandtar": bt.to_dict("records"), "bandtar_summary": bt_sum,
                                            "simam": simam, "partitions": part})
