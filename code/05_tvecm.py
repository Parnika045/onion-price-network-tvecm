"""05 - Two-regime threshold VECM for every community pair, bootstrap threshold tests,
regime-equality tests, Band-TAR asymmetry, and robustness variants.

Specification (Hansen and Seo 2002, pairwise, symmetric threshold on |ECT|):
  Y_t = [dC_i,t, dC_j,t]'
  Y_t = A_k' X_t + e_t,   k = 1 if |ECT_{t-1}| <= gamma (inner regime), k = 2 otherwise (outer regime)
  X_t = [1, ECT_{t-1}, dC_i,t-1..t-L, dC_j,t-1..t-L, 11 month dummies],  L = VAR(BIC) order - 1 = 2
gamma minimises log det Sigma(gamma) over 60 points between the 15th and 85th percentiles of |ECT|.
Adjustment of the disequilibrium itself: dECT_t = dC_j - beta dC_i, so its coefficient on ECT_{t-1}
in regime k is theta_k = rho_j,k - beta * rho_i,k (identical to OLS of dECT on X_t within regime k).
Inference: sup-LR threshold test with fixed-regressor residual bootstrap (B = 999); H0: theta_1 =
theta_2 by Newey-West (10 lags) Wald test conditional on gamma-hat.
Usage: python 05_tvecm.py            (main + robustness)
"""
import itertools
import time

import numpy as np
import pandas as pd

from common import (B_BOOT, DER, N_GRID, NW_LAGS_DAILY, SEED, TRIM, half_life, load_results,
                    month_dummies, nw_cov, save_results, write_table)
from scipy import stats

res0 = load_results()
L_MAIN = res0["cointegration"]["k_ar_diff"]
idx_aw = pd.read_parquet(DER / "community_index.parquet")
idx_fe = pd.read_parquet(DER / "community_index_fe.parquet")
cols = list(idx_aw.columns)
pairs = list(itertools.combinations(cols, 2))
eg = {r["pair"]: r for r in res0["cointegration"]["eg"]}


def design(idx, ci, cj, L=2, reverse=False, contemporaneous=False, beta_ab=None):
    """Build Y (T x 2), X (T x k), threshold var q, ECT_{t-1}, beta."""
    yi, yj = idx[ci], idx[cj]
    if reverse:
        yi, yj = yj, yi
    if beta_ab is None:
        Xc = np.column_stack([np.ones(len(yi)), yi.values])
        b, *_ = np.linalg.lstsq(Xc, yj.values, rcond=None)
    else:
        b = beta_ab
    ect = pd.Series(yj.values - b[0] - b[1] * yi.values, index=idx.index)
    d = pd.DataFrame({"dCi": yi.diff(), "dCj": yj.diff(), "ect_l": ect.shift(1)}, index=idx.index)
    regs = ["ect_l"]
    if contemporaneous:
        regs.append("dCi")
    for l in range(1, L + 1):
        d[f"dCi_l{l}"] = d["dCi"].shift(l); d[f"dCj_l{l}"] = d["dCj"].shift(l)
        regs += [f"dCi_l{l}", f"dCj_l{l}"]
    d = d.dropna()
    X = np.column_stack([np.ones(len(d)), d[regs].values, month_dummies(d.index)])
    Y = d[["dCi", "dCj"]].values
    if contemporaneous:  # original single-equation design: only dC_j is modelled
        Y = d[["dCj"]].values
    return Y, X, d["ect_l"].abs().values, d["ect_l"].values, float(b[1]), d.index


def grid_search(Y, X, q, trim=TRIM, n_grid=N_GRID, crit="logdet"):
    """Return best gamma, objective, and the grid. Uses cumulative cross-products on sorted q."""
    T = len(q)
    order = np.argsort(q, kind="stable")
    qs, Xs, Ys = q[order], X[order], Y[order]
    grid = np.linspace(np.quantile(q, trim), np.quantile(q, 1 - trim), n_grid)
    cXX = np.cumsum(np.einsum("ti,tj->tij", Xs, Xs), axis=0)
    cXY = np.cumsum(np.einsum("ti,tj->tij", Xs, Ys), axis=0)
    cYY = np.cumsum(np.einsum("ti,tj->tij", Ys, Ys), axis=0)
    tXX, tXY, tYY = cXX[-1], cXY[-1], cYY[-1]
    best = (np.inf, None)
    objs = []
    for g in grid:
        n1 = int(np.searchsorted(qs, g, side="right"))
        if n1 < X.shape[1] + 5 or T - n1 < X.shape[1] + 5:
            objs.append(np.nan); continue
        S = np.zeros((Y.shape[1], Y.shape[1]))
        for XX, XY, YY in ((cXX[n1 - 1], cXY[n1 - 1], cYY[n1 - 1]), (tXX - cXX[n1 - 1], tXY - cXY[n1 - 1], tYY - cYY[n1 - 1])):
            B = np.linalg.solve(XX + 1e-12 * np.eye(len(XX)), XY)
            S += YY - XY.T @ B
        obj = np.linalg.slogdet(S / T)[1] if crit == "logdet" else np.trace(S)
        objs.append(obj)
        if obj < best[0]:
            best = (obj, g)
    return best[1], best[0], grid, np.array(objs)


def linear_obj(Y, X, crit="logdet"):
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    E = Y - X @ B
    S = E.T @ E
    return (np.linalg.slogdet(S / len(Y))[1] if crit == "logdet" else np.trace(S)), B, E


def fit_regimes(Y, X, q, g, ect_l, beta, nw=NW_LAGS_DAILY):
    """Regime coefficients, theta, HAC test of theta1 = theta2 (and rho_j1 = rho_j2)."""
    n_mask = q <= g
    out = {"gamma": g, "n_inner": int(n_mask.sum()), "n_outer": int((~n_mask).sum())}
    # interacted single-equation regressions (equivalent to regime-split OLS)
    Z = np.column_stack([X * n_mask[:, None], X * (~n_mask)[:, None]])
    k = X.shape[1]
    dect = Y[:, -1] - beta * Y[:, 0] if Y.shape[1] == 2 else None
    targets = {"theta": dect, "rho_j": Y[:, -1]}
    if Y.shape[1] == 2:
        targets["rho_i"] = Y[:, 0]
    for name, y in targets.items():
        if y is None:
            continue
        b, u = np.linalg.lstsq(Z, y, rcond=None)[0], None
        u = y - Z @ b
        V = nw_cov(Z, u, nw)
        c1, c2 = b[1], b[k + 1]
        var_d = V[1, 1] + V[k + 1, k + 1] - 2 * V[1, k + 1]
        out[f"{name}_1"] = float(c1); out[f"{name}_2"] = float(c2)
        out[f"{name}_1_se"] = float(np.sqrt(V[1, 1])); out[f"{name}_2_se"] = float(np.sqrt(V[k + 1, k + 1]))
        out[f"{name}_diff_t"] = float((c2 - c1) / np.sqrt(var_d))
        out[f"{name}_diff_p"] = float(2 * (1 - stats.norm.cdf(abs((c2 - c1) / np.sqrt(var_d)))))
    if "theta_1" not in out:  # single-equation original design: theta not defined; use rho_j
        out["theta_1"], out["theta_2"] = out["rho_j_1"], out["rho_j_2"]
        out["theta_diff_t"], out["theta_diff_p"] = out["rho_j_diff_t"], out["rho_j_diff_p"]
        out["theta_1_se"], out["theta_2_se"] = out["rho_j_1_se"], out["rho_j_2_se"]
    out["hl_1"] = half_life(out["theta_1"]); out["hl_2"] = half_life(out["theta_2"])
    out["outer_faster"] = bool(out["theta_2"] < out["theta_1"])
    sig = out["theta_diff_p"] < 0.05
    out["verdict"] = ("outer faster*" if (sig and out["outer_faster"]) else
                      "outer slower*" if (sig and not out["outer_faster"]) else
                      "outer faster (n.s.)" if out["outer_faster"] else "outer slower (n.s.)")
    return out


def bootstrap_supLR(Y, X, q, B, seed, crit="logdet", trim=TRIM, n_grid=N_GRID):
    obj_lin, Bl, E = linear_obj(Y, X, crit)
    _, obj_thr, _, _ = grid_search(Y, X, q, trim, N_GRID, crit)
    T = len(Y)
    stat = T * (obj_lin - obj_thr) if crit == "logdet" else T * (obj_lin - obj_thr) / obj_thr
    rng = np.random.default_rng(seed)
    fitted = X @ Bl
    boots = np.empty(B)
    for b in range(B):
        Ys = fitted + E[rng.integers(0, T, T)]
        ol, _, _ = linear_obj(Ys, X, crit)
        _, ot, _, _ = grid_search(Ys, X, q, trim, n_grid, crit)
        boots[b] = T * (ol - ot) if crit == "logdet" else T * (ol - ot) / ot
    return float(stat), float((boots >= stat).mean())


def run(idx, label, L=L_MAIN, trim=TRIM, reverse=False, contemporaneous=False, boot=False, nw=NW_LAGS_DAILY):
    rows = []
    for n, (ci, cj) in enumerate(pairs):
        Y, X, q, ect_l, beta, _ = design(idx, ci, cj, L=L, reverse=reverse, contemporaneous=contemporaneous)
        crit = "ssr" if contemporaneous else "logdet"
        g, obj, grid, objs = grid_search(Y, X, q, trim, N_GRID, crit)
        r = fit_regimes(Y, X, q, g, ect_l, beta, nw)
        r.update({"pair": f"{ci}-{cj}", "beta": beta, "T": len(Y), "share_outer": r["n_outer"] / len(Y),
                  "gamma_at_grid_edge": bool(np.isclose(g, grid[0]) or np.isclose(g, grid[-1]))})
        if boot:
            t0 = time.time()
            stat, p = bootstrap_supLR(Y, X, q, B_BOOT, SEED + n, crit)
            r.update({"supLR": stat, "p_boot": p})
            print(f"  {label} {ci}-{cj}: gamma={g:.3f} th1={r['theta_1']:.3f} th2={r['theta_2']:.3f} "
                  f"pdiff={r['theta_diff_p']:.3f} supLR={stat:.1f} p={p:.3f} ({time.time()-t0:.0f}s)", flush=True)
        rows.append(r)
    df = pd.DataFrame(rows)
    return df


def summarise(df):
    s = {"n_pairs": len(df), "n_outer_faster": int(df["outer_faster"].sum()),
         "n_outer_faster_sig": int((df["verdict"] == "outer faster*").sum()),
         "n_outer_slower_sig": int((df["verdict"] == "outer slower*").sum()),
         "n_outer_slower": int((~df["outer_faster"]).sum()),
         "slower_pairs": df.loc[~df["outer_faster"], "pair"].tolist(),
         "slower_sig_pairs": df.loc[df["verdict"] == "outer slower*", "pair"].tolist(),
         "median_hl1": float(df["hl_1"].replace(np.inf, np.nan).median()),
         "median_hl2": float(df["hl_2"].replace(np.inf, np.nan).median()),
         "median_theta1": float(df["theta_1"].median()), "median_theta2": float(df["theta_2"].median()),
         "n_theta_below_minus1": int(((df["theta_1"] < -1) | (df["theta_2"] < -1)).sum()),
         "n_gamma_edge": int(df["gamma_at_grid_edge"].sum())}
    if "p_boot" in df:
        p = df["p_boot"].values
        s["n_thresh_sig05"] = int((p < 0.05).sum())
        s["n_thresh_sig_bonf"] = int((p < 0.05 / len(p)).sum())
        # Holm
        order = np.argsort(p); m = len(p); rej = np.zeros(m, bool)
        for r_, i in enumerate(order):
            if p[i] <= 0.05 / (m - r_):
                rej[i] = True
            else:
                break
        s["n_thresh_sig_holm"] = int(rej.sum())
    return s


if __name__ == "__main__":
    t0 = time.time()
    main = run(idx_aw, "main", boot=True)
    main.to_csv(DER / "tvecm_main.csv", index=False)
    s_main = summarise(main)
    print(s_main, f"{time.time()-t0:.0f}s")

    variants = {
        "fe_index": dict(idx=idx_fe),
        "lags4": dict(idx=idx_aw, L=4),
        "trim10": dict(idx=idx_aw, trim=0.10),
        "reverse": dict(idx=idx_aw, reverse=True),
        "original_spec": dict(idx=idx_aw, L=0, contemporaneous=True),
    }
    rob = {}
    for name, kw in variants.items():
        df = run(label=name, **kw)
        df.to_csv(DER / f"tvecm_{name}.csv", index=False)
        rob[name] = {**summarise(df), "verdicts": dict(zip(df["pair"], df["verdict"]))}
        print(name, rob[name])
    # weekly: mean of daily index within Friday-ending weeks
    wk = idx_aw.resample("W-FRI").mean().dropna()
    dfw = run(wk, "weekly", L=1, nw=4)
    dfw.to_csv(DER / "tvecm_weekly.csv", index=False)
    rob["weekly_mean"] = {**summarise(dfw), "verdicts": dict(zip(dfw["pair"], dfw["verdict"])), "T": int(len(wk))}
    wk_last = idx_aw.resample("W-FRI").last().dropna()
    dfwl = run(wk_last, "weekly_last", L=1, nw=4)
    rob["weekly_last"] = {**summarise(dfwl), "verdicts": dict(zip(dfwl["pair"], dfwl["verdict"]))}
    print("weekly", rob["weekly_mean"])

    save_results("tvecm", {"L": L_MAIN, "summary": s_main, "pairs": main.to_dict("records"), "robustness": rob,
                           "seconds": time.time() - t0})
    cols_out = ["pair", "gamma", "share_outer", "theta_1", "theta_1_se", "theta_2", "theta_2_se", "hl_1", "hl_2",
                "theta_diff_p", "supLR", "p_boot", "verdict"]
    write_table(main[cols_out].round(3), "table_tvecm_main")
