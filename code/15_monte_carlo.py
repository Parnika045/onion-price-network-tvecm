"""15 - Monte Carlo: what can the estimators of Section 5.3 distinguish? (Online Appendix Table B8)

Each replication simulates 28 independent community-pair gaps at weekly frequency, T = 210 weeks, calibrated to the
weekly community gaps used in 14_community_iv.py: from the autocovariances of the observed gaps, the median AR(1)
coefficient of the persistent gap is 0.825 (speed 0.175 a week), its variance 0.0141, and the variance of transitory
noise 0.0020 (about 12 percent of the observed variance). The observed gap is q_t = g_t + v_t.

True dynamics (a two-regime band model, the model the paper estimates):
    Delta g_t = -theta1 g_{t-1} 1(|g_{t-1}| <= 0.15) - theta2 g_{t-1} 1(|g_{t-1}| > 0.15) + e_t,
with theta1 = 0.175 and theta2 = 0.175 (linear), 0.35 (large gaps close twice as fast), 0.0875 (half as fast:
fragmentation) or 0.03 (large gaps barely close). About a fifth of weeks lie in the outer regime, as in the data.
e_t is Gaussian with the variance that gives var(g) = 0.0141 in the linear case.
Transitory noise v: none, Gaussian, or Student-t with 5 or 3 degrees of freedom, all with variance 0.0020.

Estimators applied to each simulated panel (as in the paper):
  conv : pooled OLS of Delta q_t on q_{t-1} and |q_{t-1}| q_{t-1}; pi > 0 means large gaps close faster;
  iv   : the same with q_{t-3} as threshold variable and (q_{t-3}, |q_{t-3}| q_{t-3}) as instruments
         (14_community_iv.py); inference for both by resampling pairs (199 draws), two-sided 5 percent tests;
  tvecm: per pair, a two-regime threshold model on |q_{t-1}| (threshold searched over its 15th-85th percentiles),
         counting pairs with a faster outer regime, and pairs with a significantly faster or slower outer regime
         (heteroskedasticity-robust t test, 5 percent).
"""
import sys

import numpy as np
import pandas as pd

from common import SEED, save_results, OUT

R = int(sys.argv[1]) if len(sys.argv) > 1 else 300
B = 199; K = 28; T = 210; BURN = 100; GAM = 0.15
RHO = 0.825; THETA = 1 - RHO; VAR_G = 0.0141; VAR_V = 0.0020
SD_E = np.sqrt(VAR_G * (1 - RHO ** 2))
rng = np.random.default_rng(SEED)


def sim_gap(theta2):
    e = rng.normal(0, SD_E, (K, T + BURN)); g = np.zeros((K, T + BURN))
    for t in range(1, T + BURN):
        sp = np.where(np.abs(g[:, t - 1]) <= GAM, THETA, theta2)
        g[:, t] = (1 - sp) * g[:, t - 1] + e[:, t]
    return g[:, BURN:]


def noise(kind):
    if kind == "none":
        return np.zeros((K, T))
    if kind == "gauss":
        return rng.normal(0, np.sqrt(VAR_V), (K, T))
    df = int(kind[1:]); x = rng.standard_t(df, (K, T))
    return x / np.sqrt(df / (df - 2)) * np.sqrt(VAR_V)


def grad(dq, q1, z, w):
    X = np.column_stack([q1, np.abs(z) * q1]); Z = np.column_stack([z, np.abs(z) * z])
    Zw = Z * w[:, None]
    return -np.linalg.solve(Zw.T @ X, Zw.T @ dq)[1]


def test(dq, q1, z, pair):
    est = grad(dq, q1, z, np.ones(len(dq))); dr = np.empty(B)
    for b in range(B):
        wk = np.bincount(rng.integers(0, K, K), minlength=K).astype(float)
        dr[b] = grad(dq, q1, z, wk[pair])
    return est, 2 * min((dr <= 0).mean(), (dr >= 0).mean())


def tvecm_counts(dqm, q1m):
    faster = slower_sig = faster_sig = 0
    for dq, q1 in zip(dqm, q1m):
        a = np.abs(q1); best = None
        for gam in np.quantile(a, np.linspace(0.15, 0.85, 30)):
            o = a > gam
            X = np.column_stack([q1 * ~o, q1 * o])
            b = np.linalg.lstsq(X, dq, rcond=None)[0]
            ssr = ((dq - X @ b) ** 2).sum()
            if best is None or ssr < best[0]:
                best = (ssr, b, X)
        _, b, X = best
        u = dq - X @ b; XtXi = np.linalg.inv(X.T @ X); V = XtXi @ (X.T * u ** 2) @ X @ XtXi
        d = b[1] - b[0]; se = np.sqrt(V[0, 0] + V[1, 1] - 2 * V[0, 1])   # coefficients are minus the speeds
        faster += d < 0; faster_sig += d / se < -1.96; slower_sig += d / se > 1.96
    return faster, faster_sig, slower_sig


SCEN = [(0.175, "none"), (0.175, "gauss"), (0.175, "t5"), (0.175, "t3"),
        (0.35, "none"), (0.35, "t5"),
        (0.0875, "none"), (0.0875, "t5"), (0.03, "none"), (0.03, "t5")]
LABEL = {0.175: "linear", 0.35: "outer twice as fast", 0.0875: "outer half as fast", 0.03: "outer barely adjusts"}
rows = []
for th2, nk in SCEN:
    rec = []
    for r in range(R):
        q = sim_gap(th2) + noise(nk); q = q - q.mean(axis=1, keepdims=True)
        dqm, q1m, q3m = q[:, 3:] - q[:, 2:-1], q[:, 2:-1], q[:, :-3]
        dq, q1, q3 = dqm.ravel(), q1m.ravel(), q3m.ravel(); pair = np.repeat(np.arange(K), dqm.shape[1])
        pc, pp = test(dq, q1, q1, pair); pi_, pip = test(dq, q1, q3, pair)
        f, fs, ss = tvecm_counts(dqm, q1m)
        rec.append((pc, pp < .05 and pc > 0, pp < .05 and pc < 0, pi_, pip < .05 and pi_ > 0, pip < .05 and pi_ < 0,
                    f, fs, ss))
    a = np.array(rec, dtype=float)
    row = {"true_dynamics": LABEL[th2], "theta2": th2, "noise": nk,
           "conv_pi_median": float(np.median(a[:, 0])), "conv_rej_faster": float(a[:, 1].mean()),
           "conv_rej_slower": float(a[:, 2].mean()),
           "iv_pi_median": float(np.median(a[:, 3])), "iv_rej_faster": float(a[:, 4].mean()),
           "iv_rej_slower": float(a[:, 5].mean()),
           "tvecm_outer_faster_median": float(np.median(a[:, 6])), "tvecm_faster_sig_median": float(np.median(a[:, 7])),
           "tvecm_slower_sig_median": float(np.median(a[:, 8])), "tvecm_any_slower_sig_share": float((a[:, 8] > 0).mean())}
    rows.append(row); print(row, flush=True)
pd.DataFrame(rows).to_csv(OUT / "tables" / "table_monte_carlo.csv", index=False)
save_results("monte_carlo", {"R": R, "B": B, "calibration": {"rho": RHO, "var_g": VAR_G, "var_v": VAR_V, "T": T, "K": K,
                                                               "gamma": GAM}, "rows": rows})
