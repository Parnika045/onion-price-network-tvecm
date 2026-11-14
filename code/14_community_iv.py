"""14 - Noise-robust check of gap-dependent adjustment between community pairs (Section VI.C).

Weekly means of the daily community price indices (04_indices_cointegration.py). For every pair of communities the
gap g_t is demeaned and the pooled model

    Delta g_t = -(theta + pi |q_t|) g_{t-1} + e_t

is estimated over all 28 pairs. Conventional estimator: q = g_{t-1}, OLS. Noise-robust estimator: q = g_{t-k}
(k = 2, 3) as an exogenous threshold variable, instruments (g_{t-k}, |g_{t-k}| g_{t-k}) (Caner and Hansen 2004).
pi > 0 means large gaps close faster. Inference resamples community pairs (999 bootstrap draws).
Speeds by quintile of the gap size are also reported.
"""
import itertools

import numpy as np
import pandas as pd

from common import DER, SEED, save_results


def pooled(df, z="g_l2", w_pair=None, iv=True):
    y = df["dg"].to_numpy(); g1 = df["g_l"].to_numpy()
    q = df[z].to_numpy() if iv else g1
    X = np.column_stack([g1, np.abs(q) * g1])
    Z = np.column_stack([q, np.abs(q) * q]) if iv else X
    w = np.ones(len(y)) if w_pair is None else w_pair[df["pair"].to_numpy()]
    Zw = Z * w[:, None]
    b = np.linalg.solve(Zw.T @ X, Zw.T @ y)
    return -b[0], -b[1]          # theta (speed at zero gap), pi (change in speed per unit |gap|)


def quintile_speeds(df, z, nb=5):
    a = df[z].abs().to_numpy()
    edges = np.quantile(a, np.linspace(0, 1, nb + 1))
    b = np.clip(np.searchsorted(edges, a, side="right") - 1, 0, nb - 1)
    y, g1, q = df["dg"].to_numpy(), df["g_l"].to_numpy(), df[z].to_numpy()
    ols = np.bincount(b, y * g1, nb) / np.bincount(b, g1 * g1, nb)
    iv = np.bincount(b, y * q, nb) / np.bincount(b, g1 * q, nb)
    return -ols, -iv


idx = pd.read_parquet(DER / "community_index.parquet")
cols = [c for c in idx.columns if c.startswith("C")]
Iw = idx[cols].resample("W-FRI").mean().dropna()
rows = []
for a, b in itertools.combinations(cols, 2):
    g = Iw[b] - Iw[a]; g = g - g.mean()
    rows.append(pd.DataFrame({"pair": len(rows), "dg": g.diff(), "g_l": g.shift(1), "g_l2": g.shift(2), "g_l3": g.shift(3)}).dropna())
CW = pd.concat(rows, ignore_index=True)


def boot(z, iv, B=999, seed=SEED):
    est = pooled(CW, z, iv=iv)
    rng = np.random.default_rng(seed)
    K = CW["pair"].nunique(); dr = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, K, K), minlength=K).astype(float)   # resample community pairs
        dr.append(pooled(CW, z, w, iv)[1])
    dr = np.array(dr)
    return {"theta": float(est[0]), "pi": float(est[1]), "pi_se": float(dr.std(ddof=1)),
            "pi_p": float(2 * min((dr <= 0).mean(), (dr >= 0).mean())),
            "half_life_weeks": float(np.log(0.5) / np.log(1 - est[0]))}


out = {"weekly_ols": boot("g_l", False), "weekly_iv2": boot("g_l2", True), "weekly_iv3": boot("g_l3", True),
       "n_pairs": int(CW["pair"].nunique()), "n_obs": int(len(CW))}
out["quintiles_conv"] = quintile_speeds(CW, "g_l")[0].tolist()
out["quintiles_iv"] = quintile_speeds(CW, "g_l2")[1].tolist()
print(out)
save_results("community_iv", out)
