"""04b - Johansen trace statistics under the four standard deterministic cases (Section 4.2).

statsmodels' coint_johansen covers an unrestricted constant (det_order=0) and an unrestricted trend (det_order=1).
This script computes the reduced-rank regression directly so that the restricted-constant and restricted-trend
cases can be compared as well. 5 percent critical values for the last rank (r <= K-1) are those of MacKinnon, Haug
and Michelis (1999): 3.84 (unrestricted constant or trend), 9.16 (restricted constant), 12.52 (restricted trend).
"""
import json
import numpy as np
import pandas as pd
from common import DER, OUT, save_results

idx = pd.read_parquet(DER / "community_index.parquet")
k = json.load(open(OUT / "results.json"))["cointegration"]["k_ar_diff"]
Y = idx.values; T0, n = Y.shape
dY = np.diff(Y, axis=0); t = np.arange(T0, dtype=float)
rows = range(k, T0 - 1)
Z0 = np.array([dY[i] for i in rows])
lag = np.array([np.concatenate([dY[i - j] for j in range(1, k + 1)]) for i in rows])
Ylag = np.array([Y[i] for i in rows]); tt = t[list(rows)][:, None]; one = np.ones((len(Z0), 1)); T = len(Z0)


def trace(Z1, Z2):
    res = lambda A: A - Z2 @ np.linalg.lstsq(Z2, A, rcond=None)[0]
    R0, R1 = res(Z0), res(Z1)
    S00, S11, S01 = R0.T @ R0 / T, R1.T @ R1 / T, R0.T @ R1 / T
    ev = np.sort(np.real(np.linalg.eigvals(np.linalg.solve(S11, S01.T @ np.linalg.solve(S00, S01)))))[::-1][:n]
    return [float(-T * np.log(1 - ev[r:]).sum()) for r in range(n)]


CV_LAST = {"restricted_constant": 9.16, "unrestricted_constant": 3.84, "restricted_trend": 12.52, "unrestricted_trend": 3.84}
cases = {"restricted_constant": (np.hstack([Ylag, one]), lag),
         "unrestricted_constant": (Ylag, np.hstack([lag, one])),
         "restricted_trend": (np.hstack([Ylag, tt]), np.hstack([lag, one])),
         "unrestricted_trend": (Ylag, np.hstack([lag, one, tt]))}
out = {}
for name, (Z1, Z2) in cases.items():
    tr = trace(Z1, Z2)
    out[name] = {"trace": tr, "trace_last": tr[-1], "cv5_last": CV_LAST[name], "reject_last": tr[-1] > CV_LAST[name]}
    print(f"{name:22s} trace(r<={n-1}) = {tr[-1]:.2f}  cv5 = {CV_LAST[name]:.2f}  reject = {tr[-1] > CV_LAST[name]}")
save_results("johansen_cases", out)
