"""02b - Sensitivity of correlations and centrality to the treatment of missing price changes (Section 3.4.1).

The baseline network sets missing log price changes to the mandi's mean change before estimating the Ledoit-Wolf
covariance matrix. With about 39 percent of mandi-days missing, this shrinks covariances toward zero for thinly
reporting mandis. Two alternatives avoid imputation altogether:
  (a) pairwise-complete sample correlations of the same changes, computed on the dates both mandis report
      (at least 60 common changes);
  (b) as (a), but using only one-day changes (consecutive trading days), so that every change spans the same horizon.
For each, the PMFG is rebuilt and eigenvector centrality recomputed, and compared with the baseline: Spearman rank
correlation of eigenvector centrality, overlap of the ten most central mandis, the rank of the Nashik-belt (C0)
mandis, and the mean off-diagonal correlation.
"""
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import DER, MIN_RETURNS, save_results

src = open(__file__.replace("02b_missing_data_check.py", "02_network.py"), encoding="utf8").read()
start = src.index("def build_pmfg"); end = src.index("t0 = time.time()")
exec(src[start:end])                                          # build_pmfg, identical to the baseline

panel = pd.read_parquet(DER / "panel240.parquet")
base = pd.read_csv(DER / "centrality_communities.csv", index_col=0)
dp = panel.pivot_table(index="Date", columns="mandi", values="dP")
mandis = [m for m in base.index if m in dp.columns]
dp = dp[mandis]
dates = pd.Series(dp.index, index=dp.index)
# one-day changes: the previous report of the mandi was on the previous trading date
td = pd.Series(np.arange(len(dp.index)), index=dp.index)
prev = dp.notna().astype(float).replace(0, np.nan).mul(td, axis=0).ffill().shift(1)
one_day = (td.values[:, None] - prev.values) == 1
dp1 = dp.where(one_day)

X = dp.fillna(dp.mean())
off = ~np.eye(len(mandis), dtype=bool)
out = {"share_missing_mandi_days": float(1 - dp.notna().values.mean()),
       "share_changes_one_day": float(np.nansum(one_day & dp.notna().values) / dp.notna().values.sum()),
       "mean_corr_imputed_sample": float(np.corrcoef(X.values, rowvar=False)[off].mean())}
for tag, D in (("pairwise", dp), ("pairwise_one_day", dp1)):
    C = D.corr(min_periods=MIN_RETURNS)
    n_missing_pairs = int(C.isna().values[off].sum() / 2)
    C = C.fillna(0); np.fill_diagonal(C.values, 1.0)
    G = build_pmfg(C)
    eig = pd.Series(nx.eigenvector_centrality_numpy(G, weight="weight"))
    rank = eig.rank(ascending=False)
    top_b = set(base["eigenvector"].nlargest(10).index); top_v = set(eig.nlargest(10).index)
    c0 = base.index[base["community"] == 0]
    out[tag] = {"mean_corr": float(C.values[off].mean()), "pairs_without_60_common": n_missing_pairs,
                "spearman_eig_vs_baseline": float(spearmanr(base["eigenvector"], eig.reindex(base.index)).correlation),
                "top10_overlap": len(top_b & top_v), "top5": [base.loc[m, "label"] for m in eig.nlargest(5).index],
                "c0_median_rank": float(rank.reindex(c0).median()), "n": len(mandis),
                "uttar_pradesh_in_top10": int(sum(base.loc[m, "State"] == "Uttar Pradesh" for m in top_v))}
    print(tag, out[tag], flush=True)
out["baseline_top5"] = [base.loc[m, "label"] for m in base["eigenvector"].nlargest(5).index]
out["baseline_c0_median_rank"] = float(base["eigenvector"].rank(ascending=False).reindex(base.index[base["community"] == 0]).median())
out["baseline_up_in_top10"] = int(sum(base.loc[m, "State"] == "Uttar Pradesh" for m in base["eigenvector"].nlargest(10).index))
print({k: v for k, v in out.items() if not isinstance(v, dict)})
save_results("missing_data_check", out)
