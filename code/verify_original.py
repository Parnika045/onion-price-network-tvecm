"""Reproduction check: re-run the ORIGINAL v5 network pipeline (no cleaning, `Market` as the
mandi key) to confirm that this package reproduces the numbers reported in the original
notebooks (240 nodes, 714 edges, Fiedler 0.039067, six communities of 58/47/47/38/25/25,
48 SIMAMs). Any difference between the paper and the corrected results is therefore due to
the data corrections, not to re-implementation.
"""
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LinearRegression

from common import DER, save_results

m = pd.read_parquet(DER / "master_raw.parquet")
m["Date"] = pd.to_datetime(m["Date"])
m = m.sort_values(["Market", "Date"]).reset_index(drop=True)
m["log_price"] = np.log(m["Modal_Price_Rs_Quintal"].clip(lower=1))
m["dP"] = m.groupby("Market")["log_price"].transform(lambda x: x.diff())
rep = m.groupby("Market")["Date"].nunique()
arr = m.groupby("Market")["Arrivals_MT"].sum()
elig = rep[rep >= 200].index
top = arr.loc[elig].sort_values(ascending=False).head(240).index.tolist()
p = m[m["Market"].isin(top)]
dp = p.pivot_table(index="Date", columns="Market", values="dP")
clean = [x for x in top if dp[x].count() >= 60]
X = dp[clean].fillna(dp[clean].mean())
lw = LedoitWolf().fit(X.values)
cov = lw.covariance_; sd = np.sqrt(np.diag(cov)); corr = cov / np.outer(sd, sd)
rho = corr; N = len(clean)
d = np.sqrt(np.clip(2 * (1 - rho), 0, None))
iu = np.triu_indices(N, 1); order = np.argsort(d[iu])
G = nx.Graph(); G.add_nodes_from(clean); ins = 0
for a in order:
    if ins >= 3 * (N - 2):
        break
    i, j = iu[0][a], iu[1][a]
    G.add_edge(clean[i], clean[j], weight=float(1 / (d[i, j] + 1e-9)))
    if not nx.check_planarity(G)[0]:
        G.remove_edge(clean[i], clean[j])
    else:
        ins += 1
L = nx.laplacian_matrix(G, weight="weight").toarray().astype(float)
f = float(np.sort(np.linalg.eigvalsh(L))[1])
comms = nx.algorithms.community.louvain_communities(G, weight="weight", seed=42)
eig = nx.eigenvector_centrality_numpy(G, weight="weight"); degc = nx.degree_centrality(G)
Gv = nx.Graph(); Gv.add_nodes_from(G.nodes())
for u, v in G.edges():
    Gv.add_edge(u, v, weight=np.sqrt(arr.get(u, 1) * arr.get(v, 1)))
ev = nx.eigenvector_centrality_numpy(Gv, weight="weight")
c = pd.DataFrame({"e": eig, "d": degc, "v": ev})
c["r"] = c.e - LinearRegression().fit(c[["d"]], c.e).predict(c[["d"]])
simam = int(((c.r >= c.r.quantile(.8)) & (c.v < c.v.quantile(.8))).sum())
out = {"n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(), "fiedler": f,
       "lw_shrinkage": float(lw.shrinkage_), "sizes": sorted([len(x) for x in comms], reverse=True),
       "n_simams": simam, "max_degree": max(dict(G.degree()).values())}
save_results("verify_original", out)
print(out)
