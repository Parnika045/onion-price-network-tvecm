"""02 - Ledoit-Wolf correlations, PMFG, centrality, SIMAMs, Louvain communities, spectra.

Design follows the original v5 notebook (returns between consecutive reports of a mandi;
missing returns set to the mandi's mean return; Ledoit-Wolf shrinkage; PMFG with
d_ij = sqrt(2(1-rho_ij)) and edge weight 1/d_ij), applied to the cleaned panel.
Usage: python 02_network.py [--shrinkage 0.15] [--tag alpha015]   (robustness variants)
"""
import argparse
import pickle
import time

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, ShrunkCovariance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import normalized_mutual_info_score

from common import DER, MIN_RETURNS, SEED, STABILITY_SEEDS, save_results, write_table

ap = argparse.ArgumentParser()
ap.add_argument("--shrinkage", type=float, default=None)
ap.add_argument("--pairwise", action="store_true", help="pairwise-complete sample correlations, no imputation")
ap.add_argument("--tag", default="baseline")
args = ap.parse_args()

panel = pd.read_parquet(DER / "panel240.parquet")
meta = pd.read_csv(DER / "meta240.csv", index_col=0)
dp = panel.pivot_table(index="Date", columns="mandi", values="dP")
mandis = [m for m in meta.index if dp[m].count() >= MIN_RETURNS]
R = dp[mandis]

if args.pairwise:
    corr = R.corr(min_periods=MIN_RETURNS).fillna(0).values
    np.fill_diagonal(corr, 1.0)
    shrink = np.nan
else:
    X = R.fillna(R.mean()).values
    if args.shrinkage is None:
        est = LedoitWolf().fit(X)
        shrink = float(est.shrinkage_)
    else:
        est = ShrunkCovariance(shrinkage=args.shrinkage).fit(X)
        shrink = args.shrinkage
    cov = est.covariance_
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    sample_corr = np.corrcoef(X, rowvar=False)
corr_df = pd.DataFrame(corr, index=mandis, columns=mandis)


def build_pmfg(cdf: pd.DataFrame) -> nx.Graph:
    nodes = cdf.columns.tolist(); N = len(nodes)
    rho = cdf.values
    d = np.sqrt(np.clip(2 * (1 - rho), 0, None))
    iu = np.triu_indices(N, k=1)
    order = np.argsort(d[iu], kind="stable")
    G = nx.Graph(); G.add_nodes_from(nodes)
    target = 3 * (N - 2); inserted = 0
    for a in order:
        if inserted >= target:
            break
        i, j = iu[0][a], iu[1][a]
        G.add_edge(nodes[i], nodes[j], weight=float(1 / (d[i, j] + 1e-9)),
                   distance=float(d[i, j]), corr=float(rho[i, j]))
        if not nx.check_planarity(G)[0]:
            G.remove_edge(nodes[i], nodes[j])
        else:
            inserted += 1
    return G


t0 = time.time()
G = build_pmfg(corr_df)
build_secs = time.time() - t0
print(f"[{args.tag}] PMFG {G.number_of_nodes()} nodes {G.number_of_edges()} edges in {build_secs:.0f}s")


def louvain(G, seed):
    comms = nx.algorithms.community.louvain_communities(G, weight="weight", seed=seed)
    return {n: cid for cid, c in enumerate(comms) for n in c}


labels = {s: louvain(G, s) for s in STABILITY_SEEDS}
nodes = list(G.nodes())
base = labels[SEED]

# ---- deterministic relabelling: C0 = community with the largest share of cumulative arrivals, etc.
raw_ids = sorted(set(base.values()))
arr_by = {r: sum(meta.loc[n, "arrivals_total_MT"] for n in nodes if base[n] == r) for r in raw_ids}
mapping = {r: k for k, r in enumerate(sorted(raw_ids, key=lambda r: -arr_by[r]))}
comm = {n: mapping[c] for n, c in base.items()}
anchors_ok = True

nmi = {f"42_vs_{s}": float(normalized_mutual_info_score([base[n] for n in nodes], [labels[s][n] for n in nodes]))
       for s in STABILITY_SEEDS[1:]}

if args.tag != "baseline":
    # robustness variant: compare with baseline partition
    with open(DER / "network_baseline.pkl", "rb") as fh:
        b = pickle.load(fh)
    common_nodes = [n for n in nodes if n in b["comm"]]
    nmi_vs_base = float(normalized_mutual_info_score([b["comm"][n] for n in common_nodes],
                                                     [comm[n] for n in common_nodes]))
    lap = nx.laplacian_matrix(G, weight="weight").toarray().astype(float)
    fied = float(np.sort(np.linalg.eigvalsh(lap))[1])
    pd.Series(comm, name="community").to_csv(DER / f"communities_{args.tag}.csv")
    save_results(f"network_{args.tag}", {"shrinkage": shrink, "n_communities": len(raw_ids),
                                          "nmi_vs_baseline": nmi_vs_base, "fiedler": fied,
                                          "sizes": sorted([list(comm.values()).count(c) for c in set(comm.values())], reverse=True)})
    print(args.tag, "NMI vs baseline", nmi_vs_base, "fiedler", fied)
    raise SystemExit

# ---------------------------------------------------------------- topology
deg = dict(G.degree())
lap = nx.laplacian_matrix(G, nodelist=nodes, weight="weight").toarray().astype(float)
ev_lap = np.sort(np.linalg.eigvalsh(lap))
adj = nx.to_numpy_array(G, nodelist=nodes, weight="weight")
ev_adj = np.sort(np.linalg.eigvalsh(adj))[::-1]
nlap = nx.normalized_laplacian_matrix(G, nodelist=nodes, weight="weight").toarray()
ev_nlap = np.sort(np.linalg.eigvalsh(nlap))
# Fiedler vector partition
fvec = np.linalg.eigh(lap)[1][:, 1]
fiedler_side = {n: int(v > 0) for n, v in zip(nodes, fvec)}

# Benchmark: Fiedler value when the same edge weights are randomly reassigned across PMFG edges
rng = np.random.default_rng(SEED)
edges = list(G.edges())
w = np.array([G[u][v]["weight"] for u, v in edges])
idx = {n: i for i, n in enumerate(nodes)}
perm_f = []
for _ in range(500):
    wp = rng.permutation(w)
    A = np.zeros((len(nodes), len(nodes)))
    for (u, v), ww in zip(edges, wp):
        A[idx[u], idx[v]] = A[idx[v], idx[u]] = ww
    L = np.diag(A.sum(1)) - A
    perm_f.append(np.sort(np.linalg.eigvalsh(L))[1])
perm_f = np.array(perm_f)

# ---------------------------------------------------------------- centrality and SIMAMs
eig = nx.eigenvector_centrality_numpy(G, weight="weight")
degc = nx.degree_centrality(G)
bet = nx.betweenness_centrality(G, weight="distance", normalized=True)
arr = meta["arrivals_total_MT"]
Gv = nx.Graph(); Gv.add_nodes_from(nodes)
for u, v in G.edges():
    Gv.add_edge(u, v, weight=float(np.sqrt(arr[u] * arr[v])))
eig_vol = nx.eigenvector_centrality_numpy(Gv, weight="weight")
cdf = pd.DataFrame({"eigenvector": eig, "degree_centrality": degc, "degree": deg,
                    "betweenness": bet, "vol_eigenvector": eig_vol}).join(meta)
lr = LinearRegression().fit(cdf[["degree_centrality"]], cdf["eigenvector"])
cdf["ec_resid"] = cdf["eigenvector"] - lr.predict(cdf[["degree_centrality"]])
cdf["SIMAM"] = (cdf["ec_resid"] >= cdf["ec_resid"].quantile(0.8)) & (cdf["vol_eigenvector"] < cdf["vol_eigenvector"].quantile(0.8))
cdf["community"] = pd.Series(comm)
cdf["fiedler_side"] = pd.Series(fiedler_side)
cdf["label"] = cdf["Market"] + " (" + cdf["State"] + ")"
cdf.to_csv(DER / "centrality_communities.csv")

# ---------------------------------------------------------------- correlations within / between communities
iu = np.triu_indices(len(mandis), 1)
ci = np.array([comm[m] for m in mandis])
same = ci[iu[0]] == ci[iu[1]]
off_lw = corr[iu]
off_s = sample_corr[iu]

# ---------------------------------------------------------------- corridors
rows = []
for u, v, dta in G.edges(data=True):
    if comm[u] != comm[v]:
        a, b = sorted([comm[u], comm[v]])
        rows.append({"corridor": f"C{a}-C{b}", "u": cdf.loc[u, "label"], "v": cdf.loc[v, "label"], "corr": dta["corr"]})
inter = pd.DataFrame(rows)
inter.to_csv(DER / "intercommunity_edges.csv", index=False)
corr_tab = inter.groupby("corridor").agg(edges=("corr", "size"), mean_corr=("corr", "mean")).reset_index()

# ---------------------------------------------------------------- community table
ctab = []
for c in sorted(set(comm.values())):
    sub = cdf[cdf["community"] == c]
    st = sub["State"].value_counts()
    ctab.append({"community": f"C{c}", "mandis": len(sub), "simams": int(sub["SIMAM"].sum()),
                 "simam_share": sub["SIMAM"].mean(),
                 "top_states": "; ".join(f"{k} {v}" for k, v in st.head(4).items()),
                 "top_ec": "; ".join(sub.nlargest(4, "eigenvector")["label"]),
                 "arrival_share": sub["arrivals_total_MT"].sum() / cdf["arrivals_total_MT"].sum()})
ctab = pd.DataFrame(ctab)
ctab.to_csv(DER / "community_summary.csv", index=False)

with open(DER / "network_baseline.pkl", "wb") as fh:
    pickle.dump({"G": G, "comm": comm, "nodes": nodes, "corr": corr_df}, fh)
nx.write_edgelist(G, DER / "pmfg_edgelist.tsv", delimiter="\t", data=["corr", "weight"])

top_deg = sorted(deg.items(), key=lambda x: -x[1])[:6]
out = {
    "N_nodes": G.number_of_nodes(), "N_edges": G.number_of_edges(), "T_returns": int(R.shape[0]),
    "N_corr_mandis": len(mandis), "is_planar": bool(nx.check_planarity(G)[0]),
    "is_connected": bool(nx.is_connected(G)), "build_seconds": build_secs,
    "lw_shrinkage": shrink, "density": nx.density(G),
    "mean_degree": float(np.mean(list(deg.values()))), "max_degree": int(max(deg.values())),
    "top_degree": [(cdf.loc[n, "label"], d, int(comm[n])) for n, d in top_deg],
    "avg_clustering_weighted": nx.average_clustering(G, weight="weight"),
    "avg_clustering_unweighted": nx.average_clustering(G),
    "avg_shortest_path": nx.average_shortest_path_length(G), "diameter": nx.diameter(G),
    "fiedler": float(ev_lap[1]), "lap_eigs_first6": ev_lap[:6].tolist(),
    "fiedler_normalized": float(ev_nlap[1]),
    "fiedler_perm_mean": float(perm_f.mean()), "fiedler_perm_p05": float(np.quantile(perm_f, 0.05)),
    "fiedler_perm_share_below": float((perm_f <= ev_lap[1]).mean()),
    "fiedler_side_sizes": [int(sum(v == k for v in fiedler_side.values())) for k in (0, 1)],
    "fiedler_side_by_community": cdf.groupby(["community", "fiedler_side"]).size().unstack(fill_value=0).to_dict(),
    "mean_edge_weight": float(w.mean()),
    "adj_eigs_top8": ev_adj[:8].tolist(), "spectral_gap": float(ev_adj[0] - ev_adj[1]),
    "n_communities": len(raw_ids), "anchors_ok": anchors_ok,
    "community_sizes": {f"C{c}": int(list(comm.values()).count(c)) for c in sorted(set(comm.values()))},
    "nmi_seeds": nmi,
    "modularity": nx.algorithms.community.modularity(G, [set(n for n in nodes if comm[n] == c) for c in set(comm.values())], weight="weight"),
    "n_simams": int(cdf["SIMAM"].sum()),
    "simam_by_comm": {f"C{c}": int(cdf.loc[cdf.community == c, "SIMAM"].sum()) for c in sorted(set(comm.values()))},
    "offdiag_lw_mean": float(off_lw.mean()), "offdiag_lw_max": float(off_lw.max()),
    "offdiag_sample_mean": float(off_s.mean()), "offdiag_sample_max": float(off_s.max()),
    "mean_abs_diff_sample_lw": float(np.abs(off_s - off_lw).mean()),
    "intra_corr_lw": float(off_lw[same].mean()), "inter_corr_lw": float(off_lw[~same].mean()),
    "intra_corr_sample": float(off_s[same].mean()), "inter_corr_sample": float(off_s[~same].mean()),
    "pmfg_edge_corr_mean": float(np.mean([d["corr"] for *_, d in G.edges(data=True)])),
    "n_inter_edges": len(inter), "corridors": corr_tab.to_dict("records"),
    "top_ec": cdf.nlargest(8, "eigenvector")[["label", "eigenvector", "betweenness", "degree", "community"]].values.tolist(),
    "top_bc": cdf.nlargest(8, "betweenness")[["label", "betweenness", "eigenvector", "degree", "community"]].values.tolist(),
    "community_table": ctab.to_dict("records"),
}
save_results("network", out)
write_table(ctab, "table_communities")
write_table(corr_tab, "table_corridors")
for k in ["fiedler", "fiedler_normalized", "fiedler_perm_mean", "fiedler_perm_share_below", "lw_shrinkage",
          "community_sizes", "nmi_seeds", "n_simams", "simam_by_comm", "intra_corr_lw", "inter_corr_lw",
          "intra_corr_sample", "inter_corr_sample", "offdiag_lw_mean", "offdiag_lw_max", "top_degree",
          "spectral_gap", "anchors_ok", "modularity", "avg_clustering_weighted", "fiedler_side_by_community"]:
    print(k, out[k])
print(ctab.to_string())
print(corr_tab.to_string())
print(pd.DataFrame(out["top_ec"]).to_string())
