"""12 - Format every number quoted in the manuscript from output/results.json.

Writes output/paper_numbers.json: {"N": {...scalar strings...}, "T": {...tables as row lists...}}.
The manuscript builder (paper/build_paper.js) reads only this file, so text and tables cannot
drift from the estimates.
"""
import json

import numpy as np
import pandas as pd

from common import DER, OUT, load_results

R = load_results()
N, T = {}, {}


def f(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "∞" if x == np.inf else "—"
    return f"{x:,.{d}f}".replace("-", "−")


def pv(p):
    return "<0.001" if p < 0.001 else f(p, 3)


d, net, co, tv, mo = R["data"], R["network"], R["cointegration"], R["tvecm"], R["moderation"]
rain, pol, asp, fr, lp = R["rainfall"], R["policy"], R["asymmetry_simam_partitions"], R["fiedler_rolling"], R["rain_timing"]
vo = R["verify_original"]
geo = R["geocode"]

LABELS = {0: "Nashik–Deccan production belt", 1: "Purvanchal–Bengal hub cluster", 2: "Central UP (Lakhimpur) cluster",
          3: "UP–Nashik fringe cluster", 4: "Maharashtra urban terminals", 5: "Kanpur–Rajasthan cluster",
          6: "Lucknow–Purvanchal secondary cluster", 7: "Western UP (Buland Shahr) belt"}

# ---------------- data
N.update(raw=f"{d['n_raw_records']:,}", names=f"{d['n_market_names']:,}", keys=f"{d['n_mandis_state_market']:,}",
         homonyms=str(len(d["homonym_market_names"])), dates=f"{d['n_trading_dates']:,}",
         outliers=f"{d['n_price_outliers_dropped']:,}", dups=f"{d['n_duplicate_extra_rows_collapsed']:,}",
         clean=f"{d['n_clean_mandi_days']:,}", eligible=f"{d['n_eligible_mandis']:,}", sample_obs=f"{d['n_sample_obs']:,}",
         arr_cov=f"{100*d['arrival_coverage_share']:.1f}", rd_min=str(d["report_days_min"]), rd_max=str(d["report_days_max"]),
         rd_med=f"{d['report_days_median']:.0f}", rd_mean=f"{d['report_days_mean']:.0f}",
         states_s=str(d["n_states_sample"]), subdiv_s=str(d["n_subdivisions_sample"]), states_raw=str(d["n_states_raw"]),
         subdiv_raw=str(d["n_subdivisions_raw"]), ceiling_share=f"{100*d['sample_share_of_ceiling']:.1f}",
         gap3=f"{100*d['share_returns_gap_gt3']:.1f}")
# ---------------- network
N.update(lw=f(net["lw_shrinkage"], 2), lw_orig=f(vo["lw_shrinkage"], 2), fiedler=f(net["fiedler"], 3),
         fiedler_orig=f(vo["fiedler"], 3), fiedler_norm=f(net["fiedler_normalized"], 3),
         fiedler_perm=f(net["fiedler_perm_mean"], 3), fiedler_perm_p05=f(net["fiedler_perm_p05"], 3),
         density=f(net["density"], 3), mean_deg=f(net["mean_degree"], 2), max_deg=str(net["max_degree"]),
         clust=f(net["avg_clustering_weighted"], 2), apl=f(net["avg_shortest_path"], 2), diam=str(net["diameter"]),
         spec_gap=f(net["spectral_gap"], 2), lam1=f(net["adj_eigs_top8"][0], 2), lam2=f(net["adj_eigs_top8"][1], 2),
         nmi123=f(net["nmi_seeds"]["42_vs_123"], 2), nmi999=f(net["nmi_seeds"]["42_vs_999"], 2),
         modularity=f(net["modularity"], 2), n_simam=str(net["n_simams"]),
         simam_c1=str(net["simam_by_comm"]["C1"]), simam_c7=str(net["simam_by_comm"]["C7"]),
         off_mean=f(net["offdiag_lw_mean"], 3), off_max=f(net["offdiag_lw_max"], 2), off_s_mean=f(net["offdiag_sample_mean"], 3),
         intra=f(net["intra_corr_lw"], 3), inter=f(net["inter_corr_lw"], 3), edge_corr=f(net["pmfg_edge_corr_mean"], 2),
         n_inter=str(net["n_inter_edges"]), fside_a=str(net["fiedler_side_sizes"][1]), fside_b=str(net["fiedler_side_sizes"][0]),
         nmi_a015=f(R["network_alpha015"]["nmi_vs_baseline"], 2), k_a015=str(R["network_alpha015"]["n_communities"]),
         nmi_a050=f(R["network_alpha050"]["nmi_vs_baseline"], 2), k_a050=str(R["network_alpha050"]["n_communities"]),
         nmi_pw=f(R["network_pairwise"]["nmi_vs_baseline"], 2), k_pw=str(R["network_pairwise"]["n_communities"]),
         K=str(co["K"]), n_pairs=str(len(tv["pairs"])))
top = {x[0]: x for x in net["top_degree"]}
N.update(basti_deg=str(net["top_degree"][0][1]), buland_deg=str(net["top_degree"][1][1]), azam_deg=str(net["top_degree"][2][1]))
# Fiedler side composition
fs = net["fiedler_side_by_community"]
N["fside_comp"] = ", ".join(f"C{c} {fs['1'][c]}" for c in sorted(fs["1"], key=int) if fs["1"][c] > 0)

# ---------------- cointegration
eg = pd.DataFrame(co["eg"])
N.update(T=f"{co['T']:,}", var_bic=str(co["var_bic"]), var_aic=str(co["var_aic"]), kdiff=str(co["k_ar_diff"]),
         eg_tmin=f(co["eg_t_range"][0], 2), eg_tmax=f(co["eg_t_range"][1], 2), eg_pmax=f(eg["eg_p"].max(), 3),
         beta_min=f(eg["beta"].min(), 2), beta_max=f(eg["beta"].max(), 2),
         n_damp=str(co["beta_counts"].get("dampening (<0.95)", 0)), n_near=str(co["beta_counts"].get("near one-to-one", 0)),
         n_amp=str(co["beta_counts"].get("amplifying (>1.05)", 0)),
         joh_eig_last=f(co["johansen"]["det1"]["eig"][-1], 4), joh_tr_last=f(co["johansen"]["det1"]["trace"][-1], 1),
         joh_tr_last0=f(co["johansen"]["det0"]["trace"][-1], 1))
ur = pd.DataFrame(co["unit_roots"])
N.update(adf_pmin=f(ur["p_level"].min(), 2), adf_pmax=f(ur["p_level"].max(), 2), kpss_min=f(ur["kpss_level"].min(), 2),
         kpss_max=f(ur["kpss_level"].max(), 2), adfd_min=f(ur["adf_diff"].max(), 1), adfd_max=f(ur["adf_diff"].min(), 1))
desc = pd.DataFrame(co["desc"])
ac = desc["first_order_autocorr_dlogp_aw"]; acfe = desc["first_order_autocorr_dlogp_fe"]
N.update(ac_min=f(ac.max(), 2), ac_max=f(ac.min(), 2), acfe_min=f(acfe.max(), 2), acfe_max=f(acfe.min(), 2))

# ---------------- TVECM
s = tv["summary"]; P = pd.DataFrame(tv["pairs"])
N.update(n_thr05=str(s["n_thresh_sig05"]), n_thr_holm=str(s["n_thresh_sig_holm"]), n_fast=str(s["n_outer_faster"]),
         n_fast_sig=str(s["n_outer_faster_sig"]), n_slow=str(s["n_outer_slower"]), n_slow_sig=str(s["n_outer_slower_sig"]),
         hl1=f(s["median_hl1"], 2), hl2=f(s["median_hl2"], 2), th1=f(s["median_theta1"], 2), th2=f(s["median_theta2"], 2),
         gamma_min=f(P["gamma"].min(), 2), gamma_max=f(P["gamma"].max(), 2), gamma_med=f(P["gamma"].median(), 2),
         n_gamma_edge=str(s["n_gamma_edge"]),
         outer_share_med=f(100 * P["share_outer"].median(), 0))
c03 = P[P.pair == "C0-C3"].iloc[0]
N.update(c03_th1=f(c03["theta_1"], 2), c03_th2=f(c03["theta_2"], 2), c03_p=f(c03["theta_diff_p"], 3), c03_g=f(c03["gamma"], 3),
         c03_outer=f(100 * c03["share_outer"], 0), c03_hl1=f(c03["hl_1"], 2), c03_hl2=f(c03["hl_2"], 2))
gm = np.exp(P["gamma"].median()) - 1
N["gamma_med_pct"] = f(100 * gm, 0)
rb = tv["robustness"]
for k, v in rb.items():
    N[f"rb_{k}_fast"] = str(v["n_outer_faster"]); N[f"rb_{k}_fastsig"] = str(v["n_outer_faster_sig"])
    N[f"rb_{k}_slowsig"] = str(v["n_outer_slower_sig"])
for k, v in asp["partitions"].items():
    N[f"pt_{k}_n"] = str(v["n_pairs"]); N[f"pt_{k}_fast"] = str(v["n_outer_faster"]); N[f"pt_{k}_fastsig"] = str(v["n_outer_faster_sig"])
    N[f"pt_{k}_slowsig"] = str(v["n_outer_slower_sig"]); N[f"pt_{k}_K"] = str(v["K"])
bts = asp["bandtar_summary"]
N.update(bt_above=str(bts["n_above_faster"]), bt_below=str(bts["n_below_faster"]), bt_above_sig=str(bts["n_above_faster_sig"]),
         bt_below_sig=str(bts["n_below_faster_sig"]))
sm_ = asp["simam"]
N.update(simam_beta=f(sm_["beta"], 2), simam_p=f(sm_["perm_p_exact"], 2), simam_olsp=f(sm_["ols_p"], 2), simam_nperm=f"{sm_['n_perm']:,}")

# ---------------- rainfall
jj = rain["system_jjas_departure_by_year"]
N.update(monsoon_ratio=f(rain["monsoon_rain_to_normal_ratio"], 2), infl=f(rain["orig_rain_inflation_median"], 1),
         def_share=f(100 * rain["deficit_share_positive"], 0), rain_cells=str(rain["n_cells"]),
         rain_cellmonths=f"{rain['n_cell_months_used']:,}", cells_per_sub=f(rain["mean_cells_per_subdiv_month"], 0),
         obs_norm_corr=f(rain["corr_cell_obs_vs_normal"], 2),
         jj21=f(100 * jj["2021"], 0), jj22=f(100 * jj["2022"], 0), jj23=f(100 * jj["2023"], 0), jj24=f(100 * jj["2024"], 0),
         old_new_diff=f(100 * rain["old_normals_vs_new"]["median_abs_pct_diff"], 0),
         imd_def=f(100 * (rain["imd_category_shares"].get("deficient", 0) + rain["imd_category_shares"].get("large deficient", 0)), 0))
M = pd.DataFrame(mo["results"])


def mrow(x, fe):
    r = M[(M.x == x) & (M.fe == fe)].iloc[0]
    return r


for key, x, fe in [("m_def_p", "deficit", "pair"), ("m_def_c", "deficit", "pair+cal"), ("m_def_t", "deficit", "pair+ym"),
                   ("m_z_p", "zdef", "pair"), ("m_z_c", "zdef", "pair+cal"), ("m_z_t", "zdef", "pair+ym"),
                   ("m_gap_t", "dz_gap", "pair+ym"), ("m_def_k", "deficit", "pair+ym (Jul-Sep only)"),
                   ("m_pd_p", "pdef", "pair"), ("m_pd_c", "pdef", "pair+cal"), ("m_pd_t", "pdef", "pair+ym"),
                   ("m_pd_k", "pdef", "pair+ym (Jul-Sep only)"), ("m_ab_c", "absz", "pair+cal"), ("m_ab_t", "absz", "pair+ym")]:
    r = mrow(x, fe)
    N[key + "_b"] = f(r["beta"], 5); N[key + "_se"] = f(r["se"], 5); N[key + "_p"] = pv(r["p"])
    N[key + "_sdpct"] = f(100 * r["beta_per_sd"] / r["mean_dep"], 1)
N.update(mod_N=f"{mo['N']:,}", def_sd=f(mo["deficit_sd"], 1), absect_mean=f(mo["abs_ect_mean"], 3))
L = pd.DataFrame(lp["results"])
l2 = L[(L.shock == "deficit") & (L.h_months == 2) & (L.calendar_fe)].iloc[0]
l2n = L[(L.shock == "deficit") & (L.h_months == 2) & (~L.calendar_fe)].iloc[0]
N.update(lp2_b=f(100 * l2["beta"] * 10, 1), lp2_p=f(l2["p"], 3), lp2n_b=f(100 * l2n["beta"] * 10, 1),
         lp_cal_pmin=f(L[(L.shock == "deficit") & (L.calendar_fe)]["p"].min(), 3),
         lpz_pmin=f(L[(L.shock == "zdef") & (L.calendar_fe)]["p"].min(), 2))
FR = pd.DataFrame(fr["regressions"])
N.update(fr_pmin=f(FR["p"].min(), 2), fr_n=str(int(FR["N"].iloc[0])))

L3 = L[(L.shock == "deficit_ins") & (L.calendar_fe)]
N.update(lp_ins_pmin=f(L3["p"].min(), 2), lp_ins2_p=f(L3[L3.h_months == 2]["p"].iloc[0], 2))
for key, x, fe in [("m_ins_t", "deficit_ins", "pair+ym"), ("m_ins_c", "deficit_ins", "pair+cal"), ("m_zins_t", "zdef_ins", "pair+ym")]:
    r = mrow(x, fe); N[key + "_p"] = pv(r["p"]); N[key + "_b"] = f(r["beta"], 5)
N.update(geo_multi=f"{geo['n_mandis_multi_variant']:,}", geo_flag_mandis=f"{geo['n_mandis_any_flagged']:,}",
         geo_flag_recs=f"{geo['n_records_flagged']:,}", geo_flag_share=f(100 * geo["share_records_flagged"], 1),
         geo_out_state=f"{geo['n_records_out_of_state']:,}", geo_s_mandis=str(geo["sample_mandis_any_flagged"]),
         geo_s_share=f(100 * geo["sample_share_records_flagged"], 1),
         geo_s_confirmed=str(sum(v for k, v in geo["sample_canonical_methods"].items() if k.startswith("variant confirmed"))),
         geo_s_relocated=str(sum(v for k, v in geo["sample_canonical_methods"].items() if k.startswith("relocated"))),
         geo_s_unver=str(geo["sample_canonical_methods"].get("dominant in-state variant (no gazetteer match)", 0)),
         geo_dummy=0)
LP = L[(L.shock == "pdef") & (L.calendar_fe)]
N.update(lp_pd_min_b=f(100 * 0.1 * LP["beta"].min(), 1), lp_pd_max_b=f(100 * 0.1 * LP["beta"].max(), 1),
         lp_pd_pos_sig=str(int(((LP["beta"] > 0) & (LP["p"] < 0.05)).sum())),
         lp_def_cal_pmin=f(L[(L.shock == "deficit") & (L.calendar_fe)]["p"].min(), 2),
         lp_def_cal_bmin=f(100 * 10 * L[(L.shock == "deficit") & (L.calendar_fe)]["beta"].min(), 1),
         lp_def_cal_bmax=f(100 * 10 * L[(L.shock == "deficit") & (L.calendar_fe)]["beta"].max(), 1),
         lp_def_nocal_bmin=f(100 * 10 * L[(L.shock == "deficit") & (~L.calendar_fe)]["beta"].min(), 1),
         lp_def_nocal_bmax=f(100 * 10 * L[(L.shock == "deficit") & (~L.calendar_fe)]["beta"].max(), 1))

# ---------------- economic interpretation
E = R["econ"]
PL = pd.DataFrame(E["prod_rain_lp"]); PD = PL[PL.shock == "deficit_mm"]
def lpr(h): return PD[PD.h == h].iloc[0]
N.update(band_pct=f(E["gamma_pct_median"], 0), band_rs=f(E["gamma_rs_median"], 0),
         band_rs_min=f(E["gamma_rs_min"], 0), band_rs_max=f(E["gamma_rs_max"], 0),
         band_pct_min=f(E["gamma_pct_min"], 0), band_pct_max=f(E["gamma_pct_max"], 0),
         band_dist_corr=f(E["corr_gamma_logdist"], 2), band_dist_p=f(E["gamma_on_logdist_perm_p"], 2),
         band_edges_p=f(E["gamma_on_edges_p"], 2), dtheta_dist_p=f(E["dtheta_on_logdist_perm_p"], 2),
         dist_min=f(E["dist_min_km"], 0), dist_max=f(E["dist_max_km"], 0),
         prod_h2=f(lpr(2)["effect_1sd_pct"], 1), prod_h2_p=f(lpr(2)["p"], 3),
         prod_h6=f(lpr(6)["effect_1sd_pct"], 1), prod_h6_p=f(lpr(6)["p"], 3),
         prod_n_sig=str(int((PD["p"] < 0.05).sum())), prod_n=str(len(PD)),
         prod_pos=str(int((PD["beta"] > 0).sum())),
         prod_pd_pmin=f(PL[PL.shock == "pdef"]["p"].min(), 2))
TD = pd.read_csv(DER.parent.parent / "output" / "tables" / "table_thresholds_distance.csv")
n_mid = int(((TD["gamma_pct"] >= 20) & (TD["gamma_pct"] <= 30)).sum())
N["band_n_20_30"] = str(n_mid)

# ---------------- policy
pp = pol["percentiles"]
N.update(p50=f(pp["50"], 0), p75=f(pp["75"], 0), p80=f(pp["80"], 0), p90=f(pp["90"], 0), p10=f(pp["10"], 0),
         pmean=f(pol["mean"], 0), psd=f(pol["sd"], 0), pcv=f(100 * pol["cv"], 1))
PT = pd.DataFrame(pol["table"]); r80 = PT[PT.percentile == 80].iloc[0]
N.update(p80_transfer=f(r80["transfer_rs_mn_per_year"], 0), p80_tri=f(r80["triangle_rs_mn_per_year_e0.6"], 0),
         p80_release=f(r80["release_kt_per_year_e0.6"], 0), p80_sd=f(r80["sd_reduction_pct"], 1),
         p80_tri_lo=f(r80["triangle_rs_mn_per_year_e0.4"], 0), p80_tri_hi=f(r80["triangle_rs_mn_per_year_e0.8"], 0),
         p80_rel_lo=f(r80["release_kt_per_year_e0.4"], 0), p80_rel_hi=f(r80["release_kt_per_year_e0.8"], 0))
N.update(vo_fiedler=f(vo["fiedler"], 3), vo_sizes="/".join(map(str, vo["sizes"])), vo_simam=str(vo["n_simams"]))

# ======================================================== tables
T["t1"] = [["Raw AGMARKNET records, Jan 2021–Dec 2024", N["raw"]],
           ["Distinct market names / distinct (state, market) mandis", f"{N['names']} / {N['keys']}"],
           ["Trading dates (of 1,461 calendar days)", N["dates"]],
           ["Records dropped as price outliers", N["outliers"]],
           ["Duplicate mandi-day records collapsed", N["dups"]],
           ["Clean mandi-day observations", N["clean"]],
           ["Mandis with ≥ 200 reporting days", N["eligible"]],
           ["Sample: top 240 by cumulative arrivals", "240"],
           ["Sample mandi-day observations", N["sample_obs"]],
           ["Share of all arrivals covered by the sample", N["arr_cov"] + "%"],
           ["Reporting days per mandi: min / median / mean / max", f"{N['rd_min']} / {N['rd_med']} / {N['rd_mean']} / {N['rd_max']}"],
           ["States / IMD subdivisions in sample", f"{N['states_s']} / {N['subdiv_s']}"],
           ["Community price indices (daily, trading days)", f"{co['K']} × {N['T']}"],
           ["Records with inconsistent coordinates (rainfall not used), sample", f"{N['geo_s_share']}% ({N['geo_s_mandis']} mandis)"],
           ["Pair-month panel for rainfall analysis", f"{mo['n_pairs']} pairs × {mo['n_months']} months = {N['mod_N']}"]]
cs = pd.DataFrame(net["community_table"])
T["t2"] = []
for _, r in cs.iterrows():
    k = int(r["community"][1:])
    de = desc.iloc[k]
    T["t2"].append([f"C{k}", LABELS[k], str(int(r["mandis"])), f(100 * r["arrival_share"], 1), f(de["mean_price_rs"], 0),
                    f(de["within_sd_logp"], 2), str(int(r["simams"])), f(rain["community_mean_deficit"][str(k)], 1),
                    r["top_ec"].replace(" APMC", "").split("; ")[0].split(" (")[0] + ", " + r["top_ec"].replace(" APMC", "").split("; ")[1].split(" (")[0]])
T["t3"] = [["Nodes / edges (3N − 6)", f"240 / {net['N_edges']}"], ["Edge density", N["density"]],
           ["Mean / maximum degree", f"{N['mean_deg']} / {N['max_deg']}"], ["Weighted clustering coefficient", N["clust"]],
           ["Average shortest path / diameter", f"{N['apl']} / {N['diam']}"],
           ["Ledoit–Wolf shrinkage intensity", N["lw"]],
           ["Mean off-diagonal correlation (LW); maximum", f"{N['off_mean']}; {N['off_max']}"],
           ["Mean correlation within / between communities (LW)", f"{N['intra']} / {N['inter']}"],
           ["Mean correlation on retained PMFG edges", N["edge_corr"]],
           ["Fiedler value λ₂(L), weighted Laplacian", N["fiedler"]],
           ["  Same topology, edge weights permuted (mean; 5th pct.)", f"{N['fiedler_perm']}; {N['fiedler_perm_p05']}"],
           ["Fiedler value, normalized Laplacian", N["fiedler_norm"]],
           ["Adjacency eigenvalues λ₁, λ₂; spectral gap", f"{N['lam1']}, {N['lam2']}; {N['spec_gap']}"],
           ["Louvain communities; modularity Q", f"{N['K']}; {N['modularity']}"],
           ["NMI, seed 42 vs 123 / vs 999", f"{N['nmi123']} / {N['nmi999']}"],
           ["NMI vs baseline: shrinkage 0.15 / 0.50 / pairwise correlations",
            f"{N['nmi_a015']} ({N['k_a015']}) / {N['nmi_a050']} ({N['k_a050']}) / {N['nmi_pw']} ({N['k_pw']})"],
           ["Inter-community PMFG edges", N["n_inter"]], ["SIMAMs", N["n_simam"]]]
T["t4"] = []
for _, r in P.iterrows():
    T["t4"].append([r["pair"].replace("-", "–"), f(r["gamma"], 3), f(100 * r["share_outer"], 0),
                    f"{f(r['theta_1'], 3)} ({f(r['theta_1_se'], 3)})", f"{f(r['theta_2'], 3)} ({f(r['theta_2_se'], 3)})",
                    f(r["hl_1"], 2), f(r["hl_2"], 2), pv(r["theta_diff_p"]), f(r["supLR"], 1), pv(r["p_boot"])])
T["t5"] = [["Baseline (arrival-weighted index, L = 2, 15% trim)", "28", N["n_fast"], N["n_fast_sig"], N["n_slow_sig"], N["hl1"], N["hl2"]]]
names = {"fe_index": "Fixed-effects (time-dummy) index", "lags4": "L = 4 lags", "trim10": "10% trimming",
         "reverse": "Reversed normalization (P_i on P_j)", "original_spec": "Single equation, contemporaneous ΔP_i, no lags",
         "weekly_mean": "Weekly means (210 weeks, L = 1)", "weekly_last": "Weekly end-of-week values (L = 1)"}
for k in ["fe_index", "lags4", "trim10", "reverse", "original_spec", "weekly_mean", "weekly_last"]:
    v = rb[k]
    T["t5"].append([names[k], str(v["n_pairs"]), str(v["n_outer_faster"]), str(v["n_outer_faster_sig"]), str(v["n_outer_slower_sig"]),
                    f(v["median_hl1"], 2), f(v["median_hl2"], 2)])
pn = {"alpha015": "Partition from shrinkage 0.15", "alpha050": "Partition from shrinkage 0.50",
      "pairwise": "Partition from pairwise correlations", "states": "State indices (states with ≥ 9 mandis)"}
for k in ["alpha015", "alpha050", "pairwise", "states"]:
    v = asp["partitions"][k]
    T["t5"].append([f"{pn[k]} (K = {v['K']})", str(v["n_pairs"]), str(v["n_outer_faster"]), str(v["n_outer_faster_sig"]),
                    str(v["n_outer_slower_sig"]), f(v["median_hl1"], 2), f(v["median_hl2"], 2)])
T["t6"] = []
lab = {"pdef": "Percentage deficit (IMD departure)", "deficit_ins": "Deficit vs 2021–24 climatology, mm", "zdef_ins": "Standardized deficit vs 2021–24 climatology", "deficit": "Deficit, mm (pair mean)", "deficit_dest": "Deficit, mm (destination)", "zdef": "Standardized deficit",
       "excess": "Excess rainfall, mm", "absz": "|Standardized anomaly|", "dz_gap": "|z_i − z_j| (differential shock)"}
for x in ["deficit", "deficit_dest", "pdef", "zdef", "excess", "absz", "dz_gap", "deficit_ins", "zdef_ins"]:
    row = [lab[x]]
    for fe in ["pair", "pair+cal", "pair+ym"]:
        r = mrow(x, fe)
        star = "***" if r["p"] < 0.01 else "**" if r["p"] < 0.05 else "*" if r["p"] < 0.1 else ""
        row.append(f"{f(r['beta'] * (100 if 'mm' in lab[x] else 1), 3)}{star} ({f(r['se'] * (100 if 'mm' in lab[x] else 1), 3)})")
    T["t6"].append(row)
T["t7"] = []
for _, r in PT.iterrows():
    T["t7"].append([f"P{int(r['percentile'])}", f(r["trigger_rs_q"], 0), f(100 * r["activation_share"], 0), f(r["mean_gap_when_active"], 0),
                    f(r["sd_reduction_pct"], 1), f(r["transfer_rs_mn_per_year"], 0), f(r["triangle_rs_mn_per_year_e0.6"], 0),
                    f(r["release_kt_per_year_e0.6"], 0)])
T["a1"] = [[r["pair"].replace("-", "–"), f(r["alpha"], 3), f(r["beta"], 3), f(r["eg_t"], 2), pv(r["eg_p"]), f(r["eg_t_reverse"], 2),
            f(r["ect_sd"], 3)] for _, r in eg.iterrows()]
T["a2"] = [[r["series"], f(r["adf_level"], 2), f(r["p_level"], 3), f(r["kpss_level"], 3), f(r["adf_diff"], 2), f(r["kpss_diff"], 3)]
           for _, r in ur.iterrows()]
BT = pd.DataFrame(asp["bandtar"])
T["a3"] = [[r["pair"].replace("-", "–"), f(r["gamma_L"], 3), f(r["gamma_U"], 3), f"{r['n_below']}/{r['n_band']}/{r['n_above']}",
            f(r["theta_below"], 3), f(r["theta_above"], 3), pv(r["p_equal"])] for _, r in BT.iterrows()]
cc = pd.read_csv(DER / "centrality_communities.csv", index_col=0)
T["a4"] = []
for k in range(co["K"]):
    s_ = cc[cc.community == k]
    st = s_["State"].value_counts()
    T["a4"].append([f"C{k}", "; ".join(f"{a} {b}" for a, b in st.head(4).items()),
                    "; ".join(x.replace(" APMC", "") for x in s_.nlargest(6, "eigenvector")["Market"])])
T["a5"] = [[r["pair"].replace("-", "–"), f(r["gamma"], 3), f(r["gamma_pct"], 1), f(r["gamma_rs"], 0), f(r["dist_km"], 0),
            str(int(r["pmfg_edges"]))] for _, r in TD.iterrows()]
T["a6"] = [[("Deficit, mm" if r["shock"] == "deficit_mm" else "Percentage deficit"), str(int(r["h"])),
            f(r["beta"] * (10 if r["shock"] == "deficit_mm" else 0.1) * 100, 2), f(r["se"] * (10 if r["shock"] == "deficit_mm" else 0.1) * 100, 2),
            pv(r["p"]), f(r["effect_1sd_pct"], 1)] for _, r in PL.iterrows()]
# ---- noise-robust check between community pairs (14_community_iv.py), Section VI.C
C = R["community_iv"]
N.update(cm_ols_pi=f(C["weekly_ols"]["pi"], 2), cm_ols_p=pv(C["weekly_ols"]["pi_p"]),
         cm_iv_pi=f(C["weekly_iv2"]["pi"], 2), cm_iv_p=pv(C["weekly_iv2"]["pi_p"]),
         cm_iv3_pi=f(C["weekly_iv3"]["pi"], 2), cm_iv3_p=pv(C["weekly_iv3"]["pi_p"]),
         cm_iv_theta=f(C["weekly_iv2"]["theta"], 2), cm_iv_hl=f(C["weekly_iv2"]["half_life_weeks"], 1),
         cm_q_iv_lo=f(C["quintiles_iv"][0], 2), cm_q_iv_hi=f(C["quintiles_iv"][-1], 2),
         cm_q_conv_lo=f(C["quintiles_conv"][0], 2), cm_q_conv_hi=f(C["quintiles_conv"][-1], 2))
N["labels"] = {f"C{k}": v for k, v in LABELS.items()}
(OUT / "paper_numbers.json").write_text(json.dumps({"N": N, "T": T}, indent=1, ensure_ascii=False))
print(len(N), "numbers;", {k: len(v) for k, v in T.items()})
