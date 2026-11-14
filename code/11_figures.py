"""11 - Figures for the paper (static, print-ready PNG at 300 dpi)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DER, FIG, load_results

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "axes.edgecolor": "#555",
                     "xtick.color": "#333", "ytick.color": "#333", "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.5, "legend.frameon": False})
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MK = ["o", "s", "^", "D", "v", "P", "X", "h"]
INK, MUTED = "#222222", "#777777"
R = load_results()

# ---- Figure 1: geography of communities
cc = pd.read_csv(DER / "centrality_communities.csv", index_col=0)
# verified coordinates (00_geocode_check.py) replace the raw-file coordinates
coords = pd.read_csv(DER / "mandi_coordinates.csv").set_index("mandi")
cc["Latitude"], cc["Longitude"] = coords.loc[cc.index, "lat"].values, coords.loc[cc.index, "lon"].values
names = R["paper_labels"] if "paper_labels" in R else {}
fig, ax = plt.subplots(figsize=(7.4, 6.2))
ok = cc["Latitude"].between(6, 37) & cc["Longitude"].between(68, 98)
for k in sorted(cc["community"].unique()):
    s = cc[(cc["community"] == k) & ok]
    ax.scatter(s["Longitude"], s["Latitude"], s=14 + 220 * s["eigenvector"], c=PAL[k], marker=MK[k],
               edgecolors="white", linewidths=0.4, alpha=0.9, label=f"C{k} (n={int((cc['community'] == k).sum())})", zorder=3)
for lab in cc.nlargest(3, "eigenvector").index.tolist() + ["Maharashtra|Lasalgaon APMC", "NCT of Delhi|APMC Azadpur"]:
    if lab in cc.index and ok[lab]:
        r = cc.loc[lab]
        ax.annotate(r["Market"].replace(" APMC", ""), (r["Longitude"], r["Latitude"]), xytext=(4, 4),
                    textcoords="offset points", fontsize=7, color=INK)
ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
ax.set_aspect("equal"); ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, title="Community", title_fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "fig1_communities_map.png", dpi=300); plt.close(fig)

# ---- Figure 2: theta_1 vs theta_2
t = pd.DataFrame(R["tvecm"]["pairs"])
fig, ax = plt.subplots(figsize=(5.2, 4.8))
lim = (-1.0, 0.2)
ax.plot(lim, lim, color=MUTED, lw=0.8, ls="--", zorder=1)
sig = t["theta_diff_p"] < 0.05
for m, face, lab in ((sig, PAL[0], "Regime difference significant (5%)"), (~sig, "white", "Not significant")):
    s = t[m]
    ax.errorbar(s["theta_1"], s["theta_2"], xerr=1.96 * s["theta_1_se"], yerr=1.96 * s["theta_2_se"], fmt="none",
                ecolor="#b5b5b5", elinewidth=0.6, zorder=2)
    ax.scatter(s["theta_1"], s["theta_2"], s=30, facecolors=face, edgecolors=PAL[0], linewidths=1.0, zorder=3, label=lab)
for _, r in t[t["theta_2"] > t["theta_1"]].iterrows():
    ax.annotate(r["pair"].replace("-", "–"), (r["theta_1"], r["theta_2"]), xytext=(5, 3), textcoords="offset points", fontsize=7)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel(r"$\theta_1$: adjustment, inner regime ($|ECT_{t-1}|\leq\gamma$)")
ax.set_ylabel(r"$\theta_2$: adjustment, outer regime ($|ECT_{t-1}|>\gamma$)")
ax.text(-0.97, 0.12, "Below the diagonal: large gaps close faster", fontsize=7, color=MUTED)
ax.legend(loc="lower right", fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "fig2_tvecm_regimes.png", dpi=300); plt.close(fig)

# ---- Figure 3: rainfall local projections
lp = pd.DataFrame(R["rain_timing"]["results"])
lp = lp[lp["shock"] == "deficit"]
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
for ax, cal, title in zip(axes, (False, True), ("Community FE only", "Community + calendar-month FE")):
    s = lp[lp["calendar_fe"] == cal]
    b = 100 * s["beta"] * 10; se = 100 * s["se"] * 10
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.errorbar(s["h_months"], b, yerr=1.96 * se, fmt="o", color=PAL[1], ms=4, lw=1.2, capsize=0)
    ax.set_title(title, fontsize=8.5); ax.set_xlabel("Horizon h (months)"); ax.set_xticks(range(4))
axes[0].set_ylabel("Price response (%) to +10 mm deficit")
fig.tight_layout(); fig.savefig(FIG / "fig3_rain_local_projections.png", dpi=300); plt.close(fig)

# ---- Figure 4: rolling Fiedler and system deficit (two panels, no dual axis)
f = pd.read_csv(DER / "fiedler_rolling.csv", index_col=0, parse_dates=True)
sysr = pd.read_parquet(DER / "rain_system_month.parquet"); sysr.index = sysr["ym"].dt.to_timestamp()
fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.0), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
axes[0].plot(f.index, f["fiedler_240"] * 1e3, color=PAL[0], lw=1.4)
axes[0].set_ylabel("Fiedler value (×10⁻³)")
axes[1].bar(sysr.index, sysr["deficit_mm"], width=20, color=PAL[1])
axes[1].set_ylabel("System deficit (mm)")
fig.tight_layout(); fig.savefig(FIG / "fig4_rolling_fiedler.png", dpi=300); plt.close(fig)
print("figures written")
