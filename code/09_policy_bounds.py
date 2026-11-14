"""09 - Buffer-stock trigger accounting (illustrative upper bounds).

What the original notebook computed: on days the system price index exceeds a trigger, price is
set equal to the trigger; "volatility reduction" is the fall in the SD of this truncated series and
"consumer surplus" is sum((P - trigger) x arrivals). The "optimal" trigger was the lowest grid
trigger with activation frequency below 20%, i.e. approximately the 80th percentile by construction.
Two errors are corrected: arrivals are in tonnes and prices in Rs/quintal (1 t = 10 q), so the
rectangle must be multiplied by 10; and the quantity is a transfer to buyers at sampled APMCs if
releases could hold the price at the trigger, not a welfare gain. We report: the transfer upper
bound, the efficiency (Harberger) triangle under a demand elasticity e, and the release volume
needed to achieve the cap, dQ = |e| * (P - trigger)/P * Q.
"""
import numpy as np
import pandas as pd

from common import DER, save_results, write_table

panel = pd.read_parquet(DER / "panel240.parquet")
panel["w"] = panel["Arrivals_MT"].clip(lower=0.01)
panel["wl"] = panel["w"] * panel["logp"]
g = panel.groupby("Date")
P = np.exp(g["wl"].sum() / g["w"].sum())
Q = g["Arrivals_MT"].sum()
years = (P.index.max() - P.index.min()).days / 365.25
pct = {q: float(np.percentile(P, q)) for q in (10, 25, 50, 75, 80, 90, 95)}
rows = []
for q in (50, 60, 70, 75, 80, 85, 90):
    trig = np.percentile(P, q)
    on = P > trig
    capped = P.where(~on, trig)
    gap = (P - trig).where(on, 0)
    transfer = (gap * Q * 10).sum()                                   # Rs
    row = {"percentile": q, "trigger_rs_q": trig, "activation_share": on.mean(),
           "mean_gap_when_active": gap[on].mean(), "sd_reduction_pct": 100 * (1 - capped.std() / P.std()),
           "transfer_rs_mn_per_year": transfer / 1e6 / years}
    for e in (0.4, 0.6, 0.8):
        dQ = e * gap / P * Q                                         # tonnes
        row[f"triangle_rs_mn_per_year_e{e}"] = (0.5 * gap * dQ * 10).sum() / 1e6 / years
        row[f"release_kt_per_year_e{e}"] = dQ.sum() / 1e3 / years
    rows.append(row)
tab = pd.DataFrame(rows)
write_table(tab.round(2), "table_policy_bounds")
save_results("policy", {"percentiles": pct, "mean": float(P.mean()), "sd": float(P.std()), "cv": float(P.std() / P.mean()),
                        "years": years, "mean_daily_arrivals_t": float(Q.mean()), "table": tab.to_dict("records"),
                        "original_transfer_p81_rs_mn_total_no_x10": 6037.8})
print(pct, P.mean(), P.std(), P.std() / P.mean())
print(tab.round(2).to_string())
