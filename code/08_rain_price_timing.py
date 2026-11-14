"""08 - Timing of rainfall effects on community price levels (monthly local projections).

The original notebook regressed h-day forward cumulative returns on a monthly deficit that is
constant within the month, using overlapping daily observations and HC3 errors, and called this a
Granger-causality horizon test. Because the deficit is observed monthly, horizons shorter than a
month are not identified, overlapping windows invalidate HC3 inference, and without calendar-month
controls the deficit proxies for the seasonal price cycle. Here:
  y_{c,m+h} - y_{c,m-1} = b_h * shock_{c,m} + community FE + calendar-month FE + e,  h = 0..3
with y the monthly mean of the daily community log-price index and standard errors clustered by
year-month. A version without calendar-month FE is reported to show the seasonal confound.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import DER, save_results, write_table

idx = pd.read_parquet(DER / "community_index.parquet")
rain = pd.read_parquet(DER / "rain_community_month.parquet")
ym = idx.index.to_period("M")
mon = idx.groupby(ym).mean()
rows = []
for c in mon.columns:
    k = int(c[1:])
    r = rain[rain["community"] == k].set_index("ym")
    s = mon[c]
    for h in range(0, 4):
        d = pd.DataFrame({"dy": s.shift(-h) - s.shift(1), "deficit": r["deficit_mm"], "zdef": r["zdef"], "z": r["z"], "deficit_ins": r["deficit_ins"], "pdef": r["pdef"]})
        d["community"] = c; d["h"] = h; d["ym"] = d.index.astype(str); d["month"] = d.index.month
        rows.append(d.dropna())
D = pd.concat(rows)

res = []
for x in ("deficit", "pdef", "zdef", "deficit_ins"):
    for h in range(4):
        dh = D[D["h"] == h]
        for cal in (False, True):
            X = [sm.add_constant(dh[[x]]), pd.get_dummies(dh["community"], drop_first=True, dtype=float)]
            if cal:
                X.append(pd.get_dummies(dh["month"], prefix="m", drop_first=True, dtype=float))
            X = pd.concat(X, axis=1)
            m = sm.OLS(dh["dy"], X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dh["ym"])[0]})
            res.append({"shock": x, "h_months": h, "calendar_fe": cal, "beta": m.params[x], "se": m.bse[x],
                        "p": m.pvalues[x], "N": int(m.nobs),
                        "effect_1sd_pct": 100 * m.params[x] * dh[x].std()})
res = pd.DataFrame(res)
write_table(res.round(4), "table_rain_price_timing")
save_results("rain_timing", {"results": res.to_dict("records")})
print(res.round(4).to_string())
