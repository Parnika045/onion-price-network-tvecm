"""00 - Check and correct mandi coordinates against an offline gazetteer (record level).

The source file attaches coordinates to every record, and the same mandi often appears with
several coordinate pairs (1,174 of 1,546 mandis have more than one). Some are far from the mandi's
state: Anand (Gujarat) has 823 records at 22.47N 72.74E and 270 records at 27.10N 83.27E, a point in
eastern Uttar Pradesh. This matters beyond maps: the file's daily rainfall is read from the IMD
grid AT the record's coordinates, so a misplaced record carries rainfall from the wrong location.

Gazetteer: GeoNames cities1000, Indian places (3,435 places with state), as bundled with the
reverse_geocoder 1.5.1 package (data/raw/geonames_cities1000_india.csv, the Indian places of the worldwide file; GeoNames, CC-BY 4.0).

Procedure
 1. Coordinate variants: each mandi's distinct (lat, lon) pairs rounded to 0.01 deg.
 2. A variant is IN-STATE if the nearest gazetteer place in the mandi's stated state is <= 100 km
    away (tolerates mandis near state borders).
 3. Canonical location = the in-state variant with the most records. If no variant is in-state,
    the mandi is relocated to a gazetteer place in the stated state matching the market name, else
    the district name, else the median location of other mandis in the same state and district.
 4. A record is FLAGGED if its variant is out of state or lies > 50 km from the canonical location.
    Flagged records keep their prices but their rainfall is not used (03_rainfall.py).
Outputs: data/derived/mandi_coordinates.csv (canonical location per mandi),
data/derived/record_location_flags.parquet, results['geocode'].
"""
import re

import numpy as np
import pandas as pd

from common import DER, RAW, save_results, write_table

STATE_KM, CANON_KM = 100.0, 50.0
STATE_MAP = {"NCT of Delhi": "NCT", "Jammu and Kashmir": "Kashmir", "Chandigarh": "Chandigarh"}

raw = pd.read_parquet(DER / "master_raw.parquet")
raw["mandi"] = raw["State"].str.strip() + "|" + raw["Market"].str.strip()
raw["vlat"], raw["vlon"] = raw["Latitude"].round(2), raw["Longitude"].round(2)
g = pd.read_csv(RAW / "geonames_cities1000_india.csv")
g = g[g["cc"].str.strip() == "IN"].copy()


def norm(s):
    s = re.sub(r"\(.*?\)", " ", str(s).lower())
    s = re.sub(r"\b(apmc|market|mandi|agriculture produce market comitee|f&v|grain|veg|yard|sub|city)\b", " ", s)
    return re.sub(r"[^a-z]", "", s)


def hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


g["key"] = g["name"].map(norm)
V = raw.groupby(["mandi", "State", "vlat", "vlon"]).size().rename("n").reset_index()
V["gstate"] = V["State"].map(lambda s: STATE_MAP.get(s, s))
dist = []
for _, r in V.iterrows():
    gs = g[g["admin1"] == r["gstate"]]
    dist.append(hav(r["vlat"], r["vlon"], gs["lat"].values, gs["lon"].values).min() if len(gs) else np.nan)
V["dist_state_km"] = dist
V["in_state"] = V["dist_state_km"] <= STATE_KM

info = raw.groupby("mandi").agg(State=("State", "first"), Market=("Market", "first"),
                                District=("District", lambda s: s.mode().iloc[0]), n=("Date", "size"))
ALIAS = {"prayagraj": "allahabad", "ahilyanagar": "ahmadnagar", "chattrapatisambhajinagar": "aurangabad",
         "raibareilly": "raebareli", "muradabad": "moradabad", "bijnaur": "bijnor", "devariya": "deoria",
         "bulandshahr": "bulandshahr", "amrawati": "amravati", "nasik": "nashik", "gurugram": "gurgaon",
         "mysuru": "mysore", "bengaluru": "bangalore", "belagavi": "belgaum", "vijayapura": "bijapur",
         "kalaburagi": "gulbarga", "hubballi": "hubli", "shivamogga": "shimoga", "tumakuru": "tumkur",
         "ballari": "bellary", "kolkata": "calcutta", "mumbai": "bombay", "badayoun": "budaun",
         "haathras": "hathras", "barabanki": "barabanki", "gazipur": "ghazipur", "sultanpur": "sultanpur",
         "faizabad": "faizabad", "ayodhya": "faizabad", "lakhimpur": "lakhimpur", "basti": "basti",
         "ambedkarnagar": "ambedkarnagar", "sonbhadra": "robertsganj", "kanpurdehat": "akbarpur"}
from difflib import SequenceMatcher


def key(s):
    k = norm(s); return ALIAS.get(k, k)


def sim(a, b):
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


g["k2"] = g["admin2"].fillna("").map(lambda s: ALIAS.get(norm(s), norm(s)))
g["key"] = g["key"].map(lambda k: ALIAS.get(k, k))


def anchor(m):
    st = STATE_MAP.get(info.loc[m, "State"], info.loc[m, "State"]); gs = g[g["admin1"] == st]
    mk, dk = key(info.loc[m, "Market"]), key(info.loc[m, "District"])
    if gs.empty:
        return None, "no gazetteer state"
    ms = gs["key"].map(lambda x: sim(x, mk))
    cand = gs[ms >= 0.85]
    if len(cand):
        dmatch = cand[cand["k2"].map(lambda x: sim(x, dk)) >= 0.8]
        if len(dmatch):
            return dmatch.iloc[0], "gazetteer: market name in district"
        if len(cand) == 1:
            return cand.iloc[0], "gazetteer: market name"
    ds = gs["key"].map(lambda x: sim(x, dk))
    dc = gs[ds >= 0.85]
    if len(dc):
        dd = dc[dc["k2"].map(lambda x: sim(x, dk)) >= 0.8]
        return (dd.iloc[0] if len(dd) else dc.iloc[0]), "gazetteer: district seat"
    return None, "no gazetteer match"


canon = []
for m, grp in V.groupby("mandi"):
    a, how = anchor(m)
    ins = grp[grp["in_state"]].sort_values("n", ascending=False)
    if a is not None:
        dv = hav(grp["vlat"].values, grp["vlon"].values, a["lat"], a["lon"])
        near = grp.assign(d=dv)[(dv <= CANON_KM) & grp["in_state"].values].sort_values("n", ascending=False)
        if len(near):
            r = near.iloc[0]
            canon.append({"mandi": m, "lat": r["vlat"], "lon": r["vlon"], "method": "variant confirmed by " + how,
                          "anchor_km": float(r["d"])})
        else:
            canon.append({"mandi": m, "lat": a["lat"], "lon": a["lon"], "method": "relocated to " + how, "anchor_km": 0.0})
    elif len(ins):
        canon.append({"mandi": m, "lat": ins.iloc[0]["vlat"], "lon": ins.iloc[0]["vlon"],
                      "method": "dominant in-state variant (no gazetteer match)", "anchor_km": np.nan})
    else:
        canon.append({"mandi": m, "lat": np.nan, "lon": np.nan, "method": "unresolved", "anchor_km": np.nan})
C = pd.DataFrame(canon).set_index("mandi").join(info)
# fill unresolved from same-district peers with a dominant in-state location
for m in C.index[C["method"] == "unresolved"]:
    peers = C[(C["State"] == C.loc[m, "State"]) & (C["District"].str.lower() == str(C.loc[m, "District"]).lower())
              & C["method"].str.startswith("variant confirmed")]
    if len(peers):
        C.loc[m, ["lat", "lon", "method"]] = [peers["lat"].median(), peers["lon"].median(), "same-district mandis"]

V = V.join(C[["lat", "lon"]].rename(columns={"lat": "clat", "lon": "clon"}), on="mandi")
V["dist_canon_km"] = hav(V["vlat"], V["vlon"], V["clat"], V["clon"])
V["flag"] = (~V["in_state"]) | (V["dist_canon_km"] > CANON_KM) | V["clat"].isna()
flags = raw[["mandi", "Date", "vlat", "vlon"]].merge(V[["mandi", "vlat", "vlon", "flag", "dist_canon_km"]],
                                                     on=["mandi", "vlat", "vlon"], how="left")
flags[["mandi", "Date", "vlat", "vlon", "flag", "dist_canon_km"]].to_parquet(DER / "record_location_flags.parquet")
C.reset_index().to_csv(DER / "mandi_coordinates.csv", index=False)
V.to_csv(DER / "coordinate_variants.csv", index=False)

meta = pd.read_csv(DER / "meta240.csv", index_col=0)
fs = flags[flags["mandi"].isin(meta.index)]
mand_flag = V[V["flag"]].groupby("mandi")["n"].sum()
out = {"state_km": STATE_KM, "canon_km": CANON_KM, "n_mandis": len(C),
       "n_mandis_multi_variant": int((V.groupby("mandi").size() > 1).sum()),
       "n_mandis_any_flagged": int(len(mand_flag)), "n_records_flagged": int(flags["flag"].sum()),
       "share_records_flagged": float(flags["flag"].mean()),
       "n_records_out_of_state": int(raw.merge(V[~V["in_state"]][["mandi", "vlat", "vlon"]], on=["mandi", "vlat", "vlon"]).shape[0]),
       "canonical_methods": C["method"].value_counts().to_dict(),
       "sample_mandis_any_flagged": int(mand_flag.index.isin(meta.index).sum()),
       "sample_records_flagged": int(fs["flag"].sum()), "sample_share_records_flagged": float(fs["flag"].mean()),
       "sample_canonical_methods": C.loc[meta.index, "method"].value_counts().to_dict()}
save_results("geocode", out)
show = (V[V["flag"] & V["mandi"].isin(meta.index)].groupby("mandi")
        .agg(flagged_records=("n", "sum"), max_km_from_canonical=("dist_canon_km", "max")).join(C[["lat", "lon", "method"]])
        .sort_values("flagged_records", ascending=False).reset_index())
write_table(show.round(2), "table_geocode_flags_sample")
print(out); print(show.head(25).round(1).to_string(index=False))


