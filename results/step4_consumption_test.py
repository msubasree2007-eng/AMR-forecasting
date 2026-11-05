"""
STEP 4 - H4: does lagged antibiotic consumption improve strict real-time AMR forecasts?
Usage: python step4_consumption_test.py [ecdc_csv] [esac_clean_v2_csv] [consumption_column]
Locked test years (2022+) are never loaded. Needs pandas, numpy, scikit-learn.

PRE-REGISTERED RULE
  model: gradient boosting, one per horizon, trained per origin O on outcomes with target year <= O only
  with-consumption vs without-consumption fitted on IDENTICAL rows (rows lacking consumption features are dropped for both)
  consumption features: DDD at origin, 1-year change, 3-year change (matched drug class, total care, rule-B clean values)
  eligible countries: >= 8 clean total-care (J01) years in 2005-2018
  H4 SUPPORTED only if relative error (with / without) <= 0.97 at horizons >= 3 AND the 95% country-cluster interval upper bound < 1.0
"""
import sys, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
warnings.filterwarnings("ignore")
CSV = sys.argv[1] if len(sys.argv) > 1 else "ECDC_surveillance_data_Antimicrobial_resistance__1_.csv"
ESAC = sys.argv[2] if len(sys.argv) > 2 else "ESAC_Net_consumption_clean_v2.csv"
COL = sys.argv[3] if len(sys.argv) > 3 else "DDD_clean_B"   # DDD_clean_B (primary) | DDD_clean_A | DDD_raw (sensitivity runs)
START, TRAIN_END, VALID_START, VALID_END = 2005, 2018, 2019, 2021
HZ = [1, 2, 3, 4, 5]; MIN_N = 30; MIN_YEARS = 8; K = 5; MIN_ROWS = 15; MIN_TRAIN = 50; B = 2000; MARGIN = 0.97
rng = np.random.default_rng(0)
PAIR = {"Carbapenems": "J01DH", "Fluoroquinolones": "J01MA", "Aminoglycosides": "J01G", "Third-generation cephalosporins": "J01DD",
        "Ceftazidime": "J01DD", "Macrolides": "J01FA", "Vancomycin": "J01XA", "Aminopenicillins": "J01CA"}

ec = pd.read_csv(CSV, encoding="utf-8-sig"); ec["Indicator"] = ec["Indicator"].str.strip()
ec["NumValue"] = pd.to_numeric(ec["NumValue"], errors="coerce"); ec["Year"] = pd.to_numeric(ec["Time"], errors="coerce")
ec = ec[ec["Year"].between(START, VALID_END)]; assert ec["Year"].max() <= VALID_END
p = ec[ec.Indicator == "R - resistant isolates, percentage"][["Population", "RegionName", "Year", "NumValue"]].rename(columns={"NumValue": "pctR"})
n = ec[ec.Indicator == "Total tested isolates"][["Population", "RegionName", "Year", "NumValue"]].rename(columns={"NumValue": "N"})
d = p.merge(n, on=["Population", "RegionName", "Year"], how="left").dropna(subset=["pctR"]); d = d[d.N >= MIN_N].copy(); d["Year"] = d.Year.astype(int)
years = list(range(START, VALID_END + 1))
W = d.pivot_table(index=["Population", "RegionName"], columns="Year", values="pctR").reindex(columns=years)
NN = d.pivot_table(index=["Population", "RegionName"], columns="Year", values="N").reindex(columns=years)
W = W[W.loc[:, START:TRAIN_END].notna().sum(axis=1) >= MIN_YEARS]; NN = NN.loc[W.index]; cm = W.groupby(level=0).mean()

A = pd.read_csv(ESAC); A = A[A.Year <= VALID_END]
tot = A[(A.ATC == "J01") & A.Year.between(START, TRAIN_END) & A.DDD_clean_B.notna()].groupby("Country").Year.nunique()
elig = set(tot[tot >= MIN_YEARS].index)
C = A.pivot_table(index=["ATC", "Country"], columns="Year", values=COL).reindex(columns=years)

def ses_level(arr):
    best = None
    for a in (0.2, 0.4, 0.6, 0.8, 1.0):
        lvl, sse = arr[0], 0.0
        for v in arr[1:]: sse += (v - lvl) ** 2; lvl = a * v + (1 - a) * lvl
        if best is None or sse < best[0]: best = (sse, lvl)
    return best[1]

rows = []
for (pop, ctry), r in W.iterrows():
    drug = pop.split("|", 1)[1]
    if drug not in PAIR or ctry not in elig or (PAIR[drug], ctry) not in C.index: continue
    cs = C.loc[(PAIR[drug], ctry)].values; v = r.values; nv = NN.loc[(pop, ctry)].values
    for j in range(K - 1, len(years)):
        hist = v[j - K + 1:j + 1]
        if np.isnan(hist).any() or j < 3 or np.isnan(cs[j]) or np.isnan(cs[j - 1]) or np.isnan(cs[j - 3]): continue
        w = 0
        while w < 8 and j - w >= 0 and not np.isnan(v[j - w]): w += 1
        lvl = ses_level(v[j - w + 1:j + 1]); slope = np.polyfit(np.arange(K), hist, 1)[0]; ch = np.diff(hist)[::-1]
        f = dict(y=hist[-1], d1=ch[0], d2=ch[1], d3=ch[2], d4=ch[3], slope5=slope, vol4=np.std(ch),
                 dev_cm=hist[-1] - cm.loc[pop, years[j]], logN=np.log(nv[j]),
                 c_t=cs[j], c_d1=cs[j] - cs[j - 1], c_d3=cs[j] - cs[j - 3])
        for h in HZ:
            if j + h < len(years) and not np.isnan(v[j + h]):
                rows.append(dict(combo=pop, country=ctry, origin=years[j], h=h, target_year=years[j + h], target=v[j + h], persist=hist[-1], ses=lvl, **f))
R = pd.DataFrame(rows); R[["pathogen", "drug"]] = R["combo"].str.split("|", n=1, expand=True); R["chg"] = R.target - R.y
base = ["y", "d1", "d2", "d3", "d4", "slope5", "vol4", "dev_cm", "logN"]; cons = ["c_t", "c_d1", "c_d3"]
dm = pd.concat([pd.get_dummies(R["pathogen"], prefix="p"), pd.get_dummies(R["drug"], prefix="d")], axis=1).astype(float)
X0 = pd.concat([R[base], dm], axis=1).astype(float); X1 = pd.concat([R[base + cons], dm], axis=1).astype(float)
va = R.target_year.between(VALID_START, VALID_END)
print("eligible countries:", len(elig), "| series used:", R.groupby(["combo", "country"]).ngroups, "| combos:", R.combo.nunique(),
      "| training+scored rows:", len(R), "| scored rows:", int(va.sum()))

def fit(X, O):
    pred = pd.Series(np.nan, index=R.index)
    for h in HZ:
        a = (R.target_year <= O) & (R.h == h); b = va & (R.origin == O) & (R.h == h)
        if b.sum() == 0 or a.sum() < MIN_TRAIN: continue
        m = GradientBoostingRegressor(loss="huber", n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8,
                                      min_samples_leaf=20, random_state=0).fit(X[a], R.loc[a, "chg"])
        pred[b] = m.predict(X[b])
    return R.y + pred
P0 = pd.Series(np.nan, index=R.index); P1 = P0.copy()
for O in range(VALID_START - 5, VALID_END):
    P0 = P0.fillna(fit(X0, O)); P1 = P1.fillna(fit(X1, O))
R["gb_without"], R["gb_with"] = P0, P1
V = R[va].dropna(subset=["gb_without", "gb_with"]).copy(); H3 = V[V.h >= 3]
combos = [c for c, k in H3.groupby("combo").size().items() if k >= MIN_ROWS]
print("scored rows (identical for all methods):", len(V), "| rows at h>=3:", len(H3), "| combos in metric:", len(combos))

def macro(sub, num, den, w=None):
    e1 = (sub.target - sub[num]).abs().values; e0 = (sub.target - sub[den]).abs().values
    cid = pd.Categorical(sub.combo, categories=combos).codes; ok_r = cid >= 0; w = np.ones(len(sub)) if w is None else w; k = len(combos)
    s1 = np.bincount(cid[ok_r], (e1 * w)[ok_r], k); s0 = np.bincount(cid[ok_r], (e0 * w)[ok_r], k); c = np.bincount(cid[ok_r], w[ok_r], k)
    ok = (c > 0) & (s0 > 0) & (s1 > 0); return float(np.exp(np.mean(np.log(s1[ok] / s0[ok]))))
def boot(sub, num, den):
    cs_ = pd.factorize(sub.country)[0]; kc = cs_.max() + 1; out = []
    for _ in range(B): out.append(macro(sub, num, den, np.bincount(rng.integers(0, kc, kc), minlength=kc).astype(float)[cs_]))
    return np.percentile(out, 2.5), np.percentile(out, 97.5)

print("\nrelative error vs persistence (1.00 = same, <1 better)")
t = {m: {f"h={h}": macro(V[V.h == h], m, "persist") for h in HZ} | {"h>=3": macro(H3, m, "persist")} for m in ["ses", "gb_without", "gb_with"]}
print(pd.DataFrame(t).T[["h=1", "h=2", "h=3", "h=4", "h=5", "h>=3"]].round(3).to_string())
print("\npooled MAE (pp):"); print(V.groupby("h").apply(lambda g: pd.Series({m: (g.target - g[m]).abs().mean() for m in ["persist", "ses", "gb_without", "gb_with"]})).round(2).to_string())
ratio = macro(H3, "gb_with", "gb_without"); lo, hi = boot(H3, "gb_with", "gb_without")
lo_p, hi_p = boot(H3, "gb_with", "persist")
print(f"\nH4 test (h>=3, {len(H3)} forecasts): with/without ratio = {ratio:.3f}  95% interval [{lo:.3f}, {hi:.3f}]")
print(f"  gb_with vs persistence = {macro(H3, 'gb_with', 'persist'):.3f} [{lo_p:.3f}, {hi_p:.3f}]")
print("  VERDICT H4:", "SUPPORTED" if (ratio <= MARGIN and hi < 1.0) else "NOT SUPPORTED")
for h in (1, 2):
    s = V[V.h == h]; print(f"  (information only) h={h}: with/without = {macro(s, 'gb_with', 'gb_without'):.3f}")
