"""
STEP 3 - THE GATE (validation years only). Usage: python step3_gate.py [ecdc_csv]
Needs: pandas, numpy, scikit-learn.
The locked test years (2022+) are NEVER loaded: the data are cut at VALID_END before anything else.
"""
import sys, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")

CSV = sys.argv[1] if len(sys.argv) > 1 else "ECDC_surveillance_data_Antimicrobial_resistance__1_.csv"
START, TRAIN_END, VALID_START, VALID_END = 2005, 2018, 2019, 2021
HZ = [1, 2, 3, 4, 5]; MIN_N = 30; MIN_YEARS = 8; K = 5; PHI = 0.8
MARGIN = 0.95; B = 2000; MIN_ROWS = 15; SEED = 0
rng = np.random.default_rng(SEED)

RULE = f"""PRE-REGISTERED GATE
 data: years {START}-{VALID_END} only | train: target year <= {TRAIN_END} | scored: target year {VALID_START}-{VALID_END}
 series: >= {MIN_YEARS} observed years in {START}-{TRAIN_END}; N tested >= {MIN_N}; origin needs {K} consecutive observed years
 metric: per pathogen-drug combo, model MAE / persistence MAE, geometric mean over combos (combos need >= {MIN_ROWS} scored forecasts)
 G1: relative error at horizons >= 3 is <= {MARGIN}
 G2: 95% country-cluster bootstrap interval for that relative error has upper bound < 1.0
 G3: beats persistence-free trend baselines (drift, damped drift, SES) AND interval vs best baseline has upper bound < 1.0
 PASS = G1 and G2 and G3 (otherwise: report as benchmark / forecastability study)"""
print(RULE, "\n")

# ------------------------------------------------------------------ load (cut BEFORE anything else)
ec = pd.read_csv(CSV, encoding="utf-8-sig")
ec["Indicator"] = ec["Indicator"].str.strip()
ec["NumValue"] = pd.to_numeric(ec["NumValue"], errors="coerce")
ec["Year"] = pd.to_numeric(ec["Time"], errors="coerce")
ec = ec[ec["Year"].between(START, VALID_END)]
assert ec["Year"].max() <= VALID_END
p = ec[ec.Indicator == "R - resistant isolates, percentage"][["Population", "RegionName", "Year", "NumValue"]].rename(columns={"NumValue": "pctR"})
n = ec[ec.Indicator == "Total tested isolates"][["Population", "RegionName", "Year", "NumValue"]].rename(columns={"NumValue": "N"})
d = p.merge(n, on=["Population", "RegionName", "Year"], how="left").dropna(subset=["pctR"])
d = d[d.N >= MIN_N].copy(); d["Year"] = d.Year.astype(int)
years = list(range(START, VALID_END + 1)); yi = {y: i for i, y in enumerate(years)}
W = d.pivot_table(index=["Population", "RegionName"], columns="Year", values="pctR").reindex(columns=years)
NN = d.pivot_table(index=["Population", "RegionName"], columns="Year", values="N").reindex(columns=years)
keep = W.loc[:, START:TRAIN_END].notna().sum(axis=1) >= MIN_YEARS
W, NN = W[keep], NN[keep]
print("eligible series:", len(W), "| combos:", W.index.get_level_values(0).nunique(), "| countries:", W.index.get_level_values(1).nunique())
cm = W.groupby(level=0).mean()            # combo x year cross-country mean (known at that year)

# ------------------------------------------------------------------ build forecasting rows
def ses_level(arr):
    best = None
    for a in (0.2, 0.4, 0.6, 0.8, 1.0):
        lvl, sse = arr[0], 0.0
        for v in arr[1:]:
            sse += (v - lvl) ** 2; lvl = a * v + (1 - a) * lvl
        if best is None or sse < best[0]: best = (sse, lvl)
    return best[1]
rows = []
for (pop, ctry), r in W.iterrows():
    v = r.values; nv = NN.loc[(pop, ctry)].values
    for j in range(K - 1, len(years)):
        hist = v[j - K + 1:j + 1]
        if np.isnan(hist).any(): continue
        w = 0                                  # consecutive observed values ending at j (max 8)
        while w < 8 and j - w >= 0 and not np.isnan(v[j - w]): w += 1
        lvl = ses_level(v[j - w + 1:j + 1])
        slope = np.polyfit(np.arange(K), hist, 1)[0]
        ch = np.diff(hist)[::-1]               # d1 = y_t - y_{t-1}, d2, d3, d4
        f = dict(y=hist[-1], d1=ch[0], d2=ch[1], d3=ch[2], d4=ch[3], slope5=slope, vol4=np.std(ch),
                 dev_cm=hist[-1] - cm.loc[pop, years[j]], logN=np.log(nv[j]))
        for h in HZ:
            if j + h < len(years) and not np.isnan(v[j + h]):
                rows.append(dict(combo=pop, country=ctry, origin=years[j], h=h, target_year=years[j + h], target=v[j + h],
                                 persist=hist[-1], drift=hist[-1] + slope * h,
                                 damped=hist[-1] + slope * sum(PHI ** i for i in range(1, h + 1)), ses=lvl, **f))
R = pd.DataFrame(rows)
R[["pathogen", "drug"]] = R["combo"].str.split("|", n=1, expand=True)
FEAT = ["y", "d1", "d2", "d3", "d4", "slope5", "vol4", "dev_cm", "logN"]
X = pd.concat([R[FEAT], pd.get_dummies(R["pathogen"], prefix="p"), pd.get_dummies(R["drug"], prefix="d")], axis=1).astype(float)
R["chg"] = R["target"] - R["y"]
tr_all = R["target_year"] <= TRAIN_END
va = R["target_year"].between(VALID_START, VALID_END)
print("training rows:", int(tr_all.sum()), "| validation (scored) rows:", int(va.sum()),
      "| validation series:", R[va].groupby(["combo", "country"]).ngroups)

# ------------------------------------------------------------------ pooled models (one per horizon, predict the CHANGE)
def fit_predict(kind, tr_mask, te_mask):
    pred = pd.Series(np.nan, index=R.index)
    for h in HZ:
        a = tr_mask & (R.h == h); b = te_mask & (R.h == h)
        if a.sum() < 50 or b.sum() == 0: continue
        if kind == "EN":
            sc = StandardScaler().fit(X[a]); g = R.loc[a, "country"].values
            cv = list(GroupKFold(n_splits=min(5, len(np.unique(g)))).split(X[a], R.loc[a, "chg"], g))
            m = ElasticNetCV(l1_ratio=[.2, .5, .8, 1.0], n_alphas=30, cv=cv, max_iter=20000, random_state=0).fit(sc.transform(X[a]), R.loc[a, "chg"])
            pred[b] = m.predict(sc.transform(X[b]))
        else:
            m = GradientBoostingRegressor(loss="huber", n_estimators=150, max_depth=2, learning_rate=0.05,
                                          subsample=0.8, min_samples_leaf=20, random_state=0).fit(X[a], R.loc[a, "chg"])
            pred[b] = m.predict(X[b])
    return R["y"] + pred                       # forecast = last value + predicted change

# ------------------------------------------------------------------ metric helpers
def macro_rel(sub, num, den, w=None):
    """geometric mean over combos of weighted-MAE(num)/weighted-MAE(den)."""
    e1 = (sub["target"] - sub[num]).abs().values; e0 = (sub["target"] - sub[den]).abs().values
    cid = pd.factorize(sub["combo"])[0]; k = cid.max() + 1
    w = np.ones(len(sub)) if w is None else w
    s1 = np.bincount(cid, e1 * w, k); s0 = np.bincount(cid, e0 * w, k); cnt = np.bincount(cid, w, k)
    ok = (cnt >= MIN_ROWS) & (s0 > 0) & (s1 > 0)
    return float(np.exp(np.mean(np.log(s1[ok] / s0[ok])))) if ok.any() else np.nan
def boot(sub, num, den):
    cs = pd.factorize(sub["country"])[0]; kc = cs.max() + 1; out = []
    for _ in range(B):
        cw = np.bincount(rng.integers(0, kc, kc), minlength=kc).astype(float)
        out.append(macro_rel(sub, num, den, cw[cs]))
    out = np.array(out); out = out[~np.isnan(out)]
    return np.percentile(out, 2.5), np.percentile(out, 97.5)

# ------------------------------------------------------------------ model selection INSIDE the training years
print("\nMODEL SELECTION (rolling-origin CV on target years 2015-2018, training years only; horizons >= 3)")
cvsub = {}
for kind in ("EN", "GB"):
    pr = pd.Series(np.nan, index=R.index)
    for T in (2015, 2016, 2017, 2018):
        te = R.target_year == T; trm = R.target_year < T
        pr[te] = fit_predict(kind, trm, te)[te]
    s = R[(R.target_year.between(2015, 2018)) & (R.h >= 3)].copy(); s[kind] = pr[s.index]
    cvsub[kind] = macro_rel(s.dropna(subset=[kind]), kind, "persist")
    print(f"  {kind}: relative error vs persistence = {cvsub[kind]:.3f}")
chosen = min(cvsub, key=cvsub.get)
print("  -> gate model (lower CV error):", chosen)

# ------------------------------------------------------------------ validation
V = R[va].copy()
for kind in ("EN", "GB"):
    V[kind] = fit_predict(kind, tr_all, va)[va]
print("\nVALIDATION (target years %d-%d): relative error vs persistence (1.00 = same; <1 better)" % (VALID_START, VALID_END))
methods = ["drift", "damped", "ses", "EN", "GB"]
tab = {m: {"h=1": macro_rel(V[V.h == 1], m, "persist"), "h=2": macro_rel(V[V.h == 2], m, "persist"),
           "h>=3": macro_rel(V[V.h >= 3], m, "persist"), "h=3": macro_rel(V[V.h == 3], m, "persist"),
           "h=4": macro_rel(V[V.h == 4], m, "persist"), "h=5": macro_rel(V[V.h == 5], m, "persist")} for m in methods}
print(pd.DataFrame(tab).T.round(3).to_string())
pm = V.groupby("h").apply(lambda g: pd.Series({m: (g.target - g[m]).abs().mean() for m in ["persist"] + methods})).round(2)
print("\nPooled MAE (percentage points) by horizon:\n", pm.to_string())

H3 = V[V.h >= 3]
print("\nGATE on %s, horizons >= 3 (%d forecasts):" % (chosen, len(H3)))
g1v = macro_rel(H3, chosen, "persist"); lo, hi = boot(H3, chosen, "persist")
base = {m: macro_rel(H3, m, "persist") for m in ("drift", "damped", "ses")}; best = min(base, key=base.get)
lo3, hi3 = boot(H3, chosen, best)
G1 = g1v <= MARGIN; G2 = hi < 1.0; G3 = all(g1v < base[m] for m in base) and hi3 < 1.0
print(f"  relative error vs persistence: {g1v:.3f}  95% interval [{lo:.3f}, {hi:.3f}]")
print(f"  best trend baseline: {best} ({base[best]:.3f}) | model vs best baseline interval [{lo3:.3f}, {hi3:.3f}]")
print(f"  G1 (<= {MARGIN}): {'PASS' if G1 else 'FAIL'} | G2 (upper bound < 1): {'PASS' if G2 else 'FAIL'} | G3 (beats all trend baselines): {'PASS' if G3 else 'FAIL'}")
print("\n  VERDICT:", "GATE PASSED - proceed to locked test ONCE" if (G1 and G2 and G3) else "GATE NOT PASSED - report as benchmark / forecastability study")

print("\nFor information only (not part of the gate): by target year, relative error of", chosen)
for ty, g in V[V.h >= 3].groupby("target_year"):
    print(f"  {ty}: {macro_rel(g, chosen, 'persist'):.3f}  (n={len(g)})")
cm_ = V[V.h >= 3].groupby("combo").apply(lambda g: pd.Series({"n": len(g), "rel": (g.target - g[chosen]).abs().mean() / (g.target - g.persist).abs().mean()}))
cm_ = cm_[cm_.n >= MIN_ROWS]
print(f"  combos where {chosen} beats persistence (h>=3): {(cm_.rel < 1).sum()} of {len(cm_)}")
