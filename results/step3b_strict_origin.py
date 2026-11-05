"""
STEP 3b - STRICT REAL-TIME RE-RUN of the gate. Usage: python step3b_strict_origin.py [ecdc_csv]
Fixes vs step3_gate.py:
  (1) for every forecast origin O, models are trained ONLY on outcomes with target year <= O (what was known at O);
  (2) all methods are scored on identical rows (rows where any model could not be fitted are dropped for everyone);
  (3) the set of pathogen-drug combos in the interval calculation is fixed from the original sample;
  (4) a second run drops the '>= 8 years up to 2018' series filter (needs only 5 consecutive years at the origin).
Locked test years (2022+) are never loaded.  Needs pandas, numpy, scikit-learn.
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
MARGIN = 0.95; B = 2000; MIN_ROWS = 15; MIN_TRAIN = 50
rng = np.random.default_rng(0)

ec = pd.read_csv(CSV, encoding="utf-8-sig")
ec["Indicator"] = ec["Indicator"].str.strip()
ec["NumValue"] = pd.to_numeric(ec["NumValue"], errors="coerce")
ec["Year"] = pd.to_numeric(ec["Time"], errors="coerce")
ec = ec[ec["Year"].between(START, VALID_END)]; assert ec["Year"].max() <= VALID_END
p = ec[ec.Indicator == "R - resistant isolates, percentage"][["Population", "RegionName", "Year", "NumValue"]].rename(columns={"NumValue": "pctR"})
n = ec[ec.Indicator == "Total tested isolates"][["Population", "RegionName", "Year", "NumValue"]].rename(columns={"NumValue": "N"})
d = p.merge(n, on=["Population", "RegionName", "Year"], how="left").dropna(subset=["pctR"])
d = d[d.N >= MIN_N].copy(); d["Year"] = d.Year.astype(int)
years = list(range(START, VALID_END + 1))
W0 = d.pivot_table(index=["Population", "RegionName"], columns="Year", values="pctR").reindex(columns=years)
N0 = d.pivot_table(index=["Population", "RegionName"], columns="Year", values="N").reindex(columns=years)

def ses_level(arr):
    best = None
    for a in (0.2, 0.4, 0.6, 0.8, 1.0):
        lvl, sse = arr[0], 0.0
        for v in arr[1:]:
            sse += (v - lvl) ** 2; lvl = a * v + (1 - a) * lvl
        if best is None or sse < best[0]: best = (sse, lvl)
    return best[1]

def build(eligibility):
    if eligibility == "registered":
        keep = W0.loc[:, START:TRAIN_END].notna().sum(axis=1) >= MIN_YEARS
    else:
        keep = pd.Series(True, index=W0.index)
    W, NN = W0[keep], N0[keep]; cm = W.groupby(level=0).mean(); rows = []
    for (pop, ctry), r in W.iterrows():
        v = r.values; nv = NN.loc[(pop, ctry)].values
        for j in range(K - 1, len(years)):
            hist = v[j - K + 1:j + 1]
            if np.isnan(hist).any(): continue
            w = 0
            while w < 8 and j - w >= 0 and not np.isnan(v[j - w]): w += 1
            lvl = ses_level(v[j - w + 1:j + 1]); slope = np.polyfit(np.arange(K), hist, 1)[0]; ch = np.diff(hist)[::-1]
            f = dict(y=hist[-1], d1=ch[0], d2=ch[1], d3=ch[2], d4=ch[3], slope5=slope, vol4=np.std(ch),
                     dev_cm=hist[-1] - cm.loc[pop, years[j]], logN=np.log(nv[j]))
            for h in HZ:
                if j + h < len(years) and not np.isnan(v[j + h]):
                    rows.append(dict(combo=pop, country=ctry, origin=years[j], h=h, target_year=years[j + h], target=v[j + h],
                                     persist=hist[-1], drift=hist[-1] + slope * h,
                                     damped=hist[-1] + slope * sum(PHI ** i for i in range(1, h + 1)), ses=lvl, **f))
    R = pd.DataFrame(rows); R[["pathogen", "drug"]] = R["combo"].str.split("|", n=1, expand=True)
    X = pd.concat([R[["y", "d1", "d2", "d3", "d4", "slope5", "vol4", "dev_cm", "logN"]],
                   pd.get_dummies(R["pathogen"], prefix="p"), pd.get_dummies(R["drug"], prefix="d")], axis=1).astype(float)
    R["chg"] = R["target"] - R["y"]
    return R, X

def fit_origin(R, X, kind, O, va):
    pred = pd.Series(np.nan, index=R.index)
    for h in HZ:
        a = (R.target_year <= O) & (R.h == h); b = va & (R.origin == O) & (R.h == h)   # only outcomes known at the origin
        if b.sum() == 0 or a.sum() < MIN_TRAIN: continue
        if kind == "EN":
            sc = StandardScaler().fit(X[a]); g = R.loc[a, "country"].values
            cv = list(GroupKFold(n_splits=min(5, len(np.unique(g)))).split(X[a], R.loc[a, "chg"], g))
            m = ElasticNetCV(l1_ratio=[.2, .5, .8, 1.0], n_alphas=30, cv=cv, max_iter=20000, random_state=0).fit(sc.transform(X[a]), R.loc[a, "chg"])
            pred[b] = m.predict(sc.transform(X[b]))
        else:
            m = GradientBoostingRegressor(loss="huber", n_estimators=150, max_depth=2, learning_rate=0.05,
                                          subsample=0.8, min_samples_leaf=20, random_state=0).fit(X[a], R.loc[a, "chg"])
            pred[b] = m.predict(X[b])
    return R["y"] + pred

def macro_rel(sub, num, den, combos, w=None):
    e1 = (sub["target"] - sub[num]).abs().values; e0 = (sub["target"] - sub[den]).abs().values
    cid = pd.Categorical(sub["combo"], categories=combos).codes; ok_row = cid >= 0
    w = np.ones(len(sub)) if w is None else w; k = len(combos)
    s1 = np.bincount(cid[ok_row], (e1 * w)[ok_row], k); s0 = np.bincount(cid[ok_row], (e0 * w)[ok_row], k); cnt = np.bincount(cid[ok_row], w[ok_row], k)
    ok = (cnt > 0) & (s0 > 0) & (s1 > 0)                    # combo list is FIXED; only zero-weight combos are skipped
    return float(np.exp(np.mean(np.log(s1[ok] / s0[ok])))) if ok.any() else np.nan

def boot(sub, num, den, combos):
    cs = pd.factorize(sub["country"])[0]; kc = cs.max() + 1; out = []
    for _ in range(B):
        cw = np.bincount(rng.integers(0, kc, kc), minlength=kc).astype(float)
        out.append(macro_rel(sub, num, den, combos, cw[cs]))
    out = np.array(out); out = out[~np.isnan(out)]
    return np.percentile(out, 2.5), np.percentile(out, 97.5)

def run(eligibility):
    print("=" * 78, f"\nRUN: eligibility = {eligibility}\n" + "=" * 78)
    R, X = build(eligibility)
    va = R["target_year"].between(VALID_START, VALID_END)
    print("series:", R.groupby(["combo", "country"]).ngroups, "| scored rows before dropping:", int(va.sum()))
    for kind in ("EN", "GB"):
        pr = pd.Series(np.nan, index=R.index)
        for O in range(VALID_START - 5, VALID_END):            # origins 2014..2020
            pr = pr.fillna(fit_origin(R, X, kind, O, va))
        R[kind] = pr
    V = R[va].dropna(subset=["EN", "GB"]).copy()
    print("scored rows with BOTH models fitted (identical rows for every method):", len(V), "| dropped:", int(va.sum()) - len(V))
    print(V.groupby("h").size().rename("rows by horizon").to_string())
    H3 = V[V.h >= 3]
    combos = [c for c, k in H3.groupby("combo").size().items() if k >= MIN_ROWS]     # fixed from the ORIGINAL sample
    print("combos in the metric (fixed):", len(combos))
    methods = ["drift", "damped", "ses", "EN", "GB"]
    tab = {m: {f"h={h}": macro_rel(V[V.h == h], m, "persist", combos) for h in HZ} | {"h>=3": macro_rel(H3, m, "persist", combos)} for m in methods}
    print("\nrelative error vs persistence (strict real-time training); 1.00 = same, <1 better")
    print(pd.DataFrame(tab).T[["h=1", "h=2", "h=3", "h=4", "h=5", "h>=3"]].round(3).to_string())
    print("\npooled MAE (percentage points):")
    print(V.groupby("h").apply(lambda g: pd.Series({m: (g.target - g[m]).abs().mean() for m in ["persist"] + methods})).round(2).to_string())
    for chosen in ("GB", "EN"):
        g = macro_rel(H3, chosen, "persist", combos); lo, hi = boot(H3, chosen, "persist", combos)
        base = {m: macro_rel(H3, m, "persist", combos) for m in ("drift", "damped", "ses")}; best = min(base, key=base.get)
        lo3, hi3 = boot(H3, chosen, best, combos)
        G1, G2, G3 = g <= MARGIN, hi < 1.0, (all(g < base[m] for m in base) and hi3 < 1.0)
        print(f"\nGATE ({chosen}, h>=3, {len(H3)} forecasts): rel.err {g:.3f} [{lo:.3f}, {hi:.3f}] | best baseline {best} {base[best]:.3f}; model vs best [{lo3:.3f}, {hi3:.3f}]")
        print(f"  G1 {'PASS' if G1 else 'FAIL'} | G2 {'PASS' if G2 else 'FAIL'} | G3 {'PASS' if G3 else 'FAIL'} -> {'PASSED' if (G1 and G2 and G3) else 'NOT PASSED'}")
    print("\nSES vs persistence (h>=3) interval:", tuple(round(x, 3) for x in boot(H3, "ses", "persist", combos)))

run("registered")
run("origin_only")
