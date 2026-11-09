"""amr_core.py - shared functions for step5 (intervals) and step6 (locked test).
Strict real-time evaluation: a model used at forecast origin O is trained ONLY on outcomes with target year <= O."""
import hashlib, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")

START, TRAIN_END = 2005, 2018
HZ = [1, 2, 3, 4, 5]; MIN_N = 30; MIN_YEARS = 8; K = 5; PHI = 0.8
MIN_ROWS = 15; MIN_TRAIN = 50; B = 2000; MARGIN = 0.95
NUM_FEATS = ["y", "d1", "d2", "d3", "d4", "slope5", "vol4", "dev_cm", "logN"]

def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]

def load_matrices(csv, last_year):
    """Loads ECDC data, CUT at last_year before anything else (guard against reading later years)."""
    ec = pd.read_csv(csv, encoding="utf-8-sig")
    ec["Indicator"] = ec["Indicator"].str.strip()
    ec["NumValue"] = pd.to_numeric(ec["NumValue"], errors="coerce")
    ec["Year"] = pd.to_numeric(ec["Time"], errors="coerce")
    ec = ec[ec["Year"].between(START, last_year)]
    assert ec["Year"].max() <= last_year
    p = ec[ec.Indicator == "R - resistant isolates, percentage"][["Population", "RegionName", "Year", "NumValue"]].rename(columns={"NumValue": "pctR"})
    n = ec[ec.Indicator == "Total tested isolates"][["Population", "RegionName", "Year", "NumValue"]].rename(columns={"NumValue": "N"})
    d = p.merge(n, on=["Population", "RegionName", "Year"], how="left").dropna(subset=["pctR"])
    d = d[d.N >= MIN_N].copy(); d["Year"] = d.Year.astype(int)
    years = list(range(START, last_year + 1))
    W0 = d.pivot_table(index=["Population", "RegionName"], columns="Year", values="pctR").reindex(columns=years)
    N0 = d.pivot_table(index=["Population", "RegionName"], columns="Year", values="N").reindex(columns=years)
    return W0, N0, years

def ses_level(arr):
    best = None
    for a in (0.2, 0.4, 0.6, 0.8, 1.0):
        lvl, sse = arr[0], 0.0
        for v in arr[1:]:
            sse += (v - lvl) ** 2; lvl = a * v + (1 - a) * lvl
        if best is None or sse < best[0]: best = (sse, lvl)
    return best[1]

def build(W0, N0, years, eligibility="registered"):
    keep = (W0.loc[:, START:TRAIN_END].notna().sum(axis=1) >= MIN_YEARS) if eligibility == "registered" else pd.Series(True, index=W0.index)
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
    X = pd.concat([R[NUM_FEATS], pd.get_dummies(R["pathogen"], prefix="p"), pd.get_dummies(R["drug"], prefix="d")], axis=1).astype(float)
    R["chg"] = R["target"] - R["y"]
    return R, X

def fit_origin(R, X, kind, O, score_mask):
    pred = pd.Series(np.nan, index=R.index)
    for h in HZ:
        a = (R.target_year <= O) & (R.h == h); b = score_mask & (R.origin == O) & (R.h == h)   # only outcomes known at O
        if b.sum() == 0 or a.sum() < MIN_TRAIN: continue
        if kind == "EN":
            sc = StandardScaler().fit(X[a]); g = R.loc[a, "country"].values
            cv = list(GroupKFold(n_splits=min(5, len(np.unique(g)))).split(X[a], R.loc[a, "chg"], g))
            m = ElasticNetCV(l1_ratio=[.2, .5, .8, 1.0], n_alphas=30, cv=cv, max_iter=20000, random_state=0).fit(sc.transform(X[a]), R.loc[a, "chg"])
            pred[b] = m.predict(sc.transform(X[b]))
        else:
            m = GradientBoostingRegressor(loss="huber", n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8,
                                          min_samples_leaf=20, random_state=0).fit(X[a], R.loc[a, "chg"])
            pred[b] = m.predict(X[b])
    return R["y"] + pred

def predict_all(R, X, kind, score_mask, first_origin, last_origin):
    pr = pd.Series(np.nan, index=R.index)
    for O in range(first_origin, last_origin + 1):
        pr = pr.fillna(fit_origin(R, X, kind, O, score_mask))
    return pr

def macro_rel(sub, num, den, combos, w=None):
    e1 = (sub["target"] - sub[num]).abs().values; e0 = (sub["target"] - sub[den]).abs().values
    cid = pd.Categorical(sub["combo"], categories=combos).codes; ok_r = cid >= 0
    w = np.ones(len(sub)) if w is None else w; k = len(combos)
    s1 = np.bincount(cid[ok_r], (e1 * w)[ok_r], k); s0 = np.bincount(cid[ok_r], (e0 * w)[ok_r], k); c = np.bincount(cid[ok_r], w[ok_r], k)
    ok = (c > 0) & (s0 > 0) & (s1 > 0)
    return float(np.exp(np.mean(np.log(s1[ok] / s0[ok])))) if ok.any() else np.nan

def boot(sub, num, den, combos, rng, n_boot=B):
    cs = pd.factorize(sub["country"])[0]; kc = cs.max() + 1; out = []
    for _ in range(n_boot):
        cw = np.bincount(rng.integers(0, kc, kc), minlength=kc).astype(float)
        out.append(macro_rel(sub, num, den, combos, cw[cs]))
    out = np.array(out); out = out[~np.isnan(out)]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

def conformal(R, score_mask, first_origin, last_origin, center="ses", alphas=(0.2, 0.1), cal_years=2, min_cal=100, min_scale_train=300):
    out = []
    for O in range(first_origin, last_origin + 1):
        for h in HZ:
            te = score_mask & (R.origin == O) & (R.h == h)
            if te.sum() == 0: continue
            cal = (R.h == h) & R.target_year.between(O - cal_years + 1, O)
            trn = (R.h == h) & (R.target_year <= O - cal_years)
            if cal.sum() < min_cal or trn.sum() < min_scale_train: continue
            e_cal = (R.loc[cal, "target"] - R.loc[cal, center]).abs().values
            gb = GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8, min_samples_leaf=20,
                                           random_state=0).fit(R.loc[trn, NUM_FEATS], (R.loc[trn, "target"] - R.loc[trn, center]).abs())
            scales = {"A_unscaled": (np.ones(cal.sum()), np.ones(te.sum())),
                      "B_volatility": (np.maximum(R.loc[cal, "vol4"].values, 0.5), np.maximum(R.loc[te, "vol4"].values, 0.5)),
                      "C_learned": (np.maximum(gb.predict(R.loc[cal, NUM_FEATS]), 0.3), np.maximum(gb.predict(R.loc[te, NUM_FEATS]), 0.3))}
            c_te = R.loc[te, center].values; idx = R.index[te]
            for a in alphas:
                for name, (sc_cal, sc_te) in scales.items():
                    s = e_cal / sc_cal; n = len(s); lvl = min(1.0, np.ceil((n + 1) * (1 - a)) / n)
                    hw = np.quantile(s, lvl, method="higher") * sc_te
                    out.append(pd.DataFrame({"idx": idx, "method": name, "alpha": a,
                                             "lo": np.clip(c_te - hw, 0, 100), "hi": np.clip(c_te + hw, 0, 100)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def interval_table(R, S):
    T = S.merge(R[["target", "h", "country", "combo", "y"]], left_on="idx", right_index=True)
    T["covered"] = ((T.target >= T.lo) & (T.target <= T.hi)).astype(float); T["width"] = T.hi - T.lo
    T["IS"] = T.width + (2 / T.alpha) * np.maximum(T.lo - T.target, 0) + (2 / T.alpha) * np.maximum(T.target - T.hi, 0)
    T["level_group"] = pd.cut(T.y, [-1, 5, 30, 101], labels=["<5%", "5-30%", ">30%"])
    return T

def interval_boot_ratio(T, num, den, alpha, rng, n_boot=B):
    a = T[(T.alpha == alpha) & (T.method == num)].set_index("idx"); b = T[(T.alpha == alpha) & (T.method == den)].set_index("idx")
    b = b.loc[a.index]; cs = pd.factorize(a.country)[0]; kc = cs.max() + 1; out = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, kc, kc), minlength=kc).astype(float)[cs]
        out.append((a.IS.values * w).sum() / (b.IS.values * w).sum())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float(a.IS.sum() / b.IS.sum())

def print_interval_report(R, S, rng, label):
    T = interval_table(R, S)
    print(f"\nINTERVAL EVALUATION ({label}) - rows scored: {T.idx.nunique()}")
    g = T.groupby(["alpha", "method"]).agg(coverage=("covered", "mean"), mean_width=("width", "mean"), mean_IS=("IS", "mean")).round(3)
    print(g.to_string())
    print("\ncoverage by horizon:"); print(T.pivot_table(index=["alpha", "method"], columns="h", values="covered").round(3).to_string())
    print("\ncoverage by level group (y at origin):"); print(T.pivot_table(index=["alpha", "method"], columns="level_group", values="covered").round(3).to_string())
    print("\nmean width by level group:"); print(T.pivot_table(index=["alpha", "method"], columns="level_group", values="width").round(2).to_string())
    verdict = []
    for a in sorted(T.alpha.unique()):
        lo, hi, r = interval_boot_ratio(T, "C_learned", "A_unscaled", a, rng)
        cov = T[(T.alpha == a) & (T.method == "C_learned")].covered.mean()
        ok = abs(cov - (1 - a)) <= 0.03 and hi < 1.0
        verdict.append(ok)
        print(f"\nH5 check, nominal {int(round((1-a)*100))}%: C_learned coverage {cov:.3f} (target {1-a:.2f} +/- 0.03); interval-score ratio C/A = {r:.3f} [{lo:.3f}, {hi:.3f}]")
    print("H5 VERDICT:", "SUPPORTED" if all(verdict) else "NOT SUPPORTED", "(needs both levels: coverage within 0.03 AND ratio interval upper bound < 1)")
    return T
