"""STEP 3c - robustness checks on the strict real-time gate (validation years only; locked test never loaded).
(1) drop the cross-country deviation feature dev_cm; (2) per-combination breakdown at horizons >= 3.
Usage: python step3c_robustness.py [ecdc_csv]   (needs step3b_strict_origin.py in the same folder)"""
import sys, re
src = open("step3b_strict_origin.py").read()
src = re.sub(r'\nrun\("registered"\)\s*\nrun\("origin_only"\)\s*$', "\n", src.strip() + "\n")
ns = {"__name__": "step3b_funcs"}; sys.argv = [sys.argv[0]] + sys.argv[1:2]
exec(compile(src, "step3b", "exec"), ns)
import numpy as np, pandas as pd
build, fit_origin, macro_rel, boot = ns["build"], ns["fit_origin"], ns["macro_rel"], ns["boot"]
VALID_START, VALID_END, MIN_ROWS = ns["VALID_START"], ns["VALID_END"], ns["MIN_ROWS"]

R, X = build("registered"); va = R["target_year"].between(VALID_START, VALID_END)
def predict(Xm):
    pr = pd.Series(np.nan, index=R.index)
    for O in range(VALID_START - 5, VALID_END): pr = pr.fillna(fit_origin(R, Xm, "GB", O, va))
    return pr
R["GB_full"] = predict(X)
R["GB_no_devcm"] = predict(X.drop(columns=["dev_cm"]))
V = R[va].dropna(subset=["GB_full", "GB_no_devcm"]).copy(); H3 = V[V.h >= 3]
combos = [c for c, k in H3.groupby("combo").size().items() if k >= MIN_ROWS]
print("scored rows:", len(V), "| h>=3:", len(H3), "| combos:", len(combos))
print("\n(1) ABLATION OF dev_cm (strict real-time, registered series, h>=3): relative error vs persistence")
for m in ("GB_full", "GB_no_devcm"):
    lo, hi = boot(H3, m, "persist", combos)
    print(f"  {m:12s} {macro_rel(H3, m, 'persist', combos):.3f}  [{lo:.3f}, {hi:.3f}]")
print("  without/with dev_cm ratio:", round(macro_rel(H3, "GB_no_devcm", "GB_full", combos), 3), tuple(round(x, 3) for x in boot(H3, "GB_no_devcm", "GB_full", combos)))

print("\n(2) PER-COMBINATION BREAKDOWN, h>=3 (relative error vs persistence; <1 = better than persistence)")
rows = []
for c, g in H3.groupby("combo"):
    if len(g) < MIN_ROWS: continue
    pm = (g.target - g.persist).abs().mean()
    rows.append((c, len(g), g.country.nunique(), round(pm, 2), round((g.target - g.GB_full).abs().mean() / pm, 3), round((g.target - g.ses).abs().mean() / pm, 3)))
T = pd.DataFrame(rows, columns=["combo", "n", "countries", "persist_MAE", "GB_rel", "SES_rel"]).sort_values("GB_rel")
print(T.to_string(index=False))
print(f"\n  GB beats persistence in {(T.GB_rel < 1).sum()} of {len(T)} combos; SES in {(T.SES_rel < 1).sum()} of {len(T)}")
print("  correlation of persistence MAE (series noisiness) with GB_rel:", round(T.persist_MAE.corr(T.GB_rel), 2), "| with SES_rel:", round(T.persist_MAE.corr(T.SES_rel), 2))
T.to_csv("per_combo_h3.csv", index=False)
