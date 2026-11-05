import sys, re
src = open("step3b_strict_origin.py").read()
src = re.sub(r'\nrun\("registered"\)\s*\nrun\("origin_only"\)\s*$', "\n", src.strip() + "\n")
sys.argv = [sys.argv[0]] + sys.argv[1:2]; ns = {"__name__": "f"}; exec(compile(src, "s", "exec"), ns)
import numpy as np, pandas as pd
build, fit_origin, macro_rel, boot = ns["build"], ns["fit_origin"], ns["macro_rel"], ns["boot"]
VS, VE, MR = ns["VALID_START"], ns["VALID_END"], ns["MIN_ROWS"]
R, X = build("registered"); va = R.target_year.between(VS, VE)
for k in ("EN", "GB"):
    pr = pd.Series(np.nan, index=R.index)
    for O in range(VS - 5, VE): pr = pr.fillna(fit_origin(R, X, k, O, va))
    R[k] = pr
V = R[va].dropna(subset=["EN", "GB"]); H3 = V[V.h >= 3]
allc = [c for c, k in H3.groupby("combo").size().items() if k >= MR]
nz = "Escherichia coli|Carbapenems"; keep = [c for c in allc if c != nz]
print("combos all/excluding near-zero E. coli carbapenems:", len(allc), len(keep))
print("\nh>=3 relative error vs persistence [95% interval]")
for m in ("drift", "damped", "ses", "EN", "GB"):
    a = macro_rel(H3, m, "persist", allc); b = macro_rel(H3, m, "persist", keep); lo, hi = boot(H3, m, "persist", keep)
    print(f"  {m:7s} all {a:.3f} | excluding E.coli-carbapenems {b:.3f} [{lo:.3f}, {hi:.3f}]")
# gate on the sensitivity set for GB
g = macro_rel(H3, "GB", "persist", keep); lo, hi = boot(H3, "GB", "persist", keep)
base = {m: macro_rel(H3, m, "persist", keep) for m in ("drift", "damped", "ses")}; best = min(base, key=base.get); lo3, hi3 = boot(H3, "GB", best, keep)
print(f"\nGATE on sensitivity set (GB): G1 {'PASS' if g<=0.95 else 'FAIL'} ({g:.3f}) | G2 {'PASS' if hi<1 else 'FAIL'} (upper {hi:.3f}) | G3 {'PASS' if (all(g<base[m] for m in base) and hi3<1) else 'FAIL'} (best baseline {best} {base[best]:.3f}; vs best [{lo3:.3f}, {hi3:.3f}])")
# scale-free alternative: pooled MAE ratio and median of per-combo ratios
pm = {m: (H3.target - H3[m]).abs().mean() for m in ("persist", "ses", "EN", "GB")}
print("\npooled MAE at h>=3 (pp):", {k: round(v, 2) for k, v in pm.items()})
rat = []
for c in allc:
    g_ = H3[H3.combo == c]; rat.append((g_.target - g_.GB).abs().mean() / (g_.target - g_.persist).abs().mean())
print("median per-combo GB ratio:", round(float(np.median(rat)), 3), "| arithmetic mean:", round(float(np.mean(rat)), 3))
