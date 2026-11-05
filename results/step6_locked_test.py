"""
STEP 6 - LOCKED TEST (2022-2024). Run ONCE, with methods frozen.
Usage (Colab, same folder as amr_core.py and the ECDC csv):
    !python step6_locked_test.py --dry-run  CSV          # tests the code on VALIDATION years (2019-2021); safe to run any time
    !python step6_locked_test.py --final --confirm I-ACCEPT-ONE-RUN  CSV | tee locked_test_results.txt
The --final run loads data up to 2024, writes LOCKED_TEST_DONE.txt and refuses to run again.
Do not edit amr_core.py or this file after the dry run, except the prediction text below (edit it BEFORE --final).
"""
import os, sys, datetime
import numpy as np, pandas as pd
import amr_core as core

# ---------------- PRE-REGISTERED PREDICTIONS: write/confirm these BEFORE running --final ----------------
PREDICTIONS = [
    "P1: pooled gradient boosting will NOT beat persistence at horizons >= 3 (macro relative error >= 1.00).",
    "P2: SES will be within +/-2% of persistence at horizons >= 3 (macro relative error between 0.98 and 1.02).",
    "P3: scaled conformal intervals (method C) will again reach coverage within 0.03 of nominal with interval-score ratio C/A < 1.",
]
# ---------------------------------------------------------------------------------------------------------
args = sys.argv[1:]
if "--final" in args:
    mode = "FINAL"
elif "--dry-run" in args:
    mode = "DRY-RUN"
else:
    sys.exit(__doc__)
CSV = [a for a in args if a.lower().endswith(".csv")]
CSV = CSV[0] if CSV else "ECDC_surveillance_data_Antimicrobial_resistance__1_.csv"
if mode == "FINAL":
    if "--confirm" not in args or "I-ACCEPT-ONE-RUN" not in args:
        sys.exit("Refusing: add  --confirm I-ACCEPT-ONE-RUN  to run the locked test.")
    if os.path.exists("LOCKED_TEST_DONE.txt"):
        sys.exit("Refusing: LOCKED_TEST_DONE.txt exists. The locked test has already been run.")
    SCORE_START, SCORE_END = 2022, 2024
else:
    SCORE_START, SCORE_END = 2019, 2021      # dry run uses VALIDATION years only (2022+ is never loaded)

print("=" * 78, f"\nSTEP 6 - {mode} | scored target years {SCORE_START}-{SCORE_END} | {datetime.datetime.now():%Y-%m-%d %H:%M}")
print("amr_core sha256[:12]:", core.sha("amr_core.py"), "| step6 sha256[:12]:", core.sha(__file__))
print("PRE-REGISTERED PREDICTIONS:"); [print("  ", p) for p in PREDICTIONS]; print("=" * 78)

W0, N0, years = core.load_matrices(CSV, SCORE_END)       # data cut at SCORE_END before anything else
R, X = core.build(W0, N0, years, "registered")
sc = R.target_year.between(SCORE_START, SCORE_END); first_o, last_o = SCORE_START - 5, SCORE_END - 1
rng = np.random.default_rng(0)
print("series:", R.groupby(["combo", "country"]).ngroups, "| scored rows:", int(sc.sum()), "| origins", first_o, "-", last_o)

# ---------------- point forecasts: persistence, drift, damped, SES (no fitting) + EN, GB (trained per origin on outcomes <= origin)
for kind in ("EN", "GB"):
    R[kind] = core.predict_all(R, X, kind, sc, first_o, last_o)
V = R[sc].dropna(subset=["EN", "GB"]).copy(); H3 = V[V.h >= 3]
combos = [c for c, k in H3.groupby("combo").size().items() if k >= core.MIN_ROWS]
print("scored rows with all methods:", len(V), "| rows at h>=3:", len(H3), "| combos in metric:", len(combos))

methods = ["drift", "damped", "ses", "EN", "GB"]
tab = {m: {f"h={h}": core.macro_rel(V[V.h == h], m, "persist", combos) for h in core.HZ} | {"h>=3": core.macro_rel(H3, m, "persist", combos)} for m in methods}
print("\nPRIMARY METRIC - macro relative error vs persistence (1.00 = same; <1 better)")
print(pd.DataFrame(tab).T[["h=1", "h=2", "h=3", "h=4", "h=5", "h>=3"]].round(3).to_string())
print("\npooled MAE (percentage points):")
print(V.groupby("h").apply(lambda g: pd.Series({m: (g.target - g[m]).abs().mean() for m in ["persist"] + methods})).round(2).to_string())

print("\nGATE (h>=3):")
base = {m: core.macro_rel(H3, m, "persist", combos) for m in ("drift", "damped", "ses")}; best = min(base, key=base.get)
gate = {}
for chosen in ("GB", "EN"):
    g = core.macro_rel(H3, chosen, "persist", combos); lo, hi = core.boot(H3, chosen, "persist", combos, rng)
    lo3, hi3 = core.boot(H3, chosen, best, combos, rng)
    G1, G2, G3 = g <= core.MARGIN, hi < 1.0, (all(g < base[m] for m in base) and hi3 < 1.0)
    gate[chosen] = G1 and G2 and G3
    print(f"  {chosen}: rel.err {g:.3f} [{lo:.3f}, {hi:.3f}] | best baseline {best} {base[best]:.3f}; model vs best [{lo3:.3f}, {hi3:.3f}]"
          f" -> G1 {'PASS' if G1 else 'FAIL'} G2 {'PASS' if G2 else 'FAIL'} G3 {'PASS' if G3 else 'FAIL'}")
lo, hi = core.boot(H3, "ses", "persist", combos, rng)
print(f"  SES vs persistence: {base['ses']:.3f} [{lo:.3f}, {hi:.3f}]")

# ---------------- pre-specified SECONDARY metric: drop combos whose 1-year persistence MAE in 2005-2018 is < 0.5 pp (training data only)
tr = R[(R.target_year <= core.TRAIN_END) & (R.h == 1)]
mae1 = tr.groupby("combo").apply(lambda g: (g.target - g.persist).abs().mean())
nearzero = set(mae1[mae1 < 0.5].index); keep = [c for c in combos if c not in nearzero]
print("\nSECONDARY METRIC (excluding near-zero combos defined from 2005-2018):", sorted(nearzero))
for m in methods:
    lo, hi = core.boot(H3, m, "persist", keep, rng)
    print(f"  {m:7s} {core.macro_rel(H3, m, 'persist', keep):.3f} [{lo:.3f}, {hi:.3f}]")

per = []
for c, g in H3.groupby("combo"):
    if len(g) >= core.MIN_ROWS:
        pm = (g.target - g.persist).abs().mean(); per.append((c, len(g), round(pm, 2), round((g.target - g.GB).abs().mean() / pm, 3), round((g.target - g.ses).abs().mean() / pm, 3)))
P = pd.DataFrame(per, columns=["combo", "n", "persist_MAE", "GB_rel", "SES_rel"]).sort_values("GB_rel")
print(f"\nGB beats persistence in {(P.GB_rel < 1).sum()} of {len(P)} combos; SES in {(P.SES_rel < 1).sum()} of {len(P)}")
print("\nby target year (h>=3), GB / SES relative error:")
for ty, g in H3.groupby("target_year"):
    print(f"  {ty}: GB {core.macro_rel(g, 'GB', 'persist', combos):.3f} | SES {core.macro_rel(g, 'ses', 'persist', combos):.3f} | n={len(g)}")

# ---------------- prediction check
p1 = tab["GB"]["h>=3"] >= 1.0; p2 = 0.98 <= tab["ses"]["h>=3"] <= 1.02
print("\nPREDICTION CHECK:  P1", "CONFIRMED" if p1 else "REFUTED", f"(GB {tab['GB']['h>=3']:.3f})", "| P2", "CONFIRMED" if p2 else "REFUTED", f"(SES {tab['ses']['h>=3']:.3f})")
print("GATE VERDICT:", "PASSED (GB)" if gate["GB"] else "NOT PASSED (GB)")

# ---------------- interval calibration (frozen design from step5)
S = core.conformal(R, sc, first_o, last_o, center="ses")
T = core.print_interval_report(R, S, rng, f"centre = ses, {mode} {SCORE_START}-{SCORE_END}")
print("(P3 is confirmed if the H5 VERDICT above is SUPPORTED)")

tag = "locked" if mode == "FINAL" else "dryrun"
V.to_csv(f"{tag}_point_forecasts.csv", index=False); T.to_csv(f"{tag}_intervals.csv", index=False); P.to_csv(f"{tag}_per_combo_h3.csv", index=False)
if mode == "FINAL":
    open("LOCKED_TEST_DONE.txt", "w").write(f"Locked test run {datetime.datetime.now().isoformat()} | amr_core {core.sha('amr_core.py')} | step6 {core.sha(__file__)}\n")
    print("\nLOCKED TEST COMPLETE. Do not re-run or change methods. Report everything above, including refuted predictions.")
else:
    print("\nDRY RUN COMPLETE (validation years). The code works; now freeze the files and run --final once.")
