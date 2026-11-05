"""
STEP 5 - H5: calibrated prediction intervals for AMR forecasts (VALIDATION years only; locked test never loaded).
Usage (Colab):  !python step5_intervals.py ECDC_surveillance_data_Antimicrobial_resistance__1_.csv
Needs amr_core.py in the same folder.

PRE-REGISTERED RULE (written before running)
  centre forecast: SES (also persistence as a secondary centre); horizons 1-5; nominal levels 80% and 90%
  intervals: split-conformal, strict real-time. At origin O, calibration = absolute errors of forecasts whose target year is in
             [O-1, O] (last two target years known at O); scale model for method C trained on target years <= O-2 only.
  methods:   A unscaled | B scaled by recent volatility, max(SD of last 4 changes, 0.5 pp) | C learned scale (gradient boosting on |error|)
  PRIMARY comparison: C vs A. H5 SUPPORTED only if, at BOTH levels, C's coverage is within 0.03 of nominal AND the interval-score
             ratio C/A has a 95% country-cluster bootstrap upper bound < 1.0. B is descriptive only.
"""
import sys
import numpy as np, pandas as pd
import amr_core as core

CSV = sys.argv[1] if len(sys.argv) > 1 else "ECDC_surveillance_data_Antimicrobial_resistance__1_.csv"
VALID_START, VALID_END = 2019, 2021
print(__doc__.split("PRE-REGISTERED RULE")[1].split("\n\n")[0] if False else "STEP 5 - interval calibration, validation years", "| amr_core sha:", core.sha("amr_core.py"))
W0, N0, years = core.load_matrices(CSV, VALID_END)          # data cut at 2021 BEFORE anything else
R, X = core.build(W0, N0, years, "registered")
va = R.target_year.between(VALID_START, VALID_END)
rng = np.random.default_rng(0)
print("series:", R.groupby(["combo", "country"]).ngroups, "| rows:", len(R), "| scored rows:", int(va.sum()))

for center in ("ses", "persist"):
    S = core.conformal(R, va, VALID_START - 5, VALID_END - 1, center=center)
    T = core.print_interval_report(R, S, rng, f"centre = {center}, validation {VALID_START}-{VALID_END}")
    if center == "ses":
        T.to_csv("intervals_validation_ses.csv", index=False)
