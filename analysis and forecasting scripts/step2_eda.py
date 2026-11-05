"""
STEP 2 - Exploratory analysis of the ECDC resistance data (EXPLORATION SLICE ONLY).

Usage:   python step2_eda.py [path_to_ecdc_csv]
Needs:   pandas, numpy   (matplotlib optional, for 3 figures)
Rule:    only years START..EXPLORE_END are loaded. Validation (2019-2021) and the
         locked test (2022-2024) are never read here. Do not change EXPLORE_END.
"""
import os, sys
import numpy as np
import pandas as pd

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "ECDC_surveillance_data_Antimicrobial_resistance__1_.csv"
START, EXPLORE_END = 2005, 2018          # <-- fixed by protocol.md
HORIZONS = [1, 2, 3, 4, 5]
OUT = "step2_outputs"
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 100)

# ---------------------------------------------------------------- load
ec = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
ec["Indicator"] = ec["Indicator"].str.strip()
ec["NumValue"] = pd.to_numeric(ec["NumValue"], errors="coerce")
ec["Year"] = pd.to_numeric(ec["Time"], errors="coerce")
ec = ec[ec["Year"].between(START, EXPLORE_END)]            # the ONLY years we look at

pct = (ec[ec["Indicator"] == "R - resistant isolates, percentage"]
       [["Population", "RegionName", "Year", "NumValue"]].rename(columns={"NumValue": "pctR"}))
nt = (ec[ec["Indicator"] == "Total tested isolates"]
      [["Population", "RegionName", "Year", "NumValue"]].rename(columns={"NumValue": "N"}))
df = pct.merge(nt, on=["Population", "RegionName", "Year"], how="left").dropna(subset=["pctR"])
df["Year"] = df["Year"].astype(int)
df[["pathogen", "drug"]] = df["Population"].str.split("|", n=1, expand=True)

wide = (df.pivot_table(index=["Population", "RegionName"], columns="Year", values="pctR")
          .reindex(columns=range(START, EXPLORE_END + 1)))   # series x years matrix

# ---------------------------------------------------------------- 1. inventory
print("=" * 70, "\n1. INVENTORY (years %d-%d)\n" % (START, EXPLORE_END), "=" * 70, sep="")
print("observations:", len(df), "| pathogen-drug combos:", df["Population"].nunique(),
      "| countries:", df["RegionName"].nunique(), "| series (combo x country):", len(wide))
n_years = wide.notna().sum(axis=1)
span = wide.apply(lambda r: (r.last_valid_index() - r.first_valid_index() + 1) if r.notna().any() else 0, axis=1)
gaps = span - n_years
print("\nYears of data per series:\n", n_years.describe().round(1).to_string())
print("series with >=10 years:", int((n_years >= 10).sum()), "| <8 years:", int((n_years < 8).sum()),
      "| with internal gaps:", int((gaps > 0).sum()))
low_n = df["N"].isna() | (df["N"] < 30)
print("\nobservations with tested isolates N<30 or missing N: %d (%.1f%%)" % (low_n.sum(), 100 * low_n.mean()))
print("  -> decision for protocol.md: drop observations with N<30? (they give unstable percentages)")
short = n_years[n_years < 8].reset_index().rename(columns={0: "years"})
short.columns = ["Population", "RegionName", "years"]
print("\nCombos with the most short series (<8 yrs):")
print(short.groupby("Population").size().sort_values(ascending=False).head(8).to_string())
n_years.rename("years").reset_index().to_csv(f"{OUT}/series_inventory.csv", index=False)

# ---------------------------------------------------------------- 2. persistence by horizon
print("\n" + "=" * 70, "\n2. PERSISTENCE ERROR BY HORIZON (|value(t+h) - value(t)|, percentage points)\n", "=" * 70, sep="")
rows, per_combo = [], {}
for h in HORIZONS:
    err = (wide.shift(-h, axis=1) - wide).abs()
    e = err.stack()
    rows.append((h, len(e), e.mean(), e.median(), e.quantile(.9)))
    per_combo[h] = e.groupby(level=0).mean()
hz = pd.DataFrame(rows, columns=["horizon", "n_pairs", "MAE", "median_AE", "p90_AE"]).set_index("horizon").round(2)
print(hz.to_string())
hz.to_csv(f"{OUT}/persistence_by_horizon.csv")

pc = pd.DataFrame({f"MAE_h{h}": per_combo[h] for h in (1, 3, 5)}).round(2)
pc["growth_h3_over_h1"] = (pc["MAE_h3"] / pc["MAE_h1"]).round(2)
pc["mean_level"] = wide.mean(axis=1).groupby(level=0).mean().round(1)
print("\nPersistence MAE per pathogen-drug combo (sorted by h=1 error):")
print(pc.sort_values("MAE_h1", ascending=False).to_string())
pc.to_csv(f"{OUT}/persistence_by_combo.csv")

# ---------------------------------------------------------------- 3. series characteristics
print("\n" + "=" * 70, "\n3. SERIES CHARACTERISTICS (level, volatility, trend)\n", "=" * 70, sep="")
yrs = np.array(wide.columns, dtype=float)
def slope(r):
    m = r.notna().values
    return np.polyfit(yrs[m], r.values[m], 1)[0] if m.sum() >= 8 else np.nan
chars = pd.DataFrame({
    "level": wide.mean(axis=1),
    "volatility": wide.diff(axis=1).std(axis=1),   # std of year-to-year changes (adjacent years only)
    "slope_pp_per_yr": wide.apply(slope, axis=1),
})
chars["abs_slope"] = chars["slope_pp_per_yr"].abs()
ch = chars.groupby(level=0).agg(series=("level", "size"), median_level=("level", "median"),
                                median_volatility=("volatility", "median"),
                                median_abs_slope=("abs_slope", "median"),
                                share_trending=("abs_slope", lambda s: (s > 0.5).mean())).round(2)
print(ch.sort_values("median_volatility", ascending=False).to_string())
print("\n(share_trending = share of series moving more than 0.5 percentage points per year)")
chars.to_csv(f"{OUT}/series_characteristics.csv")

# ---------------------------------------------------------------- 4. breaks and reporting
print("\n" + "=" * 70, "\n4. REPORTING AND POSSIBLE BREAKS BY YEAR\n", "=" * 70, sep="")
d1 = wide.diff(axis=1)
by_year = pd.DataFrame({"observations": wide.notna().sum(), "countries": df.groupby("Year")["RegionName"].nunique(),
                        "combos": df.groupby("Year")["Population"].nunique(),
                        "mean_change": d1.mean(), "median_change": d1.median(), "n_changes": d1.notna().sum()}).round(2)
z = (by_year["mean_change"] - by_year["mean_change"].mean()) / by_year["mean_change"].std()
by_year["z_mean_change"] = z.round(2)
print(by_year.to_string())
print("\nYears where the average change is unusual (|z|>2):", by_year.index[z.abs() > 2].tolist() or "none")
by_year.to_csv(f"{OUT}/by_year.csv")

# ---------------------------------------------------------------- 5. between-country differences
print("\n" + "=" * 70, "\n5. HOW MUCH OF THE VARIATION IS BETWEEN COUNTRIES?\n", "=" * 70, sep="")
rows = []
for p, g in df.groupby("Population"):
    if g["RegionName"].nunique() < 10:
        continue
    total = ((g["pctR"] - g["pctR"].mean()) ** 2).mean()
    within = ((g["pctR"] - g.groupby("RegionName")["pctR"].transform("mean")) ** 2).mean()
    rows.append((p, g["RegionName"].nunique(), round(g["pctR"].mean(), 1), round(1 - within / total, 2)))
bt = pd.DataFrame(rows, columns=["combo", "countries", "mean_level", "share_between_countries"]).set_index("combo")
print(bt.sort_values("share_between_countries", ascending=False).to_string())
print("\n(near 1.0 = countries differ in LEVEL but each country is stable over time;")
print(" lower = more of the variation is change over time within countries)")
bt.to_csv(f"{OUT}/between_country_share.csv")

# ---------------------------------------------------------------- figures (optional)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(hz.index, hz["MAE"], "o-", label="mean"); ax.plot(hz.index, hz["median_AE"], "s--", label="median")
    ax.set_xlabel("forecast horizon (years)"); ax.set_ylabel("persistence error (pp)"); ax.legend(); ax.set_title("Persistence error by horizon")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_persistence_by_horizon.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(by_year.index, by_year["mean_change"]); ax.axhline(0, color="k", lw=.5)
    ax.set_xlabel("year"); ax.set_ylabel("mean change vs previous year (pp)"); ax.set_title("Average year-to-year change across all series")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_mean_change_by_year.png", dpi=150); plt.close(fig)
    s = pc.sort_values("MAE_h1")
    fig, ax = plt.subplots(figsize=(8, 7)); ax.barh(s.index, s["MAE_h1"], label="h=1"); ax.barh(s.index, s["MAE_h3"], alpha=.4, label="h=3")
    ax.legend(); ax.set_xlabel("persistence MAE (pp)"); ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_error_by_combo.png", dpi=150); plt.close(fig)
    print("\nfigures saved in", OUT)
except ImportError:
    print("\n(matplotlib not installed - skipped figures; tables were still saved)")

print("\nDONE. CSV tables are in", OUT)
