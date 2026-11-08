# %%
# Score the frozen models on the test year, once (protocol in notes/evaluation.md)
import matplotlib.pyplot as plt
import pandas as pd
from price_forecasting.columns import HOLDOUT_YEAR, VAL_YEAR
from price_forecasting.models.evaluation import (split, evaluate, lago_naive, summarise,
                                                  negative_price_table, P)
from price_forecasting.paths import find_project_root

PROJECT = find_project_root()
RESULTS, FIG = PROJECT / "results", PROJECT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PROJECT / "data" / "processed" / "model_features_hourly_utc.csv",
                 index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index, utc=True)
_, _, test = split(df)

def load(name, col):
    s = pd.read_csv(RESULTS / name, index_col=0, parse_dates=True)[col]
    s.index = pd.to_datetime(s.index, utc=True)
    return s.reindex(test.index)

preds = {
    "naive_d1": test["price_lag_d1"],
    "naive_d7": test["price_lag_d7"],
    "naive_lago": lago_naive(test),
    "lear_avg": load("test_lear_avg.csv", "lear"),
    "lgbm_weekly": load("test_lgbm_weekly.csv", "lgbm"),
}
# Added after the 2024 error analysis; reported whatever its result
preds["average_lgbm_lear"] = (preds["lgbm_weekly"] + preds["lear_avg"]) / 2

# Every model must be scored on the same hours
missing = pd.DataFrame(preds).isna().sum()
assert not missing.any(), f"Missing forecasts:\n{missing[missing > 0]}"

# %%
# All subsets, then one summary row per model (headline: lgbm_weekly)
res = pd.concat([evaluate(test, p, name) for name, p in preds.items()])
res.to_csv(RESULTS / f"test_results_{HOLDOUT_YEAR}.csv", index=False)
print(res.round(2).to_string(index=False))

summary = pd.DataFrame([summarise(test, p, n) for n, p in preds.items()]).set_index("model")
summary.to_csv(RESULTS / f"test_summary_{HOLDOUT_YEAR}.csv")
print(summary.round(3))

# %%
# Before and after the switch to 15-minute day-ahead products on 1 October
q4 = (test.index.tz_convert("Europe/Berlin").month >= 10)
periods = {"jan_sep": ~q4, "oct_dec_15min": q4}
by_period = pd.concat([
    pd.DataFrame([summarise(test[m], p[m], n) for n, p in preds.items()]).assign(period=label)
    for label, m in periods.items()
]).set_index(["period", "model"])
by_period.to_csv(RESULTS / f"test_summary_by_period_{HOLDOUT_YEAR}.csv")
print(by_period.round(3))

# %%
# Where forecasts land for hours with an actual negative price
for n in ["lear_avg", "lgbm_weekly", "average_lgbm_lear"]:
    print(f"\n{n}:\n", negative_price_table(test, preds[n], n))

# %%
# Compare validation-year and test-year results, for the same models
val_res = pd.read_csv(RESULTS / "validation_results.csv")
val_mae = pd.concat([
    val_res[val_res["subset"] == "all"].set_index("model")["MAE"],                      # naive baselines
    pd.read_csv(RESULTS / "lear_windows_val.csv", index_col=0)["MAE"],                  # lear_avg
    pd.read_csv(RESULTS / "lgbm_experiments_val.csv", index_col=0)["MAE"],              # lgbm_weekly
    pd.read_csv(RESULTS / "average_lgbm_lear_val.csv", index_col=0)["MAE"],             # from error_analysis.py
])
val_mae = val_mae[~val_mae.index.duplicated()]   # lear_730d appears in two files with the same value
test_all = res[res["subset"] == "all"].set_index("model")
cmp = pd.DataFrame({f"MAE_{VAL_YEAR}": val_mae.reindex(preds.keys()),
                    f"MAE_{HOLDOUT_YEAR}": test_all["MAE"],
                    f"rMAE_{HOLDOUT_YEAR}": test_all["rMAE"]})
assert cmp.notna().all().all(), "A model has no validation-year result"
print(cmp.round(2))

# %%
# One two-week plot, chosen without looking at forecasts
neg_daily = (test[P] < 0).groupby(test.index.tz_convert("Europe/Berlin").date).sum()
start = neg_daily.rolling(14).sum().idxmax() - pd.Timedelta(days=13)
end = start + pd.Timedelta(days=14)

loc_date = test.index.tz_convert("Europe/Berlin").normalize().tz_localize(None)
w = (loc_date >= pd.Timestamp(start)) & (loc_date < pd.Timestamp(end))
plot_df = pd.DataFrame({"actual": test[P], "LightGBM": preds["lgbm_weekly"],
                        "LEAR": preds["lear_avg"]})[w]
plot_df.index = plot_df.index.tz_convert("Europe/Berlin")

ax = plot_df.plot(figsize=(12, 4), linewidth=1)
ax.axhline(0, color="grey", linewidth=0.6)
ax.set_ylabel("€/MWh")
ax.set_xlabel("Berlin time")
ax.set_title(f"Day-ahead price vs forecasts, {start} – {end - pd.Timedelta(days=1)}")
plt.savefig(FIG / "test_two_weeks.png", dpi=150, bbox_inches="tight")
plt.show()