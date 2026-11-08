# %%
# load and assemble error data for analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from price_forecasting.columns import VAL_YEAR
from price_forecasting.models.evaluation import split, lago_naive, summarise, P
from price_forecasting.paths import find_project_root

PROJECT = find_project_root()
RESULTS = PROJECT / "results"
FIG = PROJECT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PROJECT / "data" / "processed" / "model_features_hourly_utc.csv",
                 index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index, utc=True)
_, val, _ = split(df)

def load_lgbm(experiment="lgbm_weekly"):
    """One LightGBM experiment from the saved experiment predictions (hourly, UTC)."""
    s = pd.read_csv(RESULTS / "lgbm_experiments_val_predictions.csv",
                    index_col=0, parse_dates=True)[experiment]
    s.index = pd.to_datetime(s.index, utc=True)
    return s.reindex(val.index)

def load_lear_avg(windows=(365, 730, 1095)):
    """lear_avg: mean of the saved LEAR window forecasts, reshaped from day x hour to hourly."""
    wall = val.index.tz_convert("Europe/Berlin").tz_localize(None)
    key = pd.MultiIndex.from_arrays([wall.normalize(), wall.hour])
    parts = []
    for w in windows:
        pw = pd.read_csv(RESULTS / f"lear_{w}d_val_predictions_wide.csv", index_col=0, parse_dates=True)
        pw.columns = pw.columns.astype(int)
        parts.append(pd.Series(pw.stack().reindex(key).to_numpy(), index=val.index))
    return pd.concat(parts, axis=1).mean(axis=1)

loc = val.index.tz_convert("Europe/Berlin")
e = pd.DataFrame({
    "actual": val[P],
    "lgbm": load_lgbm(),
    "lear": load_lear_avg(),
    "naive": lago_naive(val),
    "resid_gw": val["residual_load_fc_mw"] / 1000,
    "hour": loc.hour, "month": loc.month, "date": loc.date,
    "is_weekend": loc.dayofweek >= 5,
}, index=val.index).dropna(subset=["actual", "lgbm", "lear"])

for m in ["lgbm", "lear", "naive"]:
    e[f"err_{m}"] = e[m] - e["actual"]
print(e[["err_lgbm", "err_lear"]].abs().mean().round(2))

# %%
# plot MAE by hour
by_hour = e.groupby("hour")[["err_lgbm", "err_lear", "err_naive"]].apply(
    lambda g: g.abs().mean())
ax = by_hour.plot(figsize=(9, 4), marker="o", title=f"MAE by hour ({VAL_YEAR})")
ax.set_ylabel("€/MWh")
plt.savefig(FIG / "mae_by_hour.png", dpi=150, bbox_inches="tight")
plt.show()

print(e.groupby("hour")[["err_lgbm", "err_lear"]].mean().round(1).T)   # bias by hour

# %%
# plot MAE by month
by_month = e.groupby("month")[["err_lgbm", "err_lear"]].apply(lambda g: g.abs().mean())
by_month["mean_price"] = e.groupby("month")["actual"].mean()
by_month["price_std"] = e.groupby("month")["actual"].std()
print(by_month.round(1))

# %%
# plot MAE by residual load band
bands = pd.cut(e["resid_gw"], [-40, 0, 10, 20, 30, 40, 50, 60, 90])
by_band = e.groupby(bands, observed=True).agg(
    hours=("actual", "size"),
    mae_lgbm=("err_lgbm", lambda x: x.abs().mean()),
    mae_lear=("err_lear", lambda x: x.abs().mean()),
    bias_lgbm=("err_lgbm", "mean"),
    bias_lear=("err_lear", "mean"),
)
print(by_band.round(1))

# %%
# the worst days
daily = e.groupby("date").agg(
    mae_lgbm=("err_lgbm", lambda x: x.abs().mean()),
    mae_lear=("err_lear", lambda x: x.abs().mean()),
    max_price=("actual", "max"), min_price=("actual", "min"),
    min_resid=("resid_gw", "min"), max_resid=("resid_gw", "max"),
)
worst = daily.sort_values("mae_lgbm", ascending=False).head(10)
print(worst.round(1))

n_worst = round(len(daily) * 0.05)
share = daily["mae_lgbm"].nlargest(n_worst).sum() / daily["mae_lgbm"].sum()
print(f"Worst 5% of days account for {share:.0%} of total LightGBM error")

# %%
# plot worst day
day = worst.index[0]
d = e[e["date"] == day]
ax = d.set_index("hour")[["actual", "lgbm", "lear"]].plot(
    figsize=(9, 4), marker="o", title=f"Worst day: {day}")
ax.set_ylabel("€/MWh")
plt.savefig(FIG / "worst_day.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# compare LightGBM and LEAR forecasts
e["gap"] = e["lgbm"] - e["lear"]
big = e["gap"].abs() > e["gap"].abs().quantile(0.95)
closer_lgbm = (e["err_lgbm"].abs() < e["err_lear"].abs())
print(f"In the 5% of hours with largest disagreement, LightGBM is closer in "
      f"{closer_lgbm[big].mean():.0%} of them")
print(e.loc[big].groupby(pd.cut(e.loc[big, "resid_gw"], [-40, 10, 30, 50, 90]),
                         observed=True).size())

# Simple average of both models: post-hoc (chosen after seeing 2024), so judge it on the test year only
e["avg"] = (e["lgbm"] + e["lear"]) / 2
print("MAE of the average forecast:", (e["avg"] - e["actual"]).abs().mean().round(2))
# Saved so the test report can compare the average's 2024 and test-year results
avg_val = pd.DataFrame([summarise(val, e["avg"].reindex(val.index), "average_lgbm_lear")]).set_index("model")
avg_val.to_csv(RESULTS / "average_lgbm_lear_val.csv")

# %%
# compare how often each model catches negative-price hours
# Counts shown next to the rates: some months have only a handful of negative hours
neg = e["actual"] < 0
recall = pd.DataFrame({"negative_hours": neg.groupby(e["month"]).sum()})
for m in ["lgbm", "lear"]:
    caught = (neg & (e[m] < 0)).groupby(e["month"]).sum()
    recall[f"caught_{m}"] = caught
    recall[f"recall_{m}"] = (caught / recall["negative_hours"].replace(0, np.nan)).round(2)
print(recall[recall["negative_hours"] > 0])

