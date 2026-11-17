# %%
# Export the test-year forecasts and metrics the dashboard reads (app/data, tracked in git)
import pandas as pd
from price_forecasting.columns import HOLDOUT_YEAR
from price_forecasting.models.evaluation import split, P
from price_forecasting.paths import find_project_root

PROJECT = find_project_root()
RESULTS = PROJECT / "results"
APP_DATA = PROJECT / "app" / "data"
APP_DATA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PROJECT / "data" / "processed" / "model_features_hourly_utc.csv",
                 index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index, utc=True)
_, _, test = split(df)

def load(name, col):
    s = pd.read_csv(RESULTS / name, index_col=0, parse_dates=True)[col]
    s.index = pd.to_datetime(s.index, utc=True)
    return s.reindex(test.index)

# Only what the app shows; generic file names so the app needs no project imports.
# Actual prices and wind/solar-based series are not on ENTSO-E's free re-use list, so
# they are not published: only own forecasts and the day-ahead load forecast (6.1.b,
# CC BY 4.0). Hourly errors are not exported either, as forecast minus error = price.
lgbm, lear = load("test_lgbm_weekly.csv", "lgbm"), load("test_lear_avg.csv", "lear")
out = pd.DataFrame({
    "lgbm": lgbm,
    "lear": lear,
    "load_fc_gw": test["load_fc_mw"] / 1000,
}).round(2)
out.index.name = "timestamp_utc"
assert not out.isna().any().any(), "Missing values in the app data"
out.to_csv(APP_DATA / "test_predictions.csv")

# Daily mean absolute error per model: an aggregate that does not reveal hourly prices
local_date = test.index.tz_convert("Europe/Berlin").date
daily = pd.DataFrame({
    "mae_lgbm": (lgbm - test[P]).abs().groupby(local_date).mean(),
    "mae_lear": (lear - test[P]).abs().groupby(local_date).mean(),
}).round(2)
daily.index.name = "date"
daily.to_csv(APP_DATA / "test_daily_mae.csv")

pd.read_csv(RESULTS / f"test_results_{HOLDOUT_YEAR}.csv").round(3).to_csv(
    APP_DATA / "test_results.csv", index=False)
print(out.shape, daily.shape, list(out.columns))
