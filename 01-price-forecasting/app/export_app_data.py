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

# Only what the app shows; generic file names so the app needs no project imports
out = pd.DataFrame({
    "actual": test[P],
    "lgbm": load("test_lgbm_weekly.csv", "lgbm"),
    "lear": load("test_lear_avg.csv", "lear"),
    "resid_gw": test["residual_load_fc_mw"] / 1000,
}).round(2)
out.index.name = "timestamp_utc"
assert not out.isna().any().any(), "Missing values in the app data"
out.to_csv(APP_DATA / "test_predictions.csv")

pd.read_csv(RESULTS / f"test_results_{HOLDOUT_YEAR}.csv").round(3).to_csv(
    APP_DATA / "test_results.csv", index=False)
print(out.shape, out.index.min(), out.index.max())
