# %%
import pandas as pd

from price_forecasting.models.evaluation import split, lago_naive, evaluate
from price_forecasting.paths import find_project_root

PROJECT = find_project_root()
RESULTS = PROJECT / "results"
RESULTS.mkdir(exist_ok=True)

df = pd.read_csv(PROJECT / "data" / "processed" / "model_features_hourly_utc.csv",
                 index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index, utc=True)

train, val, test = split(df)
print(len(train), len(val), len(test))

# %%
baselines = {
    "naive_d1": val["price_lag_d1"],
    "naive_d7": val["price_lag_d7"],
    "naive_lago": lago_naive(val),
}
results = pd.concat([evaluate(val, pred, name) for name, pred in baselines.items()])
print(results.round(2).to_string(index=False))

results.to_csv(RESULTS / "validation_results.csv", index=False)
