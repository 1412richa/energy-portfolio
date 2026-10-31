# %%
# Join the per-source files into the one table the models read.
# This is the only script that writes model_base_hourly_utc.csv.
import pandas as pd

from price_forecasting.paths import find_project_root

PROJECT = find_project_root()
PROCESSED = PROJECT / "data" / "processed"


def read_hourly(name):
    df = pd.read_csv(PROCESSED / name, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


prices = read_hourly("da_prices_hourly_utc.csv")
fundamentals = read_hourly("fundamentals_hourly_utc.csv")
weather = read_hourly("weather_hourly_utc.csv")
fuels = pd.read_csv(PROCESSED / "fuels_daily.csv", index_col=0, parse_dates=True)

# %%
# The fundamentals index is the German calendar period 2020–2025
base = prices.reindex(fundamentals.index).join(fundamentals)
assert base.index.equals(weather.index), "Weather hours differ from the base"
base = base.join(weather)
base["exclude_from_training"] = base["exclude_from_training"] | base["price_eur_mwh"].isna()

# Fuels are per delivery day; map them to hours by German date
local_day = base.index.tz_convert("Europe/Berlin").tz_localize(None).normalize()
base = base.join(fuels.reindex(local_day).set_axis(base.index))

# %%
# Save
base.index.name = "timestamp_utc"
base.to_csv(PROCESSED / "model_base_hourly_utc.csv")
print(base.shape, base.index.min(), base.index.max())
print("Missing values:")
print(base.isna().sum()[base.isna().sum() > 0])
