# %%
# Setup
import os
import time

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

import matplotlib.pyplot as plt
import numpy as np

from price_forecasting.paths import find_project_root

load_dotenv()
client = EntsoePandasClient(api_key=os.getenv("ENTSOE_API_KEY"))

PROJECT = find_project_root()
RAW = PROJECT / "data" / "raw"
PROCESSED = PROJECT / "data" / "processed"
YEARS = range(2020, 2026)

# %%
# Reusable download function
def fetch_year(name, query, year, retries=3):
    out = RAW / name / f"{name}_{year}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return pd.read_csv(out, index_col=0, parse_dates=True)

    start = pd.Timestamp(f"{year}-01-01", tz="Europe/Berlin")
    end = pd.Timestamp(f"{year + 1}-01-01", tz="Europe/Berlin")

    for attempt in range(1, retries + 1):
        try:
            df = query("DE_LU", start=start, end=end)
            if isinstance(df, pd.Series):
                df = df.to_frame(name)
            df = df.tz_convert("UTC")
            df.to_csv(out)
            print(f"{name} {year}: {len(df)} rows")
            return df
        except NoMatchingDataError:
            raise
        except Exception as e:
            print(f"{name} {year} attempt {attempt} failed: {e}")
            time.sleep(10)
    raise RuntimeError(f"Could not fetch {name} {year}")


def fetch_all(name, query):
    df = pd.concat([fetch_year(name, query, y) for y in YEARS]).sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="first")]
    return df.resample("h").mean()

# %%
# Download both datasets
load = fetch_all("load", client.query_load_and_forecast)
ws_fc = fetch_all("wind_solar_forecast", client.query_wind_and_solar_forecast)

load = load.rename(columns={
    "Forecasted Load": "load_fc_mw",
    "Actual Load": "load_actual_mw",
})

ws_fc = ws_fc.rename(columns={
    "Solar": "solar_fc_mw",
    "Wind Offshore": "wind_off_fc_mw",
    "Wind Onshore": "wind_on_fc_mw",
})

# Column names match check
expected_cols = {"load_fc_mw", "load_actual_mw", "solar_fc_mw",
                 "wind_off_fc_mw", "wind_on_fc_mw"}
missing = expected_cols - set(load.columns) - set(ws_fc.columns)
assert not missing, f"Columns not found after rename: {missing}"

print(load.head(), ws_fc.head(), sep="\n")

# %%
# Combine on the hourly grid of the German calendar period 2020–2025
start = pd.Timestamp("2020-01-01", tz="Europe/Berlin")
end   = pd.Timestamp("2026-01-01", tz="Europe/Berlin")
hours = pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")

df = load.reindex(hours).join(ws_fc.reindex(hours))

# %%
# Handle gaps: interpolate short ones, flag long ones
# Load forecasts for 22 February and 24 March 2022 are missing from ENTSO-E (confirmed by re-download)
fc_cols = ["load_fc_mw", "solar_fc_mw", "wind_on_fc_mw", "wind_off_fc_mw"]

def long_gap_mask(s, max_len=3):
    """True for missing values that belong to a gap longer than max_len hours."""
    gap = s.isna()
    run_length = gap.groupby((~gap).cumsum()).transform("sum")
    return gap & (run_length > max_len)

long_gaps = pd.DataFrame({c: long_gap_mask(df[c]) for c in fc_cols})
short_gaps = df[fc_cols].isna() & ~long_gaps

df[fc_cols] = df[fc_cols].interpolate(method="time", limit=3, limit_area="inside")
df[fc_cols] = df[fc_cols].mask(long_gaps)

df["fc_interpolated"] = short_gaps.any(axis=1)
df["exclude_from_training"] = long_gaps.any(axis=1)  # build_base.py adds missing prices

# Residual load, calculated once on clean data
df["residual_load_fc_mw"] = (df["load_fc_mw"] - df["solar_fc_mw"]
                             - df["wind_on_fc_mw"] - df["wind_off_fc_mw"])

# %%
# Checks
expected = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
print("Complete hourly index:", df.index.equals(expected))
print(df[fc_cols + ["residual_load_fc_mw"]].isna().sum())
print("Interpolated rows:", df["fc_interpolated"].sum())
print("Excluded from training:", df["exclude_from_training"].sum())

year_local = df.index.tz_convert("Europe/Berlin").year

yearly = df.groupby(year_local).agg(
    load_mean_gw=("load_fc_mw", lambda x: x.mean() / 1000),
    solar_max_gw=("solar_fc_mw", lambda x: x.max() / 1000),
    wind_on_mean_gw=("wind_on_fc_mw", lambda x: x.mean() / 1000),
    resid_mean_gw=("residual_load_fc_mw", lambda x: x.mean() / 1000),
)
print(yearly.round(1))

err = (df["load_fc_mw"] - df["load_actual_mw"]).abs() / df["load_actual_mw"]
print("Load forecast MAPE by year (%):")
print((err.groupby(year_local).mean() * 100).round(2))

# %%
# Plot: residual load forecast vs price
prices = pd.read_csv(PROCESSED / "da_prices_hourly_utc.csv",
                     index_col=0, parse_dates=True)
prices.index = pd.to_datetime(prices.index, utc=True)
plot_df = df.join(prices)

fig, ax = plt.subplots(figsize=(9, 6))
for year, g in plot_df.groupby(year_local):
    ax.scatter(g["residual_load_fc_mw"] / 1000, g["price_eur_mwh"],
               s=2, alpha=0.3, label=str(year))
ax.set_xlabel("Residual load forecast (GW)")
ax.set_ylabel("Day-ahead price (€/MWh)")
ax.legend(markerscale=6)
plt.show()

# %%
# Save
df.index.name = "timestamp_utc"
df.to_csv(PROCESSED / "fundamentals_hourly_utc.csv")
print(df.shape, df.index.min(), df.index.max())