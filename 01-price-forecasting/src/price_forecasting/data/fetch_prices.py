# %%
# Setup
import os
import time

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

from price_forecasting.paths import find_project_root

PROJECT = find_project_root()
RAW = PROJECT / "data" / "raw" / "prices"
PROCESSED = PROJECT / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)
print("Saving data under:", PROJECT / "data")

load_dotenv()
client = EntsoePandasClient(api_key=os.getenv("ENTSOE_API_KEY"))

# %%
# Test with one month
test = client.query_day_ahead_prices(
    "DE_LU",
    start=pd.Timestamp("2024-03-01", tz="Europe/Berlin"),
    end=pd.Timestamp("2024-04-01", tz="Europe/Berlin"),
)
print(type(test))
print(test.head())
print(test.index.tz, len(test))

# %%
# Download year by year, saving each year
def fetch_year(year: int, retries: int = 3) -> pd.Series:
    out = RAW / f"da_prices_{year}.csv"
    if out.exists():
        df = pd.read_csv(out, index_col=0, parse_dates=True)
        return df.iloc[:, 0]

    start = pd.Timestamp(f"{year}-01-01", tz="Europe/Berlin")
    end = pd.Timestamp(f"{year + 1}-01-01", tz="Europe/Berlin")

    for attempt in range(1, retries + 1):
        try:
            s = client.query_day_ahead_prices("DE_LU", start=start, end=end)
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s = s.tz_convert("UTC").rename("price_eur_mwh")
            s.to_frame().to_csv(out)
            print(f"{year}: {len(s)} rows saved")
            return s
        except Exception as e:
            print(f"{year} attempt {attempt} failed: {e}")
            time.sleep(10)
    raise RuntimeError(f"Could not fetch {year}")

series = [fetch_year(y) for y in range(2020, 2026)]

# %%
# Combine and convert to hourly
prices = pd.concat(series).sort_index()
prices.index = pd.to_datetime(prices.index, utc=True)

# Year boundaries can produce duplicate timestamps
dupes = prices.index.duplicated(keep="first")
print("Duplicates removed:", dupes.sum())
prices = prices[~dupes]

# Check which time steps occur
print(prices.index.to_series().diff().value_counts().head())

hourly = prices.resample("h").mean()

# %%
# Patch hours dropped by the year-long requests (31 Dec 00:00 local in 2020–2024);
# a short request for the same day returns them
def fetch_window(start: pd.Timestamp, end: pd.Timestamp, retries: int = 3) -> pd.Series:
    for attempt in range(1, retries + 1):
        try:
            s = client.query_day_ahead_prices("DE_LU", start=start, end=end)
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            return s.tz_convert("UTC").resample("h").mean()
        except Exception as e:
            print(f"{start.date()} attempt {attempt} failed: {e}")
            time.sleep(10)
    raise RuntimeError(f"Could not fetch {start.date()} to {end.date()}")


def fetch_patches(missing: pd.DatetimeIndex) -> pd.Series:
    out = RAW / "da_prices_patches.csv"
    if out.exists():
        cached = pd.read_csv(out, index_col=0, parse_dates=True).iloc[:, 0]
        cached.index = pd.to_datetime(cached.index, utc=True)
        if missing.isin(cached.index).all():
            return cached

    days = missing.tz_convert("Europe/Berlin").normalize().unique()
    parts = [fetch_window(day - pd.Timedelta(days=1), day + pd.Timedelta(days=1))
             for day in days]
    patches = pd.concat(parts).rename("price_eur_mwh")
    patches = patches[patches.index.isin(missing)]
    patches.to_frame().to_csv(out)
    return patches

# Only hours inside the German calendar period; the last UTC hour belongs to 2026
start = pd.Timestamp("2020-01-01", tz="Europe/Berlin")
end = pd.Timestamp("2026-01-01", tz="Europe/Berlin")
missing = hourly.index[hourly.isna() & (hourly.index >= start) & (hourly.index < end)]
if len(missing):
    patches = fetch_patches(missing)
    hourly = hourly.fillna(patches)
    print(f"Patched {hourly.loc[missing].notna().sum()} of {len(missing)} missing hours")

# %%
# Quality checks
# 1. Missing hours (resample turns gaps into NaN)
print("Missing hours:", hourly.isna().sum())
print(hourly[hourly.isna()].index[:10])

# 2. Daylight saving check in local time
local = hourly.tz_convert("Europe/Berlin")
print(local.groupby(local.index.date).size().value_counts())

# 3. Plausibility by year
summary = hourly.groupby(hourly.index.year).agg(
    mean="mean", min="min", max="max",
    negative_hours=lambda x: (x < 0).sum(),
)
print(summary.round(1))

# %%
# Save the clean file
hourly.index.name = "timestamp_utc"
hourly.to_frame().to_csv(PROCESSED / "da_prices_hourly_utc.csv")
print(hourly.index.min(), "to", hourly.index.max(), "|", len(hourly), "rows")