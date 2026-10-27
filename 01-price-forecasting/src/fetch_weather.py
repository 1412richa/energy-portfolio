# %%
# Setup
import time

import pandas as pd
import requests

from project_paths import find_project_root

PROJECT = find_project_root()
RAW = PROJECT / "data" / "raw" / "weather"
PROCESSED = PROJECT / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)

# (lat, lon, population in millions ~2023); population weights temperature towards heating demand
CITIES = {
    "berlin":    (52.52, 13.41, 3.76),
    "hamburg":   (53.55, 9.99, 1.89),
    "munich":    (48.14, 11.58, 1.51),
    "cologne":   (50.94, 6.96, 1.08),
    "frankfurt": (50.11, 8.68, 0.77),
}

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# UTC dates covering the German calendar period 2020–2025
START_DATE, END_DATE = "2019-12-31", "2025-12-31"

# A run at least ~48 h old predates the 12:00 CET D-1 auction gate closure for every delivery hour
FC_VARIABLE = "temperature_2m_previous_day2"
FC_MODEL = "gfs_seamless"  # one model throughout; its history starts 2021-03-25
HDH_BASE_C = 15.5

# %%
# Reusable download function
def get_hourly(url, lat, lon, variable, start, end, model=None, retries=3):
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": variable,
        "start_date": start, "end_date": end,
        "timezone": "UTC",
    }
    if model:
        params["models"] = model
    for attempt in range(1, retries + 1):
        r = requests.get(url, params=params, timeout=120)
        if r.ok:
            h = r.json()["hourly"]
            idx = pd.to_datetime(h["time"], utc=True)
            s = pd.Series(h[variable], index=idx, name=variable, dtype="float64")
            if s.isna().all():
                print(f"Warning: {variable} is all-null for {start} to {end}")
            return s
        print(f"Attempt {attempt}: {r.status_code} {r.text[:200]}")
        time.sleep(5)
    r.raise_for_status()


def fetch_city(city, source):
    out = RAW / f"{city}_{source}.csv"
    if out.exists():
        s = pd.read_csv(out, index_col=0, parse_dates=True).iloc[:, 0]
        s.index = pd.to_datetime(s.index, utc=True)
        return s.rename(city)

    lat, lon, _ = CITIES[city]
    if source == "actual":
        s = get_hourly(ARCHIVE_URL, lat, lon, "temperature_2m", START_DATE, END_DATE)
    else:
        s = get_hourly(PREVIOUS_RUNS_URL, lat, lon, FC_VARIABLE,
                       START_DATE, END_DATE, model=FC_MODEL)
    s.to_csv(out)
    print(f"{city} {source}: {s.notna().sum()} non-null hours")
    return s.rename(city)


def weighted_temp(source):
    temps = pd.concat([fetch_city(c, source) for c in CITIES], axis=1)
    weights = pd.Series({c: pop for c, (_, _, pop) in CITIES.items()})
    # NaN if any city is missing, so the city mix stays the same every hour
    return (temps * weights).sum(axis=1, min_count=len(CITIES)) / weights.sum()

# %%
# Download and combine
df = pd.DataFrame({
    "temp_actual_c": weighted_temp("actual"),
    "temp_fc_c": weighted_temp("forecast"),
})

# Trim to the German calendar period 2020–2025
start = pd.Timestamp("2020-01-01", tz="Europe/Berlin")
end = pd.Timestamp("2026-01-01", tz="Europe/Berlin")
df = df[(df.index >= start) & (df.index < end)].copy()

# Use the forecast where it exists; fall back to actuals before GFS history starts and in its gaps
df["temp_is_actual"] = df["temp_fc_c"].isna()
df["temp_c"] = df["temp_fc_c"].fillna(df["temp_actual_c"])
df["hdh"] = (HDH_BASE_C - df["temp_c"]).clip(lower=0)

# %%
# Checks
expected = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
print("Complete hourly index:", df.index.equals(expected))
print("Missing temp_c:", df["temp_c"].isna().sum())

year_local = df.index.tz_convert("Europe/Berlin").year
print("Share of hours using actual temperature by year (%):")
print((df["temp_is_actual"].groupby(year_local).mean() * 100).round(1))

# How much the actual-temperature fallback flatters the model: forecast error where both exist
both = df.dropna(subset=["temp_fc_c", "temp_actual_c"])
temp_err = both["temp_fc_c"] - both["temp_actual_c"]
hdh_err = ((HDH_BASE_C - both["temp_fc_c"]).clip(lower=0)
           - (HDH_BASE_C - both["temp_actual_c"]).clip(lower=0))
print(f"Overlap: {both.index.min()} to {both.index.max()}, {len(both)} h")
print(f"  temp bias {temp_err.mean():+.2f} °C, MAE {temp_err.abs().mean():.2f} °C, "
      f"P95 abs {temp_err.abs().quantile(0.95):.2f} °C")

# Heating season (Oct–Apr) is where HDH errors actually matter
local = both.index.tz_convert("Europe/Berlin")
season = pd.Series(local.month.isin([10, 11, 12, 1, 2, 3, 4]), index=both.index).map(
    {True: "heating", False: "summer"})
summary = pd.DataFrame({"temp_mae": temp_err.abs(), "temp_bias": temp_err,
                        "hdh_mae": hdh_err.abs()}).groupby([local.year, season]).mean()
print(summary.round(2))

# %%
# Save (actual temperature is kept out of the output so it can't leak in as a feature)
out = df[["temp_c", "hdh", "temp_is_actual"]]
out.index.name = "timestamp_utc"
out.to_csv(PROCESSED / "weather_hourly_utc.csv")
print(out.shape, out.index.min(), out.index.max())

# %%
# Merge into the model base (prices, load and wind/solar forecasts)
base = pd.read_csv(PROCESSED / "model_base_hourly_utc.csv",
                   index_col=0, parse_dates=True)
base.index = pd.to_datetime(base.index, utc=True)
print("Same hourly index as base:", base.index.equals(out.index))

# Drop weather columns from an earlier run so the cell can be re-run safely
merged = base.drop(columns=out.columns, errors="ignore").join(out, how="left")
print(merged[out.columns].isna().sum())

merged.index.name = "timestamp_utc"
merged.to_csv(PROCESSED / "model_base_hourly_utc.csv")
print(merged.shape, merged.index.min(), merged.index.max())
