# %%
import numpy as np
import pandas as pd
import holidays

from price_forecasting.columns import (TARGET as P, HOLDOUT_YEAR, BASE_FEATURES, OPTIONAL_FEATURES,
                                       FLAGS, DIAGNOSTICS, LAG_FEATURES, CALENDAR_FEATURES,
                                       INTERACTION_FEATURES, ALL_FEATURES, LEAR_DROP)
from price_forecasting.paths import find_project_root

PROJECT = find_project_root()
PROCESSED = PROJECT / "data" / "processed"

df = pd.read_csv(PROCESSED / "model_base_hourly_utc.csv", index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index, utc=True)

wall = df.index.tz_convert("Europe/Berlin").tz_localize(None)   # German wall-clock time
date = wall.normalize()                                          # German delivery date

# %%
# Price by wall-clock time; the repeated hour in October is averaged, and the hour
# skipped in March is the mean of its neighbours so later lags have a value
price_wall = df[P].groupby(wall).mean().asfreq("h").interpolate(limit=1, limit_area="inside")

def same_hour_lag(days):
    looked_up = price_wall.reindex(wall - pd.Timedelta(days=days))
    return pd.Series(looked_up.to_numpy(), index=df.index)

for k in [1, 2, 3, 7]:
    df[f"price_lag_d{k}"] = same_hour_lag(k)

# %%
daily = df[P].groupby(date).agg(["mean", "min", "max"])

prev_day = date - pd.Timedelta(days=1)
for stat in ["mean", "min", "max"]:
    df[f"price_d1_{stat}"] = daily[stat].reindex(prev_day).to_numpy()

# Rolling averages of daily mean prices, ending on D−1
daily_mean = daily["mean"].asfreq("D")
for window in [7, 30]:
    rolled = daily_mean.rolling(window, min_periods=int(window * 0.8)).mean()
    df[f"price_roll{window}d"] = rolled.reindex(prev_day).to_numpy()

# The start of the data has no price history yet
df["exclude_from_training"] = df["exclude_from_training"] | df[LAG_FEATURES].isna().any(axis=1)

# %%
df["hour"] = wall.hour
df["dow"] = wall.dayofweek          # 0 = Monday
df["month"] = wall.month
df["is_weekend"] = df["dow"] >= 5

# %%
years = range(2019, 2027)
de_holidays = holidays.Germany(years=years)          # national holidays only
holiday_dates = pd.to_datetime(list(de_holidays.keys()))

df["is_holiday"] = date.isin(holiday_dates)

# Bridge day: a working day between a holiday and the weekend
next_day, prev_day_ = date + pd.Timedelta(days=1), date - pd.Timedelta(days=1)
df["is_bridge_day"] = (
    ~df["is_weekend"] & ~df["is_holiday"]
    & ((df["dow"].eq(4) & prev_day_.isin(holiday_dates))     # Friday after a Thursday holiday
       | (df["dow"].eq(0) & next_day.isin(holiday_dates)))   # Monday before a Tuesday holiday
)

# 24 Dec – 1 Jan: low demand even on working days
md = wall.month * 100 + wall.day
df["is_christmas_period"] = (md >= 1224) | (md <= 101)

print(df.groupby(wall.year)[["is_holiday", "is_bridge_day", "is_christmas_period"]]
      .sum().div(24).round(1))

# %%
df["low_resid_flag"] = df["residual_load_fc_mw"] < 10_000
df["resid_x_gas_mc"] = (df["residual_load_fc_mw"] / 1000) * df["gas_mc_eur_mwh"]

# %%
# 1. Spot-check one lag against a direct lookup
t = df.index[5000]
w = t.tz_convert("Europe/Berlin").tz_localize(None)
assert np.isclose(df.loc[t, "price_lag_d1"], price_wall[w - pd.Timedelta(days=1)])

# 2. Every column has exactly one role
roles = [P, *BASE_FEATURES, *OPTIONAL_FEATURES, *FLAGS, *DIAGNOSTICS,
         *LAG_FEATURES, *CALENDAR_FEATURES, *INTERACTION_FEATURES]
dupes = {c for c in roles if roles.count(c) > 1}
assert not dupes, f"Columns with more than one role: {dupes}"
assert set(df.columns) == set(roles), f"Role mismatch: {set(df.columns) ^ set(roles)}"
assert set(LEAR_DROP) <= set(ALL_FEATURES), "LEAR_DROP names an unknown feature"

# 3. Missing values only in rows excluded from training
print(df[LAG_FEATURES].isna().sum())
usable = df.loc[~df["exclude_from_training"], ALL_FEATURES + OPTIONAL_FEATURES + [P]]
assert not usable.isna().any().any(), "Missing values in rows used for training"
print("Excluded from training:", df["exclude_from_training"].sum())

# 4. Leakage smell test: nothing should match the target almost perfectly
train = df[wall.year < HOLDOUT_YEAR]
corr = train[LAG_FEATURES].corrwith(train[P]).round(3)
print(corr.sort_values(ascending=False))
assert (corr.abs() < 0.99).all(), "A feature is suspiciously close to the target"

# %%
# Save to its own file; model_base_hourly_utc.csv belongs to build_base.py
df.index.name = "timestamp_utc"
df.to_csv(PROCESSED / "model_features_hourly_utc.csv")
print(df.shape, df.index.min(), df.index.max())
