# %%
# Read-only checks on the model base; run after any pipeline step
import pandas as pd

from columns import TARGET, FEATURES, OPTIONAL_FEATURES, FLAGS, DIAGNOSTICS
from project_paths import find_project_root

PROJECT = find_project_root()
PROCESSED = PROJECT / "data" / "processed"

df = pd.read_csv(PROCESSED / "model_base_hourly_utc.csv", index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index, utc=True)
print(df.shape)

# %%
expected = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
assert df.index.equals(expected), "Index is not a complete hourly UTC sequence"

local = df.index.tz_convert("Europe/Berlin")
print("Local range:", local.min(), "to", local.max())

hours_per_day = pd.Series(1, index=local).groupby(local.date).sum()
print(hours_per_day.value_counts())

# %%
print("Missing values:")
print(df.isna().sum()[df.isna().sum() > 0])

RANGES = {
    "price_eur_mwh":       (-500, 4000),     # EPEX day-ahead price limits
    "load_fc_mw":          (30_000, 90_000),
    "load_actual_mw":      (30_000, 90_000),
    "solar_fc_mw":         (0, 80_000),
    "wind_on_fc_mw":       (0, 70_000),
    "wind_off_fc_mw":      (0, 10_000),
    "residual_load_fc_mw": (-40_000, 90_000),
    "temp_c":              (-25, 40),
    "hdh":                 (0, 40),
    "ttf_eur_mwh":         (0, 400),
    "eua_eur_t":           (10, 120),
    "coal_eur_t":          (30, 450),
    "gas_mc_eur_mwh":      (0, 1000),
    "coal_mc_eur_mwh":     (0, 500),
    "ttf_age_d":           (2, 10),
    "eua_age_d":           (2, 60),          # auction breaks, e.g. start of 2021
    "coal_age_d":          (2, 10),
}
for col, (lo, hi) in RANGES.items():
    s = df[col].dropna()
    out = ((s < lo) | (s > hi)).sum()
    print(f"{col:22s} min={s.min():10.1f}  max={s.max():10.1f}  outside range={out}")

# %%
age_cols = ["ttf_age_d", "eua_age_d", "coal_age_d"]
print(df[age_cols].describe(percentiles=[0.5, 0.99]).round(1))
assert (df[age_cols].min() >= 2).all(), "A fuel price is newer than D-2: leakage"

# %%
year_local = local.year
print(df.groupby(year_local)[["price_eur_mwh", "gas_mc_eur_mwh",
                              "coal_mc_eur_mwh", "gas_minus_coal_mc"]].mean().round(0))

# %%
roles = [TARGET, *FEATURES, *OPTIONAL_FEATURES, *FLAGS, *DIAGNOSTICS]
dupes = {c for c in roles if roles.count(c) > 1}
assert not dupes, f"Columns with more than one role: {dupes}"

unlisted = set(df.columns) - set(roles)
assert not unlisted, f"Columns without a role: {unlisted}"
missing = set(roles) - set(df.columns)
assert not missing, f"Listed columns not in the base: {missing}"
print("All columns have exactly one role.")
