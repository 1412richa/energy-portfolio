# %%
# Figures for the README, built only from own results (no raw prices or wind/solar series)
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
from price_forecasting.columns import ALL_FEATURES, HOLDOUT_YEAR
from price_forecasting.models.evaluation import split, lago_naive, P
from price_forecasting.paths import find_project_root

PROJECT = find_project_root()
RESULTS = PROJECT / "results"
DOCS = PROJECT / "docs" / "figures"
DOCS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PROJECT / "data" / "processed" / "model_features_hourly_utc.csv",
                 index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index, utc=True)
_, _, test = split(df)

def load(name, col):
    s = pd.read_csv(RESULTS / name, index_col=0, parse_dates=True)[col]
    s.index = pd.to_datetime(s.index, utc=True)
    return s.reindex(test.index)

# %%
# 1. Test-year MAE by month: an aggregate of errors, so no prices are shown
preds = {"Naive benchmark": lago_naive(test),
         "LEAR": load("test_lear_avg.csv", "lear"),
         "LightGBM": load("test_lgbm_weekly.csv", "lgbm")}
month = test.index.tz_convert("Europe/Berlin").month
mae = pd.DataFrame({n: (p - test[P]).abs().groupby(month).mean() for n, p in preds.items()})

fig, ax = plt.subplots(figsize=(9, 4))
mae.plot(ax=ax, marker="o")
ax.set_xticks(range(1, 13))
ax.set_xlabel("Month")
ax.set_ylabel("MAE (€/MWh)")
ax.set_title(f"Forecast error by month, test year {HOLDOUT_YEAR}")
ax.set_ylim(bottom=0)
fig.savefig(DOCS / "test_mae_by_month.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)
print(mae.round(1))

# %%
# 2. What LightGBM relies on: gain importance of the model behind the last test-year week.
# Same settings, features and training window as lgbm_weekly in models/lightgbm_model.py.
PARAMS = dict(objective="l1", learning_rate=0.03, n_estimators=1500, num_leaves=63,
              min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              n_jobs=-1, verbose=-1, random_state=42)
CAT_COLS = ["hour", "dow", "month"]

local_date = df.index.tz_convert("Europe/Berlin").tz_localize(None).normalize()
d = df.copy()
d["hour"] = pd.Categorical(d["hour"], categories=range(24))
d["dow"] = pd.Categorical(d["dow"], categories=range(7))
d["month"] = pd.Categorical(d["month"], categories=range(1, 13))
bool_cols = [c for c in ALL_FEATURES if d[c].dtype == bool]
d[bool_cols] = d[bool_cols].astype(int)

last_retrain = pd.date_range(f"{HOLDOUT_YEAR}-01-01", f"{HOLDOUT_YEAR}-12-31", freq="W-MON")[-1]
mask = ((local_date >= last_retrain - pd.Timedelta(days=730)) & (local_date < last_retrain)
        & ~d["exclude_from_training"].astype(bool) & d[P].notna())
model = lgb.LGBMRegressor(**PARAMS).fit(d.loc[mask, ALL_FEATURES], d.loc[mask, P],
                                        categorical_feature=CAT_COLS)

LABELS = {
    "resid_x_gas_mc": "Residual load × gas plant cost", "residual_load_fc_mw": "Residual load forecast",
    "hour": "Hour of day", "price_lag_d1": "Price, same hour yesterday",
    "price_lag_d7": "Price, same hour last week", "month": "Month",
    "wind_on_fc_mw": "Onshore wind forecast", "wind_off_fc_mw": "Offshore wind forecast",
    "solar_fc_mw": "Solar forecast", "load_fc_mw": "Load forecast", "eua_eur_t": "Carbon price",
    "ttf_eur_mwh": "Gas price", "coal_eur_t": "Coal price", "gas_mc_eur_mwh": "Gas plant cost",
    "coal_mc_eur_mwh": "Coal plant cost", "gas_minus_coal_mc": "Gas minus coal plant cost",
    "price_d1_min": "Yesterday's lowest price", "price_d1_max": "Yesterday's highest price",
    "price_d1_mean": "Yesterday's mean price", "price_roll7d": "7-day mean price",
    "price_roll30d": "30-day mean price", "price_lag_d2": "Price, 2 days ago",
    "price_lag_d3": "Price, 3 days ago", "dow": "Day of week",
}
imp = pd.Series(model.booster_.feature_importance("gain"), index=ALL_FEATURES)
imp = (imp / imp.sum() * 100).sort_values().tail(12).rename(lambda c: LABELS.get(c, c))

fig, ax = plt.subplots(figsize=(8, 5))
imp.plot.barh(ax=ax)
ax.set_xlabel("Share of total gain (%)")
ax.set_title(f"What LightGBM relies on (model trained up to {last_retrain:%d %b %Y})")
fig.savefig(DOCS / "lgbm_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)
print(imp.sort_values(ascending=False).round(1))
