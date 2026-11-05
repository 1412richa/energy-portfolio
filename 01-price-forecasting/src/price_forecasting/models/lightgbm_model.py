# %%
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from price_forecasting.models.evaluation import split, evaluate, negative_price_table, P
from price_forecasting.columns import ALL_FEATURES
from price_forecasting.paths import find_project_root

PROJECT = find_project_root()
RESULTS = PROJECT / "results"

df = pd.read_csv(PROJECT / "data" / "processed" / "model_features_hourly_utc.csv",
                 index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index, utc=True)
local_date = df.index.tz_convert("Europe/Berlin").tz_localize(None).normalize()

X_COLS = ALL_FEATURES
CAT_COLS = ["hour", "dow", "month"]

# Fixed categories so every training window uses the same coding
df["hour"] = pd.Categorical(df["hour"], categories=range(24))
df["dow"] = pd.Categorical(df["dow"], categories=range(7))
df["month"] = pd.Categorical(df["month"], categories=range(1, 13))
bool_cols = [c for c in X_COLS if df[c].dtype == bool]
df[bool_cols] = df[bool_cols].astype(int)

print(len(X_COLS), "features")

# %%
# Define LightGBM parameters
PARAMS = dict(
    objective="l1",
    learning_rate=0.03,
    n_estimators=1500,
    num_leaves=63,
    min_child_samples=50,
    subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8,
    n_jobs=-1, verbose=-1, random_state=42,
)

# %%
# Train LightGBM model on rolling monthly windows
def train_window(month_start, window_days=730):
    start = month_start - pd.Timedelta(days=window_days)
    mask = ((local_date >= start) & (local_date < month_start)
            & ~df["exclude_from_training"].astype(bool) & df[P].notna())
    return df[mask]

months = pd.date_range("2024-01-01", "2024-12-01", freq="MS")
preds, models = [], {}
t0 = time.time()
for m in months:
    tr = train_window(m)
    te = df[(local_date >= m) & (local_date < m + pd.offsets.MonthBegin(1))]
    model = lgb.LGBMRegressor(**PARAMS).fit(tr[X_COLS], tr[P], categorical_feature=CAT_COLS)
    preds.append(pd.Series(model.predict(te[X_COLS]), index=te.index))
    models[m] = model
    print(f"{m:%Y-%m}: trained on {len(tr)} hours")
print(f"{time.time() - t0:.0f} s")

y_pred = pd.concat(preds)

# %%
# Evaluate and compare with LEAR model
_, val, _ = split(df)
y_pred = y_pred.reindex(val.index)
print("Missing forecasts:", y_pred.isna().sum())

res = evaluate(val, y_pred, "lgbm_730d_monthly")
print(res.round(2).to_string(index=False))

all_res = pd.read_csv(RESULTS / "validation_results.csv")
all_res = pd.concat([all_res[all_res["model"] != "lgbm_730d_monthly"], res])
all_res.to_csv(RESULTS / "validation_results.csv", index=False)

print("\nrMAE by model and subset:")
print(all_res.pivot(index="model", columns="subset", values="rMAE").round(2))

print("\nNegative-price hours:")
print(negative_price_table(val, y_pred, "lgbm_730d_monthly"))

y_pred.to_frame("lgbm").to_csv(RESULTS / "lgbm_730d_monthly_val_predictions.csv")

# %%
# Feature importance
last = models[months[-1]]
imp = pd.Series(last.booster_.feature_importance("gain"), index=X_COLS)
print((imp / imp.sum() * 100).sort_values(ascending=False).head(15).round(1))