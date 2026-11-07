# %%
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from price_forecasting.models.evaluation import split, evaluate, negative_price_table, summarise, P
from price_forecasting.columns import ALL_FEATURES, OPTIONAL_FEATURES, INTERACTION_FEATURES, VAL_YEAR
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

months = pd.date_range(f"{VAL_YEAR}-01-01", f"{VAL_YEAR}-12-01", freq="MS")
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

# %%
# reuseable functions for rolling-window training and evaluation
_, val, _ = split(df)

def run_lgbm(params=PARAMS, x_cols=X_COLS, window_days=730, freq="MS", target="raw"):
    # Always start on 1 January, even when a weekly freq would first land later in the week
    first = pd.Timestamp(f"{VAL_YEAR}-01-01")
    starts = pd.date_range(first, f"{VAL_YEAR}-12-31", freq=freq).union([first])
    bounds = list(starts) + [pd.Timestamp(f"{VAL_YEAR + 1}-01-01")]
    cats = [c for c in CAT_COLS if c in x_cols]
    preds = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        tr = train_window(s, window_days)
        te = df[(local_date >= s) & (local_date < e)]
        y = tr[P]
        if target == "asinh":
            med = y.median()
            mad = (y - med).abs().median()
            y = np.arcsinh((y - med) / mad)
        model = lgb.LGBMRegressor(**params).fit(tr[x_cols], y, categorical_feature=cats)
        p = model.predict(te[x_cols])
        if target == "asinh":
            p = np.sinh(p) * mad + med
        preds.append(pd.Series(p, index=te.index))
    return pd.concat(preds).reindex(val.index)


# %%
# experiment with different model configurations
no_eda = [c for c in X_COLS if c not in INTERACTION_FEATURES]
with_temp = X_COLS + OPTIONAL_FEATURES

experiments = {
    "lgbm_base":        dict(),
    "lgbm_win365":      dict(window_days=365),
    "lgbm_win1095":     dict(window_days=1095),
    "lgbm_weekly":      dict(freq="W-MON"),
    "lgbm_asinh_l2":    dict(target="asinh", params=PARAMS | {"objective": "l2"}),
    "lgbm_no_eda":      dict(x_cols=no_eda),
    "lgbm_with_temp":   dict(x_cols=with_temp),
    "lgbm_leaves31":    dict(params=PARAMS | {"num_leaves": 31}),
    "lgbm_leaves127":   dict(params=PARAMS | {"num_leaves": 127}),
    "lgbm_weekly_temp": dict(freq="W-MON", x_cols=with_temp),   # post-hoc, declared after results
}

summary, preds_store = [], {}
for name, kwargs in experiments.items():
    t0 = time.time()
    preds_store[name] = run_lgbm(**kwargs)
    summary.append(summarise(val, preds_store[name], name))
    print(f"{name}: {time.time() - t0:.0f} s")

summary = pd.DataFrame(summary).set_index("model")
print(summary.round(3).sort_values("MAE"))
summary.to_csv(RESULTS / "lgbm_experiments_val.csv")
# One column per experiment, hourly UTC index, so they can be compared hour by hour later
pd.DataFrame(preds_store).to_csv(RESULTS / "lgbm_experiments_val_predictions.csv")