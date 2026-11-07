# %%
# Housekeeping
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsIC
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from price_forecasting.models.evaluation import split, evaluate, negative_price_table, summarise, P
from price_forecasting.columns import VAL_YEAR
from price_forecasting.paths import find_project_root

PROJECT = find_project_root()
RESULTS = PROJECT / "results"

df = pd.read_csv(PROJECT / "data" / "processed" / "model_features_hourly_utc.csv",
                 index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index, utc=True)

wall = df.index.tz_convert("Europe/Berlin").tz_localize(None)
df["_date"] = wall.normalize()
df["_hour"] = wall.hour
df["ren_fc_mw"] = df["solar_fc_mw"] + df["wind_on_fc_mw"] + df["wind_off_fc_mw"]

# %%
# Create wide format DataFrames to fit LEAR model
def wide(col, fill=True):
    w = df.pivot_table(index="_date", columns="_hour", values=col, aggfunc="mean")
    w = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="D"))
    return w.interpolate(axis=1, limit_direction="both") if fill else w

P_target = wide(P, fill=False)     # target: never filled
P_feat = wide(P)                   # inputs: missing 02:00 on clock-change days filled
LOAD, REN, RES = wide("load_fc_mw"), wide("ren_fc_mw"), wide("residual_load_fc_mw")
print(P_target.shape, P_target.isna().sum().sum())

# %%
# Create features for LEAR model
parts = [P_feat.shift(k).add_prefix(f"p_d{k}_h") for k in [1, 2, 3, 7]]
for name, w in [("load", LOAD), ("ren", REN)]:
    parts += [w.add_prefix(f"{name}_d0_h"), w.shift(1).add_prefix(f"{name}_d1_h")]

daily = df.groupby("_date").agg(
    gas_mc=("gas_mc_eur_mwh", "first"),
    gas_coal=("gas_minus_coal_mc", "first"),
    holiday=("is_holiday", "max"),
    bridge=("is_bridge_day", "max"),
    xmas=("is_christmas_period", "max"),
).reindex(P_feat.index).astype(float)

dow = pd.get_dummies(P_feat.index.dayofweek, prefix="dow").set_index(P_feat.index).astype(float)

X_day = pd.concat(parts + [daily, dow], axis=1)
price_cols = [c for c in X_day.columns if c.startswith("p_d")]
excluded_days = set(df.loc[df["exclude_from_training"], "_date"])
print(X_day.shape)

# %%
# Fit LEAR model for each hour of the day
def forecast_day(D, window_days=730, min_days=300):
    start = D - pd.Timedelta(days=window_days)
    days = [d for d in X_day.index[(X_day.index >= start) & (X_day.index < D)]
            if d not in excluded_days]
    Xtr, Ytr = X_day.loc[days].copy(), P_target.loc[days]
    xD = X_day.loc[[D]].copy()

    # asinh transform with median/MAD from this training window only
    med = np.nanmedian(Ytr.values)
    mad = np.nanmedian(np.abs(Ytr.values - med)) or 1.0
    tf = lambda a: np.arcsinh((a - med) / mad)
    Xtr[price_cols], xD[price_cols] = tf(Xtr[price_cols]), tf(xD[price_cols])

    preds = {}
    for h in range(24):
        Xh, xh = Xtr.copy(), xD.copy()
        for X_ in (Xh, xh):
            res_h = RES.loc[X_.index, h]
            X_["resid_x_gas"] = res_h / 1000 * X_["gas_mc"]
            X_["low_resid"] = (res_h < 10_000).astype(float)
        y = tf(Ytr[h])
        ok = Xh.notna().all(axis=1) & y.notna()
        if ok.sum() < min_days or xh.isna().any(axis=None):
            continue
        scaler = StandardScaler().fit(Xh[ok])
        model = LassoLarsIC(criterion="aic").fit(scaler.transform(Xh[ok]), y[ok])
        z = model.predict(scaler.transform(xh))[0]
        preds[h] = np.sinh(z) * mad + med       # back to €/MWh
    return pd.Series(preds, name=D)

# %%
# Test LEAR model on January of the validation year
test_days = pd.date_range(f"{VAL_YEAR}-01-01", f"{VAL_YEAR}-01-31", freq="D")
t0 = time.time()
pred_jan = pd.DataFrame([forecast_day(D) for D in test_days])
print(f"{time.time() - t0:.0f} s for {len(test_days)} days")
print(pred_jan.iloc[:3, :6].round(1))
print(P_target.loc[test_days[:3], :5].round(1))

# %%
val_days = pd.date_range(f"{VAL_YEAR}-01-01", f"{VAL_YEAR}-12-31", freq="D")
t0 = time.time()
# Days are independent, so forecast them in parallel on all cores
pred_wide = pd.DataFrame(Parallel(n_jobs=-1, verbose=5)(delayed(forecast_day)(D) for D in val_days))
print(f"{time.time() - t0:.0f} s for {len(val_days)} days")
pred_wide.to_csv(RESULTS / "lear_730d_val_predictions_wide.csv")

_, val, _ = split(df)
key = pd.MultiIndex.from_arrays([val["_date"], val["_hour"]])
y_pred = pd.Series(pred_wide.stack().reindex(key).to_numpy(), index=val.index)
print("Missing forecasts:", y_pred.isna().sum())

res = evaluate(val, y_pred, "lear_730d")
print(res.round(2).to_string(index=False))

all_res = pd.read_csv(RESULTS / "validation_results.csv")
all_res = pd.concat([all_res[all_res["model"] != "lear_730d"], res])
all_res.to_csv(RESULTS / "validation_results.csv", index=False)

# %%
# Compare actual negative prices with LEAR forecasts
print(negative_price_table(val, y_pred, "lear_730d"))

# %%
#LEAR window comparison
# The 730-day window was already run above
lear_preds = {730: y_pred}
for w in [365, 1095]:
    t0 = time.time()
    pw = pd.DataFrame(Parallel(n_jobs=-1)(
        delayed(forecast_day)(D, window_days=w) for D in val_days))
    pw.to_csv(RESULTS / f"lear_{w}d_val_predictions_wide.csv")
    lear_preds[w] = pd.Series(pw.stack().reindex(key).to_numpy(), index=val.index)
    print(f"LEAR {w}d: {time.time() - t0:.0f} s")

lear_preds["avg"] = pd.concat([lear_preds[w] for w in [365, 730, 1095]], axis=1).mean(axis=1)

# %%
# Summarise LEAR window comparison
summary = pd.DataFrame([summarise(val, lear_preds[w], f"lear_{w}d" if w != "avg" else "lear_avg")
                        for w in [365, 730, 1095, "avg"]]).set_index("model")
print(summary.round(3).sort_values("MAE"))
summary.to_csv(RESULTS / "lear_windows_val.csv")