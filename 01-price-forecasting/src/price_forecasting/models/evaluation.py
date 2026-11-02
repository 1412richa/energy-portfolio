import numpy as np
import pandas as pd

from price_forecasting.columns import TARGET as P, HOLDOUT_YEAR

def local_year(index):
    return index.tz_convert("Europe/Berlin").year

def split(df):
    """Train up to two years before the test year, validate on the year before it,
    test on HOLDOUT_YEAR (German calendar years)."""
    y = local_year(df.index)
    return df[y < HOLDOUT_YEAR - 1], df[y == HOLDOUT_YEAR - 1], df[y == HOLDOUT_YEAR]

def lago_naive(df):
    """Naive benchmark from Lago et al. (2021): same hour one week earlier
    for Mon, Sat, Sun; same hour the previous day otherwise."""
    dow = df.index.tz_convert("Europe/Berlin").dayofweek
    use_week = np.isin(dow, [0, 5, 6])
    return pd.Series(np.where(use_week, df["price_lag_d7"], df["price_lag_d1"]),
                     index=df.index)

def metrics(y_true, y_pred, y_naive=None):
    mask = y_true.notna() & y_pred.notna()
    if y_naive is not None:
        mask &= y_naive.notna()
    err = y_pred[mask] - y_true[mask]
    out = {
        "MAE": err.abs().mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "bias": err.mean(),
        "hours": int(mask.sum()),
    }
    if y_naive is not None:
        out["rMAE"] = out["MAE"] / (y_naive[mask] - y_true[mask]).abs().mean()
    return out

def evaluate(df, y_pred, name):
    """Metrics overall and for the hours that matter most."""
    y, naive = df[P], lago_naive(df)
    hour = df.index.tz_convert("Europe/Berlin").hour
    subsets = {
        "all": np.ones(len(df), bool),
        "negative_price": (y < 0).to_numpy(),
        "low_resid_load": df["low_resid_flag"].to_numpy(),
        "evening_peak_17_20": np.isin(hour, [17, 18, 19, 20]),
    }
    rows = []
    for sub, m in subsets.items():
        r = metrics(y[m], y_pred[m], naive[m])
        rows.append({"model": name, "subset": sub, **r})
    return pd.DataFrame(rows)
