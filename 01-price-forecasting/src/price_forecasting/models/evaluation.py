import numpy as np
import pandas as pd

from price_forecasting.columns import TARGET as P, HOLDOUT_YEAR, VAL_YEAR

def local_year(index):
    return index.tz_convert("Europe/Berlin").year

def split(df):
    """Train before VAL_YEAR, validate on VAL_YEAR, test on HOLDOUT_YEAR (German calendar years)."""
    y = local_year(df.index)
    return df[y < VAL_YEAR], df[y == VAL_YEAR], df[y == HOLDOUT_YEAR]

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
        "low_resid_load": df["low_resid_flag"].astype(bool).to_numpy(),   # models may store it as 0/1
        "evening_peak_17_20": np.isin(hour, [17, 18, 19, 20]),
    }
    rows = []
    for sub, m in subsets.items():
        r = metrics(y[m], y_pred[m], naive[m])
        rows.append({"model": name, "subset": sub, **r})
    return pd.DataFrame(rows)

def negative_price_table(df, y_pred, name):
    """For hours with an actual negative price: how many fall in each price band,
    and where the forecasts for those same hours land (€/MWh)."""
    neg = df[P] < 0
    bins = [-np.inf, -50, -20, -5, -1, 0, np.inf]
    fc = y_pred[neg]
    table = pd.DataFrame({
        "actual": pd.cut(df.loc[neg, P], bins).value_counts(sort=False),
        name: pd.cut(fc, bins).value_counts(sort=False),
    })
    table.index = table.index.astype(str)
    table.loc["missing"] = [0, int(fc.isna().sum())]
    return table

def sign_skill(y_true, y_pred):
    """How well a model calls negative prices; NaN when there is nothing to score."""
    actual, forecast = y_true < 0, y_pred < 0
    hits = (actual & forecast).sum()
    return {
        "neg_recall": hits / actual.sum() if actual.any() else np.nan,          # share of negative hours caught
        "neg_precision": hits / forecast.sum() if forecast.any() else np.nan,   # share of negative forecasts that were right
    }

def summarise(df, y_pred, name):
    """One row per model: headline metrics plus negative-price sign skill."""
    r = evaluate(df, y_pred, name).set_index("subset")
    return {"model": name,
            "MAE": r.loc["all", "MAE"], "RMSE": r.loc["all", "RMSE"],
            "rMAE": r.loc["all", "rMAE"], "rMAE_neg": r.loc["negative_price", "rMAE"],
            **sign_skill(df[P], y_pred)}