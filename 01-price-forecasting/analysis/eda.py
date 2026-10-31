# %%
# Load and add local columns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from price_forecasting.columns import FEATURES, OPTIONAL_FEATURES
from price_forecasting.paths import find_project_root

# 2025 is the test year: decisions made from this EDA must not see it
HOLDOUT_YEAR = 2025

PROJECT = find_project_root()
FIG = PROJECT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PROJECT / "data" / "processed" / "model_base_hourly_utc.csv",
                 index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index, utc=True)

loc = df.index.tz_convert("Europe/Berlin")
df = df[loc.year < HOLDOUT_YEAR].copy()
loc = df.index.tz_convert("Europe/Berlin")
df["year"], df["month"], df["hour"], df["dow"] = loc.year, loc.month, loc.hour, loc.dayofweek
df["is_weekend"] = df["dow"] >= 5
df["neg"] = df["price_eur_mwh"] < 0
df["ren_share_fc"] = (df["solar_fc_mw"] + df["wind_on_fc_mw"]
                      + df["wind_off_fc_mw"]) / df["load_fc_mw"]
P = "price_eur_mwh"

# %%
# How exreme are the price tails? How do they change over time?
print(df.groupby("year")[P].describe(percentiles=[.01, .05, .5, .95, .99]).round(0))

fig, ax = plt.subplots(figsize=(9, 4))
for y, g in df.groupby("year"):
    ax.hist(g[P], bins=300, histtype="step", density=True, label=y)
ax.set_xlim(-150, 600)
ax.set_xlabel("Day-ahead price (€/MWh)")
ax.set_ylabel("Density (per €/MWh)")
ax.legend()
fig.savefig(FIG / "price_distribution_by_year.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# asinh transformation
med = df[P].median()
mad = (df[P] - med).abs().median()
df["price_asinh"] = np.arcsinh((df[P] - med) / mad)
df.groupby("year")["price_asinh"].plot(kind="hist", bins=200, histtype="step",
                                        density=True, legend=True, figsize=(9, 4))
plt.xlabel(f"asinh((price − median) / MAD), unitless (median {med:.0f}, MAD {mad:.0f} €/MWh)")
plt.ylabel("Density (per unit of transformed price)")
plt.show()

# %%
# daily price shape
prof = df.pivot_table(index="hour", columns="year", values=P, aggfunc="mean")
(prof / prof.mean()).plot(figsize=(9, 4), title="Average daily price shape (1 = daily mean)")
plt.xlabel("Hour of day (Berlin time)")
plt.ylabel("Price relative to daily mean (ratio)")
plt.savefig(FIG / "daily_shape_by_year.png", dpi=150, bbox_inches="tight")
plt.show()

df.pivot_table(index="hour", columns="is_weekend", values=P, aggfunc="mean").plot(
    figsize=(9, 4), title="Weekday vs weekend")
plt.xlabel("Hour of day (Berlin time)")
plt.ylabel("Mean day-ahead price (€/MWh)")
plt.show()

# %%
# Negative prices
print(df.groupby("year")["neg"].sum())

recent = df[df["year"] >= 2023]
heat = recent.pivot_table(index="month", columns="hour", values="neg", aggfunc="mean")
fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(heat, aspect="auto", cmap="Reds")
ax.set_xlabel("Hour of day (Berlin time)"); ax.set_ylabel("Month (1 = January)")
ax.set_yticks(range(12)); ax.set_yticklabels(range(1, 13))
fig.colorbar(im, label="Share of hours with negative price (0–1)")
fig.savefig(FIG / "negative_price_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

bins = pd.cut(df["residual_load_fc_mw"] / 1000, bins=range(-40, 95, 5))
print(df.groupby(bins, observed=True)["neg"].agg(["mean", "size"]).round(3))
print(df[df["year"] >= 2023].groupby("is_weekend")["neg"].mean().round(3))

# %%
# Correlation of features with price
cols = FEATURES + OPTIONAL_FEATURES
by_year = df.groupby("year").apply(
    lambda g: g[cols].corrwith(g[P], method="spearman"))
print(by_year.T.round(2))

# %%
# Correlation matrix of features
fc = df[cols].corr(method="spearman")
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(fc, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90)
ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
fig.colorbar(im, label="Spearman correlation (−1 to 1, unitless)")
fig.savefig(FIG / "feature_correlations.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Renewable share vs price
fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(df["ren_share_fc"], df[P], c=df["gas_mc_eur_mwh"],
                s=2, alpha=0.4, cmap="viridis")
ax.set_xlabel("Forecast renewable share of load (ratio, 1 = 100% of load)")
ax.set_ylabel("Day-ahead price (€/MWh)")
fig.colorbar(sc, label="Gas plant marginal cost (€/MWh)")
fig.savefig(FIG / "renewable_share_vs_price.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Gas cost bands, within each year so they don't just compare cheap and expensive years
d = df.dropna(subset=["residual_load_fc_mw", "gas_mc_eur_mwh", P])
d = d.assign(gas_band=d.groupby("year")["gas_mc_eur_mwh"].transform(
    lambda s: pd.qcut(s, 3, labels=False)).map({0: "low", 1: "mid", 2: "high"}))
slopes = d.groupby(["year", "gas_band"]).apply(
    lambda g: np.polyfit(g["residual_load_fc_mw"] / 1000, g[P], 1)[0])
print("Price rise per GW of residual load (€/MWh per GW), by gas cost band:")
print(slopes.unstack()[["low", "mid", "high"]].round(2))

# %%
# Which price lags carry the most information?
# Per year, so differences in price level between years don't inflate the correlation.
# shift() counts rows: good enough here, but the lag features must use the Berlin date and hour.
lags = {"D-1": 24, "D-2": 48, "D-3": 72, "D-7": 168}
print(pd.DataFrame({label: df.groupby("year")[P].apply(lambda s: s.corr(s.shift(h)))
                    for label, h in lags.items()}).round(2))