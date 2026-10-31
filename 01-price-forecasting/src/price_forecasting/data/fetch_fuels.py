# %%
# Setup
import time
import warnings

import pandas as pd
import requests
import yfinance as yf

from price_forecasting.paths import find_project_root

PROJECT = find_project_root()
RAW = PROJECT / "data" / "raw" / "fuels"
PROCESSED = PROJECT / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)

# Delivery days (German calendar); fetch earlier so the first days of 2020 have a lagged value
FIRST_DELIVERY, LAST_DELIVERY = "2020-01-01", "2025-12-31"
FETCH_START, FETCH_END = "2019-11-01", "2026-01-01"

# Settlements and auction results are published by the evening of their date,
# so at noon D-1 the latest usable value is dated D-2 or earlier
LAG_DAYS = 2

TICKERS = {
    "ttf": "TTF=F",          # Dutch TTF gas front-month, €/MWh
    "coal": "MTF=F",         # Rotterdam coal (API2) front-month, $/t
    "eurusd": "EURUSD=X",    # USD per EUR, to convert coal to €/t
}
# Files up to 2019 are old-format .xls, later ones .xlsx
EEX_URL = ("https://public.eex-group.com/eex/eua-auction-report/"
           "emission-spot-primary-market-auction-report-{year}-data.{ext}")

# EEX workbooks have no default style; openpyxl warns on every read
warnings.filterwarnings("ignore", message="Workbook contains no default style")

# %%
# Reusable download functions
def fetch_ticker(name, ticker, retries=3):
    out = RAW / f"{name}.csv"
    if out.exists():
        return pd.read_csv(out, index_col=0, parse_dates=True).iloc[:, 0]

    for attempt in range(1, retries + 1):
        d = yf.download(ticker, start=FETCH_START, end=FETCH_END,
                        progress=False, auto_adjust=False)
        if len(d):
            s = d["Close"].squeeze().dropna().rename(name)
            s.index.name = "date"
            s.to_csv(out)
            print(f"{name}: {len(s)} trading days")
            return s
        print(f"{name} attempt {attempt}: no data")
        time.sleep(10)
    raise RuntimeError(f"Could not fetch {name} ({ticker})")


def fetch_eua_year(year, retries=3):
    """EEX primary auction prices for one year, €/t, one value per auction day."""
    ext = "xls" if year <= 2019 else "xlsx"
    out = RAW / f"eua_auctions_{year}.{ext}"
    if not out.exists():
        for attempt in range(1, retries + 1):
            r = requests.get(EEX_URL.format(year=year, ext=ext),
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
            if r.ok:
                out.write_bytes(r.content)
                break
            print(f"EUA {year} attempt {attempt}: {r.status_code}")
            time.sleep(10)
        else:
            r.raise_for_status()

    # The header row sits below a few title rows; find it by its "Date" cell
    raw = pd.read_excel(out, header=None)
    header = raw.index[raw.iloc[:, 1].eq("Date")][0]
    x = raw.iloc[header + 1:].set_axis(raw.iloc[header], axis=1)
    price_col = next(c for c in x.columns if str(c).startswith("Auction Price"))

    # T3PA = general EUA (EUAA is aviation allowances); skip cancelled auctions
    # (the 2019 file has no Status column; cancelled auctions there have no price)
    ok = x["Contract"] == "T3PA"
    if "Status" in x.columns:
        ok &= x["Status"] == "successful"
    x = x.loc[ok]
    price = pd.to_numeric(x[price_col], errors="coerce")
    return price.groupby(pd.to_datetime(x["Date"])).mean().dropna()


def known_by_noon_d1(s):
    """For each delivery day D: the last value dated on or before D-2, and its age in days."""
    s = s.dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    days = pd.date_range(FIRST_DELIVERY, LAST_DELIVERY, freq="D")
    cutoff = days - pd.Timedelta(days=LAG_DAYS)
    value = s.asof(cutoff).to_numpy()
    price_date = pd.DatetimeIndex(pd.Series(s.index, index=s.index).asof(cutoff))
    return pd.DataFrame({"value": value, "age_d": (days - price_date).days}, index=days)

# %%
# Download
raw = {name: fetch_ticker(name, t) for name, t in TICKERS.items()}
raw["eua"] = pd.concat([fetch_eua_year(y) for y in range(2019, 2026)]).rename("eua")

# Convert coal to €/t with the exchange rate on the same date
raw["coal_eur"] = raw["coal"] / raw["eurusd"].asof(raw["coal"].index).to_numpy()

# %%
# Align to delivery days
fuels = pd.DataFrame(index=pd.date_range(FIRST_DELIVERY, LAST_DELIVERY, freq="D"))
for col, src in [("ttf_eur_mwh", "ttf"), ("eua_eur_t", "eua"), ("coal_eur_t", "coal_eur")]:
    lagged = known_by_noon_d1(raw[src])
    prefix = col.split("_")[0]
    fuels[col] = lagged["value"]
    fuels[f"{prefix}_age_d"] = lagged["age_d"]
fuels.index.name = "delivery_date"

# %%
# Marginal cost of power from gas and coal plants (fuel + carbon), €/MWh of electricity
# Gas plant (combined cycle)
GAS_EFF = 0.50                   # net (lower heating value) efficiency
GAS_GCV_PER_NCV = 1.108          # TTF is priced per MWh gross calorific value
CO2_PER_MWH_GAS = 0.202          # t CO2 per MWh of gas burned (net)

# Coal plant (hard coal)
COAL_EFF = 0.40
COAL_MWH_PER_T = 6.98            # API2 quality: 6,000 kcal/kg ≈ 25.1 GJ/t (net)
CO2_PER_MWH_COAL = 0.34          # t CO2 per MWh of coal burned

fuels["gas_mc_eur_mwh"] = (fuels["ttf_eur_mwh"] * GAS_GCV_PER_NCV
                           + fuels["eua_eur_t"] * CO2_PER_MWH_GAS) / GAS_EFF
fuels["coal_mc_eur_mwh"] = (fuels["coal_eur_t"] / COAL_MWH_PER_T
                            + fuels["eua_eur_t"] * CO2_PER_MWH_COAL) / COAL_EFF
# Positive: coal plants are cheaper than gas plants
fuels["gas_minus_coal_mc"] = fuels["gas_mc_eur_mwh"] - fuels["coal_mc_eur_mwh"]

# %%
# Checks
print("Missing delivery days:")
print(fuels.isna().sum())

# Age 2 = normal; higher = weekends, holidays, or auction breaks (e.g. Christmas)
age_cols = [c for c in fuels.columns if c.endswith("_age_d")]
print("Age of value used (days), max and where:")
for c in age_cols:
    print(f"  {c}: max {fuels[c].max()} on {fuels[c].idxmax().date()}, "
          f"share > 4 days {(fuels[c] > 4).mean() * 100:.1f}%")

price_cols = ["ttf_eur_mwh", "eua_eur_t", "coal_eur_t", "gas_mc_eur_mwh", "coal_mc_eur_mwh"]
print(fuels[price_cols].groupby(fuels.index.year).agg(["mean", "max"]).round(1))

# Largest day-to-day moves in the raw series; coal's 2021-11-01 drop is a contract roll
for name in ["ttf", "eua", "coal_eur"]:
    moves = raw[name].pct_change().abs().nlargest(3) * 100
    print(f"{name} largest moves (%):", {d.date(): round(v) for d, v in moves.items()})

# %%
# Save (already lagged: each row is what was known at noon before that delivery day)
fuels.to_csv(PROCESSED / "fuels_daily.csv")
print(fuels.shape, fuels.index.min().date(), fuels.index.max().date())
