# 01 - Day-Ahead Price Forecasting (in progress)

Forecasting hourly German day-ahead electricity prices (bidding zone DE-LU) from market fundamentals. The plan is to compare a statistical baseline (LEAR/SARIMA) against LightGBM and an LSTM.

**Status:** data pipeline in progress. Prices, load and wind/solar forecasts, and temperature are fetched, cleaned and merged into one hourly table for 2020–2025. Fuel and carbon prices, calendar features and modelling are next.

## Forecasting setup

- **Forecast origin:** noon CET on D-1, when the day-ahead auction closes. The model predicts all 24 hours of day D.
- **Rule:** every input must be knowable at that moment. Realised (actual) values are kept only for checks, never as model inputs.
- **Period:** German calendar years 2020–2025, i.e. 2020-01-01 00:00 to 2025-12-31 23:00 Berlin time, 52,608 hours.
- **Resolution:** hourly. The day-ahead market moved to 15-minute products on 1 October 2025; those are averaged to hourly so that all six years are on the same footing. 15-minute modelling is future work.

The full feature specification, with sources, transformations and a leakage checklist, is in [notes/features.md](notes/features.md).

## Pipeline

Run the scripts in this order; each one reads the previous one's output.

| Step | Script | Source | Output (`data/processed/`) |
|---|---|---|---|
| 1 | [src/fetch_prices.py](src/fetch_prices.py) | ENTSO-E day-ahead prices | `da_prices_hourly_utc.csv` |
| 2 | [src/fetch_fundamentals.py](src/fetch_fundamentals.py) | ENTSO-E load and wind/solar forecasts | `model_base_hourly_utc.csv` |
| 3 | [src/fetch_weather.py](src/fetch_weather.py) | Open-Meteo Previous Runs and Archive APIs | `weather_hourly_utc.csv`, and adds temperature columns to `model_base_hourly_utc.csv` |

Raw downloads are cached in `data/raw/` (per year for ENTSO-E, per city for weather), so a re-run only calls the APIs for files that are missing. CSV files are not committed to git; run the pipeline to rebuild them.

[src/project_paths.py](src/project_paths.py) holds `find_project_root`, shared by all scripts so they find `data/` from any working directory.

### Columns in `model_base_hourly_utc.csv`

| Column | Meaning | Model input? |
|---|---|---|
| `price_eur_mwh` | Day-ahead price (target) | Target; lags later |
| `load_fc_mw` | ENTSO-E day-ahead load forecast | Yes |
| `load_actual_mw` | Realised load | No, checks only |
| `solar_fc_mw`, `wind_on_fc_mw`, `wind_off_fc_mw` | ENTSO-E day-ahead generation forecasts | Yes |
| `residual_load_fc_mw` | Load forecast minus wind and solar forecasts | Yes |
| `temp_c` | Population-weighted temperature forecast (actuals where flagged) | Yes |
| `hdh` | Heating degree hours, max(0, 15.5 − `temp_c`) | Yes |
| `temp_is_actual` | `temp_c` comes from observed, not forecast, temperature | Flag |
| `fc_interpolated` | A forecast value in this hour was interpolated | Flag |
| `exclude_from_training` | Long data gap or missing price | Flag |

## Design choices and why

**Time handling**
- All data is stored with a UTC index. German local time has a 23-hour day in March and a 25-hour day in October; UTC has neither, so joins and lags stay simple.
- The period is trimmed with Berlin-time bounds (`>= 2020-01-01`, `< 2026-01-01` Europe/Berlin). That is why the files start at 2019-12-31 23:00 UTC: it is the first delivery hour of 2020 in Germany.

**Forecasts, not actuals**
- Load, wind and solar come from the ENTSO-E day-ahead forecasts, which are published before the auction. Using actuals would give the model information a trader does not have, and would make results look better than they can be in practice.
- Residual load (load minus renewables) is calculated once, after cleaning. It measures how far into the thermal part of the supply curve (gas, coal) demand reaches, which is what mainly sets the price.

**Gaps**
- Short gaps (≤3 h) in the forecasts are interpolated over time and flagged with `fc_interpolated`.
- Longer gaps are left missing and flagged with `exclude_from_training`. The main one is a full day of load forecasts missing from ENTSO-E in February 2022, confirmed by re-downloading. The rows are kept so later price lags stay on a complete hourly grid. 53 hours are excluded in total.

**Temperature**
- Temperature is used only as a proxy for heating demand. Wind and solar already come from the ENTSO-E forecasts, so no other weather variables are needed.
- **Source:** the Open-Meteo Historical Forecast API was rejected because it is stitched together from the first hours of each model run, so it behaves almost like observed weather. The Previous Runs API keeps genuinely older forecasts. `temperature_2m_previous_day2` comes from a run at least ~48 h old, which is before noon D-1 for every hour of D.
- **One model:** GFS is pinned for the whole period so that forecast quality does not change mid-sample. It is coarser than ICON or ECMWF, but temperature is the variable models forecast best, and GFS is the only one with history before 2024 (from 2021-03-25).
- **Fallback to observed temperature** for hours with no forecast: all of 2020, early 2021 and a few GFS gaps, marked by `temp_is_actual`. This keeps the same 2020–2025 window as the other data. Over the 41,315 hours with both, forecasts miss by 0.94 °C on average (0.84–1.07 °C in the heating season), so the fallback only slightly flatters the early period. The forecast also runs about 0.4–0.75 °C warm in winter, which is worth remembering if 2020 behaves differently.
- **Population-weighted average** of Berlin, Hamburg, Munich, Cologne and Frankfurt, because heating demand follows where people live.
- **Heating degree hours** rather than raw temperature, because heating demand only rises below a threshold (15.5 °C).
- **Expected value is modest.** The ENTSO-E load forecast already includes the effect of temperature on demand, and most German heating runs on gas rather than electricity. Whether temperature adds anything beyond the load forecast will be tested with and without it.

## Next steps

- TTF gas and EUA carbon prices, lagged to the last settlement before noon D-1
- Calendar and public-holiday / bridge-day features
- Price lags (same hour D-1, D-2, D-3, D-7; D-1 daily mean, min, max)
- Time-based train/validation/test split, then baseline vs LightGBM vs LSTM

## Setup

From the repository root:

```
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

Add `ENTSOE_API_KEY=your_key` to a `.env` file in the repository root. Open-Meteo needs no key.

The `de-power-price-analysis/` folder is an earlier, separate exploration of German power prices.
