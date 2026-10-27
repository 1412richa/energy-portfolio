# Project 1: Feature specification

Forecast origin: noon CET on D-1 (day-ahead gate closure).\
Rule: every feature must be knowable at that moment.

## Source and transformation
| Feature | Mechanism | Known at noon D-1? | Source | Transformation | Status |
|---|---|---|---|---|---|
| Load forecast | Demand position on merit order | Yes (forecast) | ENTSO-E | Convert to UTC; aggregate 15-min → hourly mean; handle DST | :white_check_mark: |
| Wind on/offshore forecast | Supply curve shift | Yes (forecast) | ENTSO-E | Convert to UTC; aggregate 15-min → hourly mean; handle DST | :white_check_mark: |
| Solar forecast | Supply curve shift | Yes (forecast) | ENTSO-E | Convert to UTC; aggregate 15-min → hourly mean; handle DST | :white_check_mark: |
| Residual load forecast | Depth into thermal stack | Yes (derived) | ENTSO-E | load_fc - wind_on_fc - wind_off_fc - solar_fc | :white_check_mark: |
| Temperature | Heating demand | Yes (forecast, run ≥48 h old) | Open-Meteo Previous Runs API (GFS, `temperature_2m_previous_day2`); Archive API actuals as flagged fallback | Population-weighted average of 5 cities → heating degree hours: max(0, 15.5-T); `temp_is_actual` flag | :white_check_mark: |
| TTF gas price | Thermal segment height | Yes (last settlement) | yfinance Dutch TTF Futures Tickers. TTF front-month used as proxy for German gas costs (THE). The THE–TTF spread is usually small but widened in 2022; possible source of residual error in the crisis period. | Daily settlement; forward-fill weekends and holidays; lag one trading day | [ ] |
| EUA carbon price | Thermal segment height | Yes (last settlement) | ENTSO-E | Daily settlement; forward-fill weekends and holidays; lag one trading day | [ ] |
| Calendar (hour, weekday, month) | Demand curve shape | Yes | python holidays package, country DE | One-hot hour and weekday; month as categorical | [ ] |
| Public holidays / bridge days | Low-demand days | Yes | python holidays package, country DE; national holidays plus a state-share weighting optional; bridge days are derived: weekday between holiday and weekend | Binary national-holiday flag; separate bridge-day flag | [ ] |
| Price lags D-1, D-2, D-3, D-7 | Persistence | Yes | ENTSO-E; derived from day-ahead prices | Same hour on D-1, D-2, D-3, D-7; plus D-1 daily mean, min, max | :white_check_mark: |

Hourly resolution for consistency across 2020–2025 data; 15-min extension is future work.

## Leakage checklist
- [ ] No actual (realised) load/wind/solar in model inputs - forecasts only
- [ ] Weather from Previous Runs API (run ≥48 h old), not reanalysis or Historical Forecast API; actuals only where flagged `temp_is_actual`
- [ ] No intraday price lags (no t-1h) - full day lags only
- [ ] Fuel prices lagged to the last settlement before noon D-1
- [ ] Train/test split by time, never shuffled

## Data Quality
- Load forecast missing for one full day in Feb 23 2022 (right before the Russia-Ukrain War); rows kept for lag computation but excluded from training. Short gaps (≤3 h) interpolated and flagged.

## Temperature
- **Why not the Historical Forecast API:** it is stitched together from the first hours of each model run, so it is close to observed weather, not a D-1 forecast.
- **Source used:** Previous Runs API, GFS (`gfs_seamless`), `temperature_2m_previous_day2`. A run at least ~48 h old predates noon D-1 for every delivery hour of D. One model is pinned for the whole period so forecast quality does not change mid-sample. GFS history starts 2021-03-25; ICON and ECMWF only from early 2024.
- **Fallback:** observed temperature (Archive API) fills hours with no forecast: all of 2020, 2021-01-01 to 2021-03-24, and GFS gaps (0.5% of 2023, 5.3% of 2024). Flagged as `temp_is_actual`; 22.8% of 2021 is fallback.
- **Aggregation:** population-weighted mean of Berlin, Hamburg, Munich, Cologne, Frankfurt (heating demand follows people). An hour is missing if any city is missing.
- **Saved columns:** `temp_c`, `hdh` = max(0, 15.5 − `temp_c`), `temp_is_actual`. Observed temperature is not saved, so it cannot leak in as a feature.
- **Forecast vs observed (2021-03-25 to 2025, 41,315 h):** MAE 0.94 °C, bias +0.38 °C, P95 abs error 2.52 °C. Heating season (Oct–Apr): MAE 0.84–1.07 °C, bias +0.37 to +0.75 °C (forecast too warm, HDH too low); HDH MAE 0.81–1.03 per hour. So the fallback slightly flatters 2020 and early 2021.
- **Expected value:** temperature effects on demand are already reflected in the ENTSO-E load forecast; whether temperature adds anything beyond it will be tested in Week 6. Its larger effect may be indirect, via gas demand and gas prices.