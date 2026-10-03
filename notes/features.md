# Project 1: Feature specification

Forecast origin: noon CET on D-1 (day-ahead gate closure).\
Rule: every feature must be knowable at that moment.

## Source and transformation
| Feature | Mechanism | Known at noon D-1? | Source | Transformation | Status |
|---|---|---|---|---|---|
| Load forecast | Demand position on merit order | Yes (forecast) | ENTSO-E | Convert to UTC; aggregate 15-min → hourly mean; handle DST | [ ] |
| Wind on/offshore forecast | Supply curve shift | Yes (forecast) | ENTSO-E | Convert to UTC; aggregate 15-min → hourly mean; handle DST | [ ] |
| Solar forecast | Supply curve shift | Yes (forecast) | ENTSO-E | Convert to UTC; aggregate 15-min → hourly mean; handle DST | [ ] |
| Residual load forecast | Depth into thermal stack | Yes (derived) | ENTSO-E | load_fc - wind_on_fc - wind_off_fc - solar_fc | [ ] |
| Temperature | Heating demand | Yes (forecast) | Open-Meteo Historical Forecast API | Average across cities → heating degree hours: max(0, 15-T) | [ ] |
| TTF gas price | Thermal segment height | Yes (last settlement) | yfinance Dutch TTF Futures Tickers (front-month futures used as proxies for the prices) | Daily settlement; forward-fill weekends and holidays; lag one trading day | [ ] |
| EUA carbon price | Thermal segment height | Yes (last settlement) | ENTSO-E | Daily settlement; forward-fill weekends and holidays; lag one trading day | [ ] |
| Calendar (hour, weekday, month) | Demand curve shape | Yes | python holidays package, country DE | One-hot hour and weekday; month as categorical | [ ] |
| Public holidays / bridge days | Low-demand days | Yes | python holidays package, country DE; national holidays plus a state-share weighting optional; bridge days are derived: weekday between holiday and weekend | Binary national-holiday flag; separate bridge-day flag | [ ] |
| Price lags D-1, D-2, D-3, D-7 | Persistence | Yes | ENTSO-E; derived from day-ahead prices | Same hour on D-1, D-2, D-3, D-7; plus D-1 daily mean, min, max | [ ] |

Hourly resolution for consistency across 2020–2025 data; 15-min extension is future work.

## Leakage checklist
- [ ] No actual (realised) load/wind/solar in model inputs - forecasts only
- [ ] Weather from Historical Forecast API, not reanalysis
- [ ] No intraday price lags (no t-1h) - full day lags only
- [ ] Fuel prices lagged to the last settlement before noon D-1
- [ ] Train/test split by time, never shuffled
