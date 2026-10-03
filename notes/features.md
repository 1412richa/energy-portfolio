# Project 1 — Feature specification

Forecast origin: noon CET on D−1 (day-ahead gate closure).
Rule: every feature must be knowable at that moment.

| Feature | Mechanism | Known at noon D−1? | Source | Transformation | Status |
|---|---|---|---|---|---|
| Load forecast | Demand position on merit order | Yes (forecast) | | | ☐ |
| Wind on/offshore forecast | Supply curve shift | Yes (forecast) | | | ☐ |
| Solar forecast | Supply curve shift | Yes (forecast) | | | ☐ |
| Residual load forecast | Depth into thermal stack | Yes (derived) | | | ☐ |
| Temperature | Heating demand | Yes (forecast) | | | ☐ |
| TTF gas price | Thermal segment height | Yes (last settlement) | | | ☐ |
| EUA carbon price | Thermal segment height | Yes (last settlement) | | | ☐ |
| Calendar (hour, weekday, month) | Demand curve shape | Yes | | | ☐ |
| Public holidays / bridge days | Low-demand days | Yes | | | ☐ |
| Price lags D−1, D−2, D−3, D−7 | Persistence | Yes | | | ☐ |