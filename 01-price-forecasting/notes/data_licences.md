# Data licences: checks, changes and permission requests

Record of how the project's data use was checked against the sources' terms, what was
changed in the published material, and which permissions were requested.

## Terms checked (November 2026)

| Source | Document | Finding |
|---|---|---|
| ENTSO-E Transparency Platform | "List of Data Available for Free Re-use", last modified 18 October 2023 | Listed data is under CC BY 4.0 (free re-use with attribution and noting changes). Day-ahead total load forecast (6.1.b) is listed. Day-ahead prices (12.1.d) and wind/solar day-ahead forecasts (14.1.d) are not listed, so their re-use needs the data owners' agreement. Actual load (6.1.a) is not listed either. |
| EPEX SPOT | Market data pages; General Conditions of Data Use (1 November 2025) | Commercial use only with EPEX SPOT's approval; content for internal use. The General Conditions forbid using EPEX data to train or validate machine-learning models without written consent. Not yet confirmed whether these conditions bind users who obtained prices via ENTSO-E (definitions section to be checked). |
| Open Power System Data | Time series data package, version 2020-10-06 (used in the de-power-price-analysis submodule) | The package states no licence of its own; its German series are labelled as based on the ENTSO-E Transparency Platform, so ENTSO-E's terms apply. |
| Ember | European wholesale electricity price data | Site states CC BY 4.0, but hourly prices are sourced from ENTSO-E. Not used. For Germany 2025, identical to ENTSO-E prices Jan-Sep; differs Oct-Dec (15-minute market, unknown hourly conversion). |
| Yahoo Finance, EEX, Open-Meteo | Terms of use / CC BY 4.0 (Open-Meteo) | Used for modelling only; nothing redistributed. Open-Meteo credited. |

## Changes to published material

Principle: raw data is used for modelling but not redistributed. Published material contains
only own results (forecasts, error metrics, model diagnostics), aggregates, and the openly
licensed load forecast.

| Date | Repository | Commit | Change |
|---|---|---|---|
| 17.11.2026 | energy-portfolio | `4aed83e` | Dashboard data: removed actual prices and residual load; added the load forecast (CC BY 4.0) and daily MAE per model; source attribution in the app. |
| 17.11.2026 | energy-portfolio | `cafcd2e` | README: replaced the price figures (two-week forecast vs actual, residual load vs price) with test-year MAE by month and LightGBM feature importance; added "Data sources and licences". |
| 17.11.2026 | de-power-price-analysis | `77672e4`, `55c96e4` | Added "Data and licences"; cleared notebook outputs; STL figure reduced to trend and seasonal components; residual-load scatter replaced by mean price per 5 GW band. |

Earlier versions of both repositories (before 17.11.2026) contain the raw price series and the
earlier figures in their git history. Rewriting history is deferred until the replies below.

## Permission requests

| Date sent | To | About | Status |
|---|---|---|---|
| 19.11.2026 | EPEX SPOT market data team | DE-LU day-ahead prices: model training and evaluation; displaying actual prices next to forecasts | Awaiting reply |
| 19.11.2026 | ENTSO-E Transparency Platform | Wind/solar day-ahead forecasts (14.1.d) and day-ahead prices (12.1.d): model training and evaluation, publishing own forecasts and aggregate errors; whom to contact for the data owners | Replied: contact the member TSOs directly (see below) |
| 21.11.2026 | German TSOs (50Hertz, Amprion, TenneT, TransnetBW), via the netztransparenz.de contact form, recipient "an alle Übertragungsnetzbetreiber"; in German | Wind/solar day-ahead forecasts (14.1.d) as model inputs only, publishing own forecasts and aggregate errors; attribution | Awaiting reply |
| 21.11.2026 | Creos Luxembourg (Luxembourg TSO, part of the DE-LU zone); in English | Same request as to the German TSOs: wind/solar day-ahead forecasts (14.1.d) as model inputs only; attribution | Awaiting reply |

Follow up around 03.12-10.12.2026 if there is no reply. Record replies and resulting decisions
(restoring price charts, history rewrite, attribution wording) below.

## Replies and decisions

**ENTSO-E Transparency Platform (reply received by 19.11.2026):** ENTSO-E does not decide on
re-use of this data itself; it referred the request to the member TSOs, to be contacted directly
via its members page: https://www.entsoe.eu/about/inside-entsoe/members/

Next step: contact the TSOs responsible for the DE-LU wind/solar day-ahead forecasts (14.1.d).
Day-ahead prices remain with EPEX SPOT (request of 19.11.2026, awaiting reply).

**21.11.2026:** request sent to the four German TSOs jointly via netztransparenz.de, which covers
the German TSOs only. The same request was sent separately to Creos Luxembourg (also part of the
DE-LU zone, small share), so all TSOs of the bidding zone have now been contacted.
