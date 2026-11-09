# Energy Market Analytics Portfolio

Pricing and forecasting models built on German power market data.

**Background**
Data scientist with an MSc in Economics (Universität Bonn, IIT Kanpur) and industry experience building ML pricing and forecasting systems in e-commerce. Academic foundation in econometrics, energy economics, and thermodynamics. This portfolio applies that combination to the German electricity market.

**Projects**
---
**01 - Day-Ahead Price Forecasting** (complete)
Forecasting hourly German day-ahead prices from information available at auction
gate closure. On the held-out 2025 test year, LightGBM reaches an MAE of 11.5 €/MWh,
58% below a naive benchmark and ahead of a LEAR statistical benchmark (12.3).
Features built from market fundamentals: residual load forecasts and gas, coal and
carbon costs.
**Dashboard:** [explore any day of the 2025 forecasts in a Streamlit app](https://energy-portfolio-richa.streamlit.app/)
(`streamlit run 01-price-forecasting/app/app.py` from the repository root).
Data: ENTSO-E Transparency Platform, EEX, Yahoo Finance, Open-Meteo.

---
**02 - Dynamic Tariff Design** (planned)
Customer segmentation from smart-meter data and time-of-use tariff design by segment.

---
**03 - Price Sensitivity & Churn** (planned)
Modelling customer switching behaviour under tariff changes; constraint-based tariff optimisation.

**Stack**: Python (pandas, scikit-learn, LightGBM); Streamlit planned.

---
**Setup**
```
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```
> Add ENTSOE_API_KEY=your_key to a .env file