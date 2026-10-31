1. **Daily shape changes over time.** The midday price dip deepens each year
   (solar growth) while evening peaks rise.
   -> Rolling recalibration window; recent data weighted more.

2. **Negative prices have a clear residual load threshold.** Share of negative hours:
   67% at 0-5 GW, 35% at 5-10 GW, 9% at 10-15 GW, ~0% above 20 GW.
   Weekends: 9.3% vs 2.3% on weekdays.
   -> Flag for forecast residual load < 10 GW; evaluate negative hours separately.

3. **Two negative-price regimes:** spring/summer middays (solar) and
   December nights (wind plus low holiday demand).
   -> Use a residual-load-based flag, not an hour-of-day rule.

4. **Residual load is the strongest and most stable feature**
   (Spearman 0.57-0.90 every year); solar alone is weak because its
   output coincides with high midday demand.

5. **Fuel price effects vary by year** (gas cost: 0.76 in 2021, 0.34 in 2023);
   carbon moved independently of gas in 2022.
   -> Use combined gas plant cost, not raw prices; recalibrate regularly.

6. **Fuel price features are highly collinear** (ρ ≈ 0.9+).
   -> LEAR: gas_mc_eur_mwh + gas_minus_coal_mc only. LightGBM: all.

7. **Temperature's correlation changes sign between years**, reflecting
   seasonal and crisis effects rather than temperature itself.
   -> Keep temperature optional, as decided.

8. **Test split:** 2025 held out entirely; 2024 used for validation
   and all modelling choices.