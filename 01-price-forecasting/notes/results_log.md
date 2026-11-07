LEAR (730 day window, daily recalibration): MAE 14.1 €/MWh, rMAE 0.52 on 2024. Negative-price hours: bias reversed from +28 (naive) to -18; LEAR over-predicts how far prices fall below zero, probably because 2024 negative prices cluster near zero.

## Thursday experiments (validation 2024), declared before running
Decision rule: lowest overall MAE. If two configurations are within 2%,
choose the one with better negative-price recall, then the simpler one.

LightGBM
1. Training window: 365 / 730 / 1095 days
2. Retraining: monthly vs weekly
3. Target: L1 on raw prices vs L2 on asinh-transformed prices
4. Features: without EDA features / with temperature
5. Tree size: num_leaves 31 / 63 / 127

LEAR
6. Training window: 365 / 730 / 1095 days, and the average of all three

## Thursday decisions (validation 2024)
LightGBM: lgbm_weekly -> MAE 11.69, rMAE 0.43, neg recall 0.57, precision 0.88
(10% ahead of the next configuration, so no tie-break needed)
LEAR: lear_avg (365/730/1095) -> MAE 12.75, rMAE 0.47, neg recall 0.71, precision 0.71
(lear_365d within 2% at MAE 12.94; tie-break on negative-price recall, 0.71 vs 0.64)

Sources: LightGBM in `results/lgbm_experiments_val.csv` (metrics) and
`results/lgbm_experiments_val_predictions.csv` (hourly predictions; the re-run
reproduced `thursday_lgbm_experiments.csv` exactly); LEAR in
`results/lear_windows_val.csv`, written by the window comparison in `LEAR.py`.

Findings
- Weekly retraining: largest single gain (-12% MAE vs monthly).
- Training window (LightGBM): 365 days +3.6% MAE, 1095 days -0.4%; 730 kept.
- EDA features: little effect on LightGBM MAE (+0.5% without them) despite high gain importance;
  trees rebuild the interaction. Kept for LEAR.
- Temperature: -2.7% MAE; checked against observed-fallback hours: the gain is
  the same on hours with forecast temperature (-2.7%) and on the 469 fallback
  hours using observed temperature (-2.8%), so it is not driven by the fallback.
- asinh/L2 target: worse (+3.5% MAE). Tree size: no meaningful change
  (31 leaves +1.2%, 127 leaves -0.8%).
- Models trade off negative-price recall (LEAR) vs precision (LightGBM).

Post-hoc (declared after results): weekly + temperature -> MAE 11.48, rMAE 0.42,
neg recall 0.61, precision 0.89; -1.8% MAE vs lgbm_weekly (same on forecast-temperature
and fallback hours). Within 2%, and the recall tie-break would favour it, but it was
chosen after seeing results. Decision on switching: close, but not adopted; temperature worth revisiting with better weather data.

## Friday error analysis (validation 2024)
lgbm_weekly vs lear_avg, from `analysis/error_analysis.py`.
1. Evening ramp under-forecast by both models (16-20h): LightGBM bias -3 to -7.6,
   LEAR -3 to -5. Likely partly from L1 loss targeting the median of right-skewed prices.
2. Error tracks volatility: Nov-Dec highest MAE but highest std; MAE/std in Dec (0.24)
   is close to the year's lowest (LightGBM: August 0.22; LEAR: September 0.22).
3. Scarcity hours (residual load > 60 GW): 62 hours, MAE ~150-160, bias ~−140 to -158;
   ~10% of total LightGBM error from 0.7% of hours.
4. Residual load < 0 GW: LEAR bias -37 vs LightGBM +10.
5. 64% of largest disagreements below 10 GW residual load; LightGBM closer in 66%.
6. Negative-price recall: LEAR higher Jan–Apr, Jul, Oct and Nov; LightGBM higher May, Aug, Sep;
   tied Jun and Dec. Largest gap in July (LEAR 0.78 vs LightGBM 0.38).
7. Average of both: MAE 11.10 (-5%). Post hoc; reported on 2025, not selected on 2024.
Post-hoc weekly + temperature: MAE 11.48 (-1.8%); would win under the decision rule,
but not adopted because it was chosen after seeing results.

Limitations
- Scarcity price spikes systematically under-forecast by both models.
- No actual wind/solar data, so forecast-input errors can't be separated from model errors.
- Hourly resolution; the 15-minute market since Oct 2025 is not modelled.