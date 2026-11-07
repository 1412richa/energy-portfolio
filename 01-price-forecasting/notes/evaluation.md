# Evaluation protocol
- Split by German calendar year: train 2020-2023, validation 2024, test 2025 (`VAL_YEAR` and `HOLDOUT_YEAR` in `columns.py`).
- All modelling choices made on 2024. 2025 used once, at the end.
- Rows flagged exclude_from_training are not used for training; evaluation uses all hours with an observed price.

## Retraining
- Rolling training windows, refitted before each forecast period, using only data from before it.
- LEAR: refitted every day (24 hourly models), with 365, 730 and 1095day windows; the chosen model averages all three.
- LightGBM: one model per week, 730 day window (weekly chosen over monthly on 2024).
- The test year uses the same scheme, so its training windows include 2024.

## Metrics
- MAE (main), RMSE, bias, rMAE vs Lago naive benchmark.
- Bias = forecast minus actual: positive means over-forecasting.
- MAPE not used: undefined or misleading for prices near or below zero.
- Negative-price sign skill: neg_recall (share of negative hours forecast as negative) and neg_precision (share of negative forecasts that were right). NaN when there is nothing to score.
- Reported for all hours and for negative-price, low-residual-load, and evening-peak hours.

## Model selection
Lowest overall MAE. If two configurations are within 2%, choose the one with better negative-price recall, then the simpler one. Experiments are declared before running; anything decided after seeing results is marked post-hoc.

## Caveats
- The negative-price subset is selected on the actual price, so errors there skew positive for almost any model (Lago naive: +27.9 bias vs -0.3 overall). Use it to describe errors, not to compare bias between models. The low-residual-load subset is known in advance and covers 82.7% of 2024's negative hours.
- Selecting among several configurations on 2024 makes the winner's 2024 score slightly optimistic; the 2025 test gives the unbiased estimate.
- From 1 October 2025 the day-ahead market trades 15-minute products. Forecasts are scored against their hourly average, and these months (25% of test hours) are reported separately.
- 5.3% of 2024 hours use observed rather than forecast temperature (`temp_is_actual`), which could flatter models that use temperature. Checked: the temperature gain is the same on fallback and forecast-temperature hours (-2.8% vs -2.7% MAE), and neither chosen model uses temperature.

## Test protocol (declared before running 2025)
Models reported on 2025: lgbm_weekly, lear_avg, and their simple average
(average added after 2024 error analysis; labelled as such).
Headline result: lgbm_weekly, chosen on 2024. The average is reported
alongside, whatever its result.
All subsets and sign skill reported for all three, plus naive baselines.
No changes to any configuration after viewing 2025 results.