# Evaluation protocol
- Split by German calendar year: train 2020-2023, validation 2024, test 2025.
- All modelling choices made on 2024. 2025 used once, at the end.
- Metrics: MAE (main), RMSE, bias, rMAE vs Lago naive benchmark.
- MAPE not used: undefined or misleading for prices near or below zero.
- Reported for all hours and for negative-price, low-residual-load, and evening-peak hours.
- Rows flagged exclude_from_training are not used for training; evaluation uses all hours with an observed price.