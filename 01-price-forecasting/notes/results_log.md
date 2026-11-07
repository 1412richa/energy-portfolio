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

Sources: LightGBM in `results/thursday_lgbm_experiments.csv` (later runs write
`results/lgbm_experiments_val.csv`); LEAR in `results/lear_windows_val.csv`,
written by the window comparison in `LEAR.py`.

Findings
- Weekly retraining: largest single gain (-12% MAE vs monthly).
- Training window (LightGBM): 365 days +3.6% MAE, 1095 days -0.4%; 730 kept.
- EDA features: little effect on LightGBM MAE (+0.5% without them) despite high gain importance;
  trees rebuild the interaction. Kept for LEAR.
- Temperature: -2.7% MAE; checked against observed-fallback hours [result].
- asinh/L2 target: worse (+3.5% MAE). Tree size: no meaningful change
  (31 leaves +1.2%, 127 leaves -0.8%).
- Models trade off negative-price recall (LEAR) vs precision (LightGBM).

Post-hoc (declared after results): weekly + temperature - [result]