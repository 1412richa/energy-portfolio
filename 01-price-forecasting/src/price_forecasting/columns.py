# Column roles, shared by the validation, feature and model scripts
TARGET = "price_eur_mwh"

HOLDOUT_YEAR = 2025                             # test year
VAL_YEAR = HOLDOUT_YEAR - 1                     # validation year: all modelling choices

BASE_FEATURES = [
    "load_fc_mw", "solar_fc_mw", "wind_on_fc_mw", "wind_off_fc_mw",
    "residual_load_fc_mw",
    "ttf_eur_mwh", "eua_eur_t", "coal_eur_t",
    "gas_mc_eur_mwh", "coal_mc_eur_mwh", "gas_minus_coal_mc",
]

OPTIONAL_FEATURES = ["temp_c", "hdh"]          # tested separately in Week 6

FLAGS = ["fc_interpolated", "exclude_from_training", "temp_is_actual"]

DIAGNOSTICS = [                                 # never model inputs
    "load_actual_mw", "ttf_age_d", "eua_age_d", "coal_age_d",
]

LAG_FEATURES = [
    "price_lag_d1", "price_lag_d2", "price_lag_d3", "price_lag_d7",
    "price_d1_mean", "price_d1_min", "price_d1_max",
    "price_roll7d", "price_roll30d",
]
CALENDAR_FEATURES = [
    "hour", "dow", "month", "is_weekend",
    "is_holiday", "is_bridge_day", "is_christmas_period",
]
INTERACTION_FEATURES = ["low_resid_flag", "resid_x_gas_mc"]

ALL_FEATURES = BASE_FEATURES + LAG_FEATURES + CALENDAR_FEATURES + INTERACTION_FEATURES

LEAR_DROP = [
    "ttf_eur_mwh", "eua_eur_t", "coal_eur_t", "coal_mc_eur_mwh",   # EDA finding 6: collinear fuels
    "load_fc_mw",                                                  # residual load = load - renewables
]
