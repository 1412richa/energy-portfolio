# src/columns.py
TARGET = "price_eur_mwh"

FEATURES = [
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