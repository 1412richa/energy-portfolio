## Import check
# %%
import importlib

pkgs = ["pandas", "numpy", "sklearn", "lightgbm", "statsmodels",
        "matplotlib", "plotly", "streamlit", "entsoe", "requests",
        "dotenv", "yfinance", "holidays", "ipykernel"]

for p in pkgs:
    try:
        m = importlib.import_module(p)
        print(f"{p:12s} {getattr(m, '__version__', 'ok')}")
    except Exception as e:
        print(f"{p:12s} FAILED: {e}")
# %%

## Check ENTSOE API
# %%
import os
import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient
import yfinance as yf

load_dotenv()
client = EntsoePandasClient(api_key=os.getenv("ENTSOE_API_KEY"))

start = pd.Timestamp("2024-06-01", tz="Europe/Berlin")
end = pd.Timestamp("2024-06-02", tz="Europe/Berlin")
print(client.query_day_ahead_prices("DE_LU", start=start, end=end).head())

print(yf.download("TTF=F", start="2024-06-01", end="2024-06-10").head())
