# %%
# Setup
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

load_dotenv()
client = EntsoePandasClient(api_key=os.getenv("ENTSOE_API_KEY"))

def find_project_root(name: str = "01-price-forecasting") -> Path:
    """Locate the project folder whether you run from src/, the project, or the repo root."""
    here = Path.cwd().resolve()
    for folder in [here, *here.parents]:
        if folder.name == name:
            return folder
        if (folder / name).is_dir():
            return folder / name
    raise FileNotFoundError(f"Could not find '{name}' from {here}")

PROJECT = find_project_root()
RAW = PROJECT / "data" / "raw"
PROCESSED = PROJECT / "data" / "processed"
YEARS = range(2020, 2026)

# %%
# Test one month and inspect the columns
start = pd.Timestamp("2024-03-01", tz="Europe/Berlin")
end = pd.Timestamp("2024-03-03", tz="Europe/Berlin")

load_test = client.query_load_and_forecast("DE_LU", start=start, end=end)
ws_test = client.query_wind_and_solar_forecast("DE_LU", start=start, end=end)

print(load_test.columns.tolist(), load_test.index[:3], sep="\n")
print(ws_test.columns.tolist(), ws_test.index[:3], sep="\n")

# %%
# Check out 24 hour gap in Feb 2022
retry = client.query_load_forecast(
    "DE_LU",
    start=pd.Timestamp("2022-02-21", tz="Europe/Berlin"),
    end=pd.Timestamp("2022-02-24", tz="Europe/Berlin"),
)
if isinstance(retry, pd.DataFrame):
    retry = retry.iloc[:, 0]
retry_h = retry.tz_convert("UTC").resample("h").mean()
print(retry_h.isna().sum(), len(retry_h))
print(retry_h.head(30))

