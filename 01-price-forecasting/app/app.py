from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="DE day-ahead price forecasts", layout="wide")
DATA = Path(__file__).parent / "data"
MODELS = {"actual": "Actual", "lgbm": "LightGBM", "lear": "LEAR"}

@st.cache_data
def load():
    d = pd.read_csv(DATA / "test_predictions.csv", index_col=0)
    d.index = pd.to_datetime(d.index, utc=True).tz_convert("Europe/Berlin")
    r = pd.read_csv(DATA / "test_results.csv")
    return d, r

d, r = load()
year = d.index.year.min()

st.title("German day-ahead electricity price forecasts")
st.caption(f"Test year {year}, forecasts made with information available at the noon "
           "auction on the previous day. Hourly, Berlin time.")

# Headline metrics
allh = r[r["subset"] == "all"].set_index("model")
c1, c2, c3 = st.columns(3)
c1.metric("LightGBM MAE", f"{allh.loc['lgbm_weekly', 'MAE']:.1f} €/MWh")
c2.metric("Error vs naive benchmark",
          f"-{(1 - allh.loc['lgbm_weekly', 'rMAE']) * 100:.0f}%")
c3.metric("LEAR MAE", f"{allh.loc['lear_avg', 'MAE']:.1f} €/MWh")

# Day explorer
st.subheader("Explore a day")
days = sorted(set(d.index.date))
default = pd.Timestamp(f"{year}-06-08").date()
day = st.date_input("Delivery day", value=default if default in days else days[0],
                    min_value=days[0], max_value=days[-1])
sel = d[d.index.date == day]

fig = go.Figure()
for col, name in MODELS.items():
    fig.add_scatter(x=sel.index, y=sel[col], name=name, mode="lines+markers")
fig.add_hline(y=0, line_width=1, line_color="grey")
fig.update_layout(yaxis_title="€/MWh", xaxis_title="Berlin time",
                  height=420, margin=dict(t=20))
st.plotly_chart(fig, width="stretch")

m1, m2, m3 = st.columns(3)
m1.metric("LightGBM MAE (day)", f"{(sel['lgbm'] - sel['actual']).abs().mean():.1f} €/MWh")
m2.metric("LEAR MAE (day)", f"{(sel['lear'] - sel['actual']).abs().mean():.1f} €/MWh")
m3.metric("Min residual load forecast", f"{sel['resid_gw'].min():.1f} GW")

# Results table
st.subheader(f"MAE by subset ({year}, €/MWh)")
show = r[r["model"].isin(["naive_lago", "lear_avg", "lgbm_weekly", "average_lgbm_lear"])]
st.dataframe(show.pivot(index="model", columns="subset", values="MAE").round(1),
             width="stretch")

with st.expander("Method"):
    st.markdown(
        "- Inputs: ENTSO-E load, wind and solar forecasts; TTF gas, API2 coal and "
        "EU carbon prices lagged to the last settlement before the auction.\n"
        "- Models: LEAR (per-hour LASSO, daily recalibration) and LightGBM "
        "(weekly retraining). Chosen on the previous year, tested once on this one.\n"
        "- Full code and write-up: [GitHub](https://github.com/1412richa/energy-portfolio)"
    )

st.caption("Data: ENTSO-E Transparency Platform (prices, load, wind and solar forecasts).")
