import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sktime.forecasting.arima import AutoARIMA
from sktime.utils.plotting import plot_series
from sktime.split import temporal_train_test_split

st.set_page_config(layout="wide")

data = pd.read_csv("./garan_stock.csv", index_col="Date")

data.index = pd.PeriodIndex(data.index, freq="D")
data["YYYYMM"] =  pd.PeriodIndex(data.index, freq="M")

tabMonthlyReturn, tabYearlyReturn = st.tabs(["Monthly Returns", "Yearly Returns"])
with tabMonthlyReturn:
    monthly_open_close =  (data
    .groupby("YYYYMM")
    .agg({
        "Open" : [("Open", "first")],
        "Close" : [("Close", "last")]
    })
    .droplevel(0, axis = 1)
    )

    monthly_open_close["Month_Return"] = (monthly_open_close['Close'] - monthly_open_close["Open"]) / monthly_open_close['Open']
    monthly_open_close["Positive_Return"] = np.where(monthly_open_close.Month_Return > 0, "Positive", "Negative")
    monthly_open_close["YearMonth"] = monthly_open_close.index.astype("str")
    monthly_open_close["Month_Return_Perc"] = (np.round(monthly_open_close.Month_Return,3) * 100)
    
    figMonthlyReturns = px.bar(data_frame=monthly_open_close,
                               x='YearMonth', y="Month_Return_Perc", color="Positive_Return",
                               color_discrete_sequence=["green", "red"])
    figMonthlyReturns.layout.update(showlegend=False)
    st.plotly_chart(figMonthlyReturns)

with tabYearlyReturn:
    yearly_open_close = (data
    .groupby(data.index.year)
    .agg({
        "Open" : [("Open", "first")],
        "Close" : [("Close", "last")]
    })
    .droplevel(0, axis = 1)
    )

    yearly_open_close["Year_Return"] = (yearly_open_close['Close'] - yearly_open_close["Open"]) / yearly_open_close['Open']
    yearly_open_close["Positive_Return"] = np.where(yearly_open_close.Year_Return > 0, "Positive", "Negative")
    yearly_open_close["Year"] = yearly_open_close.index.astype("str")
    yearly_open_close["Year_Return_Perc"] = (np.round(yearly_open_close.Year_Return,3) * 100)

    figYearlyReturns = px.bar(data_frame=yearly_open_close,
                              x='Year', y="Year_Return_Perc", color="Positive_Return",
                              color_discrete_sequence=["green", "red"])
    figYearlyReturns.layout.update(showlegend=False)
    st.plotly_chart(figYearlyReturns)

MIN_DATE = pd.to_datetime(str(data.index.min())).to_pydatetime()
MAX_DATE = pd.to_datetime(str(data.index.max())).to_pydatetime()

with st.sidebar:
    data_col = st.selectbox(label="choose a col", options=data.columns,
                             index=int(np.where(data.columns == "Close")[0][0]))
    data_date = st.slider(label="hopefully date slider",
                           min_value=MIN_DATE, max_value=MAX_DATE,
                           value=(MIN_DATE, MAX_DATE))
    input_test_size = st.slider(label="Test Size Percentage", min_value=0.1, max_value=0.3, value=0.2, step=0.05)

full_data = data.loc[(data.index >= str(data_date[0])) & (data.index <= str(data_date[1])), data_col]

## FULL DATA MODEL
arima = AutoARIMA(maxiter=100)
arima.fit(full_data)
fh = np.arange(1, 20)
preds = arima.predict(fh)
intervals = arima.predict_interval(fh, coverage=[0.5, 0.9])

conf_50 = pd.concat([intervals['Close'][0.5].upper, intervals['Close'][0.5].lower[::-1]])
conf_90 = pd.concat([intervals['Close'][0.9].upper, intervals['Close'][0.9].lower[::-1]])

fig_all_data = go.Figure()
fig_all_data.add_trace(go.Scatter(
    x=full_data.index.to_timestamp(), y=full_data,
    line_color="steelblue", mode="lines+markers", name="Observations"
))
fig_all_data.add_trace(go.Scatter(
    x=conf_90.index.to_timestamp(), y=conf_90, line_color="yellow",
    fill="toself", mode="lines+markers", name="90% Confidence"
))
fig_all_data.add_trace(go.Scatter(
    x=conf_50.index.to_timestamp(), y=conf_50, line_color="lightgreen",
    fill="toself", mode="lines+markers", name="50% Confidence"
))
fig_all_data.add_trace(go.Scatter(
    x=preds.index.to_timestamp(), y=preds, line_color="orange", mode="lines+markers", name="Predictions"
))
st.plotly_chart(fig_all_data)

train_data, test_data = temporal_train_test_split(full_data, test_size=input_test_size)

arima = AutoARIMA(maxiter=100)
arima.fit(train_data)
fh = np.arange(1, test_data.shape[0]+1)
preds = arima.predict(fh)
intervals = arima.predict_interval(fh, coverage=[0.5, 0.9])
preds.index = test_data.index
intervals.index = test_data.index
figARIMATT, axARIMATT = plot_series(train_data, preds, test_data, pred_interval=intervals, labels=["train", "pred", "actual"])
st.pyplot(figARIMATT)