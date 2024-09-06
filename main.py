import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sktime.forecasting.arima import AutoARIMA
from sktime.split import temporal_train_test_split
import yfinance as yf

st.set_page_config(layout="wide")

with st.sidebar:
    ticker_name = st.text_input(label="Yahoo Finance Symbol", value="Example Data", max_chars=20)

@st.cache_data
def pull_data(ticker_name, period="2y"):
    ticker = yf.ticker.Ticker(ticker=ticker_name)
    data = ticker.history(period)
    info_data = ticker.get_info()
    return data, info_data


if ticker_name == "Example Data":
    with open("./toy_info.json", "r") as toy:
        info_data = json.load(toy)
    data = pd.read_csv("./toy_data.csv", index_col="Date")
else:
    data, info_data = pull_data(ticker_name=ticker_name)
    
if "symbol" in info_data:
    pass
else:
    st.write("The given symbol could not been found. Please check and try again.")
    st.stop()

data.index = pd.PeriodIndex(data.index, freq="D")
data["YYYYMM"] =  pd.PeriodIndex(data.index, freq="M")

st.title(f"{info_data["longName"]} ({info_data["symbol"]}) | :gray[{info_data["city"]}, {info_data["country"]}]")

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
                               color_discrete_map={"Positive" : "green", "Negative" : "red"}, text_auto=True,
                               labels={
                                   "Month_Return_Perc" : "Monthly Return %",
                                   "YearMonth" : ""})
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
                              color_discrete_map={"Positive" : "green", "Negative" : "red"}, text_auto=True,
                              labels={
                                  "Year_Return_Perc" : "Yearly Return %",
                                  "Year" : ""
                              })
    figYearlyReturns.layout.update(showlegend=False)
    st.plotly_chart(figYearlyReturns)

MIN_DATE = pd.to_datetime(str(data.index.min())).to_pydatetime()
MAX_DATE = pd.to_datetime(str(data.index.max())).to_pydatetime()
data_col = "Close"

with st.sidebar:
    # data_col = st.selectbox(label="choose a col", options=data.columns,
    #                          index=int(np.where(data.columns == "Close")[0][0]))
    data_date = st.slider(label="Data Date Range",
                           min_value=MIN_DATE, max_value=MAX_DATE,
                           value=(MIN_DATE, MAX_DATE))
    input_test_size = st.slider(label="Test Size Percentage", min_value=0.1, max_value=0.3, value=0.2, step=0.05)

full_data = data.loc[(data.index >= str(data_date[0])) & (data.index <= str(data_date[1])), data_col]

## TRAIN TEST
st.header("Hypotetical Investment with ARIMA")
train_data, test_data = temporal_train_test_split(full_data, test_size=input_test_size)

arima_testtrain = AutoARIMA(maxiter=100, update_pdq=False)
arima_testtrain.fit(train_data)
fh = np.arange(1, test_data.shape[0]+1)
preds = arima_testtrain.predict(fh)
intervals = arima_testtrain.predict_interval(fh, coverage=[0.5, 0.9])
preds.index = test_data.index
intervals.index = test_data.index


last_observation = train_data.iloc[-1]
last_predictions =  preds.iloc[-1]
last_actual = test_data.iloc[-1]
expected_return = (last_predictions - last_observation) / last_observation
actual_return = (last_actual - last_observation) / last_observation
money_at_hand = 100
exp_money_return = (money_at_hand * expected_return)
actual_money_return = (money_at_hand * actual_return)
predicted_money = money_at_hand + exp_money_return
realized_money = money_at_hand + actual_money_return

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Investment", value=money_at_hand)
with col2:
    st.metric("Predicted Money", value=np.round(predicted_money, 2), delta=f"{np.round(expected_return, 2) * 100}%")
with col3:
    st.metric("Actual Money", value=np.round(realized_money, 2), delta=f"{np.round(actual_return, 2) * 100}%")
with col4:
    st.metric("Prediction Discrepancy", value=np.round(realized_money - predicted_money, 2),
            delta=f"{np.round((realized_money - predicted_money) / predicted_money, 2) * 100}%")

conf_50 = pd.concat([intervals[data_col][0.5].upper, intervals[data_col][0.5].lower[::-1]])
conf_90 = pd.concat([intervals[data_col][0.9].upper, intervals[data_col][0.9].lower[::-1]])

tabTrainTest_Graph, tabTrainTest_ModelInfo = st.tabs(["Model Plot", "Model Information"])

with tabTrainTest_Graph:
    fig_traintest = go.Figure()
    fig_traintest.add_trace(go.Scatter(
        x=train_data.index.to_timestamp(), y=train_data,
        line_color="steelblue", mode="lines+markers", name="Training", legendrank=1
    ))
    fig_traintest.add_trace(go.Scatter(
        x=conf_90.index.to_timestamp(), y=conf_90, line_color="yellow",
        fill="toself", mode="lines+markers", name="90% Confidence", marker={"size":1}, line={"width":1}, legendrank=4
    ))
    fig_traintest.add_trace(go.Scatter(
        x=conf_50.index.to_timestamp(), y=conf_50, line_color="lightgreen",
        fill="toself", mode="lines+markers", name="50% Confidence", marker={"size":1}, line={"width":1}, legendrank=3
    ))
    fig_traintest.add_trace(go.Scatter(
        x=preds.index.to_timestamp(), y=preds, line_color="orange", mode="lines+markers", name="Predictions", legendrank=2
    ))
    fig_traintest.add_trace(go.Scatter(
        x=test_data.index.to_timestamp(), y=test_data,
        line_color="darkgreen", mode="lines+markers", name="Actual", legendrank=5
    ))
    fig_traintest.layout.update(legend={
        "orientation":'h',"x":0.25
    })
    st.plotly_chart(fig_traintest)
with tabTrainTest_ModelInfo:
    st.text(arima_testtrain.summary().as_text())

## FULL DATA MODEL
st.header("Forecast")
tabFullModel_Graph, tabFullModel_ModelInfo, tabUpdatedModel, tabUpdatedModel_Info = st.tabs(
    ["Model Plot", "Model Information", "Updated Model", "Updated Model Information"]
)

with tabFullModel_Graph:
    arima = AutoARIMA(maxiter=100)
    arima.fit(full_data)
    fh = np.arange(1, 20)
    preds = arima.predict(fh)
    intervals = arima.predict_interval(fh, coverage=[0.5, 0.9])

    conf_50 = pd.concat([intervals[data_col][0.5].upper, intervals[data_col][0.5].lower[::-1]])
    conf_90 = pd.concat([intervals[data_col][0.9].upper, intervals[data_col][0.9].lower[::-1]])

    fig_all_data = go.Figure()
    fig_all_data.add_trace(go.Scatter(
        x=full_data.index.to_timestamp(), y=full_data,
        line_color="steelblue", mode="lines+markers", name="Observations", legendrank=1
    ))
    fig_all_data.add_trace(go.Scatter(
        x=conf_90.index.to_timestamp(), y=conf_90, line_color="yellow",
        fill="toself", mode="lines+markers", name="90% Confidence", marker={"size":1}, line={"width":1}, legendrank=4
    ))
    fig_all_data.add_trace(go.Scatter(
        x=conf_50.index.to_timestamp(), y=conf_50, line_color="lightgreen",
        fill="toself", mode="lines+markers", name="50% Confidence", marker={"size":1}, line={"width":1}, legendrank=3
    ))
    fig_all_data.add_trace(go.Scatter(
        x=preds.index.to_timestamp(), y=preds, line_color="orange", mode="lines+markers", name="Predictions", legendrank=2
    ))
    fig_all_data.layout.update(legend={
        "orientation":'h', "x":0.25
    })
    st.plotly_chart(fig_all_data)
with tabFullModel_ModelInfo:
    st.text(arima.summary().as_text())
with tabUpdatedModel:
    arima_testtrain.update(test_data)
    fh = np.arange(1, 20)
    preds = arima_testtrain.predict(fh)
    intervals = arima_testtrain.predict_interval(fh, coverage=[0.5, 0.9])

    conf_50 = pd.concat([intervals[data_col][0.5].upper, intervals[data_col][0.5].lower[::-1]])
    conf_90 = pd.concat([intervals[data_col][0.9].upper, intervals[data_col][0.9].lower[::-1]])

    fig_all_dataFitted = go.Figure()
    fig_all_dataFitted.add_trace(go.Scatter(
        x=full_data.index.to_timestamp(), y=full_data,
        line_color="steelblue", mode="lines+markers", name="Observations", legendrank=1
    ))
    fig_all_dataFitted.add_trace(go.Scatter(
        x=conf_90.index.to_timestamp(), y=conf_90, line_color="yellow",
        fill="toself", mode="lines+markers", name="90% Confidence", marker={"size":1}, line={"width":1}, legendrank=4
    ))
    fig_all_dataFitted.add_trace(go.Scatter(
        x=conf_50.index.to_timestamp(), y=conf_50, line_color="lightgreen",
        fill="toself", mode="lines+markers", name="50% Confidence", marker={"size":1}, line={"width":1}, legendrank=3
    ))
    fig_all_dataFitted.add_trace(go.Scatter(
        x=preds.index.to_timestamp(), y=preds, line_color="orange", mode="lines+markers", name="Predictions", legendrank=2
    ))
    fig_all_dataFitted.layout.update(legend={
        "orientation":'h', "x":0.25
    })
    st.plotly_chart(fig_all_dataFitted)
with tabUpdatedModel_Info:
    st.text(arima_testtrain.summary().as_text())