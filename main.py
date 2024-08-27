import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sktime.forecasting.arima import AutoARIMA
from sktime.utils.plotting import plot_series
from sktime.split import temporal_train_test_split

st.set_page_config(layout="wide")

data = pd.read_csv("./garan_stock.csv", index_col="Date")

data.index = pd.PeriodIndex(data.index, freq="D")
data["YYYYMM"] =  pd.PeriodIndex(data.index, freq="M")

monthly_open_close =  (data
 .groupby("YYYYMM")
 .agg({
     "Open" : [("Open", "first")],
     "Close" : [("Close", "last")]
 })
 .droplevel(0, axis = 1)
)

monthly_open_close["Month_Return"] = (monthly_open_close['Close'] - monthly_open_close["Open"]) / monthly_open_close['Open']
monthly_open_close["Positive_Return"] = np.where(monthly_open_close.Month_Return > 0, 1, 0)

yearly_open_close = (data
 .groupby(data.index.year)
 .agg({
     "Open" : [("Open", "first")],
     "Close" : [("Close", "last")]
 })
 .droplevel(0, axis = 1)
)

yearly_open_close["Year_Return"] = (yearly_open_close['Close'] - yearly_open_close["Open"]) / yearly_open_close['Open']
yearly_open_close["Positive_Return"] = np.where(yearly_open_close.Year_Return > 0, 1, 0)
yearly_open_close.index.name = "YYYY"


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

# fig, ax = plot_series(training_data)
# st.pyplot(fig)

arima = AutoARIMA(maxiter=100)
arima.fit(full_data)
fh = np.arange(1, 20)
preds = arima.predict(fh)
intervals = arima.predict_interval(fh, coverage=[0.5, 0.9])
figARIMA, axARIMA = plot_series(full_data, preds, pred_interval=intervals)
st.pyplot(figARIMA)

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

figMonthlyReturn, axMonthlyReturn = plt.subplots()
sns.barplot(data=monthly_open_close,
            x="YYYYMM", y="Month_Return",hue="Positive_Return", hue_order=[1,0],
             ax=axMonthlyReturn)

figYearlyReturn, axYearlyReturn = plt.subplots()
sns.barplot(data=yearly_open_close,
            x="YYYY", y="Year_Return",hue="Positive_Return", hue_order=[1,0],
             ax=axYearlyReturn)

col1, col2 =  st.columns([3,2])
with col1:
    st.pyplot(figMonthlyReturn)
with col2:
    st.pyplot(figYearlyReturn)
