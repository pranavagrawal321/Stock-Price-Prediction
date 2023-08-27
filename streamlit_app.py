import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Price Prediction")

stocks = ("AAPL", "GOOG", "MSFT", "GME")
selected_stocks = st.selectbox("Select Stock", stocks)

n_years = st.slider("Select years", 1, 4)
period = n_years * 365


@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data = load_data(selected_stocks)

st.subheader("Raw Data")
st.write(data.tail())


def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="Stock_Open"))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Stock_Close"))
    fig.update_layout(
        title_text="Stock Data",
        xaxis_rangeslider=dict(visible=True),
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )
    st.plotly_chart(fig)


plot_raw_data(data)

df_train = data[["Date", "Close"]]
df_train = df_train.rename(columns={
    "Date": "ds",
    "Close": "y"
})

m = Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Forecast Data")
st.write(forecast.tail())

f1 = plot_plotly(m, forecast)
st.plotly_chart(f1)

st.write("Forecast Components")
f2 = m.plot_components(forecast)
st.write(f2)