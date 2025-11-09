Python 3.10.1 (tags/v3.10.1:2cd268a, Dec  6 2021, 19:10:37) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# =======================
# Load Datasets
# =======================
@st.cache_data
def load_data():
    btc = pd.read_csv("cleaned_btc_data.csv")
    eth = pd.read_csv("cleaned_eth_data.csv")
    doge = pd.read_csv("cleaned_doge_data.csv")

    btc_arima = pd.read_csv("btc_arima_forecast.csv")
    btc_sarima = pd.read_csv("btc_sarima_future_30days.csv")
    btc_lstm = pd.read_csv("btc_lstm_forecast.csv")
    btc_prophet = pd.read_csv("btc_prophet_forecast_30days.csv")

    eth_arima = pd.read_csv("eth_arima_forecast.csv")
    eth_sarima = pd.read_csv("eth_sarima_forecast.csv")
    eth_lstm = pd.read_csv("eth_lstm_forecast.csv")
    eth_prophet = pd.read_csv("eth_prophet_forecast.csv")

    doge_arima = pd.read_csv("doge_arima_forecast.csv")
    doge_sarima = pd.read_csv("doge_sarima_forecast.csv")
    doge_lstm = pd.read_csv("doge_lstm_forecast.csv")
    doge_prophet = pd.read_csv("doge_prophet_forecast.csv")

    model_eval = pd.read_csv("model_evaluation_summary.csv")
    return btc, eth, doge, btc_arima, btc_sarima, btc_lstm, btc_prophet, \
           eth_arima, eth_sarima, eth_lstm, eth_prophet, \
           doge_arima, doge_sarima, doge_lstm, doge_prophet, model_eval

btc, eth, doge, btc_arima, btc_sarima, btc_lstm, btc_prophet, \
eth_arima, eth_sarima, eth_lstm, eth_prophet, \
doge_arima, doge_sarima, doge_lstm, doge_prophet, model_eval = load_data()

# =======================
# Sidebar
# =======================
st.sidebar.title("📊 Crypto Dashboard")
menu = st.sidebar.radio("Navigate:", ["Overview", "Data View", "EDA", "Forecasts", "Model Evaluation"])

# =======================
# Overview
# =======================
if menu == "Overview":
    st.title("📈 Cryptocurrency Analysis Dashboard")
    st.markdown("""
    **Project:** Time Series Analysis of BTC, ETH, DOGE

    **Key Features:**
    - Data cleaning and preparation
    - Exploratory Data Analysis (EDA)
    - Time Series Forecasting: ARIMA, SARIMA, LSTM, Prophet
    - Model evaluation (RMSE, MAE, MAPE)
    """)
    st.image("https://miro.medium.com/v2/resize:fit:1200/format:webp/1*oH-8EsNZhhrkTxhnLJ1uRA.png", use_column_width=True)

# =======================
# Data View
# =======================
elif menu == "Data View":
    st.title("💾 Data View")
    coin = st.selectbox("Select Cryptocurrency:", ["BTC", "ETH", "DOGE"])
    if coin == "BTC":
        st.dataframe(btc.head(50))
    elif coin == "ETH":
        st.dataframe(eth.head(50))
    else:
        st.dataframe(doge.head(50))

# =======================
# EDA
# =======================
elif menu == "EDA":
    st.title("🔍 Exploratory Data Analysis")
    coin = st.selectbox("Select Cryptocurrency for EDA:", ["BTC", "ETH", "DOGE"])

    if coin == "BTC":
        df = btc
    elif coin == "ETH":
        df = eth
    else:
        df = doge

    st.subheader("Closing Price Trend")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df['Date'], df['Close'], color='blue')
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Price Distribution")
    fig2 = px.histogram(df, x='Close', nbins=50, color_discrete_sequence=['orange'])
    st.plotly_chart(fig2)

    st.subheader("Correlation Heatmap")
    corr = df.select_dtypes(include=np.number).corr()
    fig3, ax3 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)

    if 'Volume' in df.columns:
        st.subheader("Volume vs Price Scatter")
        fig4 = px.scatter(df, x='Volume', y='Close', color='Close', size='Volume')
        st.plotly_chart(fig4)

# =======================
# Forecasts
# =======================
elif menu == "Forecasts":
    st.title("📊 Forecast Models")
    coin = st.selectbox("Select Cryptocurrency:", ["BTC", "ETH", "DOGE"])
    model = st.selectbox("Select Model:", ["ARIMA", "SARIMA", "LSTM", "PROPHET"])

    # Mapping
    forecast_map = {
        "BTC": {"ARIMA": btc_arima, "SARIMA": btc_sarima, "LSTM": btc_lstm, "PROPHET": btc_prophet},
        "ETH": {"ARIMA": eth_arima, "SARIMA": eth_sarima, "LSTM": eth_lstm, "PROPHET": eth_prophet},
        "DOGE": {"ARIMA": doge_arima, "SARIMA": doge_sarima, "LSTM": doge_lstm, "PROPHET": doge_prophet}
    }

    df_forecast = forecast_map[coin][model]
    st.dataframe(df_forecast.head(50))

    if 'Date' in df_forecast.columns and 'Predicted_Close' in df_forecast.columns:
        st.line_chart(df_forecast.set_index('Date')[['Predicted_Close']])
    elif 'Forecast' in df_forecast.columns:
        st.line_chart(df_forecast.set_index('Date')[['Forecast']])

# =======================
# Model Evaluation
# =======================
elif menu == "Model Evaluation":
    st.title("🏆 Model Evaluation Summary")
    st.dataframe(model_eval)
