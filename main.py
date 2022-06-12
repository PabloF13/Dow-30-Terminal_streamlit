import streamlit as st
from datetime import datetime
import yahoo_fin.stock_info as si
from functions import news, predict, inter_can, financials, backtesting, it, rsi, vol, rent
import nltk

nltk.download('vader_lexicon')
st.set_option('deprecation.showPyplotGlobalUse', False)
today=datetime.today().strftime("%Y-%m-%d")
stocks=si.tickers_dow()
dates=("2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01", "2020-01-01",
       "2021-01-01", "2022-01-01","2022-05-01")

with st.sidebar.container():
    st.title("Dow Jones 30 Terminal")
    st.markdown("---")

with st.sidebar.container():
    menu=st.radio(label="Menu", options=['DJIA %', 'Stock News',"Financials", 'RSI & Volatility',
                                         'Technical Indicators','Backtesting', 'Forecasting'])
with st.sidebar.container():
    st.markdown("---")
    st.markdown(("https://pablof13.github.io/"))
    st.markdown("Pablo Faura Sanz")

if menu=="Stock News":
    header = st.container()
    st.header("Stock News")
    selected_stock = st.selectbox("Pick a stock", stocks)
    tablat = rent([selected_stock])
    st.metric(label="Day %", value=tablat.loc[selected_stock, "Close"], delta=tablat.loc[selected_stock, "% day"])
    news(selected_stock)

if menu == "Financials":
    selected_stock = st.selectbox("Pick a stock", stocks)
    financials(selected_stock)

if menu=="RSI & Volatility":

    header = st.container()
    selected_stock=st.selectbox("Pick a stock", stocks)
    selected_date=st.select_slider("Since", dates)
    st.header("Relative Strength Index (RSI)")
    st.text("Measures the magnitude of recent price changes to evaluate overbought or oversold\n"
            "conditions in the price of the stock.")
    rsi(selected_stock, selected_date)
    st.header("Volatility")
    st.text("The rate at which the price of the stock increases or decreases over a particular\n"
            "period.")
    vol(selected_stock, selected_date)

if menu == "DJIA %":

    header = st.container()
    tablat = rent(stocks)
    selected_stock=st.selectbox("Pick a stock", stocks)
    st.metric(label="Day %", value=tablat.loc[selected_stock, "Close"], delta=tablat.loc[selected_stock, "% day"])
    inter_can(selected_stock)
    st.table(tablat)

if menu == "Technical Indicators":

    header = st.container()
    with header:
        st.title("Technical Indicators")
        st.text("Sum of technical indicators of each value of index provided by Investing.com")
        st.text("Click on upper right for full screen")
    with st.spinner("Loading..."):
        it(stocks)

if menu == "Backtesting":

    header = st.container()
    with header:
        st.title("Backtesting")
        st.subheader("Test a trading strategy to historical data to determine its accuracy.")
    selectedma1, selectedma2 ,selectedma3 ,selectedma4 = 5,34,120,200
    selected_date=st.select_slider("Since", dates)
    st.caption("This trading strategy have a buy signal when the close price is above the selected\n"
               "moving averages and a sell signal when it cross down again. Starting with a 100k US$ in cash,\n"
               "every operation have a fee of 0.5 + 0.003 per stock.")
    with st.spinner("Loading..."):
        backtesting(stocks, selected_date, selectedma1,selectedma2,selectedma3,selectedma4)

if menu=="Forecasting":

    header = st.container()
    selected_stock=st.selectbox("Pick a stock", stocks)
    with header:
        st.title("Forecasting")
    with st.spinner("Loading..."):
        predict(selected_stock)
        st.caption("Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) used in many fields of machine and deep learning.\n"
                   " Stock Prediction is extremely challenging, the best stock predictors might get 75% accurate, rarely more.\n"
                   " But with LSTM model we can hit more than 90%. But this is for just one observation ahead. The model takes as input the \n"
                   "last 10 observations. For that reason, when we try to predict N observations ahead, the error multiply and the prediction\n"
                   "is totally incorrect.")





















