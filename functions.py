import pandas_datareader.data as wb
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import datetime, timedelta
import yfinance as yf
import investpy
import time
import numpy as np
import seaborn as sns
import feedparser
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.graph_objects as go

def news(ticker):
    #get news and analyze
    vader = SentimentIntensityAnalyzer()
    titles_list=[]
    compound=[]
    sentiment_list=[]
    url = "https://seekingalpha.com/api/sa/combined/" + ticker + ".xml"
    f = feedparser.parse(url)
    for entry in f.entries:
        titles_list.append(str(entry.title))
    for title in titles_list:
        compound.append(vader.polarity_scores(title))
    for i in compound:
        sentiment_list.append(i['compound'])
    sentiment_list = [i for i in sentiment_list if i != 0]
    s = sum(sentiment_list) / len(sentiment_list)

    #plot
    fig=plt.figure(figsize=(4, 0.4))
    plt.imshow(np.linspace(0, 1, 256).reshape(1, -1), extent=[-1, 1, -1, 1], aspect='auto', cmap='RdYlGn')
    plt.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color='black')  # surrounding rectangle
    plt.axvline(x=s, color='black', ls=':', linewidth=1)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.8, 1.8)
    plt.yticks([0], ['\"Bullish\"'])
    plt.tick_params(axis='y', colors='white')
    plt.tight_layout()
    plt.xticks([])
    fig.set_facecolor("#0E1117")
    st.pyplot(fig)
    st.caption("A sentiment analysis based on the headlines by Vader SentimentIntensityAnalyzer")

    #show news
    st.subheader("News Feed")
    for entry in f.entries:
        st.write("Title:", entry.title)
        st.write("Link:", entry.link)

def predict(ticker):
    stock_df = pd.DataFrame()
    stock = []
    stock = yf.download(ticker, start="2015-01-01", progress=False)
    stock_df = stock_df.append(stock, sort=False)
    data = stock_df.filter(["Close"])
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * .95))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    train_data = scaled_data[0:int(training_data_len), :]
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])
        if i <= 61:
            print(x_train)
            print(y_train)
            print()

    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    test_data = scaled_data[training_data_len - 60:, :]

    x_test = []
    y_test = dataset[training_data_len:, :]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid["Predictions"] = predictions

    fig=plt.figure(figsize=(13, 5))
    plt.title("Stock Price Forecast LSTM Model", color="white")
    plt.xlabel("Date", fontsize=13, color="white")
    plt.ylabel("Close Price", fontsize=13, color="white")
    plt.plot(train["Close"].tail(180))
    plt.plot(valid[["Close", "Predictions"]])
    plt.legend(["Train", " Val", "Predictions"], loc="upper left", facecolor='k', labelcolor='w')
    plt.tick_params(axis='both', colors='white')
    fig.set_facecolor("#262730")
    plt.grid(False)


    st.pyplot(fig)

def inter_can(ticker):
    stock_df = pd.DataFrame()
    stock = []
    stock = yf.download(ticker, start="2022-01-01", progress=False)
    stock_df = stock_df.append(stock, sort=False)
    fig=go.Figure(data=[go.Candlestick(x=stock_df.index,low=stock_df["Low"],high=stock_df["High"],
                                          close=stock_df["Close"],open=stock_df["Open"],
                                          increasing_line_color="green",decreasing_line_color="red")],
                  layout=go.Layout(height=600, width=700,title=str(ticker),
                                   yaxis_title="Price (USD $)", xaxis_title="Date"))
    plt.rcParams.update({'axes.facecolor':'black'})
    plt.rcParams["figure.facecolor"] = "black"
    st.plotly_chart(fig)

def financials(stock):

    ticker = yf.Ticker(stock)
    st.subheader("Balance Sheet (Billions)")
    bs=ticker.balance_sheet/1000000
    bs=bs.astype(int)
    bs.columns=[2021,2020, 2019, 2018]
    st.table(bs)
    st.subheader("Earnings (Billions)")
    ea=ticker.earnings/1000000
    ea=ea.astype(int)
    st.table(ea)
    st.subheader("Cash Flow (Billions)")
    cf=ticker.cashflow/1000000
    cf=cf.astype(int)
    cf.columns=[2021,2020, 2019, 2018]
    st.table(cf)

def backtesting(acciones, fecha, mean1, mean2, mean3, mean4):


    resumen = pd.DataFrame(columns=["Ticker", "ReturnMA"+str(mean1), "ReturnMA"+str(mean1)+"(%)", "ReturnMA"+str(mean2),
                                    "ReturnMA"+str(mean2)+"(%)", "ReturnMA"+str(mean3), "ReturnMA"+str(mean3)+"(%)",
                                    "ReturnMA"+str(mean4), "ReturnMA"+str(mean4)+"(%)"])
    capital=float(100000)
    cf=0.5
    cv=0.0003
    ct=float()
    tit=int()
    resultado=float()

    for i in range(len(acciones)):
        df = yf.download(acciones[i], fecha, progress=False)["Close"]
        resumen.loc[i, "Ticker"] = acciones[i]
        mean_1 = df.rolling(mean1).mean()
        mean_2 = df.rolling(mean2).mean()
        mean_3 = df.rolling(mean3).mean()
        mean_4 = df.rolling(mean4).mean()

        p = 0
        x = 1
        while p < 4:
            if p == 0:
                data = pd.DataFrame(np.where((mean_1 < 0.98 * df), 1, 0), index=df.index)
            if p == 1:
                data = pd.DataFrame(np.where((mean_2 < 0.97 * df), 1, 0), index=df.index)
            if p == 2:
                data = pd.DataFrame(np.where((mean_3 < 0.96 * df), 1, 0), index=df.index)
            else:
                data = pd.DataFrame(np.where((mean_4 < 0.95 * df), 1, 0), index=df.index)

            signal = data.diff()
            signal["Títulos"] = 0
            signal["Coste"] = 0
            signal["Venta"] = 0
            signal["Comisión"] = 0

            cash = capital
            for z in range(len(df)):
                if signal.iloc[z, 0] > 0:
                    signal.iloc[z, 1] = cash // df[z]
                    tit += signal.iloc[z, 1]
                    signal.iloc[z, 2] = round(signal.iloc[z, 1] * df[z], 2)
                    signal.iloc[z, 4] = round(cf + signal.iloc[z, 2] * cv, 2)
                    ct = ct + cf + signal.iloc[z, 2] * cv
                    cash += -signal.iloc[z, 2] - signal.iloc[z, 4]

                if signal.iloc[z, 0] < 0:
                    signal.iloc[z, 3] = round(tit * df[z], 2)
                    signal.iloc[z, 4] = round(cf + signal.iloc[z, 3] * cv, 2)
                    ct = ct + cf + signal.iloc[z, 2] * cv
                    signal.iloc[z, 1] = -tit
                    tit = 0
                    cash += signal.iloc[z, 3] - signal.iloc[z, 4]

            signal = signal.fillna(0)
            signal = signal.loc[(signal != 0).any(axis=1)]


            ct = 0
            resultado = round(float(tit * df[-1:] + cash), 2)
            porcentaje = round((resultado / capital - 1) * 100, 2)


            resumen.iloc[i, (x)] = resultado
            resumen.iloc[i, (x + 1)] = porcentaje
            tit = 0
            x += 2
            p += 1

    st.table(resumen)

def it(tickers):

    df1 = pd.DataFrame()
    investpy.search_quotes(text=str(tickers[0]), products=["stocks"], n_results=1)

    for i in range(len(tickers)):
        time.sleep(2)
        df = investpy.search_quotes(text=str(tickers[i]), products=["stocks"], n_results=1)
        df = df.retrieve_technical_indicators(interval="weekly")
        df.index = df["indicator"]
        df = df.drop(["indicator"], axis=1)
        df["ticker"] = str(tickers[i])
        df1 = pd.concat([df, df1])

    buy = pd.DataFrame(np.where(df1["signal"] == "buy", 1, 0))
    sell = pd.DataFrame(np.where(df1["signal"] == "sell", 1, 0))
    neutral = pd.DataFrame(np.where(df1["signal"] == "neutral", 1, 0))
    overbought = pd.DataFrame(np.where(df1["signal"] == "overbought", 1, 0))
    oversold = pd.DataFrame(np.where(df1["signal"] == "oversold", 1, 0))
    less_volatility = pd.DataFrame(np.where(df1["signal"] == "less_volatility", 1, 0))
    high_volatility = pd.DataFrame(np.where(df1["signal"] == "high_volatility", 1, 0))

    sentiment = pd.concat([buy, sell, neutral, overbought, oversold, less_volatility, high_volatility], axis=1)
    sentiment.columns = ["buy", "sell", "neutral", "overbought", "oversold", "less_volatility", "high_volatility"]
    sentiment["indicator"] = df1.index
    sentiment.sum().drop("indicator")

    indicators = pd.DataFrame()
    l = list(set(sentiment["indicator"]))

    for i in range(len(l)):
        indicators[i] = sentiment[sentiment["indicator"] == str(l[i])].sum(axis=0).drop(["indicator"])
    indicators.columns = l
    indicators["total"] = indicators.sum(axis=1)

    st.table(indicators)

def rsi(acciones, fecha):

    data = wb.DataReader(acciones, "yahoo", fecha)["Adj Close"]
    returns = data.pct_change()
    up = returns.clip(lower=0)
    down = -1 * returns.clip(upper=0)
    ema_up = up.ewm(com=14, adjust=False).mean()
    ema_down = down.ewm(com=14, adjust=False).mean()
    rs = ema_up / ema_down

    rsi = 100 - (100 / (1 + rs))

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.get_xaxis().set_visible(False)
    fig.suptitle(str(acciones), color="white")
    fig.set_size_inches(18.5, 10.5)

    data.plot(ax=ax1)
    ax1.set_ylabel(str(acciones) + " Price", color="white")
    ax1.spines['top'].set_color('white')

    rsi.plot(ax=ax2)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI", color="white")
    ax2.set_xlabel("Date", color="white")
    ax2.axhline(70, color="r", linestyle="--")
    ax2.axhline(30, color="r", linestyle="--")
    ax2.tick_params(axis='both', colors='white')
    ax1.tick_params(axis='both', colors='white')
    ax1.grid(False)
    ax2.grid(False)
    fig.set_facecolor("#262730")

    st.pyplot(fig)

def vol(acciones, fecha):

    data = wb.DataReader(acciones, "yahoo", fecha)["Adj Close"]
    returns = data.pct_change()[1:]
    period = 100
    vol = returns.rolling(period).std() * np.sqrt(period)
    vol = vol.iloc[period:]

    fig, ax = plt.subplots()
    ax = vol.plot(figsize=(8, 6))
    plt.title("Volatility", color="white")
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='both', colors='white')

    fig2=sns.displot(returns.dropna(), bins=100, color="blue", height=3, aspect=18/6)
    sns.set(rc={'axes.facecolor': '#0E1117', 'figure.facecolor': '#0E1117'})
    plt.title("Daily Returns", color="white")
    ax.grid(False)
    plt.grid(False)
    plt.tick_params(axis='both', colors='white')
    plt.xlabel("Adj Close", color="white")
    plt.ylabel("Count", color="white")
    fig.set_facecolor("#262730")

    st.pyplot(fig)
    st.pyplot(fig2)

def rent(acciones):
    to = datetime.today()
    we = timedelta(days=7)
    ye = timedelta(days=365)

    lw = to - we  # Hoy
    ly1 = to - we - ye  # Año y una semana atrás
    ly2 = ly1 + we  # Año atrás

    lw = lw.strftime("%Y-%m-%d")
    ly1 = ly1.strftime("%Y-%m-%d")
    ly2 = ly2.strftime("%Y-%m-%d")

    tablat = pd.DataFrame()

    for i in range(len(acciones)):
        tabla = yf.download(acciones[i], lw, to, progress=False)[["Open", "Close", "Volume"]]
        for w in range(len(tabla) - 2):
            tabla = tabla.drop(tabla.index[0])

        ap = yf.download(acciones[i], ly1, ly2, progress=False)[["Open", "Close", "Volume"]]
        for w in range(len(ap) - 2):
            ap = ap.drop(ap.index[0])

        tabla["% day"] = 0
        tabla["% Last Close."] = 0
        tabla["% 52 week"] = 0
        tabla["Vol (m)"] = 0
        tabla["Ticker"] = acciones[i]

        tabla.iloc[1, 0] = round(tabla.iloc[1, 0], 2)
        tabla.iloc[1, 1] = round(tabla.iloc[1, 1], 2)
        tabla.iloc[1, 3] = round((((tabla.iloc[1, 1] / tabla.iloc[1, 0]) - 1) * 100), 2)
        tabla.iloc[1, 4] = round((((tabla.iloc[1, 1] / tabla.iloc[0, 1]) - 1) * 100), 2)
        tabla.iloc[1, 5] = round((((tabla.iloc[1, 1] / ap.iloc[1, 1]) - 1) * 100), 2)
        tabla.iloc[1, 6] = round((tabla.iloc[1, 0] + tabla.iloc[1, 1]) * tabla.iloc[1, 2] * 0.0005, 2)
        tabla = tabla.drop(["Volume"], axis=1)
        tabla = tabla.drop(tabla.index[0])
        tablat = pd.concat([tablat, tabla], axis=0)

    tablat = tablat.set_index("Ticker")
    tablat = tablat.sort_values(by="% day", ascending=False)
    return tablat

def graf(accion, fecha):

    df = wb.DataReader(accion, "yahoo", fecha)["Close"]

    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(111)
    ax1.set_title("Evolución precio " + str(accion))
    ax1.set_ylabel("Precio")
    ax1.set_xlabel("Fecha")
    ax1.plot(df, color="b", lw=1)

    st.pyplot(fig)






