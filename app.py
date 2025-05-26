import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import time
import requests
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import threading
import winsound

st.set_page_config(layout="wide")
st.title("🐉 Dragon Trading AI Dashboard")
st.markdown("""
Real-time candlestick chart, volume tracking, technical indicators, live prediction alerts,
backtesting, portfolio simulation, and model switching.
""")

# Sidebar
st.sidebar.header("Configuration")
crypto = st.sidebar.selectbox("Select Cryptocurrency", ["BTC", "ETH", "SOL", "ADA"])
ticker = crypto + "USDT"
interval = "1m"
limit = 200
model_type = st.sidebar.selectbox("Select Model", ["Random Forest", "XGBoost"])

# Sound alert on prediction
ALERT_SOUND_UP = 880  # Frequency for UP
ALERT_SOUND_DOWN = 220  # Frequency for DOWN
DURATION = 400  # Milliseconds

portfolio = {"capital": 10000, "position": 0, "holding": False, "entry_price": 0.0}

@st.cache_data(ttl=60)
def fetch_binance_data(symbol="BTCUSDT", interval="1m", limit=200):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=['time','open','high','low','close','volume','close_time','qav','num_trades','taker_base_vol','taker_quote_vol','ignore'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('time', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

def compute_indicators(df):
    df['return'] = df['close'].pct_change()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['rsi_14'] = compute_rsi(df['close'])
    df['macd'], df['macd_signal'] = compute_macd(df['close'])
    df['momentum'] = df['close'] - df['close'].shift(10)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    return df.dropna()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series):
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def train_model(df, model_type="Random Forest"):
    features = ['return', 'sma_20', 'sma_50', 'rsi_14', 'macd', 'macd_signal', 'momentum']
    X = df[features]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100)
    else:
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return model, acc

def predict_next(model, df):
    last = df.iloc[-1:]
    features = ['return', 'sma_20', 'sma_50', 'rsi_14', 'macd', 'macd_signal', 'momentum']
    pred = model.predict(last[features])[0]
    return pred

def alert(pred):
    if pred == 1:
        winsound.Beep(ALERT_SOUND_UP, DURATION)
        st.success("🔥 Model predicts price will go UP!")
    else:
        winsound.Beep(ALERT_SOUND_DOWN, DURATION)
        st.error("📉 Model predicts price will go DOWN!")

def simulate_portfolio(df, model):
    capital = portfolio['capital']
    holding = False
    entry_price = 0.0
    for i in range(len(df)-1):
        row = df.iloc[i]
        features = ['return', 'sma_20', 'sma_50', 'rsi_14', 'macd', 'macd_signal', 'momentum']
        pred = model.predict([row[features]])[0]
        if pred == 1 and not holding:
            entry_price = row['close']
            holding = True
        elif pred == 0 and holding:
            capital *= row['close'] / entry_price
            holding = False
    return capital

# Real-time section
with st.spinner("Fetching live data and predicting..."):
    df = fetch_binance_data(ticker, interval, limit)
    df = compute_indicators(df)
    model, acc = train_model(df, model_type=model_type)
    pred = predict_next(model, df)
    alert(pred)
    st.metric("Model Accuracy", f"{acc:.2%}")
    st.metric("Prediction Time", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    simulated = simulate_portfolio(df, model)
    st.metric("Simulated Portfolio", f"${simulated:,.2f}")

# Chart
fig = go.Figure(data=[
    go.Candlestick(x=df.index,
                   open=df['open'],
                   high=df['high'],
                   low=df['low'],
                   close=df['close'],
                   name='Candles'),
    go.Bar(x=df.index, y=df['volume'], name='Volume', yaxis='y2', marker_color='rgba(100,100,255,0.3)'),
    go.Scatter(x=df.index, y=df['sma_20'], mode='lines', name='SMA 20', line=dict(color='orange')),
    go.Scatter(x=df.index, y=df['sma_50'], mode='lines', name='SMA 50', line=dict(color='cyan'))
])
fig.update_layout(
    xaxis_rangeslider_visible=False,
    title=f"{ticker} Live Candlestick Chart with Volume and Indicators",
    yaxis_title="Price (USD)",
    yaxis2=dict(overlaying='y', side='right', showgrid=False, title='Volume'),
    template="plotly_dark"
)
st.plotly_chart(fig, use_container_width=True)
