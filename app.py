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
import winsound

st.set_page_config(layout="wide")
st.title("🐉 Dragon Trading AI Dashboard")
st.markdown("""
Real-time candlestick chart, volume tracking, technical indicators, live prediction alerts, model switching, and backtesting.
""")

# Sidebar Configuration
st.sidebar.header("Configuration")
crypto = st.sidebar.selectbox("Select Cryptocurrency", ["BTC", "ETH", "SOL", "ADA"])
ticker = crypto + "USDT"
interval = "1m"
limit = 200
model_choice = st.sidebar.radio("Select Model", ["Random Forest", "XGBoost"])

# Sound Alert
ALERT_SOUND = 523  # C note frequency
DURATION = 600  # Milliseconds

def fetch_binance_data(symbol="BTCUSDT", interval="1m", limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=['time','open','high','low','close','volume','close_time','qav','num_trades','taker_base_vol','taker_quote_vol','ignore'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('time', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

def compute_indicators(df):
    df['return'] = df['close'].pct_change()
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
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
    features = ['return', 'sma_5', 'sma_10', 'rsi_14', 'macd', 'macd_signal', 'momentum']
    X = df[features]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    if model_type == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    else:
        model = RandomForestClassifier(n_estimators=100)

    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return model, acc, X_test, y_test

def predict_next(model, df):
    last = df.iloc[-1:]
    features = ['return', 'sma_5', 'sma_10', 'rsi_14', 'macd', 'macd_signal', 'momentum']
    pred = model.predict(last[features])[0]
    return pred

def alert(pred):
    if pred == 1:
        winsound.Beep(ALERT_SOUND, DURATION)
        st.success("🔊 Prediction: Price will go UP")
    else:
        st.error("🔊 Prediction: Price will go DOWN")

# Fetch and process
with st.spinner("Fetching data and making predictions..."):
    df = fetch_binance_data(ticker, interval, limit)
    df = compute_indicators(df)
    model, acc, X_test, y_test = train_model(df, model_choice)
    pred = predict_next(model, df)
    alert(pred)
    st.metric("Model Accuracy", f"{acc:.2%}")

# Real-time Chart
fig = go.Figure(data=[
    go.Candlestick(x=df.index,
                   open=df['open'], high=df['high'],
                   low=df['low'], close=df['close'], name='Candles'),
    go.Bar(x=df.index, y=df['volume'], name='Volume', yaxis='y2', marker_color='rgba(100,100,255,0.3)')
])
fig.update_layout(
    title=f"{ticker} Live Candlestick Chart with Volume",
    xaxis_rangeslider_visible=False,
    yaxis_title="Price (USD)",
    yaxis2=dict(overlaying='y', side='right', showgrid=False, title='Volume'),
    template="plotly_dark"
)
st.plotly_chart(fig, use_container_width=True)

# Backtesting visualization
st.subheader("📈 Backtesting Strategy vs. Market")
df['prediction'] = model.predict(df[['return', 'sma_5', 'sma_10', 'rsi_14', 'macd', 'macd_signal', 'momentum']])
df['strategy_returns'] = df['return'] * df['prediction'].shift(1)
df['market_returns'] = df['return']

cumulative_strategy = (df['strategy_returns'] + 1).cumprod()
cumulative_market = (df['market_returns'] + 1).cumprod()

bt_fig = go.Figure()
bt_fig.add_trace(go.Scatter(x=df.index, y=cumulative_strategy, name='Strategy'))
bt_fig.add_trace(go.Scatter(x=df.index, y=cumulative_market, name='Market'))
bt_fig.update_layout(title="Backtesting Cumulative Returns", template="plotly_dark")
st.plotly_chart(bt_fig, use_container_width=True)
