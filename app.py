import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import time
import requests
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import os
import platform
from io import BytesIO
import base64

# Cross-platform sound alerts
def play_sound(frequency, duration):
    try:
        if platform.system() == 'Windows':
            import winsound
            winsound.Beep(frequency, duration)
        else:
            os.system(f'play -nq -t alsa synth {duration/1000} sine {frequency}')
    except Exception as e:
        st.warning(f"Sound alert failed: {str(e)}")

st.set_page_config(layout="wide")
st.title("🐉 Dragon Trading AI Dashboard 2.0")
st.markdown("""
Enhanced with real-time updates, cross-platform alerts, advanced portfolio tracking, 
and additional technical indicators. Supports both Windows and Unix systems.
""")

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {
        "capital": 10000,
        "position": 0,
        "holding": False,
        "entry_price": 0.0,
        "history": []
    }

if 'last_processed' not in st.session_state:
    st.session_state.last_processed = None

# Sidebar configuration
st.sidebar.header("Configuration")
crypto = st.sidebar.selectbox("Cryptocurrency", ["BTC", "ETH", "SOL", "ADA", "BNB"])
ticker = crypto + "USDT"
interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h"])
model_type = st.sidebar.selectbox("AI Model", ["Random Forest", "XGBoost", "LSTM (Experimental)"])
enable_sound = st.sidebar.checkbox("Enable Sound Alerts", True)
enable_shorting = st.sidebar.checkbox("Enable Short Positions", False)

# Backtest date range
backtest_start = st.sidebar.date_input("Backtest Start Date", pd.to_datetime("2023-01-01"))
backtest_end = st.sidebar.date_input("Backtest End Date", pd.to_datetime("today"))

@st.cache_data(ttl=30)
def fetch_binance_data(symbol="BTCUSDT", interval="1m", limit=500):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=['time','open','high','low','close','volume','close_time',
                                    'qav','num_trades','taker_base_vol','taker_quote_vol','ignore'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    return df.set_index('time')[['open', 'high', 'low', 'close', 'volume']].astype(float)

def compute_indicators(df):
    # Price transformations
    df['returns'] = df['close'].pct_change()
    
    # Trend indicators
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    
    # Volatility indicators
    df['atr'] = df['high'].combine(df['low'], np.maximum) - df['high'].combine(df['low'], np.minimum)
    df['bb_upper'] = df['sma_20'] + 2*df['close'].rolling(20).std()
    df['bb_lower'] = df['sma_20'] - 2*df['close'].rolling(20).std()
    
    # Momentum indicators
    df['rsi_14'] = compute_rsi(df['close'])
    df['macd'], df['macd_signal'] = compute_macd(df['close'])
    df['stoch_k'], df['stoch_d'] = compute_stoch(df['high'], df['low'], df['close'])
    df['adx'] = compute_adx(df['high'], df['low'], df['close'])
    
    # Volume indicators
    df['obv'] = compute_obv(df['close'], df['volume'])
    
    # Target variable
    df['target'] = (df['close'].shift(-5) > df['close']).astype(int)
    return df.dropna()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal).mean()
    return macd, signal

def compute_stoch(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(d_period).mean()
    return k, d

def compute_adx(high, low, close, period=14):
    pass  # Implementation omitted for brevity

def compute_obv(close, volume):
    return np.sign(close.diff()) * volume

def train_model(df, model_type):
    features = ['returns', 'sma_20', 'sma_50', 'ema_12', 'ema_26', 
               'rsi_14', 'macd', 'macd_signal', 'stoch_k', 'stoch_d',
               'bb_upper', 'bb_lower', 'obv', 'atr']
    X = df[features]
    y = df['target']
    
    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, class_weight='balanced')
    elif model_type == "XGBoost":
        model = XGBClassifier(n_estimators=300, learning_rate=0.05, use_label_encoder=False)
    else:
        from tensorflow.keras.models import Sequential
        model = Sequential()  # Simplified LSTM implementation
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model.fit(X_train, y_train)
    return model, accuracy_score(y_test, model.predict(X_test))

def execute_trade(prediction, current_price, portfolio, fee=0.0002):
    if prediction == 1 and not portfolio['holding']:
        # Long entry
        portfolio['position'] = (portfolio['capital'] * (1 - fee)) / current_price
        portfolio['capital'] = 0
        portfolio['holding'] = 'long'
        portfolio['entry_price'] = current_price
        portfolio['history'].append(('long', current_price, datetime.now()))
        
    elif prediction == -1 and enable_shorting and not portfolio['holding']:
        # Short entry
        portfolio['position'] = (portfolio['capital'] * (1 - fee)) / current_price
        portfolio['capital'] = 0
        portfolio['holding'] = 'short'
        portfolio['entry_price'] = current_price
        portfolio['history'].append(('short', current_price, datetime.now()))
        
    elif prediction == 0 and portfolio['holding']:
        # Exit position
        multiplier = 1 if portfolio['holding'] == 'long' else -1
        exit_value = portfolio['position'] * current_price * (1 + multiplier * (current_price - portfolio['entry_price'])/portfolio['entry_price'])
        portfolio['capital'] = exit_value * (1 - fee)
        portfolio['position'] = 0
        portfolio['holding'] = False
        portfolio['history'][-1] = (*portfolio['history'][-1], current_price, datetime.now()))
        
    return portfolio

# Main dashboard
try:
    df = fetch_binance_data(ticker, interval)
    df = compute_indicators(df)
    current_price = df['close'].iloc[-1]
    
    model, accuracy = train_model(df, model_type)
    prediction = model.predict(df.iloc[-1:][features])[0]
    
    # Update portfolio
    if st.session_state.last_processed != df.index[-1]:
        st.session_state.portfolio = execute_trade(prediction, current_price, st.session_state.portfolio)
        st.session_state.last_processed = df.index[-1]
    
    # Alert system
    if enable_sound:
        if prediction == 1:
            play_sound(880, 500)
        elif prediction == -1:
            play_sound(440, 500)
        else:
            play_sound(660, 300)
    
    # Portfolio metrics
    portfolio_value = st.session_state.portfolio['capital'] + \
                    st.session_state.portfolio['position'] * current_price
    st.metric("Portfolio Value", f"${portfolio_value:,.2f}", 
             delta=f"{(portfolio_value - 10000)/100:.2f}%")
    
    # Main chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                                low=df['low'], close=df['close'], name="Price"))
    fig.add_trace(go.Scatter(x=df.index, y=df['sma_20'], line=dict(color='orange'), name="SMA 20"))
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_upper'], line=dict(color='purple', dash='dot'), name="Bollinger Upper"))
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_lower'], line=dict(color='purple', dash='dot'), name="Bollinger Lower"))
    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    
    # Indicator subplots
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name="Volume"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi_14'], line=dict(color='blue'), name="RSI"), row=3, col=1)
    fig.update_layout(grid={"rows": 3, "columns": 1, "pattern": "independent"})
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Trading history
    st.subheader("Trade History")
    history_df = pd.DataFrame(st.session_state.portfolio['history'], 
                             columns=['Type', 'Entry', 'Entry Time', 'Exit', 'Exit Time'])
    st.dataframe(history_df.style.format({
        'Entry': '${:.2f}', 'Exit': '${:.2f}',
        'Entry Time': lambda x: x.strftime('%Y-%m-%d %H:%M'),
        'Exit Time': lambda x: x.strftime('%Y-%m-%d %H:%M') if x else ''
    }))
    
except Exception as e:
    st.error(f"Error initializing dashboard: {str(e)}")

# Auto-refresh every 60 seconds
time.sleep(60)
st.experimental_rerun()
