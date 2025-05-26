import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import requests
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from streamlit.components.v1 import html

# === CONFIGURATION ===
MODEL_PATH = Path("crypto_model.pkl")
START_DATE = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')  # 2 years data
COIN_IDS = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Cardano (ADA)": "ADA-USD"
}
FEATURES = ['rsi', 'macd', 'macd_signal', 'macd_hist', 'sma_20', 'sma_50', 'momentum']

# === DRAGON THEME SETUP ===
st.set_page_config(page_title="Dragon Trading AI", layout="wide", page_icon="🐉")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=MedievalSharp&display=swap');

:root {
    --dragon-red: #8B0000;
    --dragon-gold: #FFD700;
    --scrollbar: #4a0000;
}

.stApp {
    background: linear-gradient(45deg, #1a1a1a, #2d0d06);
    color: var(--dragon-gold);
}

h1, h2, h3 {
    font-family: 'MedievalSharp', cursive !important;
    color: var(--dragon-red) !important;
    text-shadow: 2px 2px 4px var(--dragon-gold);
}

.stButton>button {
    background: linear-gradient(45deg, var(--dragon-red), var(--dragon-gold)) !important;
    border: 1px solid var(--dragon-gold) !important;
    color: #000 !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px var(--dragon-red) !important;
}
</style>
""", unsafe_allow_html=True)

html("""
<div style="position: fixed; top: -100px; right: -200px; opacity: 0.3; z-index: -1; transform: scaleX(-1);">
  <img src="https://www.freeiconspng.com/uploads/dragon-png-5.png" width="600" height="600" style="animation: breathe 4s ease-in-out infinite;">
</div>
<style>
@keyframes breathe {
  0%, 100% { transform: scale(1) rotate(-10deg); }
  50% { transform: scale(1.05) rotate(-15deg); }
}
</style>
""", height=600)

# === FUNCTIONS ===

def fetch_data_yfinance(symbol):
    """Fetch historical OHLCV data from yfinance"""
    try:
        df = yf.download(symbol, start=START_DATE, progress=False)
        df.columns = df.columns.str.lower()
        df = df[['open','high','low','close','volume']].ffill().dropna()
        return df
    except Exception as e:
        st.warning(f"yFinance data fetch error: {e}")
        return pd.DataFrame()

def fetch_data_coingecko(symbol):
    """Fallback data fetch from CoinGecko API"""
    coin_id = symbol.lower().replace("-usd", "")
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {'vs_currency':'usd','days':'730','interval':'daily'}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        prices = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
        prices['date'] = pd.to_datetime(prices['timestamp'], unit='ms')
        prices.set_index('date', inplace=True)
        prices['open'] = prices['close'].shift(1).fillna(method='bfill')
        prices['high'] = prices[['open', 'close']].max(axis=1)
        prices['low'] = prices[['open', 'close']].min(axis=1)
        prices['volume'] = prices['close'].rolling(7).std().abs()*1000
        return prices[['open', 'high', 'low', 'close', 'volume']].dropna()
    except Exception as e:
        st.warning(f"CoinGecko fetch error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_data(symbol):
    df = fetch_data_yfinance(symbol)
    if df.empty or len(df) < 100:
        st.info("Falling back to CoinGecko data source...")
        df = fetch_data_coingecko(symbol)
    return df

def calculate_features(df):
    df = df.copy()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['momentum'] = df['close'] - df['close'].shift(4)
    df = df.dropna()
    return df

def load_model():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

def train_model(df, n_estimators=100, max_depth=5):
    df_feat = calculate_features(df)
    df_feat['target'] = (df_feat['close'].shift(-1) > df_feat['close']).astype(int)
    X = df_feat[FEATURES]
    y = df_feat['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    joblib.dump(model, MODEL_PATH)
    return model, accuracy, df_feat

def predict(model, df):
    df_feat = calculate_features(df)
    latest_features = df_feat[FEATURES].iloc[-1:].values
    pred = model.predict(latest_features)[0]
    proba = model.predict_proba(latest_features)[0]
    return pred, proba, df_feat

def backtest_strategy(df_feat, model):
    df_feat['predicted'] = model.predict(df_feat[FEATURES])
    df_feat['strategy_return'] = df_feat['predicted'] * df_feat['close'].pct_change().shift(-1)
    df_feat['buy_and_hold_return'] = df_feat['close'].pct_change().shift(-1)
    df_feat = df_feat.dropna()
    cum_strategy_return = (1 + df_feat['strategy_return']).cumprod() - 1
    cum_buy_hold = (1 + df_feat['buy_and_hold_return']).cumprod() - 1
    return cum_strategy_return, cum_buy_hold

# === MAIN APP ===

def main():
    st.title("🐉 Dragon Trading AI")
    st.markdown("#### Breathing Fire into Crypto Investments")

    coin_name = st.sidebar.selectbox("Select Your Treasure", list(COIN_IDS.keys()))
    symbol = COIN_IDS[coin_name]

    st.sidebar.markdown("---")
    st.sidebar.header("Dragon's Training Ground")
    n_estimators = st.sidebar.slider("Number of Dragon Eggs (Trees)", 50, 300, 100)
    max_depth = st.sidebar.slider("Dragon's Wisdom Depth (Max Tree Depth)", 3, 30, 7)

    df = load_data(symbol)

    if df.empty:
        st.error("Failed to load data. Please try again or select another crypto.")
        return

    tab1, tab2 = st.tabs(["🐉 Dragon's Prophecy", "⚔ Dragon Training"])

    with tab1:
        st.header(f"{coin_name} Price History")
        st.line_chart(df['close'])

        model = load_model()
        if model:
            pred, proba, df_feat = predict(model, df)
            col1, col2 = st.columns(2)
            col1.metric("🔥 Predicted Next Day Move", "Up" if pred == 1 else "Down")
            col2.metric("🔥 Confidence", f"{proba[pred]*100:.2f}%")

            st.subheader("Backtesting Your Dragon's Strategy")
            cum_strategy_return, cum_buy_hold = backtest_strategy(df_feat, model)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(cum_strategy_return.index, cum_strategy_return, label='Dragon Strategy')
            ax.plot(cum_buy_hold.index, cum_buy_hold, label='Buy & Hold')
            ax.set_ylabel("Cumulative Returns")
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("No trained Dragon found. Please train your dragon in the next tab!")

    with tab2:
        st.header("Train Your Dragon")
        if st.button("Summon Dragon and Train"):
            with st.spinner("Training your dragon on ancient crypto scrolls..."):
                model, accuracy, _ = train_model(df, n_estimators, max_depth)
                st.success(f"Dragon trained with an accuracy of {accuracy*100:.2f}%!")
        else:
            st.info("Adjust the dragon eggs and wisdom depth, then train!")

    st.markdown("---")
    st.caption("Powered by 🐉 Dragon AI Labs")

if __name__ == "__main__":
    main()
