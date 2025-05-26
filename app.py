import streamlit as st
import yfinance as yf
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import requests
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from streamlit.components.v1 import html

# ====================== DRAGON THEME SETUP ======================
st.set_page_config(page_title="Dragon Trading AI", layout="wide", page_icon="🐉")

# Custom CSS for dragon theme
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

h1 {
    font-family: 'MedievalSharp', cursive !important;
    color: var(--dragon-red) !important;
    text-shadow: 2px 2px 4px var(--dragon-gold);
    font-size: 3.5rem !important;
    border-bottom: 3px solid var(--dragon-gold);
    padding-bottom: 0.5rem;
}

.dragon-container {
    position: fixed;
    top: -100px;
    right: -200px;
    z-index: -1;
    opacity: 0.4;
    transform: scaleX(-1);
}

.dragon-img {
    width: 600px;
    height: 600px;
    animation: breathe 4s ease-in-out infinite;
}

.money-fall {
    position: fixed;
    top: 220px;
    right: 240px;
    font-size: 24px;
    opacity: 0.7;
    transform: rotate(-45deg);
    animation: fall 2s linear infinite;
}

@keyframes breathe {
    0%, 100% { transform: scale(1) rotate(-10deg); }
    50% { transform: scale(1.05) rotate(-15deg); }
}

@keyframes fall {
    0% { transform: translateY(-100vh) rotate(0deg); opacity: 1; }
    100% { transform: translateY(100vh) rotate(360deg); opacity: 0; }
}

.stButton>button {
    background: linear-gradient(45deg, var(--dragon-red), var(--dragon-gold)) !important;
    border: 1px solid var(--dragon-gold) !important;
    color: #000 !important;
    font-weight: bold !important;
    transition: all 0.3s ease;
    border-radius: 8px !important;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px var(--dragon-red) !important;
}

.st-emotion-cache-1qg05tj {
    background-color: rgba(139, 0, 0, 0.3) !important;
    border: 2px solid var(--dragon-gold) !important;
    border-radius: 15px !important;
    box-shadow: 0 0 15px var(--dragon-red) !important;
}
</style>
""", unsafe_allow_html=True)

# Dragon graphic and animated money
html(f"""
<div class="dragon-container">
    <img src="https://www.freeiconspng.com/uploads/dragon-png-5.png" class="dragon-img">
    <div class="money-fall">💵💵💵💵💵</div>
    <div class="money-fall" style="animation-delay: -0.5s">💵💵💵💵💵</div>
    <div class="money-fall" style="animation-delay: -1s">💵💵💵💵💵</div>
</div>
""")

# ====================== APP TITLE ======================
st.title("🐉 Dragon Trading AI")
st.markdown("#### Breathing Fire into Crypto Investments")

# ====================== ORIGINAL FUNCTIONALITY ======================
MODEL_PATH = Path("crypto_model.pkl")
START_DATE = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
COIN_IDS = {'btc-usd': 'bitcoin', 'eth-usd': 'ethereum', 'ada-usd': 'cardano'}
FEATURES = ['rsi', 'macd', 'macd_signal', 'macd_hist', 'sma_20', 'sma_50', 'momentum']

coin_mapping = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Cardano (ADA)": "ADA-USD"
}

def normalize_columns(df):
    column_map = {
        'open': ['open', 'opening', 'start', 'first'],
        'high': ['high', 'highest', 'max', 'peak'],
        'low': ['low', 'lowest', 'min', 'bottom'],
        'close': ['close', 'closing', 'end', 'last', 'final'],
        'volume': ['volume', 'vol', 'quantity', 'amount']
    }
    
    normalized = {}
    for standard, variants in column_map.items():
        for col in df.columns:
            if any(variant in col.lower() for variant in variants):
                normalized[standard] = df[col]
                break
    return pd.DataFrame(normalized)

def fetch_yfinance_data(symbol):
    try:
        df = yf.download(symbol, start=START_DATE, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(map(str, col)).lower() for col in df.columns.values]
        else:
            df.columns = df.columns.str.lower()
        df = normalize_columns(df)
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            st.warning(f"yFinance missing columns. Found: {df.columns.tolist()}")
            return pd.DataFrame()
        df = df[required].ffill().dropna()
        return df
    except Exception as e:
        st.warning(f"yFinance error: {str(e)}")
        return pd.DataFrame()

def fetch_coingecko_data(symbol):
    try:
        coin_id = COIN_IDS.get(symbol.lower(), symbol.lower().replace('-usd', ''))
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {'vs_currency': 'usd', 'days': '365', 'interval': 'daily'}
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        prices = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
        prices['date'] = pd.to_datetime(prices['timestamp'], unit='ms')
        prices.set_index('date', inplace=True)
        prices['open'] = prices['close'].shift(1).fillna(method='bfill')
        prices['high'] = prices[['open', 'close']].max(axis=1)
        prices['low'] = prices[['open', 'close']].min(axis=1)
        prices['volume'] = prices['close'].rolling(7).std().abs() * 1000
        return prices[['open', 'high', 'low', 'close', 'volume']].dropna()
    except Exception as e:
        st.warning(f"CoinGecko error: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_data(symbol):
    df = fetch_yfinance_data(symbol)
    if df.empty or len(df) < 50:
        st.warning("Falling back to CoinGecko...")
        df = fetch_coingecko_data(symbol)
    if df.empty or len(df) < 50:
        st.error("Data loading failed. Try checking your connection or selecting another crypto.")
        return pd.DataFrame()
    return df

def calculate_features(df):
    try:
        df = df.copy()
        df['sma_20'] = df['close'].rolling(20, min_periods=1).mean()
        df['sma_50'] = df['close'].rolling(50, min_periods=1).mean()
        delta = df['close'].diff().fillna(0)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14, min_periods=1).mean()
        avg_loss = loss.rolling(14, min_periods=1).mean()
        rs = (avg_gain + 1e-10) / (avg_loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['momentum'] = df['close'] - df['close'].shift(4)
        return df.dropna()
    except Exception as e:
        st.error(f"Feature error: {str(e)}")
        return pd.DataFrame()

def load_model():
    try:
        if MODEL_PATH.exists():
            return joblib.load(MODEL_PATH)
        return None
    except Exception as e:
        st.error(f"Model load error: {str(e)}")
        return None

def train_model(data, n_estimators=100, max_depth=5):
    try:
        df = calculate_features(data)
        if len(df) < 100:
            raise ValueError("Minimum 100 data points required")
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        X = df[FEATURES].dropna()
        y = df['target'].loc[X.index]
        if len(X) < 50:
            raise ValueError("Insufficient training samples")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        joblib.dump(model, MODEL_PATH)
        return model, accuracy, df
    except Exception as e:
        st.error(f"Training failed: {str(e)}")
        return None, 0, pd.DataFrame()

def predict(model, data):
    try:
        df = calculate_features(data)
        latest = df[FEATURES].iloc[[-1]]
        pred = model.predict(latest)[0]
        proba = model.predict_proba(latest)[0]
        return pred, proba, df
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, pd.DataFrame()

def main():
    st.sidebar.header("Dragon's Lair Configuration")
    coin_name = st.sidebar.selectbox("Select Treasure", list(coin_mapping.keys()))
    symbol = coin_mapping[coin_name]
    df = fetch_data(symbol)

    tab1, tab2 = st.tabs(["📜 Dragon's Scrolls", "⚔ Training Grounds"])

    with tab1:
        st.header("Dragon's Prophecy")
        if not df.empty:
            st.subheader(f"{coin_name} Gold Hoard Value")
            st.line_chart(df['close'])

            if st.button("Seek Dragon's Wisdom", type="primary"):
                model = load_model()
                if model:
                    pred, proba, df_feat = predict(model, df)
                    col1, col2 = st.columns(2)
                    with col1:
                        current_price = df['close'].iloc[-1]
                        st.metric("Current Treasure Value", f"${current_price:,.2f}")
                    with col2:
                        if pred is not None:
                            direction = "🐉 FIRE BREATH (BUY)" if pred == 1 else "💤 HIBERNATE (HOLD)"
                            confidence = proba[1] if pred == 1 else proba[0]
                            st.metric("Dragon's Verdict", 
                                    f"{direction}\n{confidence*100:.1f}% Confidence")

                    if pred is not None:
                        try:
                            df_feat['predicted'] = model.predict(df_feat[FEATURES])
                            df_feat['target'] = (df_feat['close'].shift(-1) > df_feat['close']).astype(int)
                            accuracy = (df_feat['predicted'] == df_feat['target']).mean()
                            st.metric("Historical Accuracy", f"{accuracy*100:.1f}%")
                        except Exception as e:
                            st.warning(f"Accuracy calculation skipped: {str(e)}")

    with tab2:
        st.header("Dragon Training Ritual")
        st.subheader("Forge Your Predictor")
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("Number of Dragon Eggs", 50, 200, 100)
        with col2:
            max_depth = st.slider("Dragon's Wisdom Depth", 3, 20, 7)

        if st.button("Begin Training Ritual", type="primary"):
            if not df.empty:
                with st.spinner("Feeding the dragon..."):
                    model, accuracy, _ = train_model(df, n_estimators, max_depth)
                    if model:
                        st.success(f"🐉 Dragon trained! Accuracy: {accuracy*100:.1f}%")
                        st.subheader("Dragon's Knowledge Map")
                        importances = model.feature_importances_
                        fig, ax = plt.subplots()
                        ax.barh(FEATURES, importances)
                        st.pyplot(fig)

    with st.expander("Ancient Cryptic Writings"):
        if not df.empty:
            st.write("*Data Summary*")
            st.dataframe(df.describe())
            st.write("*Recent Features*")
            st.dataframe(calculate_features(df).tail(3))

if _name_ == "_main_":
    main()
