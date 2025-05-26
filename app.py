import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Constants
START_DATE = "2021-01-01"
TODAY = datetime.today().strftime('%Y-%m-%d')

# ========== Data Fetching ==========

def fetch_data_yfinance(symbol):
    try:
        df = yf.download(symbol, start=START_DATE, progress=False)
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)
        df.columns = df.columns.str.lower()
        df = df[['open','high','low','close','volume']].ffill().dropna()
        return df
    except Exception as e:
        st.warning(f"yFinance data fetch error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_data(symbol):
    df = fetch_data_yfinance(symbol)
    if df.empty or len(df) < 100:
        st.error("Failed to load enough data from yFinance. Please try again later or select another crypto.")
    return df

# ========== Feature Engineering ==========

def compute_rsi(data, window=14):
    delta = data['close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(data, slow=26, fast=12, signal=9):
    exp1 = data['close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_sma(data, window):
    return data['close'].rolling(window=window).mean()

def compute_momentum(data, window=10):
    momentum = data['close'] - data['close'].shift(window)
    return momentum

def add_features(df):
    df['rsi'] = compute_rsi(df)
    df['macd'], df['macd_signal'] = compute_macd(df)
    df['sma_10'] = compute_sma(df, 10)
    df['sma_20'] = compute_sma(df, 20)
    df['momentum'] = compute_momentum(df)
    df = df.dropna()
    return df

# ========== Prepare Dataset for ML ==========

def prepare_dataset(df):
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)  # 1 = price up tomorrow, else 0
    features = ['rsi', 'macd', 'macd_signal', 'sma_10', 'sma_20', 'momentum']
    df = df.dropna()
    X = df[features]
    y = df['target']
    return X, y

# ========== Train Model ==========

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc

# ========== Predict Next Day Movement ==========

def predict_next_day(model, df):
    latest = df.iloc[-1]
    features = ['rsi', 'macd', 'macd_signal', 'sma_10', 'sma_20', 'momentum']
    X_latest = latest[features].values.reshape(1, -1)
    pred = model.predict(X_latest)[0]
    prob = model.predict_proba(X_latest)[0][pred]
    return pred, prob

# ========== Streamlit App ==========

st.title("🔥 Breathing Fire into Crypto Investments 🔥")
st.markdown("Predict next day price movement for popular cryptocurrencies using Random Forest.")

# Select crypto
options = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Litecoin (LTC-USD)": "LTC-USD",
    "Ripple (XRP-USD)": "XRP-USD",
    "Bitcoin Cash (BCH-USD)": "BCH-USD",
}

selected_name = st.selectbox("Select Cryptocurrency", list(options.keys()))
symbol = options[selected_name]

# Load data
with st.spinner("Loading data..."):
    data = load_data(symbol)

if not data.empty:
    st.subheader(f"Data Preview for {selected_name}")
    st.line_chart(data['close'])

    # Feature engineering
    data_feat = add_features(data)

    # Prepare ML dataset
    X, y = prepare_dataset(data_feat)

    if len(X) > 100:
        # Train model
        model, accuracy = train_model(X, y)
        st.success(f"Model trained with accuracy: {accuracy:.2%}")

        # Predict next day movement
        pred, prob = predict_next_day(model, data_feat)
        movement = "UP 📈" if pred == 1 else "DOWN 📉"
        st.markdown(f"**Prediction for next trading day:** {movement} with probability {prob:.2%}")

    else:
        st.error("Not enough data after feature engineering to train model.")

else:
    st.error("Failed to load data. Please try again later or select another crypto.")
