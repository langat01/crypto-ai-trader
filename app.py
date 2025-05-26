import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

st.title("🔥 Breathing Fire into Crypto Investments (CryptoCompare API) 🔥")
st.write("Predict next day price movement for popular cryptocurrencies using Random Forest.")

cryptos = {
    "Bitcoin": "BTC",
    "Ethereum": "ETH",
    "Binance Coin": "BNB",
    "Cardano": "ADA",
    "Solana": "SOL"
}

selected_crypto_name = st.selectbox("Select Cryptocurrency", list(cryptos.keys()))
selected_crypto_symbol = cryptos[selected_crypto_name]

@st.cache_data(show_spinner=False)
def fetch_data_cryptocompare(symbol, limit=730):
    """
    Fetch daily OHLCV data from CryptoCompare API.
    limit: number of days to fetch (max 2000)
    """
    url = f"https://min-api.cryptocompare.com/data/v2/histoday"
    params = {
        "fsym": symbol,
        "tsym": "USD",
        "limit": limit,
        "aggregate": 1
    }
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        if data['Response'] != 'Success':
            raise ValueError(f"CryptoCompare API error: {data.get('Message', 'Unknown error')}")
        raw = data['Data']['Data']
        df = pd.DataFrame(raw)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        # Rename columns to match our feature engineering expectations
        df.rename(columns={'open':'open', 'high':'high', 'low':'low', 'close':'close', 'volumefrom':'volume', 'volumeto':'volumeto'}, inplace=True)
        df = df[['open','high','low','close','volume']].astype(float)
        return df
    except Exception as e:
        st.error(f"Failed to fetch data from CryptoCompare: {e}")
        return pd.DataFrame()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def create_features(df):
    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['rsi_14'] = compute_rsi(df['close'], 14)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()
    return df

def train_model(df):
    features = ['return', 'sma_5', 'sma_10', 'rsi_14']
    X = df[features]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    return model, acc, report

def predict_next_day(model, df):
    features = ['return', 'sma_5', 'sma_10', 'rsi_14']
    last_row = df.iloc[-1:]
    X_last = last_row[features]
    prediction = model.predict(X_last)[0]
    return prediction

def main():
    st.info("Fetching data from CryptoCompare API...")
    df = fetch_data_cryptocompare(selected_crypto_symbol)
    if df.empty:
        st.error("Failed to load data. Please try again later or select another crypto.")
        return

    df = create_features(df)
    model, acc, report = train_model(df)
    st.write(f"Model accuracy on test set: **{acc:.2%}**")
    st.text("Classification report:\n" + report)

    prediction = predict_next_day(model, df)
    if prediction == 1:
        st.success("Model predicts the price will **go UP** tomorrow.")
    else:
        st.error("Model predicts the price will **go DOWN** tomorrow.")

if __name__ == "__main__":
    main()
