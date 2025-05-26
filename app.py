import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

START_DATE = '2021-01-01'

st.title("🔥 Breathing Fire into Crypto Investments 🔥")
st.write("Predict next day price movement for popular cryptocurrencies using Random Forest.")

cryptos = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Binance Coin (BNB-USD)": "BNB-USD",
    "Cardano (ADA-USD)": "ADA-USD",
    "Solana (SOL-USD)": "SOL-USD"
}

selected_crypto_name = st.selectbox("Select Cryptocurrency", list(cryptos.keys()))
selected_crypto_symbol = cryptos[selected_crypto_name]

@st.cache_data
def fetch_data_yfinance(symbol):
    try:
        df = yf.download(symbol, start=START_DATE, progress=False)
        # Handle MultiIndex columns by flattening
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)
        df.columns = df.columns.str.lower()
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing columns in data: {df.columns}")
        df = df[required_cols].ffill().dropna()
        return df
    except Exception as e:
        st.warning(f"yFinance data fetch error: {e}")
        return pd.DataFrame()

@st.cache_data
def fetch_data_coingecko(id):
    url = f"https://api.coingecko.com/api/v3/coins/{id}/market_chart"
    params = {"vs_currency": "usd", "days": "730", "interval": "daily"}
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        prices = data['prices']  # list of [timestamp, price]
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
        df.set_index('date', inplace=True)
        df = df[['price']]
        df['close'] = df['price']
        df['open'] = df['price']
        df['high'] = df['price']
        df['low'] = df['price']
        df['volume'] = np.nan  # CoinGecko doesn't provide volume here
        df = df.drop(columns=['price'])
        df = df.loc[START_DATE:]
        df = df.ffill().dropna()
        return df
    except Exception as e:
        st.warning(f"CoinGecko fetch error: {e}")
        return pd.DataFrame()

def create_features(df):
    df['return'] = df['close'].pct_change()
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['rsi_14'] = compute_rsi(df['close'], 14)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

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
    df = fetch_data_yfinance(selected_crypto_symbol)
    if df.empty:
        st.info("Falling back to CoinGecko data source...")
        # Map ticker symbol to CoinGecko ID
        cg_map = {
            "BTC-USD": "bitcoin",
            "ETH-USD": "ethereum",
            "BNB-USD": "binancecoin",
            "ADA-USD": "cardano",
            "SOL-USD": "solana"
        }
        df = fetch_data_coingecko(cg_map.get(selected_crypto_symbol, "bitcoin"))

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
