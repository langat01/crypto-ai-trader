import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import plotly.graph_objs as go

st.set_page_config(page_title="🔥 Crypto Price Movement Predictor 🔥", layout="wide")

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

days = st.slider("Select number of days of historical data:", min_value=180, max_value=1000, value=730, step=30)

@st.cache_data(show_spinner=False)
def fetch_data_cryptocompare(symbol, limit=730):
    url = f"https://min-api.cryptocompare.com/data/v2/histoday"
    params = {
        "fsym": symbol,
        "tsym": "USD",
        "limit": limit,
        "aggregate": 1
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    if data['Response'] != 'Success':
        raise ValueError(f"CryptoCompare API error: {data.get('Message', 'Unknown error')}")
    raw = data['Data']['Data']
    df = pd.DataFrame(raw)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'open':'open', 'high':'high', 'low':'low', 'close':'close', 'volumefrom':'volume'}, inplace=True)
    df = df[['open','high','low','close','volume']].astype(float)
    return df

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
    df.dropna(inplace=True)
    return df

@st.cache_data(show_spinner=False)
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
    return model, acc, report, X_test, y_test

def plot_price_and_indicators(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['open'],
                                 high=df['high'],
                                 low=df['low'],
                                 close=df['close'],
                                 name='Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['sma_5'], line=dict(color='blue', width=1), name='SMA 5'))
    fig.add_trace(go.Scatter(x=df.index, y=df['sma_10'], line=dict(color='orange', width=1), name='SMA 10'))
    fig.update_layout(title="Price Chart with SMAs", xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig, use_container_width=True)

    # RSI plot
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df['rsi_14'], line=dict(color='purple', width=2), name='RSI 14'))
    fig2.update_layout(title="Relative Strength Index (RSI 14)", xaxis_title="Date", yaxis_title="RSI",
                       yaxis=dict(range=[0,100]))
    st.plotly_chart(fig2, use_container_width=True)

def plot_feature_importance(model, features):
    importances = model.feature_importances_
    fig = go.Figure([go.Bar(x=features, y=importances, marker_color='firebrick')])
    fig.update_layout(title="Feature Importances", yaxis_title="Importance")
    st.plotly_chart(fig, use_container_width=True)

def main():
    try:
        with st.spinner("Fetching data from CryptoCompare API..."):
            df = fetch_data_cryptocompare(selected_crypto_symbol, limit=days)
        if df.empty:
            st.error("Failed to load data. Please try again later or select another crypto.")
            return

        df = create_features(df)

        with st.spinner("Training model..."):
            model, acc, report, X_test, y_test = train_model(df)

        st.markdown(f"### Model accuracy on test set: **{acc:.2%}**")
        st.text("Classification report:\n" + report)

        prediction = model.predict(X_test.tail(1))[0]
        if prediction == 1:
            st.success("Model predicts the price will **go UP** tomorrow.")
        else:
            st.error("Model predicts the price will **go DOWN** tomorrow.")

        plot_price_and_indicators(df)
        plot_feature_importance(model, ['return', 'sma_5', 'sma_10', 'rsi_14'])

    except Exception as e:
        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
