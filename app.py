import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import plotly.graph_objs as go

st.set_page_config(page_title="Dragon Trading AI", page_icon="🐉", layout="wide")
st.title("🐉 Dragon Trading AI - Real-Time Price & Prediction")
st.markdown("Predict next day price movement and watch live price updates for popular cryptocurrencies.")

cryptos = {
    "Bitcoin (BTC)": "BTC",
    "Ethereum (ETH)": "ETH",
    "Binance Coin (BNB)": "BNB",
    "Cardano (ADA)": "ADA",
    "Solana (SOL)": "SOL"
}

selected_crypto_name = st.selectbox("Select Cryptocurrency", list(cryptos.keys()))
selected_crypto_symbol = cryptos[selected_crypto_name]

days = st.slider("Select number of days of historical data", min_value=180, max_value=1000, value=730, step=30)

@st.cache_data(show_spinner=False)
def fetch_data_cryptocompare(symbol, limit=730):
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
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
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
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

def fetch_realtime_price(symbol):
    url = "https://min-api.cryptocompare.com/data/price"
    params = {"fsym": symbol, "tsyms": "USD"}
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        price = r.json().get("USD", None)
        return price
    except Exception as e:
        st.warning(f"Failed to fetch real-time price: {e}")
        return None

def main():
    # Historical data and model training
    try:
        with st.spinner("Fetching historical data..."):
            df = fetch_data_cryptocompare(selected_crypto_symbol, limit=days)
        if df.empty:
            st.error("Failed to load historical data. Try again or select another cryptocurrency.")
            return

        df = create_features(df)

        with st.spinner("Training model..."):
            model, acc, report, X_test, y_test = train_model(df)

        st.markdown(f"### Model Accuracy on Test Set: *{acc:.2%}*")
        st.text("Classification Report:\n" + report)

        prediction = model.predict(X_test.tail(1))[0]
        if prediction == 1:
            st.success("🟢 Model predicts the price will *go UP* tomorrow.")
        else:
            st.error("🔴 Model predicts the price will *go DOWN* tomorrow.")
    except Exception as e:
        st.error(f"An error occurred while loading or training: {e}")
        return

    # Real-time price live update toggle
    st.markdown("---")
    st.markdown("## Real-Time Price Monitoring")

    run_live = st.checkbox("Enable Live Price Updates", value=False)
    price_threshold_up = st.number_input("Alert if price rises above:", min_value=0.0, value=0.0, step=1.0)
    price_threshold_down = st.number_input("Alert if price falls below:", min_value=0.0, value=0.0, step=1.0)

    price_placeholder = st.empty()
    chart_placeholder = st.empty()
    alert_placeholder = st.empty()

    if run_live:
        prices = []
        times = []

        for _ in range(60*5):  # Run for ~5 mins (60*5 iterations at 1 sec)
            price = fetch_realtime_price(selected_crypto_symbol)
            if price is not None:
                current_time = datetime.now()
                prices.append(price)
                times.append(current_time)

                # Display price
                price_placeholder.markdown(f"*Current {selected_crypto_name} Price:* ${price:,.2f} USD (Updated: {current_time.strftime('%H:%M:%S')})")

                # Check alerts
                if price_threshold_up > 0 and price > price_threshold_up:
                    alert_placeholder.success(f"🚀 Price Alert! Price rose above ${price_threshold_up:,.2f}!")
                elif price_threshold_down > 0 and price < price_threshold_down:
                    alert_placeholder.error(f"⚠ Price Alert! Price fell below ${price_threshold_down:,.2f}!")
                else:
                    alert_placeholder.empty()

                # Update live price chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=times, y=prices, mode='lines+markers', name='Price USD'))
                fig.update_layout(
                    title=f"Live Price Chart of {selected_crypto_name}",
                    xaxis_title="Time",
                    yaxis_title="Price (USD)",
                    template="plotly_dark",
                    height=400
                )
                chart_placeholder.plotly_chart(fig, use_container_width=True)

            else:
                price_placeholder.warning("Failed to fetch live price.")

            time.sleep(1)  # Update every 1 second

    else:
        st.info("Check the box above to enable live price updates.")

if __name__ == "__main__":
    main()
