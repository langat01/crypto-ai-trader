import streamlit as st
import yfinance as yf
import joblib
from features import add_features, prepare_training_data
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="🚀 Crypto AI Trading", layout="wide")

# Sidebar controls
st.sidebar.header("Settings")
crypto_symbol = st.sidebar.selectbox("Select Cryptocurrency", ['BTC-USD', 'ETH-USD', 'LTC-USD', 'DOGE-USD'])
start_date = st.sidebar.date_input("Training Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("Training End Date", value=pd.to_datetime("today"))
action = st.sidebar.radio("Choose Action", ["Run Prediction", "Retrain Model"])

st.title(f"🚀 Crypto AI Trading Strategy: {crypto_symbol}")

@st.cache_data(ttl=3600)
def fetch_data(symbol, start, end):
    return yf.download(symbol, start=start, end=end)

def load_model(symbol):
    try:
        return joblib.load(f"models/{symbol}_model.pkl")
    except:
        st.warning("No existing model found for this coin. Please retrain.")
        return None

if action == "Run Prediction":
    with st.spinner("Fetching data and predicting..."):
        df = fetch_data(crypto_symbol, start_date, end_date)
        if df.empty:
            st.error("❌ Failed to load data.")
        else:
            model = load_model(crypto_symbol)
            if model is None:
                st.info("Please retrain the model first.")
            else:
                data_feat = add_features(df)
                latest = data_feat.iloc[[-1]]
                features = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'SMA_20', 'SMA_50', 'Momentum']
                X_latest = latest[features]
                pred = model.predict(X_latest)[0]
                prob = model.predict_proba(X_latest)[0]

                st.metric("Latest Close Price", f"${df['Close'].iloc[-1]:,.2f}")
                st.metric("Prediction for Tomorrow", "UP" if pred == 1 else "DOWN")
                st.write(f"Confidence - Up: {prob[1]*100:.2f}%, Down: {prob[0]*100:.2f}%")

                st.subheader("Close Price History")
                st.line_chart(df['Close'])

elif action == "Retrain Model":
    st.info("Training model with selected data range. This may take some time.")
    df = fetch_data(crypto_symbol, start_date, end_date)
    if df.empty:
        st.error("❌ Failed to load data.")
    else:
        X, y = prepare_training_data(df)
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, f"models/{crypto_symbol}_model.pkl")
        st.success("✅ Model retrained and saved successfully!")

