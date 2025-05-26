# Dragon Trading AI
# app.py - Main Streamlit App

import streamlit as st
from data_loader import get_crypto_data
from indicators import compute_indicators
from model import train_model, predict_next_day, backtest_strategy
from plots import plot_candlestick_with_volume
from utils import play_sound_alert

st.set_page_config(page_title="Dragon Trading AI", layout="wide")
st.title("🔥 Dragon Trading AI - Crypto Prediction Dashboard 🔥")

crypto_symbol = st.selectbox("Choose your crypto", ["BTC", "ETH", "SOL", "BNB", "ADA"])
st.info(f"Fetching data for {crypto_symbol}...")

df = get_crypto_data(crypto_symbol)

if df.empty:
    st.error("Failed to fetch crypto data. Try again later.")
    st.stop()

# Compute technical indicators
df = compute_indicators(df)

# Display interactive candlestick chart
st.subheader("🌍 Live Candlestick Chart with Volume")
st.plotly_chart(plot_candlestick_with_volume(df), use_container_width=True)

# Train prediction model
model, acc, report = train_model(df)

st.subheader("🌟 Model Performance")
st.write(f"**Accuracy:** {acc:.2%}")
st.text(report)

# Prediction
prediction = predict_next_day(model, df)

if prediction == 1:
    st.success("📈 Model predicts the price will go **UP** tomorrow!")
    play_sound_alert('up')
else:
    st.error("📉 Model predicts the price will go **DOWN** tomorrow.")
    play_sound_alert('down')

# Backtesting
st.subheader("⏳ Strategy Backtest")
st.dataframe(backtest_strategy(df, model))

st.caption("Built with ❤️ by Dragon AI")
