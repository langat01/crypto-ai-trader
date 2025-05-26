import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import threading
from src.data.fetch_data import fetch_crypto_data_realtime
from src.utils.sound_alert import play_alert_sound
from src.models.predict import load_model_and_predict

st.set_page_config(page_title="Dragon Trading AI", layout="wide")
st.title("🔥 Dragon Trading AI Dashboard 🔥")

selected_crypto = st.selectbox("Select Cryptocurrency", ["BTC", "ETH", "SOL", "ADA", "BNB"])
model_option = st.radio("Model Type", ["Random Forest", "XGBoost"], horizontal=True)

# Live Chart Area
st.subheader(f"Live Candlestick Chart for {selected_crypto}")
chart_area = st.empty()
data_load_state = st.empty()

# Real-time Prediction Output
prediction_area = st.empty()
sound_state = st.empty()

# Technical Indicator Summary
st.subheader("Technical Indicator Snapshot")
indicator_metrics = st.columns(4)

# Load once
model = load_model_and_predict(model_option, selected_crypto)

# Real-time chart and prediction loop
def update_dashboard():
    while True:
        data_load_state.info("Fetching real-time data...")
        df = fetch_crypto_data_realtime(selected_crypto)
        if df is not None and not df.empty:
            # Candlestick Chart with Volume
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Candlestick'))
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color='blue',
                yaxis='y2'))

            fig.update_layout(
                xaxis_rangeslider_visible=False,
                height=600,
                yaxis_title='Price (USD)',
                yaxis2=dict(overlaying='y', side='right', title='Volume'),
                template='plotly_dark'
            )
            chart_area.plotly_chart(fig, use_container_width=True)

            # Technical Indicators
            indicators = ['rsi_14', 'macd', 'bb_upper', 'bb_lower', 'sma_5', 'momentum']
            for i, metric in enumerate(indicators[:4]):
                indicator_metrics[i].metric(label=metric.upper(), value=f"{df.iloc[-1][metric]:.2f}")

            # Model Prediction
            prediction = model.predict(df.iloc[[-1]][model.feature_names_in_])[0]
            if prediction == 1:
                prediction_area.success("Prediction: Price will go UP 📈")
                play_alert_sound("up")
            else:
                prediction_area.error("Prediction: Price will go DOWN 📉")
                play_alert_sound("down")
        else:
            chart_area.warning("Failed to load data. Try again later.")

        data_load_state.empty()
        time.sleep(30)

# Run the dashboard in a background thread
threading.Thread(target=update_dashboard, daemon=True).start()
