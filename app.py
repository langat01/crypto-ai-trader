import streamlit as st
import yfinance as yf
import pandas as pd
import shap
import plotly.graph_objects as go

from model_utils import load_model
from backtest import backtest
from plot_utils import plot_price, plot_macd
from features import add_features
from email_utils import send_email_alert
from sms_utils import send_sms_alert

st.set_page_config(page_title="🚀 AI Crypto Trading", layout="wide")

st.title("📈 AI-Powered Crypto Trading Strategy")

# --- Inputs ---
st.sidebar.header("🔧 Settings")
symbol = st.sidebar.text_input("Crypto Symbol", value="BTC-USD")
model_name = st.sidebar.selectbox("Select Model", ["RandomForest", "XGBoost"])
show_features = st.sidebar.checkbox("Show Technical Indicators")
run = st.sidebar.button("Predict")

@st.cache_data(ttl=3600)
def load_data(symbol):
    return yf.download(symbol, start="2023-01-01")

def explain_prediction(model, X_input):
    st.subheader("📊 Model Explainability (SHAP)")
    try:
        explainer = shap.Explainer(model, X_input)
        shap_values = explainer(X_input)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(bbox_inches='tight')
    except Exception as e:
        st.warning(f"SHAP not supported for this model: {e}")

def run_prediction():
    df = load_data(symbol)
    if df.empty:
        st.error("❌ Could not load data.")
        return

    df_feat = add_features(df, scale=True, include_target=False)
    latest = df_feat.iloc[[-1]]

    model = load_model(model_name)
    prediction = model.predict(latest)[0]
    proba = model.predict_proba(latest)[0]

    st.metric("Latest Price", f"${df['Close'].iloc[-1]:,.2f}")
    st.info(f"Prediction: {'📈 UP' if prediction == 1 else '📉 DOWN'}")
    st.info(f"Confidence: Up = {proba[1]*100:.2f}%, Down = {proba[0]*100:.2f}%")

    if show_features:
        st.subheader("🧮 Technical Indicators (Latest)")
        st.dataframe(latest.T.rename(columns={latest.index[-1]: "Value"}))

    st.plotly_chart(plot_price(df_feat), use_container_width=True)
    st.plotly_chart(plot_macd(df_feat), use_container_width=True)

    st.subheader("📉 Backtest Results")
    bt_df, acc = backtest(model, df_feat)
    st.success(f"Backtest Accuracy: {acc * 100:.2f}%")
    st.dataframe(bt_df[['Close', 'Target', 'Predicted']].tail(20))

    explain_prediction(model, latest)

    # --- Email Alert Section ---
    st.subheader("📧 Email Prediction Alert")
    with st.expander("Send email notification"):
        to_email = st.text_input("Recipient Email Address")
        from_email = st.text_input("Sender Gmail Address")
        app_password = st.text_input("Gmail App Password", type="password")
        if st.button("Send Email Alert"):
            if to_email and from_email and app_password:
                result = send_email_alert(
                    "UP" if prediction == 1 else "DOWN",
                    proba[prediction] * 100,
                    to_email,
                    from_email,
                    app_password
                )
                if result is True:
                    st.success("✅ Email sent successfully.")
                else:
                    st.error(result)
            else:
                st.warning("Please fill in all email fields.")

    # --- SMS Alert Section ---
    st.subheader("📱 SMS Prediction Alert")
    with st.expander("Send SMS notification"):
        to_number = st.text_input("Recipient Phone Number (e.g. +2547XXXXXXX)")
        from_number = st.text_input("Twilio Phone Number")
        twilio_sid = st.text_input("Twilio Account SID")
        twilio_token = st.text_input("Twilio Auth Token", type="password")
        if st.button("Send SMS Alert"):
            if to_number and from_number and twilio_sid and twilio_token:
                result = send_sms_alert(
                    "UP" if prediction == 1 else "DOWN",
                    proba[prediction] * 100,
                    to_number,
                    from_number,
                    twilio_sid,
                    twilio_token
                )
                if result is True:
                    st.success("✅ SMS sent successfully.")
                else:
                    st.error(result)
            else:
                st.warning("Please fill in all SMS fields.")

# Trigger prediction
if run:
    with st.spinner("Running prediction..."):
        run_prediction()
else:
    st.info("Use the sidebar to select options and run a prediction.")
