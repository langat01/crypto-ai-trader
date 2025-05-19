import streamlit as st
import yfinance as yf
import pandas as pd
import shap
import plotly.graph_objects as go

from model_utils import load_model
from backtest import backtest
from plot_utils import plot_price, plot_macd
from features import add_features

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

# Trigger prediction
if run:
    with st.spinner("Running prediction..."):
        run_prediction()
else:
    st.info("Use the sidebar to select options and run a prediction.")
