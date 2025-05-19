import streamlit as st
import yfinance as yf
import pandas as pd

from model_utils import load_model
from backtest import backtest
from plot_utils import plot_price, plot_macd
from features import add_features

st.set_page_config(page_title="🚀 Crypto AI Trading", layout="wide")

st.title("Crypto AI Trading Strategy")
st.markdown("**Predicting Bitcoin (BTC-USD) Next Day Movement with Machine Learning**")

@st.cache_data(ttl=3600)
def load_data():
    return yf.download('BTC-USD', start='2023-01-01')

def make_prediction():
    df = load_data()
    if df.empty:
        st.error("❌ BTC data could not be loaded.")
        return
    
    model = load_model()
    df_feat = add_features(df, scale=True, include_target=False)
    
    latest = df_feat.iloc[[-1]]  # latest row
    prediction = model.predict(latest)[0]
    proba = model.predict_proba(latest)[0]

    # Display metrics
    st.metric("📊 Latest BTC Close", f"${df['Close'].iloc[-1]:,.2f}")
    st.write("### Technical Indicators (Latest Day)")
    st.dataframe(df_feat.iloc[[-1]].T.rename(columns={df_feat.index[-1]: "Value"}))

    if prediction == 1:
        st.success(f"📈 **Prediction: UP**")
    else:
        st.error(f"📉 **Prediction: DOWN**")
    
    st.info(f"🔍 Confidence: Up = `{proba[1]*100:.2f}%`, Down = `{proba[0]*100:.2f}%`")

    st.plotly_chart(plot_price(df_feat), use_container_width=True)
    st.plotly_chart(plot_macd(df_feat), use_container_width=True)

    st.subheader("🔁 Backtest Performance")
    bt_df, acc = backtest(model, df_feat)
    st.write(f"✅ Model Accuracy: **{acc * 100:.2f}%**")
    st.dataframe(bt_df[['Close', 'Target', 'Predicted']].tail(20))


# --- UI Buttons ---
if st.button("Run Prediction"):
    with st.spinner("Loading data and predicting..."):
        make_prediction()
else:
    st.info("Click the button above to predict tomorrow's BTC movement.")
