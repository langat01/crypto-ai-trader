import streamlit as st
import yfinance as yf
from model_utils import load_model, predict
from backtest import backtest
from plot_utils import plot_price, plot_macd

st.set_page_config(page_title="🚀 Crypto AI Trading", layout="wide")

st.title("Crypto AI Trading Strategy")
st.markdown("Predicting Bitcoin (BTC-USD) Next Day Movement with Machine Learning")

@st.cache_data(ttl=3600)
def load_data():
    return yf.download('BTC-USD', start='2023-01-01')

if st.button("Run Prediction"):
    st.spinner("Loading data and predicting...")
    df = load_data()
    if df.empty:
        st.error("❌ BTC data could not be loaded.")
    else:
        model = load_model()
        prediction, prob, df_feat = predict(model, df)

        st.metric("Latest BTC Close", f"${df['Close'].iloc[-1]:,.2f}")
        if prediction == 1:
            st.success(f"📈 Prediction: **UP**")
        else:
            st.error(f"📉 Prediction: **DOWN**")
        st.write(f"Confidence: Up = {prob[1]*100:.2f}%, Down = {prob[0]*100:.2f}%")

        st.plotly_chart(plot_price(df_feat), use_container_width=True)
        st.plotly_chart(plot_macd(df_feat), use_container_width=True)

        st.subheader("Backtest Performance")
        bt_df, acc = backtest(model, df_feat)
        st.write(f"Model Accuracy: **{acc*100:.2f}%**")
        st.dataframe(bt_df[['Close', 'Target', 'Predicted']].tail(20))
else:
    st.info("Click the button above to predict tomorrow's BTC movement.")
