import streamlit as st
import yfinance as yf
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from features import add_features

st.set_page_config(page_title="Crypto AI Trading", layout="wide")
st.title("🚀 Crypto AI Trading Strategy")
st.markdown("Predicting Cryptocurrency Next-Day Movement with Machine Learning")

CRYPTO_OPTIONS = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Cardano (ADA-USD)": "ADA-USD"
}

selected_crypto = st.selectbox("Select Cryptocurrency", list(CRYPTO_OPTIONS.keys()))
ticker = CRYPTO_OPTIONS[selected_crypto]

@st.cache_data(ttl=60)
def fetch_data(symbol):
    return yf.download(symbol, period="3mo", interval="1d")

def load_model():
    return joblib.load("crypto_model.pkl")

def predict_next_day_movement(model, data):
    data_feat = add_features(data)
    latest = data_feat.iloc[[-1]]
    features = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'SMA_20', 'SMA_50', 'Momentum']
    X_latest = latest[features]
    pred = model.predict(X_latest)[0]
    prob = model.predict_proba(X_latest)[0]
    return pred, prob, data_feat

def backtest_model(model, data_feat):
    data_feat['Target'] = (data_feat['Close'].shift(-1) > data_feat['Close']).astype(int)
    features = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'SMA_20', 'SMA_50', 'Momentum']
    X = data_feat[features]
    y = data_feat['Target']
    preds = model.predict(X)
    data_feat['Predicted'] = preds
    data_feat['Correct'] = (preds == y).astype(int)
    accuracy = data_feat['Correct'].mean()
    return data_feat, accuracy

if st.button("🔮 Predict Next Movement"):
    with st.spinner(f"Fetching latest {selected_crypto} data and predicting..."):
        df = fetch_data(ticker)
        if df.empty:
            st.error("❌ Failed to load crypto data.")
        else:
            model = load_model()
            prediction, probabilities, df_feat = predict_next_day_movement(model, df)

            close_price = float(df['Close'].iloc[-1])
            st.markdown(f"### Latest {selected_crypto} Close: ${close_price:,.2f}")

            if prediction == 1:
                st.success("📈 Prediction: **UP**")
            else:
                st.error("📉 Prediction: **DOWN**")

            st.markdown(f"""
            **Confidence**  
            - Up = {probabilities[1]*100:.2f}%  
            - Down = {probabilities[0]*100:.2f}%
            """)

            st.subheader("📊 Price History")
            st.line_chart(df['Close'])

            st.subheader("📉 Backtest Accuracy")
            backtested_df, acc = backtest_model(model, df_feat.copy())
            st.write(f"Accuracy: **{acc*100:.2f}%**")

            st.subheader("🆚 Predictions vs Actual")
            plot_df = backtested_df[-50:].copy()
            plot_df['Actual'] = plot_df['Target'].map({1: 'Up', 0: 'Down'})
            plot_df['Predicted'] = plot_df['Predicted'].map({1: 'Up', 0: 'Down'})
            st.dataframe(plot_df[['Close', 'Actual', 'Predicted']])

            st.subheader("📉 Technical Indicators")
            fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            ax[0].plot(df_feat['Close'], label='Close Price')
            ax[0].plot(df_feat['SMA_20'], label='SMA 20')
            ax[0].plot(df_feat['SMA_50'], label='SMA 50')
            ax[0].legend()
            ax[0].set_title(f'{selected_crypto} Price and SMAs')
            ax[1].plot(df_feat['MACD'], label='MACD')
            ax[1].plot(df_feat['MACD_Signal'], label='Signal')
            ax[1].bar(df_feat.index, df_feat['MACD_Hist'], label='Hist')
            ax[1].legend()
            ax[1].set_title('MACD Indicators')
            st.pyplot(fig)
else:
    st.info("Click the button above to predict the selected crypto's next movement.")
