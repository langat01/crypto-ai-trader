import streamlit as st
import yfinance as yf
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from features import add_features
from datetime import datetime, timedelta

# Configuration
st.set_page_config(page_title="Crypto AI Trading Pro", layout="wide")
st.title("🚀 Crypto AI Trading Strategy Pro")

# Model and data caching
@st.cache_resource(ttl=3600)
def load_model(symbol: str):
    """Load pre-trained model with error handling"""
    try:
        model = joblib.load(f"models/{symbol}_model.pkl")
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

@st.cache_data(ttl=1800)
def fetch_data(symbol: str, days_back: int = 365):
    """Fetch data with retry logic"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    try:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if df.empty:
            raise ValueError("Empty DataFrame returned")
        return df
    except Exception as e:
        st.error(f"Data fetch failed: {str(e)}")
        return pd.DataFrame()

# UI Components
def render_metrics(current_price: float, prediction: int, probabilities: list):
    """Display prediction metrics"""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${current_price:,.2f}")
    with col2:
        st.metric(
            "Prediction", 
            "UP ↗️" if prediction == 1 else "DOWN ↘️",
            f"{probabilities[1]*100:.1f}%" if prediction == 1 else f"{probabilities[0]*100:.1f}%"
        )
    with col3:
        st.metric("Confidence", f"{max(probabilities)*100:.1f}%")

def render_charts(df: pd.DataFrame, df_feat: pd.DataFrame):
    """Visualize data and indicators"""
    tab1, tab2, tab3 = st.tabs(["Price History", "Technical Indicators", "Backtest Results"])
    
    with tab1:
        st.line_chart(df['Close'], use_container_width=True)
    
    with tab2:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Price and SMAs
        ax1.plot(df_feat.index, df_feat['Close'], label='Price', color='blue')
        ax1.plot(df_feat.index, df_feat['SMA_20'], label='SMA 20', color='orange')
        ax1.plot(df_feat.index, df_feat['SMA_50'], label='SMA 50', color='green')
        ax1.set_title('Price and Moving Averages')
        ax1.legend()
        
        # MACD
        ax2.plot(df_feat.index, df_feat['MACD'], label='MACD', color='blue')
        ax2.plot(df_feat.index, df_feat['MACD_Signal'], label='Signal', color='red')
        ax2.bar(df_feat.index, df_feat['MACD_Hist'], label='Histogram', color='gray')
        ax2.axhline(0, color='black', linestyle='--')
        ax2.set_title('MACD Indicators')
        ax2.legend()
        
        st.pyplot(fig)
    
    with tab3:
        st.line_chart(df_feat[['RSI']], use_container_width=True)
        st.write(f"Latest RSI: {df_feat['RSI'].iloc[-1]:.2f}")

# Main App Logic
def main():
    st.sidebar.header("Configuration")
    
    # Coin selection
    coin_mapping = {
        "Bitcoin (BTC)": "BTC-USD",
        "Ethereum (ETH)": "ETH-USD",
        "Cardano (ADA)": "ADA-USD"
    }
    coin_name = st.sidebar.selectbox("Select Cryptocurrency", list(coin_mapping.keys()))
    symbol = coin_mapping[coin_name]
    
    # Time range selection
    days_back = st.sidebar.slider("Analysis Period (days)", 30, 365, 180)
    
    if st.sidebar.button("Run Analysis", type="primary"):
        with st.spinner(f"Analyzing {coin_name} data..."):
            # Data pipeline
            df = fetch_data(symbol, days_back)
            if df.empty:
                st.error("No data available for analysis")
                return
                
            model = load_model(symbol)
            if not model:
                return
                
            df_feat = add_features(df)
            
            # Prediction
            features = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'SMA_20', 'SMA_50', 'Momentum']
            X_latest = df_feat[features].iloc[[-1]]
            
            prediction = model.predict(X_latest)[0]
            probabilities = model.predict_proba(X_latest)[0]
            
            # Display results
            st.header(f"{coin_name} Analysis Results")
            render_metrics(df['Close'].iloc[-1], prediction, probabilities)
            
            st.divider()
            render_charts(df, df_feat)
            
            # Backtest results
            st.subheader("Model Performance")
            df_feat['Prediction'] = model.predict(df_feat[features])
            df_feat['Actual'] = (df_feat['Close'].shift(-1) > df_feat['Close']).astype(int)
            accuracy = (df_feat['Prediction'] == df_feat['Actual']).mean()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Historical Accuracy", f"{accuracy*100:.1f}%")
            with col2:
                st.metric("Recent Correct Predictions", 
                          f"{(df_feat['Prediction'][-10:] == df_feat['Actual'][-10:]).mean()*100:.1f}%",
                          "Last 10 days")

if __name__ == "__main__":
    main()
