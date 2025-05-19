import streamlit as st
import yfinance as yf
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import datetime, timedelta
from features import add_features

# Configuration
st.set_page_config(page_title="Crypto AI Trading Pro", layout="wide")
st.title("🚀 Crypto AI Trading Strategy Pro")

# Constants
MODELS_DIR = "models"
DEFAULT_MODEL = "default_model.pkl"
COIN_MAPPING = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Cardano (ADA)": "ADA-USD"
}

def ensure_models_directory():
    """Create models directory if it doesn't exist"""
    Path(MODELS_DIR).mkdir(exist_ok=True)
    if not Path(f"{MODELS_DIR}/{DEFAULT_MODEL}").exists():
        st.sidebar.warning(f"Please add model files to {MODELS_DIR}/ directory")

@st.cache_resource(ttl=3600)
def load_model(symbol: str):
    """Enhanced model loading with fallback"""
    ensure_models_directory()
    
    model_path = f"{MODELS_DIR}/{symbol}_model.pkl"
    fallback_path = f"{MODELS_DIR}/{DEFAULT_MODEL}"
    
    try:
        if Path(model_path).exists():
            return joblib.load(model_path)
        elif Path(fallback_path).exists():
            st.warning(f"Using default model for {symbol}")
            return joblib.load(fallback_path)
        else:
            st.error("No model files found")
            return None
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
        return df if not df.empty else pd.DataFrame()
    except Exception as e:
        st.error(f"Data fetch failed: {str(e)}")
        return pd.DataFrame()

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
    tab1, tab2, tab3 = st.tabs(["Price History", "Technical Indicators", "Market Conditions"])
    
    with tab1:
        st.line_chart(df['Close'], use_container_width=True)
    
    with tab2:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.plot(df_feat.index, df_feat['Close'], label='Price')
        ax1.plot(df_feat.index, df_feat['SMA_20'], label='SMA 20')
        ax1.plot(df_feat.index, df_feat['SMA_50'], label='SMA 50')
        ax1.set_title('Price and Moving Averages')
        ax1.legend()
        
        ax2.plot(df_feat.index, df_feat['MACD'], label='MACD')
        ax2.plot(df_feat.index, df_feat['MACD_Signal'], label='Signal')
        ax2.bar(df_feat.index, df_feat['MACD_Hist'], label='Histogram')
        ax2.axhline(0, color='black', linestyle='--')
        ax2.set_title('MACD Indicators')
        ax2.legend()
        
        st.pyplot(fig)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("RSI", f"{df_feat['RSI'].iloc[-1]:.1f}")
            st.line_chart(df_feat['RSI'], use_container_width=True)
        with col2:
            st.metric("Volatility", f"{df_feat['Volatility'].iloc[-1]*100:.2f}%")
            st.line_chart(df_feat['Volatility'], use_container_width=True)

def main():
    st.sidebar.header("Configuration")
    ensure_models_directory()
    
    # User inputs
    coin_name = st.sidebar.selectbox("Select Cryptocurrency", list(COIN_MAPPING.keys()))
    symbol = COIN_MAPPING[coin_name]
    days_back = st.sidebar.slider("Analysis Period (days)", 30, 365, 180)
    
    st.sidebar.markdown("""
    **First-time setup:**
    1. Create a 'models' directory
    2. Add your trained models:
       - `BTC-USD_model.pkl`
       - `ETH-USD_model.pkl`
       - `default_model.pkl` (fallback)
    """)
    
    if st.sidebar.button("Run Analysis", type="primary"):
        with st.spinner(f"Analyzing {coin_name}..."):
            # Data pipeline
            df = fetch_data(symbol, days_back)
            if df.empty:
                st.error("No data available")
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
            st.header(f"{coin_name} Analysis")
            render_metrics(df['Close'].iloc[-1], prediction, probabilities)
            
            st.divider()
            render_charts(df, df_feat)
            
            # Performance metrics
            st.subheader("Model Performance")
            df_feat['Prediction'] = model.predict(df_feat[features])
            df_feat['Actual'] = (df_feat['Close'].shift(-1) > df_feat['Close']).astype(int)
            
            col1, col2 = st.columns(2)
            with col1:
                accuracy = (df_feat['Prediction'] == df_feat['Actual']).mean()
                st.metric("Historical Accuracy", f"{accuracy*100:.1f}%")
            with col2:
                recent_accuracy = (df_feat['Prediction'][-10:] == df_feat['Actual'][-10:]).mean()
                st.metric("Recent Accuracy", f"{recent_accuracy*100:.1f}%", "Last 10 days")

if __name__ == "__main__":
    main()
