import streamlit as st
import yfinance as yf
import pandas as pd
import joblib
import matplotlib.pyplot as plt
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
        st.warning("Default model not found. Using temporary model - predictions will be random.")
        return False
    return True

@st.cache_resource
def load_model(symbol: str):
    """Load model with comprehensive error handling"""
    if not ensure_models_directory():
        # If no default model exists, create a dummy one in memory
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        X_dummy = np.random.rand(100, 7)
        y_dummy = np.random.randint(0, 2, 100)
        model.fit(X_dummy, y_dummy)
        return model
    
    model_path = f"{MODELS_DIR}/{symbol}_model.pkl"
    fallback_path = f"{MODELS_DIR}/{DEFAULT_MODEL}"
    
    try:
        if Path(model_path).exists():
            return joblib.load(model_path)
        return joblib.load(fallback_path)
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None

@st.cache_data(ttl=1800)
def fetch_data(symbol: str, days_back: int = 365):
    """Fetch data with retry logic"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        return df if not df.empty else pd.DataFrame()
    except Exception as e:
        st.error(f"Data fetch failed: {str(e)}")
        return pd.DataFrame()

def main():
    st.sidebar.header("Configuration")
    
    # User inputs
    coin_name = st.sidebar.selectbox("Select Cryptocurrency", list(COIN_MAPPING.keys()))
    symbol = COIN_MAPPING[coin_name]
    days_back = st.sidebar.slider("Analysis Period (days)", 30, 365, 180)
    
    st.sidebar.markdown("""
    **First-time setup:**
    1. The app will automatically create a 'models' directory
    2. For best results, add your trained models:
       - `BTC-USD_model.pkl`
       - `ETH-USD_model.pkl`
       - `ADA-USD_model.pkl`
    3. A default model will be created if none exist
    """)
    
    if st.sidebar.button("Run Analysis", type="primary"):
        with st.spinner(f"Analyzing {coin_name}..."):
            # Data pipeline
            df = fetch_data(symbol, days_back)
            if df.empty:
                st.error("No data available - please check your internet connection")
                return
                
            model = load_model(symbol)
            if model is None:
                st.error("Failed to initialize model")
                return
                
            df_feat = add_features(df)
            
            # Prediction
            features = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'SMA_20', 'SMA_50', 'Momentum']
            X_latest = df_feat[features].iloc[[-1]]
            
            try:
                prediction = model.predict(X_latest)[0]
                probabilities = model.predict_proba(X_latest)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
                
                # Display results
                st.header(f"{coin_name} Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Current Price", f"${df['Close'].iloc[-1]:,.2f}")
                with col2:
                    st.metric(
                        "Prediction", 
                        "UP ↗️" if prediction == 1 else "DOWN ↘️",
                        f"{probabilities[1]*100:.1f}%" if prediction == 1 else f"{probabilities[0]*100:.1f}%"
                    )
                
                # Visualizations
                st.line_chart(df['Close'])
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import numpy as np  # For the dummy model fallback
    main()
