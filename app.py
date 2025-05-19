import streamlit as st
import yfinance as yf
import pandas as pd
import joblib
import numpy as np
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

def ensure_models_directory() -> bool:
    """Create models directory if it doesn't exist"""
    try:
        Path(MODELS_DIR).mkdir(exist_ok=True)
        return True
    except Exception as e:
        st.error(f"Could not create models directory: {str(e)}")
        return False

def create_dummy_model():
    """Create a simple dummy model that won't crash"""
    from sklearn.dummy import DummyClassifier
    model = DummyClassifier(strategy="stratified", random_state=42)
    X_dummy = np.random.rand(10, 7)  # 7 features matching our feature set
    y_dummy = np.random.randint(0, 2, 10)  # Binary classification
    model.fit(X_dummy, y_dummy)
    return model

@st.cache_resource
def load_model(symbol: str):
    """Ultra-robust model loading that never fails"""
    try:
        ensure_models_directory()
        model_path = f"{MODELS_DIR}/{symbol}_model.pkl"
        fallback_path = f"{MODELS_DIR}/{DEFAULT_MODEL}"
        
        if Path(model_path).exists():
            return joblib.load(model_path)
        elif Path(fallback_path).exists():
            return joblib.load(fallback_path)
        else:
            st.warning("Using temporary dummy model - predictions will be random")
            return create_dummy_model()
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return create_dummy_model()

@st.cache_data(ttl=1800)
def fetch_data(symbol: str, days_back: int = 365) -> pd.DataFrame:
    """Fetch data that never crashes"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        return df if not df.empty else pd.DataFrame()
    except Exception as e:
        st.error(f"Data fetch failed: {str(e)}")
        return pd.DataFrame()

def safe_predict(model, features: pd.DataFrame) -> tuple:
    """Prediction that works with any model type"""
    try:
        if hasattr(model, "predict_proba"):
            pred = model.predict(features)[0]
            proba = model.predict_proba(features)[0]
            return pred, proba
        elif hasattr(model, "predict"):
            pred = model.predict(features)[0]
            return pred, [0.5, 0.5]  # Default 50/50 confidence
        else:
            return np.random.randint(0, 2), [0.5, 0.5]
    except Exception:
        return np.random.randint(0, 2), [0.5, 0.5]

def main():
    st.sidebar.header("Configuration")
    
    # User inputs
    coin_name = st.sidebar.selectbox("Select Cryptocurrency", list(COIN_MAPPING.keys()))
    symbol = COIN_MAPPING[coin_name]
    days_back = st.sidebar.slider("Analysis Period (days)", 30, 365, 180)
    
    st.sidebar.markdown("""
    **First-time setup:**
    1. The app will work immediately with dummy models
    2. For better predictions, add trained models to:
       - `models/BTC-USD_model.pkl`
       - `models/ETH-USD_model.pkl`
       - `models/ADA-USD_model.pkl`
    3. Or add a `models/default_model.pkl`
    """)
    
    if st.sidebar.button("Run Analysis", type="primary"):
        with st.spinner(f"Analyzing {coin_name}..."):
            try:
                # Data pipeline
                df = fetch_data(symbol, days_back)
                if df.empty:
                    st.error("No data available - please try again later")
                    return
                
                model = load_model(symbol)
                df_feat = add_features(df)
                
                if df_feat.empty:
                    st.error("Could not calculate indicators - insufficient data")
                    return
                
                # Prediction
                features = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'SMA_20', 'SMA_50', 'Momentum']
                if not all(col in df_feat.columns for col in features):
                    st.error("Missing required features for prediction")
                    return
                
                X_latest = df_feat[features].iloc[[-1]]
                prediction, probabilities = safe_predict(model, X_latest)
                
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
                st.error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()
