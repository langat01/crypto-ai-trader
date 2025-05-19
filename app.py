import streamlit as st
import yfinance as yf
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import plotly.graph_objects as go
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice

# Configuration
st.set_page_config(
    page_title="🚀 Advanced Crypto AI Trading",
    layout="wide",
    page_icon="📈"
)

# Constants
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
CRYPTO_SYMBOLS = {
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum', 
    'BNB-USD': 'Binance Coin',
    'SOL-USD': 'Solana',
    'XRP-USD': 'Ripple',
    'ADA-USD': 'Cardano',
    'DOGE-USD': 'Dogecoin',
    'DOT-USD': 'Polkadot'
}

# Technical Indicators Calculator
def calculate_technical_indicators(df):
    """Calculate various technical indicators for the given DataFrame"""
    # Momentum Indicators
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    
    # Trend Indicators
    macd = MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Volatility Indicators
    bb = BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Width'] = bb.bollinger_wband()
    
    # Volume Indicators
    df['VWAP'] = VolumeWeightedAveragePrice(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        volume=df['Volume'],
        window=20
    ).volume_weighted_average_price()
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # Price Momentum
    df['Momentum'] = df['Close'].pct_change(periods=5)
    
    return df.dropna()

# Data Preparation
def prepare_training_data(df):
    """Prepare features and target variable for training"""
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    
    features = [
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Upper', 'BB_Lower', 'BB_Width',
        'VWAP', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
        'Momentum', 'Volume'
    ]
    
    X = df[features]
    y = df['Target']
    
    return X, y

# Model Management
def save_model(model, symbol):
    """Save trained model to disk"""
    model_path = os.path.join(MODEL_DIR, f"{symbol}_model.pkl")
    joblib.dump(model, model_path)
    return model_path

def load_model(symbol):
    """Load trained model from disk"""
    model_path = os.path.join(MODEL_DIR, f"{symbol}_model.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# Data Fetching with caching
@st.cache_data(ttl=3600, show_spinner="Fetching market data...")
def fetch_data(symbol, start_date, end_date):
    """Fetch historical market data"""
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            st.error("No data returned for the selected symbol and date range.")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Visualization Functions
def plot_candlestick(df):
    """Create interactive candlestick chart"""
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])
    
    fig.update_layout(
        title="Price History",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        height=600,
        template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_technical_indicators(df):
    """Plot technical indicators"""
    with st.expander("Technical Indicators"):
        tab1, tab2, tab3 = st.tabs(["RSI & MACD", "Bollinger Bands", "Moving Averages"])
        
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
            fig.add_hline(y=30, line_dash="dot", line_color="green")
            fig.add_hline(y=70, line_dash="dot", line_color="red")
            fig.update_layout(title="RSI (14 days)", height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'))
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal'))
            fig.update_layout(title="MACD (12,26,9)", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='Upper Band'))
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='Lower Band'))
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price'))
            fig.update_layout(title="Bollinger Bands (20,2)", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20'))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50'))
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price'))
            fig.update_layout(title="Moving Averages", height=300)
            st.plotly_chart(fig, use_container_width=True)

# Sidebar Configuration
def sidebar_controls():
    """Render sidebar controls and return user inputs"""
    st.sidebar.header("⚙️ Trading Parameters")
    
    crypto_symbol = st.sidebar.selectbox(
        "Select Cryptocurrency",
        options=list(CRYPTO_SYMBOLS.keys()),
        format_func=lambda x: f"{CRYPTO_SYMBOLS[x]} ({x})"
    )
    
    default_end = datetime.today()
    default_start = default_end - timedelta(days=365)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=default_start)
    with col2:
        end_date = st.date_input("End Date", value=default_end)
    
    if start_date >= end_date:
        st.sidebar.error("End date must be after start date")
        st.stop()
    
    action = st.sidebar.radio(
        "Choose Action",
        ["Run Prediction", "Retrain Model", "Model Analysis"],
        index=0
    )
    
    if action == "Retrain Model":
        st.sidebar.subheader("Model Parameters")
        n_estimators = st.sidebar.slider("Number of Trees", 50, 500, 100, 50)
        max_depth = st.sidebar.slider("Max Depth", 2, 20, 5, 1)
        test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20, 5)
        
        model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'test_size': test_size / 100,
            'random_state': 42
        }
    else:
        model_params = None
    
    return crypto_symbol, start_date, end_date, action, model_params

# Main App Logic
def main():
    """Main application logic"""
    st.title("🚀 Advanced Crypto AI Trading Strategy")
    
    # Get user inputs from sidebar
    crypto_symbol, start_date, end_date, action, model_params = sidebar_controls()
    
    # Fetch and prepare data
    with st.spinner("Processing data..."):
        df = fetch_data(crypto_symbol, start_date, end_date)
        if df is None:
            st.error("Failed to load data. Please try different parameters.")
            st.stop()
        
        df = calculate_technical_indicators(df)
    
    # Display basic info
    st.subheader(f"{CRYPTO_SYMBOLS[crypto_symbol]} ({crypto_symbol})")
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${df['Close'].iloc[-1]:,.2f}")
    col2.metric("24h Change", 
                f"${df['Close'].iloc[-1] - df['Close'].iloc[-2]:,.2f}", 
                f"{(df['Close'].iloc[-1]/df['Close'].iloc[-2]-1)*100:.2f}%")
    col3.metric("Market Cap", 
                f"${df['Close'].iloc[-1] * df['Volume'].sum() / 1e9:,.1f}B" if 'Volume' in df else "N/A")
    
    # Action handling
    if action == "Run Prediction":
        run_prediction(df, crypto_symbol)
    elif action == "Retrain Model":
        retrain_model(df, crypto_symbol, model_params)
    elif action == "Model Analysis":
        analyze_model(df, crypto_symbol)
    
    # Visualizations
    plot_candlestick(df)
    plot_technical_indicators(df)

def run_prediction(df, symbol):
    """Run prediction with existing model"""
    model = load_model(symbol)
    if model is None:
        st.warning("No trained model found. Please retrain the model first.")
        return
    
    features = [
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Upper', 'BB_Lower', 'BB_Width',
        'VWAP', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
        'Momentum', 'Volume'
    ]
    
    latest_data = df.iloc[[-1]][features]
    prediction = model.predict(latest_data)[0]
    probabilities = model.predict_proba(latest_data)[0]
    
    st.subheader("📊 Prediction Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Tomorrow's Prediction",
            value="🟢 BUY" if prediction == 1 else "🔴 SELL",
            delta=f"{probabilities[1]*100:.1f}% confidence" if prediction == 1 else f"{probabilities[0]*100:.1f}% confidence"
        )
    
    with col2:
        st.write(f"**Probability Distribution:**")
        st.write(f"UP: {probabilities[1]*100:.2f}% | DOWN: {probabilities[0]*100:.2f}%")
    
    st.progress(probabilities[1] if prediction == 1 else probabilities[0])

def retrain_model(df, symbol, params):
    """Retrain the model with current data"""
    st.subheader("🔄 Model Training")
    
    X, y = prepare_training_data(df)
    if len(X) < 100:
        st.warning(f"Limited data available ({len(X)} samples). Consider expanding date range.")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=params['test_size'],
        shuffle=False
    )
    
    with st.spinner("Training model..."):
        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            random_state=params['random_state']
        )
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save model
        model_path = save_model(model, symbol)
        
    st.success(f"✅ Model trained and saved to `{model_path}`")
    
    # Display training results
    st.subheader("📈 Training Results")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Training Samples", len(y_train))
    col2.metric("Test Samples", len(y_test))
    col3.metric("Test Accuracy", f"{accuracy*100:.2f}%")
    
    st.write("**Classification Report:**")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
    
    # Feature importance
    st.write("**Feature Importance:**")
    feature_imp = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig = go.Figure(go.Bar(
        x=feature_imp['Importance'],
        y=feature_imp['Feature'],
        orientation='h'
    ))
    fig.update_layout(height=500, title="Feature Importance")
    st.plotly_chart(fig, use_container_width=True)

def analyze_model(df, symbol):
    """Analyze model performance"""
    st.subheader("🔍 Model Analysis")
    
    model = load_model(symbol)
    if model is None:
        st.warning("No trained model found. Please train a model first.")
        return
    
    X, y = prepare_training_data(df)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    # Performance metrics
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)
    
    col1, col2 = st.columns(2)
    col1.metric("Overall Accuracy", f"{accuracy*100:.2f}%")
    col2.metric("Prediction Confidence", 
               f"{(y_proba[y==1].mean()*100 if sum(y)>0 else 0):.1f}% (UP) / " + 
               f"{(100 - y_proba[y==0].mean()*100 if sum(y)<len(y) else 0):.1f}% (DOWN)")
    
    # Detailed report
    with st.expander("Detailed Classification Report"):
        st.dataframe(pd.DataFrame(report).transpose())
    
    # Prediction distribution
    st.write("**Prediction Distribution:**")
    pred_dist = pd.DataFrame({
        'Actual': y,
        'Predicted': y_pred,
        'Probability_Up': y_proba
    })
    st.bar_chart(pred_dist['Predicted'].value_counts())
    
    # Historical predictions
    st.write("**Historical Predictions vs Actual:**")
    history_df = df[['Close']].copy()
    history_df['Prediction'] = y_pred
    history_df['Correct'] = (y_pred == y).astype(int)
    st.line_chart(history_df[['Close']])
    
    # Add markers for correct/incorrect predictions
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_df.index,
        y=history_df['Close'],
        mode='lines',
        name='Price'
    ))
    fig.add_trace(go.Scatter(
        x=history_df[history_df['Correct'] == 1].index,
        y=history_df[history_df['Correct'] == 1]['Close'],
        mode='markers',
        marker=dict(color='green', size=8),
        name='Correct Prediction'
    ))
    fig.add_trace(go.Scatter(
        x=history_df[history_df['Correct'] == 0].index,
        y=history_df[history_df['Correct'] == 0]['Close'],
        mode='markers',
        marker=dict(color='red', size=8),
        name='Incorrect Prediction'
    ))
    fig.update_layout(height=600, title="Prediction Accuracy Over Time")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
