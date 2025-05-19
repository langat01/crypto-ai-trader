import pandas as pd
import numpy as np
from typing import Tuple, Union

def compute_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI) with input validation."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame")
    if 'Close' not in data.columns:
        raise ValueError("Data must contain 'Close' column")
    if len(data) < window:
        raise ValueError(f"Need at least {window} data points for RSI calculation")

    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).abs()
    loss = delta.where(delta < 0, 0).abs()
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(
    data: pd.DataFrame, 
    span_short: int = 12, 
    span_long: int = 26, 
    span_signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD, Signal Line, and Histogram with validation."""
    required_length = max(span_short, span_long, span_signal)
    if len(data) < required_length:
        raise ValueError(f"Need at least {required_length} data points for MACD")
    
    ema_short = data['Close'].ewm(span=span_short, adjust=False).mean()
    ema_long = data['Close'].ewm(span=span_long, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=span_signal, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def add_features(
    data: pd.DataFrame,
    rsi_window: int = 14,
    macd_params: Tuple[int, int, int] = (12, 26, 9),
    sma_windows: Tuple[int, ...] = (20, 50),
    momentum_window: int = 5,
    volatility_window: int = 10
) -> pd.DataFrame:
    """
    Enhanced feature engineering with configurable parameters.
    Returns DataFrame with technical indicators and cleaned data.
    """
    # Input validation
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if 'Close' not in data.columns:
        raise ValueError("Input must contain 'Close' column")
    
    data = data.copy()
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data = data[data['Close'].notna()]
    
    # Calculate all indicators
    data['RSI'] = compute_rsi(data, window=rsi_window)
    
    macd, signal, hist = compute_macd(
        data,
        span_short=macd_params[0],
        span_long=macd_params[1],
        span_signal=macd_params[2]
    )
    data['MACD'] = macd
    data['MACD_Signal'] = signal
    data['MACD_Hist'] = hist
    
    for window in sma_windows:
        data[f'SMA_{window}'] = data['Close'].rolling(window=window).mean()
    
    data['Momentum'] = data['Close'].pct_change(periods=momentum_window)
    data['Volatility'] = data['Close'].pct_change().rolling(volatility_window).std()
    
    return data.dropna()
