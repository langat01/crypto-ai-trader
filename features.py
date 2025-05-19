import pandas as pd
import numpy as np
from typing import Tuple

def safe_convert_to_series(data) -> pd.Series:
    """Convert any input to pandas Series safely"""
    if isinstance(data, pd.Series):
        return data
    if isinstance(data, pd.DataFrame):
        return data.iloc[:, 0] if len(data.columns) > 0 else pd.Series(dtype=float)
    if isinstance(data, (list, tuple, np.ndarray)):
        return pd.Series(data)
    return pd.Series(dtype=float)

def compute_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """Safe RSI calculation that never fails"""
    try:
        close = safe_convert_to_series(data.get('Close', []))
        if len(close) < window:
            return pd.Series(np.nan, index=close.index)
        
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    except Exception:
        return pd.Series(dtype=float)

def compute_macd(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Safe MACD calculation that never fails"""
    try:
        close = safe_convert_to_series(data.get('Close', []))
        if len(close) < 26:  # Minimum required for MACD
            empty = pd.Series(dtype=float, index=close.index)
            return empty, empty, empty
            
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal, macd - signal
    except Exception:
        empty = pd.Series(dtype=float)
        return empty, empty, empty

def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Completely safe feature engineering"""
    try:
        if not isinstance(data, pd.DataFrame) or data.empty:
            return pd.DataFrame()
            
        df = data.copy()
        df['Close'] = pd.to_numeric(df.get('Close', np.nan), errors='coerce')
        df = df.dropna(subset=['Close'])
        
        if df.empty:
            return pd.DataFrame()
            
        # Calculate indicators
        df['RSI'] = compute_rsi(df)
        macd, signal, hist = compute_macd(df)
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Hist'] = hist
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['Momentum'] = df['Close'].pct_change(5)
        
        return df.dropna()
    except Exception:
        return pd.DataFrame()
