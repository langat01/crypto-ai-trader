
import pandas as pd
import numpy as np
from typing import Tuple


def safe_convert_to_series(data) -> pd.Series:
    """Safely convert input to pandas Series."""
    if isinstance(data, pd.Series):
        return data
    if isinstance(data, pd.DataFrame):
        return data.iloc[:, 0] if not data.empty else pd.Series(dtype=float)
    if isinstance(data, (list, tuple, np.ndarray)):
        return pd.Series(data)
    return pd.Series(dtype=float)


def compute_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """Compute RSI using exponential moving average for stability."""
    try:
        close = safe_convert_to_series(data.get('Close', []))
        if len(close) < window:
            return pd.Series(np.nan, index=close.index)

        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(span=window, adjust=False).mean()
        avg_loss = loss.ewm(span=window, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception:
        return pd.Series(dtype=float)


def compute_macd(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD, Signal Line, and Histogram."""
    try:
        close = safe_convert_to_series(data.get('Close', []))
        if len(close) < 26:
            empty = pd.Series(np.nan, index=close.index)
            return empty, empty, empty

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal

        return macd, signal, hist
    except Exception:
        empty = pd.Series(dtype=float)
        return empty, empty, empty


def compute_bollinger_bands(close: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands: Upper, Middle, Lower."""
    try:
        middle = close.rolling(window).mean()
        std = close.rolling(window).std()
        upper = middle + 2 * std
        lower = middle - 2 * std
        return upper, middle, lower
    except Exception:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)


def add_features(data: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Apply safe and robust feature engineering to market data."""
    try:
        if not isinstance(data, pd.DataFrame) or data.empty:
            return pd.DataFrame()

        df = data.copy()
        df['Close'] = pd.to_numeric(df.get('Close', np.nan), errors='coerce')
        df = df.dropna(subset=['Close'])

        if df.empty:
            return pd.DataFrame()

        # Indicators
        df['RSI'] = compute_rsi(df)
        macd, signal, hist = compute_macd(df)
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Hist'] = hist
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['Momentum'] = df['Close'].pct_change(periods=5)

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = compute_bollinger_bands(df['Close'])
        df['BB_Upper'] = bb_upper
        df['BB_Middle'] = bb_middle
        df['BB_Lower'] = bb_lower

        return df.dropna()
    except Exception as e:
        if verbose:
            print(f"[add_features] Error: {e}")
        return pd.DataFrame()
