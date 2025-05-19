import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD

def compute_rsi(data, window=14):
    """
    Compute the Relative Strength Index (RSI).

    Parameters:
    - data: pd.Series or pd.DataFrame['Close']
    - window: int, the window length to calculate RSI

    Returns:
    - pd.Series with RSI values
    """
    return RSIIndicator(close=data['Close'], window=window).rsi()

def compute_macd(data, span_short=12, span_long=26, span_signal=9):
    """
    Compute MACD, Signal line and MACD Histogram using TA-Lib.

    Parameters:
    - data: pd.DataFrame with 'Close' column
    - span_short: short EMA period
    - span_long: long EMA period
    - span_signal: signal EMA period

    Returns:
    - macd: pd.Series
    - signal: pd.Series
    - histogram: pd.Series
    """
    macd_indicator = MACD(close=data['Close'],
                          window_slow=span_long,
                          window_fast=span_short,
                          window_sign=span_signal)
    return macd_indicator.macd(), macd_indicator.macd_signal(), macd_indicator.macd_diff()

def add_technical_indicators(df):
    """
    Add all technical indicators (RSI, MACD, SMAs, Momentum) to a price DataFrame.

    Parameters:
    - df: pd.DataFrame with a 'Close' column

    Returns:
    - df: pd.DataFrame with new feature columns
    """
    if df.empty or 'Close' not in df.columns:
        raise ValueError("DataFrame is empty or missing 'Close' column")

    df = df.copy()
    df = df.dropna(subset=['Close'])  # Drop rows with NaNs in 'Close'

    # Compute indicators
    df['RSI'] = compute_rsi(df)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = compute_macd(df)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Momentum'] = df['Close'] - df['Close'].shift(5)

    df.dropna(inplace=True)  # Drop rows with any remaining NaNs
    return df

def prepare_training_data(df):
    """
    Prepares features and binary target variable for ML model training.

    Target is 1 if next day's closing price is higher than today, else 0.

    Parameters:
    - df: pd.DataFrame with price history

    Returns:
    - X: pd.DataFrame of features
    - y: pd.Series of target
    """
    df = add_technical_indicators(df)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    feature_cols = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'SMA_20', 'SMA_50', 'Momentum']
    X = df[feature_cols]
    y = df['Target']

    return X, y

# Optional standalone test
if __name__ == "__main__":
    import yfinance as yf

    # Download historical price data
    df = yf.download('BTC-USD', start='2023-01-01', end='2025-05-18')

    if df.empty:
        print("Failed to fetch BTC-USD data.")
    else:
        X, y = prepare_training_data(df)
        print("Features sample:")
        print(X.tail())
        print("\nTarget sample:")
        print(y.tail())
