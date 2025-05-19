import pandas as pd

def compute_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI)."""
    delta = data['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = -delta.clip(upper=0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(data: pd.DataFrame):
    """Compute MACD, Signal Line, and MACD Histogram."""
    ema12 = data['Close'].ewm(span=12).mean()
    ema26 = data['Close'].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    hist = macd - signal
    return macd, signal, hist


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the dataset."""
    data = data.copy()
    
    # RSI
    data['RSI'] = compute_rsi(data)
    
    # MACD
    macd, signal, hist = compute_macd(data)
    data['MACD'] = macd
    data['MACD_Signal'] = signal
    data['MACD_Hist'] = hist
    
    # Simple Moving Averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # Momentum (Price change over 5 days)
    data['Momentum'] = data['Close'] - data['Close'].shift(5)

    # Drop rows with NaN (from rolling/ewm ops)
    data.dropna(inplace=True)
    return data


# Optional: Run as script for testing
if __name__ == "__main__":
    import yfinance as yf
    df = yf.download('BTC-USD', start='2023-01-01', end='2025-05-18')
    df_feat = add_features(df)
    print(df_feat.tail())
