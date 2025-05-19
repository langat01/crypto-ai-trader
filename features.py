import pandas as pd
from sklearn.preprocessing import StandardScaler

def compute_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    delta = data['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = -delta.clip(upper=0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(data: pd.DataFrame):
    ema12 = data['Close'].ewm(span=12).mean()
    ema26 = data['Close'].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    hist = macd - signal
    return macd, signal, hist

def add_features(data: pd.DataFrame, scale: bool = False, include_target: bool = True) -> pd.DataFrame:
    """Add features and optionally scale and include target."""
    df = data.copy()
    
    # RSI
    df['RSI'] = compute_rsi(df)
    
    # MACD
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = compute_macd(df)

    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Bollinger Bands
    rolling_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + 2 * rolling_std
    df['BB_Lower'] = df['SMA_20'] - 2 * rolling_std

    # Momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(5)
    
    # Rate of Change
    df['ROC'] = df['Close'].pct_change(periods=5)

    # Exponential Moving Average
    df['EMA_10'] = df['Close'].ewm(span=10).mean()
    
    # Optional target for classification
    if include_target:
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Drop rows with NaNs from rolling/ewm ops
    df.dropna(inplace=True)

    # Optional scaling
    if scale:
        scaler = StandardScaler()
        features = [col for col in df.columns if col not in ['Target']]
        df[features] = scaler.fit_transform(df[features])

    return df

# Optional: Run for testing
if __name__ == "__main__":
    import yfinance as yf
    df = yf.download('BTC-USD', start='2023-01-01', end='2025-05-18')
    df_feat = add_features(df)
    print(df_feat.tail())
