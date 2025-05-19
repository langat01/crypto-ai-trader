import pandas as pd

def compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.clip(lower=0)).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(data):
    ema12 = data['Close'].ewm(span=12).mean()
    ema26 = data['Close'].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    hist = macd - signal
    return macd, signal, hist

def add_features(df):
    df = df.copy()
    df['RSI'] = compute_rsi(df)
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['Momentum'] = df['Close'] - df['Close'].shift(5)
    df['Returns'] = df['Close'].pct_change()
    macd, signal, hist = compute_macd(df)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = macd, signal, hist
    df.dropna(inplace=True)
    return df
