import pandas as pd

def compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(data, span_short=12, span_long=26, span_signal=9):
    ema_short = data['Close'].ewm(span=span_short, adjust=False).mean()
    ema_long = data['Close'].ewm(span=span_long, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=span_signal, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def add_features(data):
    data = data.copy()
    data['RSI'] = compute_rsi(data)
    macd, signal, hist = compute_macd(data)
    data['MACD'] = macd
    data['MACD_Signal'] = signal
    data['MACD_Hist'] = hist
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['Momentum'] = data['Close'] - data['Close'].shift(5)
    data.dropna(inplace=True)
    return data

# Optional: test features.py alone
if __name__ == "__main__":
    import yfinance as yf
    df = yf.download('BTC-USD', start='2023-01-01', end='2025-05-18')
    df_feat = add_features(df)
    print(df_feat.tail())
