# feature_engineering.py

import pandas as pd
import numpy as np

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_bollinger_bands(series, window=20):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper_band = sma + (2 * std)
    lower_band = sma - (2 * std)
    return upper_band, lower_band

def compute_momentum(series, window=10):
    return series.diff(periods=window)

def create_features(df):
    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['rsi_14'] = compute_rsi(df['close'], 14)
    df['momentum_10'] = compute_momentum(df['close'], 10)
    df['macd'], df['macd_signal'] = compute_macd(df['close'])
    df['bb_upper'], df['bb_lower'] = compute_bollinger_bands(df['close'])
    
    # Target: 1 if price will go up next day, else 0
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)
    return df
