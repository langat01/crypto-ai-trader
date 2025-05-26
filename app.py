import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Fetch historical price data from CryptoCompare
def fetch_crypto_data(symbol, limit=730):
    url = f'https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym=USD&limit={limit}'
    response = requests.get(url)
    data = response.json()
    if data['Response'] != 'Success':
        raise Exception(f"Failed to fetch data: {data.get('Message', 'Unknown error')}")
    df = pd.DataFrame(data['Data']['Data'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df[['open', 'high', 'low', 'close', 'volumeto', 'volumefrom']]

# Calculate technical indicators
def add_technical_indicators(df):
    df = df.copy()
    # Simple Moving Averages
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_30'] = df['close'].rolling(window=30).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    df['RSI'] = 100 - (100 / (1 + RS))
    
    # Momentum
    df['Momentum'] = df['close'] - df['close'].shift(10)
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    
    # Volume change
    df['Vol_Change'] = df['volumeto'].pct_change()
    
    # Drop rows with NaN
    df.dropna(inplace=True)
    return df

# Prepare labels: 1 if next day's close > today close, else 0
def prepare_labels(df):
    df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)
    return df

# Prepare features and labels for ML
def prepare_features_labels(df):
    X = df[['SMA_10', 'SMA_30', 'RSI', 'Momentum', 'MACD', 'Vol_Change']]
    y = df['Target']
    return X, y

# Train/test split and model training function
def train_model(X, y, model_type='random_forest'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'xgboost':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    else:
        raise ValueError("Invalid model_type, choose 'random_forest' or 'xgboost'")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return model, accuracy, report, X_test, y_test, y_pred

# Plot feature importance for the model
def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print("Model does not support feature importances")
        return
    
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
    plt.title('Feature Importances')
    plt.show()

# Example usage
if __name__ == "__main__":
    symbol = 'BTC'  # Change to any symbol supported by CryptoCompare, e.g. ETH, SOL
    print(f"Fetching data for {symbol}...")
    df = fetch_crypto_data(symbol)
    df = add_technical_indicators(df)
    df = prepare_labels(df)
    X, y = prepare_features_labels(df)
    
    print("Training model...")
    model, accuracy, report, X_test, y_test, y_pred = train_model(X, y, model_type='xgboost')
    
    print(f"Model accuracy on test set: {accuracy*100:.2f}%")
    print("Classification report:")
    print(report)
    
    plot_feature_importance(model, X.columns)
