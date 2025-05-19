import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from features import add_features

# --- Coins to train models for ---
coins = {
    "BTC-USD": "btc_model.pkl",
    "ETH-USD": "eth_model.pkl",
    "ADA-USD": "ada_model.pkl"
}

# --- Feature Columns ---
FEATURES = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'SMA_20', 'SMA_50', 'Momentum']

# --- Model Training Function ---
def train_and_save_model(ticker, filename):
    print(f"\n📥 Fetching data for {ticker}")
    df = yf.download(ticker, start="2021-01-01")
    if df.empty:
        print(f"⚠️ No data for {ticker}")
        return

    df_feat = add_features(df)
    df_feat['Target'] = (df_feat['Close'].shift(-1) > df_feat['Close']).astype(int)
    df_feat.dropna(inplace=True)

    X = df_feat[FEATURES]
    y = df_feat['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Trained {ticker} model with accuracy: {acc*100:.2f}%")

    joblib.dump(model, filename)
    print(f"💾 Saved model as {filename}")

# --- Loop through coins ---
for ticker, filename in coins.items():
    train_and_save_model(ticker, filename)
