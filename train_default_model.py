import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Parameters
symbol = "BTCUSDT"
interval = "1m"
dataset_path = f"datasets/{symbol}_{interval}_xgboost_dataset.csv"
model_dir = "models"
model_filename = f"{model_dir}/{symbol}_{interval}_xgboost_model.pkl"

# Ensure model directory exists
os.makedirs(model_dir, exist_ok=True)

# Load dataset
df = pd.read_csv(dataset_path)

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Feature engineering
df['return'] = df['close'].pct_change()
df['sma'] = df['close'].rolling(window=5).mean()
df['momentum'] = df['close'] - df['close'].shift(4)
df['volatility'] = df['return'].rolling(window=5).std()

# Target: 1 if next close > current close else 0
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

# Drop rows with NaN values
df = df.dropna()

# Features and target
features = ['return', 'sma', 'momentum', 'volatility']
X = df[features]
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Parameters for XGBoost
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 5,
    "eta": 0.1,
    "verbosity": 1,
    "seed": 42
}

# Train model
model = xgb.train(params, dtrain, num_boost_round=100)

# Predict on test set
y_pred_prob = model.predict(dtest)
y_pred = (y_pred_prob > 0.5).astype(int)

# Metrics
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")
