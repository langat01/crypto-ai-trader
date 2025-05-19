import joblib
from features import add_features

def load_model(path="crypto_model.pkl"):
    return joblib.load(path)

def predict(model, df):
    df_feat = add_features(df)
    latest = df_feat.iloc[[-1]]
    X = latest[['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'SMA_20', 'SMA_50', 'Momentum']]
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    return pred, prob, df_feat
