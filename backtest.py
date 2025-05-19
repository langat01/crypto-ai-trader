def backtest(model, df_feat):
    df_feat = df_feat.copy()
    df_feat['Target'] = (df_feat['Close'].shift(-1) > df_feat['Close']).astype(int)
    X = df_feat[['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'SMA_20', 'SMA_50', 'Momentum']]
    y = df_feat['Target']
    df_feat['Predicted'] = model.predict(X)
    df_feat['Correct'] = (df_feat['Predicted'] == y).astype(int)
    accuracy = df_feat['Correct'].mean()
    return df_feat, accuracy
