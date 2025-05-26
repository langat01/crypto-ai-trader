# === data_loader.py ===
import requests
import pandas as pd

def fetch_cryptocompare_data(symbol="BTC", limit=730):
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    params = {
        "fsym": symbol,
        "tsym": "USD",
        "limit": limit,
        "aggregate": 1
    }
    r = requests.get(url, params=params)
    data = r.json()
    if data['Response'] != 'Success':
        raise ValueError(f"CryptoCompare error: {data.get('Message')}")
    df = pd.DataFrame(data['Data']['Data'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close',
                       'volumefrom': 'volume'}, inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    return df.astype(float)
