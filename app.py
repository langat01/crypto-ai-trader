import streamlit as st
import base64
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import plotly.graph_objs as go

# ─────────────────────────────────────────────────────────────────────────────
# 1) PAGE CONFIG: must come before any other st.* calls
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dragon trading AI",
    page_icon="🐉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# 2) Inject a full‐page, transparent dragon background (CSS)
#    (No caching here so updates to dragon.png are picked up immediately)
# ─────────────────────────────────────────────────────────────────────────────
def get_base64_image(path: str) -> str:
    """
    Read a local image file (dragon.png) and return it as a Base64‐encoded string.
    """
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Make sure dragon.png is in the same folder as this script
dragon_b64 = get_base64_image("dragon.png")

page_bg_style = f"""
<style>
.stApp {{
  background-image: url("data:image/png;base64,{dragon_b64}");
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
}}
</style>
"""
st.markdown(page_bg_style, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3) MAIN HEADING below CSS
# ─────────────────────────────────────────────────────────────────────────────
st.title("Dragon trading AI")

# ─────────────────────────────────────────────────────────────────────────────
# 4) SIDEBAR & HEADER
# ─────────────────────────────────────────────────────────────────────────────
# Sidebar header with a small dragon icon
st.sidebar.markdown(
    """
    <div style="text-align:center">
      <h1 style="margin-bottom:0">🐉</h1>
      <h3 style="margin-top:5px">Dragon AI</h3>
    </div>
    <hr style="border:1px solid #444" />
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Configuration")
cryptos = {
    "Bitcoin (BTC)": "BTC",
    "Ethereum (ETH)": "ETH",
    "Binance Coin (BNB)": "BNB",
    "Cardano (ADA)": "ADA",
    "Solana (SOL)": "SOL"
}
selected_crypto_name = st.sidebar.selectbox("Cryptocurrency", list(cryptos.keys()))
selected_crypto_symbol = cryptos[selected_crypto_name]

days = st.sidebar.slider(
    "Historical days for model training",
    min_value=180, max_value=1000, value=730, step=30
)

st.sidebar.markdown("---")
st.sidebar.write("By _Your Name_")

# ─────────────────────────────────────────────────────────────────────────────
# 5) UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_data_cryptocompare(symbol, limit=730):
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    params = {"fsym": symbol, "tsym": "USD", "limit": limit, "aggregate": 1}
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    if data["Response"] != "Success":
        raise ValueError(f"CryptoCompare API error: {data.get('Message', 'Unknown error')}")
    raw = data["Data"]["Data"]
    df = pd.DataFrame(raw)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df.rename(
        columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volumefrom": "volume",
        },
        inplace=True,
    )
    return df[["open", "high", "low", "close", "volume"]].astype(float)

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def create_features(df):
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["sma_5"] = df["close"].rolling(window=5).mean()
    df["sma_10"] = df["close"].rolling(window=10).mean()
    df["rsi_14"] = compute_rsi(df["close"], 14)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df.dropna(inplace=True)
    return df

@st.cache_data(show_spinner=False)
def train_model(df):
    features = ["return", "sma_5", "sma_10", "rsi_14"]
    X = df[features]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=False)
    return model, acc, report, X_test, y_test

def fetch_realtime_price(symbol):
    url = "https://min-api.cryptocompare.com/data/price"
    params = {"fsym": symbol, "tsyms": "USD"}
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        return r.json().get("USD", None)
    except Exception as e:
        st.warning(f"Failed to fetch live price: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 6) TABS: Model & Predictions vs Live Monitor
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📉 Model & Predictions", "📈 Live Price Monitor"])

# ─────────────────────────────────────────────────────────────────────────────
# 7) TAB 1: MODEL & PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("## 🧠 Train & Evaluate Model")
    try:
        with st.spinner("Fetching historical data..."):
            df_hist = fetch_data_cryptocompare(selected_crypto_symbol, limit=days)
        if df_hist.empty:
            st.error("No historical data. Try a different crypto or fewer days.")
        else:
            df_features = create_features(df_hist)
            with st.spinner("Training model..."):
                model, acc, report, X_test, y_test = train_model(df_features)

            st.markdown(f"### Model Accuracy on Test Set: **{acc:.2%}**")
            st.text_area("Classification Report", report, height=150)

            # Daily prediction
            st.markdown("### 🔮 Daily Prediction")
            pred_daily = model.predict(X_test.tail(1))[0]
            col_a, col_b, col_c = st.columns(3)
            if pred_daily == 1:
                col_a.metric("Tomorrow (24 h)", "⬆️ Up", delta=None)
            else:
                col_a.metric("Tomorrow (24 h)", "⬇️ Down", delta=None)

            # Short-Term (10 min / 1 hr)
            st.markdown("### ⏱️ Short‐Term Predictions")
            try:
                url_min = "https://min-api.cryptocompare.com/data/v2/histominute"
                params_min = {
                    "fsym": selected_crypto_symbol,
                    "tsym": "USD",
                    "limit": 70,
                    "aggregate": 1,
                }
                r_min = requests.get(url_min, params=params_min)
                r_min.raise_for_status()
                data_min = r_min.json()
                if data_min["Response"] != "Success":
                    raise ValueError("Minute-level data fetch failed.")
                raw_min = data_min["Data"]["Data"]

                df_min = pd.DataFrame(raw_min)
                df_min["time"] = pd.to_datetime(df_min["time"], unit="s")
                df_min.set_index("time", inplace=True)
                df_min.rename(
                    columns={
                        "open": "open",
                        "high": "high",
                        "low": "low",
                        "close": "close",
                        "volumefrom": "volume",
                    },
                    inplace=True,
                )
                df_min = df_min[["open", "high", "low", "close", "volume"]].astype(float)

                df_min["return"] = df_min["close"].pct_change()
                df_min["sma_5"] = df_min["close"].rolling(window=5).mean()
                df_min["sma_10"] = df_min["close"].rolling(window=10).mean()
                delta = df_min["close"].diff()
                gain = delta.clip(lower=0).rolling(window=14).mean()
                loss = -delta.clip(upper=0).rolling(window=14).mean()
                rs = gain / loss
                df_min["rsi_14"] = 100 - (100 / (1 + rs))
                df_min.dropna(inplace=True)

                X_short = df_min[["return", "sma_5", "sma_10", "rsi_14"]].iloc[-1].values.reshape(
                    1, -1
                )
                pred_10min = model.predict(X_short)[0]
                pred_1hr = model.predict(X_short)[0]

                if pred_10min == 1:
                    col_b.metric("Next 10 min", "⬆️ Up")
                else:
                    col_b.metric("Next 10 min", "⬇️ Down")

                if pred_1hr == 1:
                    col_c.metric("Next 1 hr", "⬆️ Up")
                else:
                    col_c.metric("Next 1 hr", "⬇️ Down")

            except Exception as e:
                st.warning(f"Short-term fetch failed: {e}")

            # Historical 30-day candlestick
            st.markdown("### 📊 Historical 30 Day Candlestick")
            df_30 = df_hist.tail(30)
            fig_hist = go.Figure(
                data=[
                    go.Candlestick(
                        x=df_30.index,
                        open=df_30["open"],
                        high=df_30["high"],
                        low=df_30["low"],
                        close=df_30["close"],
                        increasing_line_color="lightgreen",
                        decreasing_line_color="salmon",
                    )
                ]
            )
            fig_hist.update_layout(
                template="plotly_dark",
                margin=dict(l=0, r=0, t=30, b=0),
                yaxis_title="Price (USD)",
                xaxis_title="Date",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    except Exception as e:
        st.error(f"Error in Model & Predictions: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 7) TAB 2: LIVE PRICE MONITOR
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("## ⏲️ Live Price Monitoring")
    run_live = st.checkbox("Enable Live Price Updates", value=False)
    price_threshold_up = st.number_input(
        "Alert if price > … USD", min_value=0.0, value=0.0, step=1.0
    )
    price_threshold_down = st.number_input(
        "Alert if price < … USD", min_value=0.0, value=0.0, step=1.0
    )

    price_placeholder = st.empty()
    chart_placeholder = st.empty()
    alert_placeholder = st.empty()

    if run_live:
        prices = []
        times = []

        for _ in range(60 * 5):
            price = fetch_realtime_price(selected_crypto_symbol)
            if price is not None:
                current_time = datetime.now()
                prices.append(price)
                times.append(current_time)

                price_placeholder.markdown(
                    f"<div style='font-size:18px'>"
                    f"<b>Current {selected_crypto_name}:</b> ${price:,.2f} USD  "
                    f"<span style='color:gray;font-size:14px'>({current_time.strftime('%H:%M:%S')})</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                if price_threshold_up > 0 and price > price_threshold_up:
                    alert_placeholder.success(
                        f"🚀 Price Alert: {selected_crypto_name} > ${price_threshold_up:,.2f}"
                    )
                elif price_threshold_down > 0 and price < price_threshold_down:
                    alert_placeholder.error(
                        f"⚠ Price Alert: {selected_crypto_name} < ${price_threshold_down:,.2f}"
                    )
                else:
                    alert_placeholder.empty()

                fig_live = go.Figure()
                fig_live.add_trace(
                    go.Scatter(
                        x=times,
                        y=prices,
                        mode="lines+markers",
                        line=dict(color="cyan", width=2),
                        marker=dict(size=4, color="cyan"),
                    )
                )
                fig_live.update_layout(
                    template="plotly_dark",
                    margin=dict(l=0, r=0, t=30, b=0),
                    xaxis_title="Time",
                    yaxis_title="Price (USD)",
                    height=400,
                )
                chart_placeholder.plotly_chart(fig_live, use_container_width=True)

            else:
                price_placeholder.warning("Unable to fetch live price.")

            time.sleep(1)

    else:
        st.info("Toggle the checkbox above to start live monitoring.")

# ─────────────────────────────────────────────────────────────────────────────
# 8) FOOTER SPACING
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
