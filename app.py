import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import lightgbm as lgb
import xgboost as xgb
import optuna
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_aggrid import AgGrid
from streamlit_option_menu import option_menu
from dotenv import load_dotenv
import os
import requests

load_dotenv()  # load environment variables if any from .env


def calculate_technical_indicators(df):
    """
    Adds RSI, MACD, MACD Signal, MACD Histogram, SMA_20, SMA_50, and Momentum columns.
    Ensures data is clean and suitable for TA calculations.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    
    if 'Close' not in df.columns:
        raise ValueError("Missing 'Close' column in the DataFrame.")

    df = df.copy()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(subset=['Close'], inplace=True)

    rsi_indicator = RSIIndicator(close=df['Close'].astype(float), window=14)
    df['RSI'] = rsi_indicator.rsi()

    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()

    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Momentum'] = df['Close'] - df['Close'].shift(5)

    df.dropna(inplace=True)
    return df


def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


def main():
    st.title("Crypto AI Trader - Technical Indicator Dashboard")

    menu = option_menu("Main Menu", ["Dashboard", "Model Training", "Prediction"],
                       icons=['bar-chart-line', 'gear', 'rocket'], menu_icon="cast", default_index=0)

    if menu == "Dashboard":
        st.header("Price and Technical Indicators")
        ticker = st.text_input("Enter ticker symbol", "BTC-USD")
        start_date = st.date_input("Start date", pd.to_datetime("2023-01-01"))
        end_date = st.date_input("End date", pd.to_datetime("today"))

        if start_date >= end_date:
            st.error("Start date must be before end date")
            return

        data_load_state = st.text("Loading data...")
        try:
            df = load_data(ticker, start_date, end_date)
            if df.empty:
                st.warning("No data found for this ticker and date range.")
                return
            df = calculate_technical_indicators(df)
            data_load_state.text("Data loaded and indicators calculated!")

            # Show price chart with RSI overlay
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index,
                                         open=df['Open'], high=df['High'],
                                         low=df['Low'], close=df['Close'],
                                         name='Price'))
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', yaxis="y2"))
            fig.update_layout(
                yaxis_title='Price',
                yaxis2=dict(title='RSI', overlaying='y', side='right'),
                legend=dict(x=0, y=1.1, orientation='h')
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show data table with AgGrid
            st.subheader("Raw Data with Indicators")
            AgGrid(df)

        except Exception as e:
            st.error(f"Error loading data or calculating indicators: {e}")

    elif menu == "Model Training":
        st.header("Train Your Model")
        st.info("This section will allow you to train machine learning models like XGBoost, LightGBM, Random Forest.")

        # For brevity, example training with RandomForest here
        ticker = st.text_input("Enter ticker symbol for training", "BTC-USD")
        start_date = st.date_input("Training start date", pd.to_datetime("2023-01-01"))
        end_date = st.date_input("Training end date", pd.to_datetime("2025-01-01"))

        if st.button("Train Model"):
            try:
                df = load_data(ticker, start_date, end_date)
                df = calculate_technical_indicators(df)
                # Prepare target variable: 1 if next day's close > today's close else 0
                df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
                df.dropna(inplace=True)

                features = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'SMA_20', 'SMA_50', 'Momentum']
                X = df[features]
                y = df['Target']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X_train, y_train)
                accuracy = clf.score(X_test, y_test)
                st.success(f"Model trained with accuracy: {accuracy:.2%}")

                # Save the model for later use
                joblib.dump(clf, "rf_model.pkl")
                st.info("Model saved as rf_model.pkl")

            except Exception as e:
                st.error(f"Error during training: {e}")

    elif menu == "Prediction":
        st.header("Make Predictions")
        st.info("Load your model and predict the next day's price movement.")

        ticker = st.text_input("Enter ticker for prediction", "BTC-USD")
        end_date = pd.to_datetime("today")
        start_date = end_date - pd.Timedelta(days=365)

        if st.button("Predict"):
            try:
                df = load_data(ticker, start_date, end_date)
                df = calculate_technical_indicators(df)
                features = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'SMA_20', 'SMA_50', 'Momentum']
                X = df[features]

                # Load the trained model
                clf = joblib.load("rf_model.pkl")

                preds = clf.predict(X)
                df['Prediction'] = preds

                st.subheader("Predictions on Historical Data")
                AgGrid(df[['Close', 'Prediction']].tail(20))

                # Show prediction distribution
                sns.countplot(x='Prediction', data=df)
                plt.title("Prediction Distribution")
                st.pyplot(plt.gcf())

            except FileNotFoundError:
                st.error("Model file not found. Please train the model first.")
            except Exception as e:
                st.error(f"Prediction error: {e}")


if __name__ == "__main__":
    main()
