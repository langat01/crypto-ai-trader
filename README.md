# 🐉 Dragon Trading AI

A Streamlit-based, fully transparent crypto-trading dashboard featuring historical backtesting, daily and short-term price predictions (Random Forest), and live price monitoring. Everything is styled with a dragon-themed background and a clear, modern UI.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Demo Screenshot](#demo-screenshot)  
4. [Getting Started](#getting-started)  
   - [Prerequisites](#prerequisites)  
   - [Installation](#installation)  
   - [Directory Structure](#directory-structure)  
5. [Usage](#usage)  
   - [Launch the App](#launch-the-app)  
   - [Configuration & Sidebar](#configuration--sidebar)  
   - [Tab 1: Model & Predictions](#tab-1-model--predictions)  
   - [Tab 2: Live Price Monitor](#tab-2-live-price-monitor)  
6. [Technical Details](#technical-details)  
   - [Data Sources](#data-sources)  
   - [Feature Engineering](#feature-engineering)  
   - [Model Training](#model-training)  
   - [Visualization](#visualization)  
7. [How to Customize](#how-to-customize)  
   - [Changing Background Image](#changing-background-image)  
   - [Adjusting Indicators & Model Parameters](#adjusting-indicators--model-parameters)  
   - [Adding New Cryptocurrencies](#adding-new-cryptocurrencies)  
8. [Future Enhancements](#future-enhancements)  
9. [Troubleshooting](#troubleshooting)  
10. [License](#license)  
11. [Acknowledgements](#acknowledgements)  

---

## Project Overview

**Dragon Trading AI** is a Streamlit application that provides:
- **Historical Analysis**: Fetches OHLCV data from CryptoCompare to train a Random Forest classifier.
- **Predictions**:  
  - **Daily Prediction**: Will Bitcoin (or another selected cryptocurrency) close higher than today?  
  - **Short-Term Prediction**: Next 10 minutes and next 1 hour direction (Up/Down).
- **Live Price Monitoring**: Continuously polls CryptoCompare for real-time price, renders a live-updating chart, and fires alerts when price crosses user-defined thresholds.
- **Fully Transparent UI**: Every widget container, sidebar, button, and chart is rendered with a transparent background, overlayed on a dragon-themed full-screen backdrop.

---

## Features

- 🧠 **Machine Learning**  
  - Random Forest classifier trained on:  
    - Daily returns  
    - 5-day & 10-day simple moving averages (SMA)  
    - 14-period Relative Strength Index (RSI)  
  - Displays test-set accuracy and classification report.
- 🔮 **Daily & Short-Term Price Predictions**  
  - “Up” or “Down” for tomorrow (24 h).  
  - “Up” or “Down” for next 10 minutes and next 1 hour.
- 📊 **Historical Candlestick Chart**  
  - 30-day candlestick chart on a dark-trading theme with transparent background.
- ⏲️ **Live Price Monitor**  
  - Real-time price fetch (once per second for up to 5 minutes by default).  
  - Line chart of live price, updating each second.  
  - Price alerts (above/below user-set thresholds).
- 🐉 **Dragon-Themed Transparent UI**  
  - Full-page dragon background (dragon.png).  
  - All Streamlit containers (metrics, buttons, charts, sidebar) are transparent, with white text and minimal borders.

---

## Demo Screenshot

*(Include one or two screenshots or GIFs here to show off the transparent dashboard with the dragon background, the candlestick chart, and live monitor. For example:)*

<p align="center">
  <img src="./screenshots/dashboard.png" alt="Dragon Trading AI Dashboard" width="800px">
</p>

---

## Getting Started

### Prerequisites

1. **Python 3.8+**  
2. **Streamlit 1.20+**  
3. **Packages** (install via `pip`):  
   ```bash
   pip install streamlit pandas numpy requests scikit-learn plotly
