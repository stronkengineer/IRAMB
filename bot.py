import threading
import time
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
from binance.spot import Spot
from tradingview_ta import TA_Handler, Interval
from pymongo import MongoClient
import streamlit.components.v1 as components
import requests

# Google Tag Manager
components.html("""
<script>
(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
'https://www.googletagmanager.com/gtm.js?id=GTM-KVZFT9ZW'+dl;f.parentNode.insertBefore(j,f);
})(window,document,'script','dataLayer','GTM-KVZFT9ZW');
</script>
<noscript>
<iframe src="https://www.googletagmanager.com/ns.html?id=GTM-KVZFT9ZW"
height="0" width="0" style="display:none;visibility:hidden"></iframe>
</noscript>
""", height=0)

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY") or "YXDUUN2VOTJULXZN"
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY") or "PKN03W8Z3MONNVHA6BP4"
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY") or "0XqyJ6905DEviEH2oTUrunTMB9KmDBfLHFdpZPSl"

# Mongo
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
user_collection = db["user_credentials"]
trade_collection = db["trade_history"]

# Auth
st.sidebar.header("User Authentication")
use_binance = st.sidebar.checkbox("Use Binance", value=False)
api_key, api_secret = None, None
client = None
if use_binance:
    api_key = st.sidebar.text_input("Binance API Key", type="password")
    api_secret = st.sidebar.text_input("Binance API Secret", type="password")
    if api_key and api_secret:
        user = user_collection.find_one({"api_key": api_key})
        if not user:
            user_collection.insert_one({"api_key": api_key, "api_secret": api_secret})
            st.sidebar.success("API credentials saved.")
        client = Spot(api_key=api_key, api_secret=api_secret)
    else:
        st.sidebar.warning("Please enter your Binance API credentials.")

# Parameters
SYMBOLS = {
    "crypto": ["BTCUSDT", "ETHUSDT"],
    "forex": ["EURUSD", "USDJPY"],
    "stocks": ["AAPL", "TSLA"]
}
INTERVAL = Interval.INTERVAL_1_MINUTE
CHECK_FREQUENCY = 60
TRADE_PERCENT = st.sidebar.slider("Trade Percentage", 1, 10, 5) / 100
STOP_LOSS_PERCENT = st.sidebar.slider("Stop Loss %", 1, 10, 2) / 100
TAKE_PROFIT_PERCENT = st.sidebar.slider("Take Profit %", 1, 10, 5) / 100

# Algorithm selectors for both stocks and crypto
ALGO_OPTIONS = [
    "SMA Crossover",
    "RSI",
    "Momentum",
    "TradingView Only",
    "Hybrid (TV + RSI)"
]

# Allow the user to choose algorithms for both stocks and crypto
selected_crypto_algo = st.sidebar.selectbox("Select Crypto Trading Algorithm", ALGO_OPTIONS)
selected_stock_algo = st.sidebar.selectbox("Select Stock Trading Algorithm", ALGO_OPTIONS)

# Price fetcher
def get_price(symbol):
    if symbol in SYMBOLS["crypto"]:
        handler = TA_Handler(symbol=symbol, screener="crypto", exchange="BINANCE", interval=INTERVAL)
        try:
            analysis = handler.get_analysis()
            return float(analysis.indicators["close"])
        except:
            return None
    elif symbol in SYMBOLS["stocks"]:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}"
        try:
            response = requests.get(url).json()
            return float(response["Global Quote"]["05. price"])
        except:
            return None
    elif symbol in SYMBOLS["forex"]:
        from_symbol, to_symbol = symbol[:3], symbol[3:]
        url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={from_symbol}&to_currency={to_symbol}&apikey={ALPHA_VANTAGE_KEY}"
        try:
            response = requests.get(url).json()
            return float(response["Realtime Currency Exchange Rate"]["5. Exchange Rate"])
        except:
            return None
    return None

# Strategy logic for crypto and stocks based on selected algorithm
def should_trade(symbol, category):
    # Select the appropriate algorithm based on category (crypto or stocks)
    selected_algo = selected_crypto_algo if category == "crypto" else selected_stock_algo
    screener = "crypto" if category == "crypto" else "forex" if category == "forex" else "america"
    price = get_price(symbol)
    signal = get_tradingview_signal(symbol, screener)
    rec = signal["RECOMMENDATION"]

    if selected_algo == "TradingView Only":
        return rec in ["BUY", "SELL"], rec, price

    elif selected_algo == "SMA Crossover":
        try:
            handler = TA_Handler(symbol=symbol, screener=screener, interval=INTERVAL, exchange="BINANCE")
            indicators = handler.get_analysis().indicators
            sma_50 = indicators.get("SMA50")
            sma_200 = indicators.get("SMA200")
            if sma_50 and sma_200:
                if sma_50 > sma_200:
                    return True, "BUY", price
                elif sma_50 < sma_200:
                    return True, "SELL", price
        except:
            pass
        return False, None, price

    elif selected_algo == "RSI":
        try:
            rsi = TA_Handler(symbol=symbol, screener=screener, interval=INTERVAL).get_analysis().indicators['RSI']
            if rsi < 30:
                return True, "BUY", price
            elif rsi > 70:
                return True, "SELL", price
        except:
            pass
        return False, None, price

    elif selected_algo == "Momentum":
        try:
            handler = TA_Handler(symbol=symbol, screener=screener, interval=INTERVAL)
            mom = handler.get_analysis().indicators.get("Mom")
            if mom:
                if mom > 0:
                    return True, "BUY", price
                elif mom < 0:
                    return True, "SELL", price
        except:
            pass
        return False, None, price

    elif selected_algo == "Hybrid (TV + RSI)":
        if rec in ["BUY", "SELL"]:
            try:
                rsi = TA_Handler(symbol=symbol, screener=screener, interval=INTERVAL).get_analysis().indicators['RSI']
                if rec == "BUY" and rsi < 35:
                    return True, rec, price
                elif rec == "SELL" and rsi > 65:
                    return True, rec, price
            except:
                pass
        return False, None, price

    return False, None, price

# Trading Bot
def trading_bot():
    while use_binance and client:
        for category, symbols in SYMBOLS.items():
            for symbol in symbols:
                try:
                    do_trade, signal, price = should_trade(symbol, category)
                    if do_trade and price:
                        if category == "crypto" and client:
                            balance = client.account()["balances"]
                            asset = symbol[:-4]
                            asset_balance = next((b for b in balance if b["asset"] == asset), None)
                            quantity = float(asset_balance["free"]) * TRADE_PERCENT if asset_balance else 0
                            if quantity > 0:
                                place_order(symbol, signal, quantity, category)
                        else:
                            simulated_quantity = 1000 * TRADE_PERCENT
                            place_order(symbol, signal, simulated_quantity, category)
                    print(f"{symbol} - ${price:.2f} - Signal: {signal}")
                except Exception as e:
                    print(f"Error on {symbol}: {e}")
        time.sleep(CHECK_FREQUENCY)

# Start thread
if st.button("Start Trading Bot"):
    if use_binance and client:
        thread = threading.Thread(target=trading_bot, daemon=True)
        thread.start()
        st.success("Trading Bot Started!")
    else:
        st.warning("Trading bot only runs with Binance enabled and configured.")

# UI Overview
st.title("IRAM-B: Intelligent Risk-Aware Market Bot")
st.subheader("Live Market Overview")
for category, symbols in SYMBOLS.items():
    for symbol in symbols:
        screener = "crypto" if category == "crypto" else "forex" if category == "forex" else "america"
        try:
            price = get_price(symbol)
            signal = get_tradingview_signal(symbol, screener)
            recommendation = signal["RECOMMENDATION"]
            st.metric(label=f"{symbol} ({category})", value=f"${price:.2f}" if price else "N/A", delta=recommendation)
        except:
            st.warning(f"{symbol} signal unavailable")

# Candlestick chart for Crypto and Stock symbols
if client:
    st.subheader("BTC/USDT Candlestick Chart")
    data = client.klines(symbol="BTCUSDT", interval="1m", limit=50)
    if data:
        df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume", "_", "_", "_", "_", "_", "_"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)

        fig = go.Figure(data=[go.Candlestick(
            x=df["time"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"]
        )])
        st.plotly_chart(fig)

# RSI plot for Crypto and Stock symbols
def plot_rsi(symbol, category):
    rsi_values = []
    if category == "crypto":
        screener = "crypto"
    elif category == "stocks":
        screener = "america"
    handler = TA_Handler(symbol=symbol, screener=screener, interval=INTERVAL)
    for i in range(14):  # Collect last 14 RSI values
        rsi_values.append(handler.get_analysis().indicators['RSI'])
        time.sleep(1)
    
    # Plot RSI values
    st.subheader(f"{symbol} RSI")
    st.line_chart(rsi_values)

# Show RSI plot for the first symbol
plot_rsi("BTCUSDT", "crypto")
