import threading
import time
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
from binance.client import Client
from tradingview_ta import TA_Handler, Interval
from pymongo import MongoClient

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")

# Initialize MongoDB
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
trade_collection = db["trade_history"]

# Constants
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVAL = Interval.INTERVAL_1_MINUTE
CHECK_FREQUENCY = 60

# User Authentication
st.sidebar.header("User Authentication")
api_key = st.sidebar.text_input("Binance API Key", type="password")
api_secret = st.sidebar.text_input("Binance API Secret", type="password")

if api_key and api_secret:
    client = Client(api_key, api_secret)
else:
    st.sidebar.warning("Please enter your Binance API credentials.")
    st.stop()

# Trading Parameters
TRADE_PERCENT = st.sidebar.slider("Trade Percentage", 1, 10, 5) / 100
STOP_LOSS_PERCENT = st.sidebar.slider("Stop Loss %", 1, 10, 2) / 100
TAKE_PROFIT_PERCENT = st.sidebar.slider("Take Profit %", 1, 10, 5) / 100

# Get latest price
def get_price(symbol):
    ticker = client.get_symbol_ticker(symbol=symbol)
    return float(ticker["price"])

# Get TradingView signal
def get_tradingview_signal(symbol):
    analysis = TA_Handler(
        symbol=symbol, exchange="BINANCE", screener="crypto", interval=INTERVAL
    )
    return analysis.get_analysis().summary

# Place trade order
def place_order(symbol, side, quantity):
    try:
        order = client.order_market(symbol=symbol, side=side, quantity=quantity)
        trade_collection.insert_one(order)
        return order
    except Exception as e:
        st.error(f"Order failed: {e}")
        return None

# Trading bot logic
def trading_bot():
    while True:
        for symbol in SYMBOLS:
            try:
                price = get_price(symbol)
                signal = get_tradingview_signal(symbol)
                recommendation = signal["RECOMMENDATION"]
                balance = client.get_asset_balance(asset=symbol[:-4])
                quantity = float(balance["free"]) * TRADE_PERCENT
                
                if recommendation == "BUY":
                    place_order(symbol, "BUY", quantity)
                elif recommendation == "SELL":
                    place_order(symbol, "SELL", quantity)
                
                print(f"{symbol} - Price: ${price} | Signal: {recommendation}")
            except Exception as e:
                print(f"Error in trading bot: {e}")
        time.sleep(CHECK_FREQUENCY)

# Start bot function
def start_trading_bot():
    thread = threading.Thread(target=trading_bot, daemon=True)
    thread.start()

# Streamlit Dashboard
st.title("IRAM-B: Intelligent Risk-Aware Market Bot")
st.subheader("Live Market Overview")

live_data = {symbol: get_price(symbol) for symbol in SYMBOLS}
st.metric(label="Bitcoin (BTC)", value=f"${live_data['BTCUSDT']:.2f}")
st.metric(label="Ethereum (ETH)", value=f"${live_data['ETHUSDT']:.2f}")

# Candlestick Chart
st.subheader("BTC Price Candlestick Chart")
data = client.get_klines(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1MINUTE, limit=50)
df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume", "ignore", "ignore", "ignore", "ignore", "ignore", "ignore"])
df["time"] = pd.to_datetime(df["time"], unit="ms")
df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)

fig = go.Figure()
fig.add_trace(go.Candlestick(x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="BTC Candlestick"))
st.plotly_chart(fig)

# Start Trading Bot Button
if st.button("Start Trading Bot"):
    start_trading_bot()
    st.success("Trading Bot Started!")
