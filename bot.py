import threading
import time
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
from binance.spot import Spot  # Updated import statement
from tradingview_ta import TA_Handler, Interval
from pymongo import MongoClient
import streamlit as st
import streamlit.components.v1 as components

# Google Tag Manager Snippet
def inject_gtm():
    gtm_code = """
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
    """
    components.html(gtm_code, height=0)

# Inject GTM
inject_gtm()

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")

# Initialize MongoDB
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
user_collection = db["user_credentials"]
trade_collection = db["trade_history"]

# User Authentication
st.sidebar.header("User Authentication")
api_key = st.sidebar.text_input("Binance API Key", type="password")
api_secret = st.sidebar.text_input("Binance API Secret", type="password")

client = None
if api_key and api_secret:
    user = user_collection.find_one({"api_key": api_key})
    if not user:
        user_collection.insert_one({"api_key": api_key, "api_secret": api_secret})
        st.sidebar.success("API credentials saved.")
    client = Spot(api_key=api_key, api_secret=api_secret)
else:
    st.sidebar.warning("Please enter your Binance API credentials.")

# Constants
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVAL = Interval.INTERVAL_1_MINUTE
CHECK_FREQUENCY = 60

# Trading parameters
TRADE_PERCENT = st.sidebar.slider("Trade Percentage", 1, 10, 5) / 100
STOP_LOSS_PERCENT = st.sidebar.slider("Stop Loss %", 1, 10, 2) / 100
TAKE_PROFIT_PERCENT = st.sidebar.slider("Take Profit %", 1, 10, 5) / 100

def get_price(symbol):
    if client:
        ticker = client.ticker_price(symbol)
        return float(ticker["price"])
    return None

def get_tradingview_signal(symbol):
    analysis = TA_Handler(
        symbol=symbol, exchange="BINANCE", screener="crypto", interval=INTERVAL
    )
    return analysis.get_analysis().summary

def place_order(symbol, side, quantity):
    if client:
        try:
            order = client.new_order(symbol=symbol, side=side, type="MARKET", quantity=quantity)
            trade_collection.insert_one(order)
            return order
        except Exception as e:
            st.error(f"Order failed: {e}")
    return None

def trading_bot():
    while client:
        for symbol in SYMBOLS:
            try:
                price = get_price(symbol)
                if price is None:
                    continue
                signal = get_tradingview_signal(symbol)
                recommendation = signal["RECOMMENDATION"]
                balance = client.account()["balances"]
                asset_balance = next((b for b in balance if b["asset"] == symbol[:-4]), None)
                quantity = float(asset_balance["free"]) * TRADE_PERCENT if asset_balance else 0
                
                if recommendation == "BUY" and quantity > 0:
                    place_order(symbol, "BUY", quantity)
                elif recommendation == "SELL" and quantity > 0:
                    place_order(symbol, "SELL", quantity)
                
                print(f"{symbol} - Price: ${price} | Signal: {recommendation}")
            except Exception as e:
                print(f"Error in trading bot: {e}")
        time.sleep(CHECK_FREQUENCY)

def start_trading_bot():
    if client:
        thread = threading.Thread(target=trading_bot, daemon=True)
        thread.start()
    else:
        st.error("Please enter valid Binance API credentials to start trading.")

# Streamlit Dashboard
st.title("IRAM-B: Intelligent Risk-Aware Market Bot")
st.subheader("Live Market Overview")

# Fetch live data and visualize
live_data = {symbol: get_price(symbol) for symbol in SYMBOLS if client}
if live_data:
    st.metric(label="Bitcoin (BTC)", value=f"${live_data.get('BTCUSDT', 0):.2f}")
    st.metric(label="Ethereum (ETH)", value=f"${live_data.get('ETHUSDT', 0):.2f}")
    
    # Price Trend Visualization
    history_data = pd.DataFrame({"Timestamp": [time.time()], "BTC Price": [live_data.get('BTCUSDT', 0)], "ETH Price": [live_data.get('ETHUSDT', 0)]})
    st.line_chart(history_data.set_index("Timestamp"))
    
    # Candlestick Chart
    st.subheader("BTC Price Candlestick Chart")
    data = client.klines(symbol="BTCUSDT", interval="1m", limit=50) if client else []
    if data:
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
