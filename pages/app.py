import threading
import time
import os
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from dotenv import load_dotenv
from binance.client import Client
from tradingview_ta import TA_Handler, Interval
from pymongo import MongoClient
import streamlit.components.v1 as components
import requests
import random
from alpaca_trade_api.rest import REST as Alpaca
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
import pandas as pd
import random
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import random
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from scipy.signal import argrelextrema
from auth import IGAuthenticator
from sklearn.ensemble import RandomForestClassifier

if "total_profit" not in st.session_state:
    st.session_state.total_profit = 0.0
if "open_positions" not in st.session_state:
    st.session_state.open_positions = {}
if "bot_thread" not in st.session_state:
    st.session_state.bot_thread = None

if not st.session_state.get("logged_in"):
    st.warning("🔐 Please log in first via the Home page.")
    st.stop()

    if "user_id" not in st.session_state:
     st.session_state.user_id = None


st.title("📊 Main Dashboard")
st.write(f"Hello, {st.session_state.username}!")

if "ml_model" not in st.session_state:
    st.session_state.ml_model = RandomForestClassifier()
    # Dummy fit for demonstration
    X_demo = np.random.rand(100, 5)
    y_demo = np.random.randint(0, 2, 100)
    st.session_state.ml_model.fit(X_demo, y_demo)
# --- Load environment variables ---
load_dotenv(".env")
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY") or "YXDUUN2VOTJULXZN"
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY") or "pub_877205437754424e9f9b6279373344598fe13"

# --- MongoDB ---
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
user_collection = db["user_credentials"]
trade_collection = db["trade_history"]

# --- Alpaca client ---
alpaca = None
if ALPACA_API_KEY and ALPACA_SECRET_KEY:
    alpaca = Alpaca(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url="https://paper-api.alpaca.markets")

# --- Streamlit Google Tag Manager (placeholder) ---
components.html("""<script>...</script><noscript>...</noscript>""", height=0)

# --- Sidebar: Authentication & Settings ---
# --- Sidebar: Authentication & Settings ---
st.sidebar.header("User Authentication")

# --- Binance API ---
use_binance = st.sidebar.checkbox("Use Binance", value=False)
use_testnet = st.sidebar.checkbox("Use Binance Testnet", value=False)

client = None
api_key_binance, api_secret_binance = None, None

if use_binance:
    if use_testnet:
        st.sidebar.markdown("### 🔧 Testnet API Credentials")
        api_key_binance = st.sidebar.text_input("Testnet API Key", type="password")
        api_secret_binance = st.sidebar.text_input("Testnet API Secret", type="password")
    else:
        st.sidebar.markdown("### 🔐 Live API Credentials")
        api_key_binance = st.sidebar.text_input("Live API Key", type="password")
        api_secret_binance = st.sidebar.text_input("Live API Secret", type="password")

    if api_key_binance and api_secret_binance:
        user_binance = user_collection.find_one({
            "api_key_binance": api_key_binance,
            "is_testnet": use_testnet
        })

        if not user_binance:
            user_collection.insert_one({
                "api_key_binance": api_key_binance,
                "api_secret_binance": api_secret_binance,
                "is_testnet": use_testnet
            })
            st.sidebar.success("✅ API credentials saved.")

        # Initialize Binance client
        client = Client(api_key=api_key_binance, api_secret=api_secret_binance)

        # Use Testnet endpoint if selected
        if use_testnet:
            client.API_URL = "https://testnet.binance.vision/api"
            st.sidebar.success("Connected to Binance Testnet ✅")
        else:
            st.sidebar.success("Connected to Binance Live ✅")
    else:
        st.sidebar.warning("⚠️ Please enter your Binance API credentials.")

# --- Alpaca API ---
use_alpaca = st.sidebar.checkbox("Use Alpaca", value=False)
api_key_alpaca, api_secret_alpaca = None, None
alpaca_client = None
if use_alpaca:
    api_key_alpaca = st.sidebar.text_input("Alpaca API Key", type="password")
    api_secret_alpaca = st.sidebar.text_input("Alpaca Secret Key", type="password")
    if api_key_alpaca and api_secret_alpaca:
        user_alpaca = user_collection.find_one({"api_key_alpaca": api_key_alpaca})
        if not user_alpaca:
            user_collection.insert_one({"api_key_alpaca": api_key_alpaca, "api_secret_alpaca": api_secret_alpaca})
            st.sidebar.success("Alpaca API credentials saved.")
        alpaca_client = Alpaca(api_key_alpaca, api_secret_alpaca, base_url="https://paper-api.alpaca.markets")
    else:
        st.sidebar.warning("Please enter your Alpaca API credentials.")

# --- IG API ---
use_ig = st.sidebar.checkbox("Use IG", value=False)

ig_api_key, ig_account_id, ig_access_token = None, None, None
ig_env = "Demo"  # default environment

if use_ig:
    ig_env = st.sidebar.selectbox("Select IG Environment", ["Demo", "Live"])
    
    ig_api_key = st.sidebar.text_input("IG API Key", type="password")
    ig_username = st.sidebar.text_input("IG Username")
    ig_password = st.sidebar.text_input("IG Password", type="password")

    if ig_api_key and ig_account_id and ig_access_token:
        # Save user credentials in DB if not exist
        user_ig = user_collection.find_one({"ig_api_key": ig_api_key})
        if not user_ig:
            user_collection.insert_one({
                "ig_api_key": ig_api_key,
                "ig_username": ig_username,
                "ig_password": ig_password,
                "ig_environment": ig_env  # save environment too
            })
            st.sidebar.success("IG API credentials saved.")
    else:
        st.sidebar.warning("Please enter your IG API credentials.")

    # Setup base URL depending on environment
    if ig_env == "Demo":
        ig_base_url = "https://demo-api.ig.com/gateway/deal"
    else:
        ig_base_url = "https://api.ig.com/gateway/deal"
    igClient=IGAuthenticator(
        api_key=ig_api_key,username=ig_username,password=ig_password, base_url=ig_base_url)


    # Initialize IGClient here if needed, e.g.:
    # ig_client = IGClient(ig_api_key, ig_account_id, ig_access_token, base_url=ig_base_url)


# --- Trade Settings ---
TRADE_PERCENT = st.sidebar.slider("Trade Percentage (%)", 1, 10, 5) / 100
STOP_LOSS_PERCENT = st.sidebar.slider("Stop Loss %", 1, 10, 2) / 100
TAKE_PROFIT_PERCENT = st.sidebar.slider("Take Profit %", 1, 20, 5) / 100  # Added TP slider


ALGO_OPTIONS = [
    "SMA Crossover",
    "RSI",
    "Momentum",
    "TradingView Only",
    "Hybrid (TV + RSI)"
]
selected_algo = st.sidebar.selectbox("Select Trading Algorithm", ALGO_OPTIONS)

# --- Language toggle ---
lang = st.sidebar.selectbox("Language / اللغة", ["English", "Arabic"])
def t(text_en, text_ar):
    return text_ar if lang == "Arabic" else text_en

# --- Symbols ---
DEFAULTS = {
    "crypto": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
    "forex": ["EURUSD", "USDJPY", "GBPUSD"],
    "stocks": ["AAPL", "TSLA", "MSFT"],
    "commodities": ["GOLD", "SILVER", "OIL"]
}

target_prices = {}

for category, symbols in DEFAULTS.items():
    for symbol in symbols:
        key = f"target_{category}_{symbol}"
        target_prices[key] = st.sidebar.number_input(
            f"Target Price for {symbol} ({category})", min_value=0.0, value=0.0, step=0.01
        )

def get_user_symbols():
    categories = ["crypto", "forex", "stocks", "commodities"]
    user_symbols = {}

    st.write("### Select default symbols or enter your own (comma-separated)")

    for category in categories:
        default_str = ", ".join(DEFAULTS[category])
        choice = st.selectbox(
            f"Choose default for {category.capitalize()}",
            options=["Custom input"] + [default_str],
            index=1 if category in DEFAULTS else 0
        )
        if choice == "Custom input":
            user_input = st.text_input(f"Enter {category} symbols", placeholder=default_str)
            symbols_list = [sym.strip().upper() for sym in user_input.split(",") if sym.strip()] if user_input else []
        else:
            symbols_list = [sym.strip().upper() for sym in choice.split(",")]
        user_symbols[category] = symbols_list

    return user_symbols

SYMBOLS = get_user_symbols()

INTERVAL = Interval.INTERVAL_1_MINUTE
CHECK_FREQUENCY = 60

# --- Sentiment analyzer ---
sentiment_analyzer = SentimentIntensityAnalyzer()

# --- Fetch news headlines ---
def get_news_headlines():
    try:
        url = (
            f"https://newsapi.org/v2/top-headlines?"
            f"category=business&language=en&apiKey={NEWSAPI_KEY}"
        )
        response = requests.get(url)
        data = response.json()
        headlines = [article["title"] for article in data.get("articles", [])]
        return headlines
    except Exception as e:
        print(f"News fetch error: {e}")
        return []

# --- Web scraping additional news ---
def scrape_additional_news():
    news_list = []
    try:
        # Example scraping Reuters business headlines
        url = "https://www.reuters.com/business/"
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.content, "html.parser")
        articles = soup.find_all("h3", class_="MediaStoryCard__heading___2A8G4")
        for art in articles[:10]:
            text = art.get_text(strip=True)
            if text:
                news_list.append(text)
    except Exception as e:
        print(f"Scraping error: {e}")
    return news_list

# --- Sentiment analysis on news for symbol ---
def get_sentiment_for_symbol(symbol):
    headlines = get_news_headlines() + scrape_additional_news()
    if not headlines:
        return None

    base_symbol = symbol.replace("USDT", "").replace("USD", "").lower()
    relevant_headlines = [h for h in headlines if base_symbol in h.lower()]
    if not relevant_headlines:
        return None

    scores = [sentiment_analyzer.polarity_scores(h)["compound"] for h in relevant_headlines]
    avg_score = sum(scores) / len(scores) if scores else 0

    if avg_score > 0.2:
        return "positive"
    elif avg_score < -0.2:
        return "negative"
    else:
        return "neutral"

# --- Price fetcher ---
import yfinance as yf

import yfinance as yf
import time

#price_cache = {}
#CACHE_EXPIRY = 60  # seconds

import requests
import time

API_NINJAS_KEY = 'zRmrQxu5V3uueK/Ws/F/tA==PZTRPDW7AxpoAyhA'

@st.cache_data(ttl=300, show_spinner=False)
def get_price(symbol):
    try:
        if symbol in SYMBOLS["crypto"]:
            handler = TA_Handler(symbol=symbol, screener="crypto", exchange="BINANCE", interval=INTERVAL)
            analysis = handler.get_analysis()
            return float(analysis.indicators["close"])

        elif symbol in SYMBOLS["forex"]:
            handler = TA_Handler(symbol=symbol, screener="forex", exchange="FX_IDC", interval=INTERVAL)
            analysis = handler.get_analysis()
            return float(analysis.indicators["close"])

        elif symbol in SYMBOLS["stocks"]:
            handler = TA_Handler(symbol=symbol, screener="america", exchange="NASDAQ", interval=INTERVAL)
            analysis = handler.get_analysis()
            return float(analysis.indicators["close"])

        elif symbol in SYMBOLS["commodities"]:
            commodity_source_map = {
                "GOLD": {"source": "api_ninjas", "id": "gold"},
                "SILVER": {"source": "yfinance", "id": "SI=F"},
                "OIL": {"source": "yfinance", "id": "CL=F"},
                "NATGAS": {"source": "yfinance", "id": "NG=F"},
                "COPPER": {"source": "api_ninjas", "id": "copper"},
                "PLATINUM": {"source": "api_ninjas", "id": "platinum"},
                "PALLADIUM": {"source": "api_ninjas", "id": "palladium"},
                "COCOA": {"source": "api_ninjas", "id": "cocoa"},
                "COFFEE": {"source": "api_ninjas", "id": "coffee"},
                "CORN": {"source": "api_ninjas", "id": "corn"},
                "SUGAR": {"source": "api_ninjas", "id": "sugar"},
                "WHEAT": {"source": "api_ninjas", "id": "wheat"},
            }

            info = commodity_source_map.get(symbol.upper())
            if not info:
                st.warning(f"Unsupported commodity symbol: {symbol}")
                return None

            if info["source"] == "yfinance":
                try:
                    data = yf.Ticker(info["id"]).history(period="1d")
                    if not data.empty:
                        return float(data["Close"].iloc[-1])
                    else:
                        st.warning(f"No data found in yfinance for {symbol}")
                        return None
                except Exception as e:
                    st.warning(f"yfinance error for {symbol}: {e}")
                    return None

            elif info["source"] == "api_ninjas":
                try:
                    url = f"https://api.api-ninjas.com/v1/commodityprice?name={info['id']}"
                    headers = {'X-Api-Key': API_NINJAS_KEY}
                    response = requests.get(url, headers=headers)
                    if response.status_code == 200:
                        data = response.json()
                        price = data.get("price")
                        if price is not None:
                            return float(price)
                        else:
                            st.warning(f"No price in API response for {symbol}")
                            return None
                    else:
                        st.warning(f"API error for {symbol}: {response.status_code} {response.text}")
                        return None
                except Exception as e:
                    st.warning(f"API Ninjas error for {symbol}: {e}")
                    return None

        else:
            return None

    except Exception as e:
        st.warning(f"Price fetch error for {symbol}: {e}")
        return None

# --- TradingView signal ---
INTERVAL = "1d"  # or whatever interval you want

commodity_tv_map = {
    "OIL": {"symbol": "CL1!", "exchange": "NYMEX", "screener": "commodities"},
    "GOLD": {"symbol": "GC1!", "exchange": "COMEX", "screener": "commodities"},
    "SILVER": {"symbol": "SI1!", "exchange": "COMEX", "screener": "commodities"},
    # Add more commodities here as needed
}

@st.cache_data(ttl=300, show_spinner=False)
def get_tradingview_signal(symbol, screener, category):  
    try:
        if category == "commodities":
            # override symbol and exchange if in map
            commodity_info = commodity_tv_map.get(symbol.upper())
            if commodity_info:
                tv_symbol = commodity_info["symbol"]
                exchange = commodity_info["exchange"]
                screener = commodity_info["screener"]
            else:
                tv_symbol = symbol
                exchange = "NYMEX"  # fallback default
        elif category == "crypto":
            exchange = "BINANCE"
            tv_symbol = symbol
        elif category == "forex":
            exchange = "FX_IDC"
            tv_symbol = symbol
        else:
            exchange = "NASDAQ"
            tv_symbol = symbol

        handler = TA_Handler(symbol=tv_symbol, screener=screener, interval=INTERVAL, exchange=exchange)
        analysis = handler.get_analysis()
        return analysis.summary
    except Exception as e:
        print(f"TV signal error for {symbol}: {e}")
        return {"RECOMMENDATION": "NEUTRAL"}

# -- IG Converter
def build_ig_symbol_map(igClient):
    symbol_map = {}
    markets = igClient.fetch_markets()  # assuming your IG client has this method from your earlier code

    for market in markets:
        # Example normalization:
        # market['instrumentName'] could be "Gold" or "EUR/USD"
        # market['epic'] is the IG epic string you need

        # Normalize symbol to uppercase and remove spaces/slashes for consistency
        symbol = market['instrumentName'].upper().replace(" ", "").replace("/", "")
        symbol_map[symbol] = market['epic']

    return symbol_map
# Build IG symbol map
if use_ig and igClient:
 ig_symbol_map= build_ig_symbol_map(igClient)

# --- Place order ---
def place_order(symbol, side, quantity, category, price=None, close=False):
    now = time.time()
    try:
        if not close:
            # --- OPENING a trade ---
            open_price = price
            close_price = None
            order = {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "category": category,
                "timestamp": now,
                "open_price": open_price,
                "close_price": close_price,
                "close_timestamp": None
            }
            if category == "crypto" and client:
                binance_order = client.new_order(symbol=symbol, side=side, type="MARKET", quantity=quantity)
                order.update({"platform": "Binance", "binance_response": binance_order})
            elif category == "stocks" and alpaca:
                alpaca_side = "buy" if side.upper() == "BUY" else "sell"
                alpaca_order = alpaca.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    type="market",
                    time_in_force="gtc"
                )
                order.update({"platform": "Alpaca", "alpaca_id": alpaca_order.id})
            elif category == "forex" and igClient:
                epic = ig_symbol_map.get(symbol.upper(), symbol)
                ig_direction = side.upper()
                ig_response = igClient.place_order(
                    epic=epic,
                    size=quantity,
                    direction=ig_direction,
                    order_type='MARKET',
                    currency_code='USD',
                    expiry='DFB'
                )
                order.update({"platform": "IG", "ig_response": ig_response})
            trade_collection.insert_one(order)
            return order

        else:
            # --- CLOSING a trade ---
            close_price = price
            # Find the most recent open trade for this symbol, opposite side, and category
            open_trade = trade_collection.find_one({
                "symbol": symbol,
                "side": {"$ne": side},  # Opposite side
                "category": category,
                "close_price": None
            }, sort=[("timestamp", -1)])
            if open_trade:
                trade_collection.update_one(
                    {"_id": open_trade["_id"]},
                    {"$set": {"close_price": close_price, "close_timestamp": now}}
                )
                return open_trade
            else:
                # No open trade found, insert as a closed trade for record
                order = {
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "category": category,
                    "timestamp": now,
                    "open_price": None,
                    "close_price": close_price,
                    "close_timestamp": now
                }
                trade_collection.insert_one(order)
                return order

    except Exception as e:
        st.error(t(f"Order failed: {e}", f"فشل الطلب: {e}"))
        return None


import yfinance as yf
@st.cache_data(ttl=120)
def get_yf_symbol(symbol, category):
    """
    Map any input symbol/category to a yfinance-compatible ticker.
    Handles dynamic crypto, forex, commodities, and stocks.
    """
    symbol = symbol.upper().replace(" ", "").replace("/", "")
    if category == "crypto":
        # Try to split into base/quote, e.g. BTCUSDT, ETHBTC, DOGEUSD, etc.
        # yfinance supports many as BASE-QUOTE (e.g. BTC-USD, ETH-BTC)
        if symbol.endswith("USDT"):
            return symbol.replace("USDT", "-USD")
        elif symbol.endswith("USD"):
            return symbol.replace("USD", "-USD")
        elif len(symbol) > 6:
            # Try to split base/quote for other pairs (e.g. ETHBTC, LTCETH)
            # Heuristic: first 3 or 4 chars as base, rest as quote
            for split in [3, 4]:
                base, quote = symbol[:split], symbol[split:]
                # Check if both base and quote are alphabetic and reasonable length
                if base.isalpha() and quote.isalpha() and 2 <= len(quote) <= 5:
                    return f"{base}-{quote}"
            return symbol  # fallback
        else:
            return symbol
    elif category == "forex":
        # yfinance expects e.g. EURUSD=X, GBPJPY=X, etc.
        return symbol + "=X"
    elif category == "commodities":
        # Try to map common commodities, fallback to symbol=F
        yf_map = {
            "GOLD": "GC=F",
            "SILVER": "SI=F",
            "OIL": "CL=F",
            "NATGAS": "NG=F",
            "COPPER": "HG=F",
            "PLATINUM": "PL=F",
            "PALLADIUM": "PA=F",
            "COCOA": "CC=F",
            "COFFEE": "KC=F",
            "CORN": "ZC=F",
            "SUGAR": "SB=F",
            "WHEAT": "ZW=F",
        }
        return yf_map.get(symbol, symbol + "=F")
    else:
        # stocks: use as-is
        return symbol

def sma_crossover_signal(symbol, short_window=10, long_window=30, category="stocks"):
    yf_symbol = get_yf_symbol(symbol, category)
    data = yf.Ticker(yf_symbol).history(period=f"{long_window*3}d", interval="1d")
    if data.shape[0] < long_window + 1:
        return False, None  # Not enough data

    data['SMA_short'] = data['Close'].rolling(window=short_window).mean()
    data['SMA_long'] = data['Close'].rolling(window=long_window).mean()

    prev_short = data['SMA_short'].iloc[-2]
    prev_long = data['SMA_long'].iloc[-2]
    curr_short = data['SMA_short'].iloc[-1]
    curr_long = data['SMA_long'].iloc[-1]

    if prev_short < prev_long and curr_short > curr_long:
        return True, "BUY"
    elif prev_short > prev_long and curr_short < curr_long:
        return True, "SELL"
    else:
        return False, None

def momentum_signal(symbol, window=10, threshold=0.01, category="stocks"):
    yf_symbol = get_yf_symbol(symbol, category)
    data = yf.Ticker(yf_symbol).history(period=f"{window+1}d", interval="1d")
    if data.shape[0] < window + 1:
        return False, None  # Not enough data

    price_now = data['Close'].iloc[-1]
    price_past = data['Close'].iloc[-(window+1)]
    momentum = (price_now - price_past) / price_past

    if momentum > threshold:
        return True, "BUY"
    elif momentum < -threshold:
        return True, "SELL"
    else:
        return False, None
# --- Trading decision logic ---
def should_trade(symbol, category):
    screener = "crypto" if category == "crypto" else "forex" if category == "forex" else "america"
    price = get_price(symbol)
    signal = get_tradingview_signal(symbol, screener, category)
    rec = signal.get("RECOMMENDATION", "NEUTRAL")
    sentiment = get_sentiment_for_symbol(symbol)

    algo_trade = False
    direction = None

    if selected_algo == "TradingView Only":
        algo_trade = rec in ["BUY", "SELL"]
        direction = rec

    elif selected_algo == "SMA Crossover":
        algo_trade = random.choice([True, False])
        direction = random.choice(["BUY", "SELL"])

    elif selected_algo == "RSI":
        try:
            rsi = TA_Handler(symbol=symbol, screener=screener, interval=INTERVAL).get_analysis().indicators['RSI']
            if rsi < 30:
                algo_trade, direction = True, "BUY"
            elif rsi > 70:
                algo_trade, direction = True, "SELL"
        except:
            pass

    elif selected_algo == "Momentum":
        algo_trade = random.choice([True, False])
        direction = random.choice(["BUY", "SELL"])

    elif selected_algo == "Hybrid (TV + RSI)":
        if rec in ["BUY", "SELL"]:
            try:
                rsi = TA_Handler(symbol=symbol, screener=screener, interval=INTERVAL).get_analysis().indicators['RSI']
                if rec == "BUY" and rsi < 35:
                    algo_trade, direction = True, "BUY"
                elif rec == "SELL" and rsi > 65:
                    algo_trade, direction = True, "SELL"
            except:
                pass

    # Sentiment filter: block contradicting trades
    if algo_trade and direction and sentiment:
        if (direction == "BUY" and sentiment == "negative") or (direction == "SELL" and sentiment == "positive"):
            st.info(t(f"[Blocked] {symbol} - {direction} blocked due to sentiment: {sentiment}",
                      f"[تم الحظر] {symbol} - تم حظر {direction} بسبب الشعور: {sentiment}"))
            return False, None, price

    return algo_trade, direction, price

# --- Trading Bot Thread ---
# --- In-memory state for open positions and daily loss ---
if "open_positions" not in st.session_state:
    st.session_state.open_positions = {}
if "daily_loss" not in st.session_state:
    st.session_state.daily_loss = 0
if "last_trade_time" not in st.session_state:
    st.session_state.last_trade_time = {}

DAILY_LOSS_LIMIT = st.sidebar.number_input("Daily Loss Limit ($)", min_value=10, value=100)
TRADE_COOLDOWN = st.sidebar.number_input("Trade Cooldown (seconds)", min_value=10, value=60)

stop_bot = threading.Event()

import joblib
import numpy as np
def compute_qty(category, price):
    try:
        if category == "crypto":
            if not client:
                return 0
            balance = client.account().get("balances", [])
            usdt_balance = next((b for b in balance if b.get("asset") == "USDT"), None)
            if not usdt_balance:
                return 0
            trade_usdt = float(usdt_balance["free"]) * TRADE_PERCENT
            qty = round(trade_usdt / price, 6)
            return qty

        elif category == "stocks":
            if not alpaca:
                return 0
            account = alpaca.get_account()
            cash = float(account.cash)
            trade_cash = cash * TRADE_PERCENT
            qty = max(1, int(trade_cash / price))
            return qty

        elif category == "forex":
            if not igClient:
                return 0
            return 1000  # Or dynamic based on your margin settings

        elif category == "commodities":
            return 1  # Or dynamic based on contract spec

    except Exception as e:
        print(f"[compute_qty] Error calculating quantity for {category}: {e}")
    return 0


# Load your pre-trained Random Forest model
try:
    rf_model = joblib.load("models/random_forest_trader.pkl")
except Exception as e:
    print(f"Failed to load Random Forest model: {e}")
    rf_model = None

def get_rf_features(symbol, category):
    """
    Build feature array for Random Forest prediction.
    For simplicity, let's use: [price, volatility, volume, time].
    Extend this with your actual engineered features.
    """
    price = get_price(symbol) or 0
    volatility = np.random.uniform(0, 1)  # dummy placeholder, replace with real
    volume = np.random.uniform(100, 1000)  # dummy placeholder, replace with real
    hour_of_day = (time.localtime().tm_hour + time.localtime().tm_min / 60.0) / 24.0
    return np.array([[price, volatility, volume, hour_of_day]])

def trading_bot(status_area):
    while not stop_bot.is_set():
        to_close = []
        for pos_key, pos in list(st.session_state.open_positions.items()):
            symbol = pos_key.split(":", 1)[1]
            category = pos_key.split(":", 1)[0]
            current_price = get_price(symbol)
            if current_price is None:
                continue
            sl = pos.get("stop_loss_price")
            tp = pos.get("take_profit_price")
            side = pos["side"]
            qty = pos["qty"]

            # SL / TP enforcement
            if side == "BUY" and (current_price <= sl or current_price >= tp):
                reason = "SL" if current_price <= sl else "TP"
                pnl = (current_price - pos["entry_price"]) * qty
                realized_loss = -pnl if pnl < 0 else 0
                st.session_state.daily_loss += realized_loss
                place_order(symbol, "SELL", qty, category, price=current_price, close=True)
                status_area.text(f"{symbol} ({category}): {reason} hit at {current_price:.2f}. Closing.")
                to_close.append(pos_key)
            elif side == "SELL" and (current_price >= sl or current_price <= tp):
                reason = "SL" if current_price >= sl else "TP"
                pnl = (pos["entry_price"] - current_price) * qty
                realized_loss = -pnl if pnl < 0 else 0
                st.session_state.daily_loss += realized_loss
                place_order(symbol, "BUY", qty, category, price=current_price, close=False)
                status_area.text(f"{symbol} ({category}): {reason} hit at {current_price:.2f}. Closing.")
                to_close.append(pos_key)

            # Target price
            target_price = pos.get("target_price", 0.0)
            if target_price > 0:
                if side == "BUY" and current_price >= target_price:
                    pnl = (current_price - pos["entry_price"]) * qty
                    realized_loss = -pnl if pnl < 0 else 0
                    st.session_state.daily_loss += realized_loss
                    place_order(symbol, "SELL", qty, category, price=current_price, close=True)
                    status_area.text(f"{symbol} ({category}): Target {target_price:.2f} reached. Closing.")
                    to_close.append(pos_key)
                elif side == "SELL" and current_price <= target_price:
                    pnl = (pos["entry_price"] - current_price) * qty
                    realized_loss = -pnl if pnl < 0 else 0
                    st.session_state.daily_loss += realized_loss
                    place_order(symbol, "BUY", qty, category, price=current_price, close=False)
                    status_area.text(f"{symbol} ({category}): Target {target_price:.2f} reached. Closing.")
                    to_close.append(pos_key)

        for pos_key in to_close:
            del st.session_state.open_positions[pos_key]

        # ============================
        # === Trading decision loop ===
        # ============================
        for category, symbols in SYMBOLS.items():
            for symbol in symbols:
                now = time.time()
                pos_key = f"{category}:{symbol}"
                last_time = st.session_state.last_trade_time.get(pos_key)
                if last_time and now - last_time < TRADE_COOLDOWN:
                    continue

                # --- Original strategy ---
                do_trade, signal, price = should_trade(symbol, category)

                # --- Random Forest override / assist ---
                if rf_model and price is not None:
                    features = get_rf_features(symbol, category)
                    prediction = rf_model.predict(features)[0]
                    probas = rf_model.predict_proba(features)[0]
                    confidence = max(probas)
                    # If high confidence, override original signal
                    if confidence > 0.75:
                        signal = "BUY" if prediction == 1 else "SELL"
                        do_trade = True
                        status_area.text(f"RF override for {symbol}: {signal} (conf: {confidence:.2f})")

                if not do_trade or signal is None:
                    continue

                # --- Proceed as before ---
                qty = compute_qty(category, symbol, price, client, alpaca, igClient)
                if qty <= 0:
                    continue

                stop_loss_price = price * (1 - STOP_LOSS_PERCENT) if signal == "BUY" else price * (1 + STOP_LOSS_PERCENT)
                take_profit_price = price * (1 + TAKE_PROFIT_PERCENT) if signal == "BUY" else price * (1 - TAKE_PROFIT_PERCENT)
                target_key = f"target_{category}_{symbol}"
                target_price = target_prices.get(target_key, 0.0)

                st.session_state.open_positions[pos_key] = {
                    "side": signal,
                    "entry_price": price,
                    "qty": qty,
                    "timestamp": now,
                    "stop_loss_price": stop_loss_price,
                    "take_profit_price": take_profit_price,
                    "target_price": target_price
                }

                place_order(symbol, signal, qty, category, price=price)
                status_area.text(
                    f"Traded {symbol} ({category}): Signal {signal}, Price {price}, "
                    f"SL {stop_loss_price:.2f}, TP {take_profit_price:.2f}, Target {target_price:.2f}"
                )
                st.session_state.last_trade_time[pos_key] = now

                if st.session_state.daily_loss >= DAILY_LOSS_LIMIT:
                    status_area.text("Daily loss limit reached. Stopping.")
                    stop_bot.set()
                    break

        time.sleep(CHECK_FREQUENCY)


# Streamlit app controls

st.write("## Open Positions")
for pos_key, pos in st.session_state.open_positions.items():
    symbol = pos_key.split(":", 1)[1]
    category = pos_key.split(":", 1)[0]
    side = pos["side"]
    entry = pos["entry_price"]
    qty = pos["qty"]
    open_time = pos.get("timestamp", None)
    if open_time:
        duration_sec = int(time.time() - open_time)
        hours, remainder = divmod(duration_sec, 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_str = f"{hours}h {minutes}m {seconds}s"
    else:
        duration_str = "N/A"
    st.write(
        f"{symbol} ({category}) | {side} | Entry: {entry} | Qty: {qty} | Open for: {duration_str}"
    )
    
st.title("Automated Trading Bot")

status_area = st.empty()

if "bot_thread" not in st.session_state:
    st.session_state.bot_thread = None

if st.button("Start Trading Bot"):
    if st.session_state.bot_thread is None or not st.session_state.bot_thread.is_alive():
        stop_bot.clear()
        st.session_state.bot_thread = threading.Thread(target=trading_bot, args=(status_area,), daemon=True)
        st.session_state.bot_thread.start()
        status_area.text("Trading bot started.")
    else:
        status_area.text("Trading bot is already running.")

if st.button("Stop Trading Bot"):
    if st.session_state.bot_thread and st.session_state.bot_thread.is_alive():
        stop_bot.set()
        st.session_state.bot_thread.join()
        status_area.text("Trading bot stopped.")
    else:
        status_area.text("Trading bot is not running.")

# --- Visualization helpers ---


@st.cache_data(ttl=600, show_spinner=False)
def fetch_historical_prices(symbol, category):
    try:
        if category == "crypto":
            yf_symbol = symbol.replace("USDT", "-USD") if symbol.endswith("USDT") else symbol
            data = yf.Ticker(yf_symbol).history(period="1d", interval="1m")
        elif category == "stocks":
            data = yf.Ticker(symbol).history(period="1d", interval="1m")
        elif category == "forex":
            yf_symbol = symbol + "=X"
            data = yf.Ticker(yf_symbol).history(period="1d", interval="1m")
        elif category == "commodities":
            yf_map = {"GOLD": "GC=F", "SILVER": "SI=F", "OIL": "CL=F"}
            yf_symbol = yf_map.get(symbol.upper(), symbol)
            data = yf.Ticker(yf_symbol).history(period="1d", interval="1m")
        else:
            return None
        if data.empty:
            return None
        data = data.reset_index()
        return data
    except Exception as e:
        st.warning(f"Error fetching historical prices for {symbol}: {e}")
        return None

def get_time_col(df):
    for col in ["Datetime", "Date", "time"]:
        if col in df.columns:
            return col
    return df.columns[0]

def plot_price_chart(symbol, current_price, category):
    df = fetch_historical_prices(symbol, category)
    if df is None or df.empty:
        st.warning(f"No historical data for {symbol}")
        return go.Figure()
    time_col = get_time_col(df)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        row_heights=[0.7, 0.3], vertical_spacing=0.05,
                        subplot_titles=(t("Price Chart", "مخطط السعر"), t("Volume", "الحجم")))
    fig.add_trace(
        go.Scatter(
            x=df[time_col],
            y=df["Close"],
            mode='lines',
            line=dict(color='deepskyblue', width=2),
            name=t("Price", "السعر")
        ),
        row=1, col=1
    )
    if "Volume" in df.columns:
        fig.add_trace(
            go.Bar(
                x=df[time_col],
                y=df["Volume"],
                marker_color='gray',
                name=t("Volume", "الحجم")
            ),
            row=2, col=1
        )
    fig.update_layout(
        height=500,
        title_text=f"{t('Live Data', 'البيانات الحية')} - {symbol}",
        xaxis=dict(color='white'),
        yaxis=dict(color='white'),
        plot_bgcolor='rgba(0,0,0,1)',
        paper_bgcolor='rgba(0,0,0,1)',
        font=dict(color='white'),
        margin=dict(t=50, b=40),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='gray')
    )
    return fig

def plot_candlestick_chart(symbol, current_price, category):
    df = fetch_historical_prices(symbol, category)
    if df is None or df.empty:
        st.warning(f"No historical data for {symbol}")
        return go.Figure()
    time_col = get_time_col(df)
    fig = go.Figure(
        data=[go.Candlestick(
            x=df[time_col],
            open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name=t("Candlesticks", "الشموع"),
            increasing_line_color='lime', decreasing_line_color='red'
        )]
    )
    fig.update_layout(
        title=f"{t('Candlestick Chart', 'مخطط الشموع')} - {symbol}",
        xaxis_title=t("Time", "الوقت"),
        yaxis_title=t("Price", "السعر"),
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        margin=dict(t=50, b=40),
        xaxis=dict(color='white'),
        yaxis=dict(color='white')
    )
    return fig
# ...existing code...
def show_all_symbols_visuals():
    st.header(t("Live Price Visualizations for All Symbols", "مخططات الأسعار الحية لجميع الرموز"))
    for category, symbols in SYMBOLS.items():
        with st.expander(f"{category.upper()} {t('Symbols', 'الرموز')}"):
            for symbol in symbols:
                price = get_price(symbol)
                if price:
                    st.subheader(symbol)

                    # 🌟 Show Line + Volume Chart
                    st.plotly_chart(plot_price_chart(symbol, price, category), use_container_width=True)

                    # 🌟 Show Candlestick Chart
                    st.plotly_chart(plot_candlestick_chart(symbol, price, category), use_container_width=True)

                    # 🌟 Trade Signals
                    do_trade, direction, _ = should_trade(symbol, category)
                    sentiment = get_sentiment_for_symbol(symbol) or t("neutral", "محايد")
                    if do_trade:
                        if direction == "BUY":
                            st.success(t(f"Signal: Strong Buy 🟢 | Sentiment: {sentiment}",
                                         f"إشارة: شراء قوي 🟢 | الشعور: {sentiment}"))
                        else:
                            st.error(t(f"Signal: Strong Sell 🔴 | Sentiment: {sentiment}",
                                       f"إشارة: بيع قوي 🔴 | الشعور: {sentiment}"))
                    else:
                        st.info(t(f"Signal: Neutral ⚪ | Sentiment: {sentiment}",
                                  f"إشارة: محايد ⚪ | الشعور: {sentiment}"))
                else:
                    st.warning(t(f"No price data available for {symbol}",
                                 f"لا تتوفر بيانات السعر لـ {symbol}"))
# ...existing code...
show_all_symbols_visuals()

import pandas as pd

st.header(t("Recent Trades", "الصفقات الأخيرة"))

# Category filter
category_options = ["all", "crypto", "stocks", "forex", "commodities"]
selected_category = st.selectbox(t("Show trades for category:", "عرض الصفقات حسب الفئة:"), category_options)

# Build query
query = {"user_id": st.session_state.user_id}
if selected_category != "all":
    query["category"] = selected_category.lower()

# Projection
projection = {
    "_id": 0,
    "timestamp": 1,
    "symbol": 1,
    "side": 1,
    "quantity": 1,
    "category": 1,
    "open_price": 1,
    "close_price": 1
}

try:
    trades_cursor = db["trade_history"].find(query, projection).sort("timestamp", -1).limit(20)
    trades = list(trades_cursor)
except Exception as e:
    st.error(f"Database error: {e}")
    trades = []

if trades:
    df = pd.DataFrame(trades)

    # Ensure columns
    for col in ["timestamp", "symbol", "side", "quantity", "open_price", "close_price"]:
        if col not in df.columns:
            df[col] = ""

    # Compute timestamp and side
    df["Timestamp"] = pd.to_datetime(df["timestamp"], unit='s').dt.strftime("%Y-%m-%d %H:%M:%S")
    df["Side"] = df["side"].str.upper()
    df["Quantity"] = df["quantity"]
    df["Symbol"] = df["symbol"]
    df["Open Price"] = df["open_price"]
    df["Close Price"] = df["close_price"]

    # Compute profit
    def calc_profit(row):
        try:
            qty = float(row["Quantity"])
            open_p = float(row["Open Price"])
            close_p = float(row["Close Price"])
            if row["Side"] == "BUY":
                return (close_p - open_p) * qty
            elif row["Side"] == "SELL":
                return (open_p - close_p) * qty
        except:
            return 0.0
    df["Profit"] = df.apply(calc_profit, axis=1)

    display_df = df[["Timestamp", "Symbol", "Side", "Quantity", "Open Price", "Close Price", "Profit"]]

    # Highlight both side & profit
    def highlight_side(val):
        color = 'green' if val == "BUY" else 'red' if val == "SELL" else 'black'
        return f'color: {color}; font-weight: bold'
    
    def highlight_profit(val):
        try:
            color = 'green' if val > 0 else 'red' if val < 0 else 'black'
            return f'color: {color}; font-weight: bold'
        except:
            return ''
    
    styled_df = display_df.style \
        .map(highlight_side, subset=["Side"]) \
        .map(highlight_profit, subset=["Profit"])

    st.dataframe(styled_df, use_container_width=True)

else:
    st.info(t("No trades available.", "لا توجد صفقات متاحة."))

# Footer
st.write(t("IRAM-B :: Adam Elnaba", "تم التطوير بواسطة IRAM-B"))
