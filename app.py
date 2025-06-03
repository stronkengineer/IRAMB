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
api_key_binance, api_secret_binance = None, None
client = None
if use_binance:
    api_key_binance = st.sidebar.text_input("Binance API Key", type="password")
    api_secret_binance = st.sidebar.text_input("Binance API Secret", type="password")
    if api_key_binance and api_secret_binance:
        user_binance = user_collection.find_one({"api_key_binance": api_key_binance})
        if not user_binance:
            user_collection.insert_one({"api_key_binance": api_key_binance, "api_secret_binance": api_secret_binance})
            st.sidebar.success("Binance API credentials saved.")
        client = Client(api_key=api_key_binance, api_secret=api_secret_binance)
    else:
        st.sidebar.warning("Please enter your Binance API credentials.")

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
price_cache = {}
CACHE_EXPIRY = 300  # seconds

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
            now = time.time()
            if symbol in price_cache:
                cached_price, timestamp = price_cache[symbol]
                if now - timestamp < CACHE_EXPIRY:
                    return cached_price

            # Extended commodity mapping with source and symbol/id
            commodity_source_map = {
                "GOLD": {"source": "api_ninjas", "id": "gold"},
                "SILVER": {"source": "yfinance", "id": "SI=F"},
                "OIL": {"source": "yfinance", "id": "CL=F"},
                "NATGAS": {"source": "yfinance", "id": "NG=F"},          # Natural Gas
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
                print(f"Unsupported commodity symbol: {symbol}")
                return None

            if info["source"] == "yfinance":
                try:
                    data = yf.Ticker(info["id"]).history(period="1d")
                    if not data.empty:
                        price = data["Close"].iloc[-1]
                        price_cache[symbol] = (price, now)
                        return float(price)
                    else:
                        print(f"No data found in yfinance for {symbol}")
                        return None
                except Exception as e:
                    print(f"yfinance error for {symbol}: {e}")
                    return None

            elif info["source"] == "api_ninjas":
                url = f"https://api.api-ninjas.com/v1/commodityprice?name={info['id']}"
                headers = {'X-Api-Key': API_NINJAS_KEY}
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    price = data.get("price")
                    if price is not None:
                        price_cache[symbol] = (price, now)
                        return float(price)
                    else:
                        print(f"No price in API response for {symbol}")
                        return None
                else:
                    print(f"API error for {symbol}: {response.status_code} {response.text}")
                    return None

        else:
            return None

    except Exception as e:
        print(f"Price fetch error for {symbol}: {e}")
        return None



# --- TradingView signal ---
INTERVAL = "1d"  # or whatever interval you want

commodity_tv_map = {
    "OIL": {"symbol": "CL1!", "exchange": "NYMEX", "screener": "commodities"},
    "GOLD": {"symbol": "GC1!", "exchange": "COMEX", "screener": "commodities"},
    "SILVER": {"symbol": "SI1!", "exchange": "COMEX", "screener": "commodities"},
    # Add more commodities here as needed
}

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
def place_order(symbol, side, quantity, category):
    order = {
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "category": category,
        "timestamp": time.time()
    }
    try:
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
            # Use your pre-built IG symbol->epic map here
            epic = ig_symbol_map.get(symbol.upper(), symbol)  # convert symbol to epic if possible

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

        # Insert order record in DB
        trade_collection.insert_one(order)
        return order

    except Exception as e:
        st.error(t(f"Order failed: {e}", f"فشل الطلب: {e}"))
        return None


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
stop_bot = threading.Event()

def trading_bot(status_area):
    while not stop_bot.is_set():
        for category, symbols in SYMBOLS.items():
            if not symbols:
                status_area.text(f"No symbols found for category {category}, skipping.")
                continue

            for symbol in symbols:
                try:
                    if not symbol or not isinstance(symbol, str):
                        status_area.text(f"Invalid symbol: {symbol} in category {category}, skipping.")
                        continue

                    do_trade, signal, price = should_trade(symbol, category)
                    if not do_trade:
                        continue
                    if price is None:
                        status_area.text(f"No price for {symbol}, skipping trade.")
                        continue
                    if signal is None or signal.upper() not in {"BUY", "SELL"}:
                        status_area.text(f"Invalid signal {signal} for {symbol}, skipping.")
                        continue

                    if category == "crypto":
                        if not client:
                            status_area.text("Binance client not initialized, skipping crypto trade.")
                            continue
                        balance = client.account().get("balances", [])
                        asset = symbol[:-4]
                        asset_balance = next((b for b in balance if b.get("asset") == asset), None)
                        if not asset_balance:
                            status_area.text(f"No balance found for asset {asset}")
                        qty = 0.001  # TODO: improve qty calc
                        place_order(symbol, signal.upper(), qty, category)

                    elif category == "stocks":
                        if not alpaca:
                            status_area.text("Alpaca client not initialized, skipping stock trade.")
                            continue
                        qty = 1
                        place_order(symbol, signal.upper(), qty, category)

                    elif category == "forex":
                        if not igClient:
                            status_area.text("IG client not initialized, skipping forex trade.")
                            continue
                        normalized_symbol = symbol.upper().replace(" ", "").replace("/", "")
                        epic = ig_symbol_map.get(normalized_symbol)
                        if not epic:
                            status_area.text(f"No IG epic mapping found for forex symbol: {symbol}")
                            continue
                        qty = 1000
                        place_order(epic, signal.upper(), qty, category)

                    else:
                        status_area.text(f"Category '{category}' not handled for trading.")

                    status_area.text(f"Traded {symbol} ({category}): Signal {signal}, Price {price}")

                except Exception as e:
                    status_area.text(f"Error trading {symbol} in {category}: {e}")

        time.sleep(CHECK_FREQUENCY)

# Streamlit app controls

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


def plot_price_chart(symbol, current_price):
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=60, freq='T')
    prices = [current_price * (1 + 0.005 * (random.random() - 0.5)) for _ in range(60)]
    df = pd.DataFrame({"time": timestamps, "price": prices})

    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=(t("Price Chart", "مخطط السعر"),)
    )

    fig.add_trace(
        go.Scatter(
            x=df["time"], y=df["price"], mode='lines',
            line=dict(color='deepskyblue', width=2),
            name=t("Price", "السعر")
        ),
        row=1, col=1
    )

    fig.update_layout(
        height=400,
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

def plot_candlestick_chart(symbol, current_price):
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=60, freq='T')
    df = pd.DataFrame({
        "time": timestamps,
        "open": [current_price * (1 + 0.005 * (random.random() - 0.5)) for _ in range(60)],
        "high": [current_price * (1 + 0.01 * random.random()) for _ in range(60)],
        "low": [current_price * (1 - 0.01 * random.random()) for _ in range(60)],
        "close": [current_price * (1 + 0.005 * (random.random() - 0.5)) for _ in range(60)]
    })

    fig = go.Figure(
        data=[go.Candlestick(
            x=df["time"],
            open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
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

def show_all_symbols_visuals():
    st.header(t("Live Price Visualizations for All Symbols", "مخططات الأسعار الحية لجميع الرموز"))
    for category, symbols in SYMBOLS.items():
        with st.expander(f"{category.upper()} {t('Symbols', 'الرموز')}"):
            for symbol in symbols:
                price = get_price(symbol)
                if price:
                    st.subheader(symbol)

                    # 🌟 Show Line + Volume Chart
                    st.plotly_chart(plot_price_chart(symbol, price), use_container_width=True)

                    # 🌟 Show Candlestick Chart
                    st.plotly_chart(plot_candlestick_chart(symbol, price), use_container_width=True)

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
# --- Main UI ---
st.title(t("Multi-Asset AI Trading Bot", "بوت التداول الذكي متعدد الأصول"))
st.write(t("Trading algorithm:", "خوارزمية التداول:"), selected_algo)

# Show live visuals with signals and sentiment
show_all_symbols_visuals()


st.header(t("Recent Trades", "الصفقات الأخيرة"))

# Fetch latest 10 trades from the database
trades = list(trade_collection.find().sort("timestamp", -1).limit(10))

if trades:
    # Convert to DataFrame and format columns
    df = pd.DataFrame(trades)
    df["Timestamp"] = pd.to_datetime(df["timestamp"], unit='s').dt.strftime("%Y-%m-%d %H:%M:%S")
    df["Side"] = df["side"].str.upper()
    df["Quantity"] = df["quantity"]
    df["Symbol"] = df["symbol"]

    display_df = df[["Timestamp", "Symbol", "Side", "Quantity"]]

    # Style BUY green, SELL red
    def highlight_side(val):
        color = 'green' if val == "BUY" else 'red' if val == "SELL" else 'black'
        return f'color: {color}; font-weight: bold'

    styled_df = display_df.style.map(highlight_side, subset=["Side"])

    st.dataframe(styled_df, use_container_width=True)
else:
    st.info(t("No trades available.", "لا توجد صفقات متاحة."))

# Footer 

st.write(t("IRAM-B :: Adam Elnaba", "تم التطوير بواسطة IRAM-B"))
