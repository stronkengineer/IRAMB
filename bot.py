import threading
import time
import os
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from dotenv import load_dotenv
from binance.spot import Spot
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

TRADE_PERCENT = st.sidebar.slider("Trade Percentage (%)", 1, 10, 5) / 100
STOP_LOSS_PERCENT = st.sidebar.slider("Stop Loss %", 1, 10, 2) / 100
TAKE_PROFIT_PERCENT = st.sidebar.slider("Take Profit %", 1, 10, 5) / 100

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
SYMBOLS = {
    "crypto": ["BTCUSDT", "ETHUSDT"],
    "forex": ["EURUSD", "USDJPY"],
    "stocks": ["AAPL", "TSLA"]
}
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
def get_price(symbol):
    try:
        if symbol in SYMBOLS["crypto"]:
            handler = TA_Handler(symbol=symbol, screener="crypto", exchange="BINANCE", interval=INTERVAL)
        elif symbol in SYMBOLS["forex"]:
            handler = TA_Handler(symbol=symbol, screener="forex", exchange="FX_IDC", interval=INTERVAL)
        elif symbol in SYMBOLS["stocks"]:
            handler = TA_Handler(symbol=symbol, screener="america", exchange="NASDAQ", interval=INTERVAL)
        else:
            return None
        analysis = handler.get_analysis()
        return float(analysis.indicators["close"])
    except Exception as e:
        print(f"Price fetch error for {symbol}: {e}")
        return None

# --- TradingView signal ---
def get_tradingview_signal(symbol, screener, category):
    exchange = "BINANCE" if category == "crypto" else "FX_IDC" if category == "forex" else "NASDAQ"
    try:
        handler = TA_Handler(symbol=symbol, screener=screener, interval=INTERVAL, exchange=exchange)
        return handler.get_analysis().summary
    except Exception as e:
        print(f"TV signal error {symbol}: {e}")
        return {"RECOMMENDATION": "NEUTRAL"}

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
            alpaca_side = "buy" if side == "BUY" else "sell"
            alpaca_order = alpaca.submit_order(
                symbol=symbol,
                qty=quantity,
                side=alpaca_side,
                type="market",
                time_in_force="gtc"
            )
            order.update({"platform": "Alpaca", "alpaca_id": alpaca_order.id})
        # For forex or other categories, could add more platforms here.
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
def trading_bot():
    while not stop_bot.is_set():
        for category, symbols in SYMBOLS.items():
            for symbol in symbols:
                try:
                    do_trade, signal, price = should_trade(symbol, category)
                    if do_trade and price:
                        if category == "crypto" and client:
                            balance = client.account()["balances"]
                            asset = symbol[:-4]
                            asset_balance = next((b for b in balance if b["asset"] == asset), None)
                            qty = 0.001  # Replace with actual quantity calc logic
                            place_order(symbol, signal, qty, category)
                        elif category == "stocks" and alpaca:
                            qty = 1  # Replace with actual logic
                            place_order(symbol, signal, qty, category)
                        # For forex, add your own trade execution logic
                except Exception as e:
                    print(f"Error trading {symbol}: {e}")
        time.sleep(CHECK_FREQUENCY)

# --- Visualization helpers ---
def find_support_resistance_levels(prices, distance=5, prominence=0.01):
    prices = np.array(prices)

    resistances, _ = find_peaks(prices, distance=distance, prominence=prominence)
    supports, _ = find_peaks(-prices, distance=distance, prominence=prominence)

    return supports, resistances

def plot_price_chart(symbol, current_price):
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=60, freq='T')
    prices = [current_price * (1 + 0.01 * (random.random() - 0.5)) for _ in range(60)]
    volumes = [random.randint(100, 1000) for _ in range(60)]
    df = pd.DataFrame({"time": timestamps, "price": prices, "volume": volumes})

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.2,
        subplot_titles=(t("Price Chart", "مخطط السعر"), t("Volume", "الحجم"))
    )

    # Plot price line
    fig.add_trace(
        go.Scatter(
            x=df["time"], y=df["price"], mode='lines+markers',
            name=t("Price", "السعر")
        ),
        row=1, col=1
    )

    # Plot volume bars
    fig.add_trace(
        go.Bar(
            x=df["time"], y=df["volume"], name=t("Volume", "الحجم"),
            marker=dict(color='lightblue')
        ),
        row=2, col=1
    )

    # Find support and resistance levels for the price data
    supports_idx, resistances_idx = find_support_resistance_levels(
        df["price"], distance=5, prominence=0.01
    )

    # Add support lines (green dotted)
    for idx in supports_idx:
        price_level = df["price"].iloc[idx]
        fig.add_hline(
            y=price_level, line_dash="dot", line_color="green",
            annotation_text="", annotation_position="bottom right",
            row=1, col=1
        )

    # Add resistance lines (red dotted)
    for idx in resistances_idx:
        price_level = df["price"].iloc[idx]
        fig.add_hline(
            y=price_level, line_dash="dot", line_color="red",
            annotation_text="", annotation_position="top right",
            row=1, col=1
        )

    fig.update_layout(
        height=600,
        title_text=f"{t('Live Data', 'البيانات الحية')} - {symbol}",
        xaxis_title=t("Time", "الوقت"),
        yaxis_title=t("Price", "السعر")
    )

    return fig


def plot_candlestick_chart(symbol, current_price):
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=60, freq='T')
    df = pd.DataFrame({
        "time": timestamps,
        "open": [current_price * (1 + 0.01 * (random.random() - 0.5)) for _ in range(60)],
        "high": [current_price * (1 + 0.02 * random.random()) for _ in range(60)],
        "low": [current_price * (1 - 0.02 * random.random()) for _ in range(60)],
        "close": [current_price * (1 + 0.01 * (random.random() - 0.5)) for _ in range(60)],
        "volume": [random.randint(100, 1000) for _ in range(60)]
    })

    fig = go.Figure(
        data=[go.Candlestick(
            x=df["time"],
            open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            name=symbol
        )]
    )

    # Find support and resistance levels
    supports_idx, resistances_idx = find_support_resistance_levels(
        df['close'], distance=5, prominence=0.01
    )

    # Plot support levels (green dotted lines)
    for idx in supports_idx:
        price_level = df['close'].iloc[idx]
        fig.add_hline(
            y=price_level, line_dash="dot", line_color="green",
            annotation_text="", annotation_position="bottom right"
        )

    # Plot resistance levels (red dotted lines)
    for idx in resistances_idx:
        price_level = df['close'].iloc[idx]
        fig.add_hline(
            y=price_level, line_dash="dot", line_color="red",
            annotation_text="", annotation_position="top right"
        )

    fig.update_layout(
        title=f"{t('Candlestick Chart', 'مخطط الشموع')} - {symbol}",
        xaxis_title=t("Time", "الوقت"),
        yaxis_title=t("Price", "السعر")
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

# Start/stop buttons for bot control
if st.button(t("Start Trading Bot", "تشغيل البوت")):
    stop_bot.clear()
    threading.Thread(target=trading_bot, daemon=True).start()
    st.success(t("Trading bot started.", "تم تشغيل البوت."))

if st.button(t("Stop Trading Bot", "إيقاف البوت")):
    stop_bot.set()
    st.warning(t("Trading bot stopped.", "تم إيقاف البوت."))

# Display recent trades
st.header(t("Recent Trades", "الصفقات الأخيرة"))
trades = trade_collection.find().sort("timestamp", -1).limit(10)
for t_ in trades:
    side = t_["side"]
    sym = t_["symbol"]
    qty = t_["quantity"]
    ts = pd.to_datetime(t_["timestamp"], unit='s').strftime("%Y-%m-%d %H:%M:%S")
    st.write(f"{ts} - {sym} - {side} - Qty: {qty}")

# Footer
st.write(t("IRAM-B :: Adam Elnaba", "تم التطوير بواسطة IRAM-B"))