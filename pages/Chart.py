import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from binance.client import Client
import yfinance as yf

# Your API keys here
ALPHA_VANTAGE_KEY = "B6762CER4FORROBH"
FMP_API_KEY = "gzoC7P7Ly3pWMoKHgLc1hXkPq7kMOafk"

# Setup Alpha Vantage clients
fx_client = ForeignExchange(key=ALPHA_VANTAGE_KEY)
ts_client = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format='pandas')
ti_client = TechIndicators(key=ALPHA_VANTAGE_KEY, output_format='pandas')

# Setup Binance client

# Fetch stock data from Financial Modeling Prep
def fetch_stock_data(symbol):
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={FMP_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"Error fetching stock data for {symbol}")
        return None
    data = response.json()
    if "historical" not in data:
        st.error(f"No historical data found for {symbol}")
        return None
    df = pd.DataFrame(data["historical"])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.sort_index()
    return df

# Fetch forex or commodity data from Alpha Vantage
def fetch_alpha_vantage_fx_or_commodity(symbol, is_forex=True):
    try:
        if is_forex:
            from_symbol = symbol[:3]
            to_symbol = symbol[3:]
            data, _ = fx_client.get_currency_exchange_daily(from_symbol, to_symbol, outputsize='compact')
            # data keys: '1. open', '2. high', '3. low', '4. close'
        else:
            data, _ = ts_client.get_daily(symbol, outputsize='compact')
            # same keys for daily data

        df = pd.DataFrame.from_dict(data, orient='index')
        df.index = pd.to_datetime(df.index)
        # Rename columns according to Alpha Vantage keys
        df.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
        }, inplace=True)

        df = df[['open', 'high', 'low', 'close']]  # reorder columns
        df = df.astype(float)
        df = df.sort_index()
        return df
    except Exception as e:
        st.error(f"Alpha Vantage API error: {e}")
        return None

# Fetch commodity data using yfinance
def fetch_commodity_yfinance(symbol):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1y")  # 1 year daily data
        if df.empty:
            st.error(f"No data found for commodity symbol {symbol}")
            return None
        df = df[['Open', 'High', 'Low', 'Close']].rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close'
        })
        df.index.name = 'date'
        df = df.sort_index()
        return df
    except Exception as e:
        st.error(f"yfinance error fetching commodity {symbol}: {e}")
        return None


# Fetch crypto data from CryptoCompare (daily)
def fetch_crypto_data_cryptocompare(symbol, limit=100):
    try:
        # Assume symbol like BTCUSDT, we take BTC as base and USD as quote
        if len(symbol) > 3:
            base = symbol[:-4]  # e.g. BTC from BTCUSDT
            quote = symbol[-4:]  # e.g. USDT
            # CryptoCompare uses USD as quote, so convert USDT -> USD for API
            if quote.upper() in ['USDT', 'USD']:
                quote = 'USD'
        else:
            base = symbol
            quote = 'USD'
        url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={base.upper()}&tsym={quote.upper()}&limit={limit}"
        response = requests.get(url)
        data = response.json()
        if data['Response'] != 'Success':
            st.error(f"CryptoCompare API error: {data.get('Message', 'Unknown error')}")
            return None
        df = pd.DataFrame(data['Data']['Data'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df = df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volumeto': 'volume'
        })
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df
    except Exception as e:
        st.error(f"CryptoCompare API error: {e}")
        return None

# Calculate indicators: SMA, RSI, MACD, Bollinger Bands
def add_technical_indicators(df):
    # SMA 14
    df['SMA14'] = df['close'].rolling(window=14).mean()

    # RSI 14
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI14'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # Bollinger Bands (20, 2 std)
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    df['BB_std'] = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']

    return df

# Plot chart with indicators
def plot_chart_with_indicators(df, title="Price Chart"):
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25],
                        specs=[[{"type": "candlestick"}],
                               [{"type": "scatter"}],
                               [{"type": "bar"}]])

    # Candlestick on first subplot
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'
    ), row=1, col=1)

    # SMA 14
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA14'], line=dict(color='blue', width=1), name='SMA 14'), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], line=dict(color='rgba(0,0,255,0.2)'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], line=dict(color='rgba(0,0,255,0.2)'),
                             fill='tonexty', fillcolor='rgba(0,0,255,0.1)', showlegend=False), row=1, col=1)

    # RSI subplot with 30/70 lines
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI14'], line=dict(color='orange', width=1), name='RSI 14'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # MACD histogram & lines
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='purple', width=1), name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], line=dict(color='black', width=1), name='Signal Line'), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], marker_color='gray', name='MACD Hist'), row=3, col=1)

    fig.update_layout(height=900, title=title, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# Streamlit UI
st.title("Multi-Asset Trading Analytics")

asset_type = st.selectbox("Select Asset Type", ["Stocks", "Forex", "Cryptocurrency", "Commodities"])

if asset_type == "Stocks":
    symbol = st.text_input("Enter Stock Symbol", "AAPL")
    if st.button("Fetch and Plot"):
        df = fetch_stock_data(symbol)
        if df is not None:
            df = add_technical_indicators(df)
            plot_chart_with_indicators(df, title=f"{symbol.upper()} Stock Price")

elif asset_type == "Forex":
    symbol = st.text_input("Enter Forex Pair (e.g., EURUSD)", "EURUSD")
    if st.button("Fetch and Plot"):
        df = fetch_alpha_vantage_fx_or_commodity(symbol, is_forex=True)
        if df is not None:
            df = add_technical_indicators(df)
            plot_chart_with_indicators(df, title=f"{symbol.upper()} Forex Price")

elif asset_type == "Cryptocurrency":
    symbol = st.text_input("Enter Crypto Symbol (e.g., BTCUSDT)", "BTCUSDT")
    if st.button("Fetch and Plot"):
        df = fetch_crypto_data_cryptocompare(symbol)
        if df is not None:
            df = add_technical_indicators(df)
            plot_chart_with_indicators(df, title=f"{symbol.upper()} Crypto Price")


elif asset_type == "Commodities":
    symbol = st.text_input("Enter Commodity Symbol (yfinance format, e.g. GC=F for Gold)", "GC=F")
    if st.button("Fetch and Plot"):
        df = fetch_commodity_yfinance(symbol)
        if df is not None:
            df = add_technical_indicators(df)
            plot_chart_with_indicators(df, title=f"{symbol.upper()} Commodity Price")
