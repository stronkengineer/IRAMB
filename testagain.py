import time
import numpy as np
from binance.client import Client
from stable_baselines3 import PPO
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# === Load env vars ===
load_dotenv(".env")
BINANCE_API_KEY = "4OxjZnaxjslBROnhw2m1oqdV83fmb4fwFJa2q6K1orpgnKhQYEKL43xyBZuC1Uwz"
BINANCE_API_SECRET = "Qtpo4HEQnDgAhBbEBDQQZ60G0792QrSKBYlML6B6AP4ZdL5FN2srAPsAQJCOCuKp"
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")

# === Connect to Binance Testnet ===
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
client.API_URL = "https://testnet.binance.vision/api"

# === Connect to MongoDB ===
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
trade_collection = db["trade_history"]
summary_collection = db["trade_summary"]

# === Load your trained PPO model ===
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
quantities = {"BTCUSDT": 0.0003, "ETHUSDT": 0.003, "BNBUSDT": 0.02}

# === Track positions in memory ===
positions = {symbol: 0 for symbol in symbols}  # 1=long, -1=short

# === Helper ===
def get_closes(symbol, limit=15):
    klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=limit)
    closes = np.array([float(k[4]) for k in klines])
    return closes

# === Main HFT loop ===
while True:
    for symbol in symbols:
        closes = get_closes(symbol, limit=15)
        if len(closes) < 8:
            continue

        fast_sma = np.mean(closes[-3:])
        slow_sma = np.mean(closes[-8:])
        current_pos = positions[symbol]

        # === Flip logic: always in market
        if fast_sma > slow_sma and current_pos != 1:
            # Flip to LONG
            order = client.order_market_buy(symbol=symbol, quantity=quantities[symbol])
            price = float(order['fills'][0]['price'])
            positions[symbol] = 1
            print(f"✅ FLIP to BUY {symbol} at {price:.2f} (Fast SMA {fast_sma:.2f} > Slow SMA {slow_sma:.2f})")
            trade_collection.insert_one({
                "symbol": symbol,
                "side": "BUY",
                "opening_price": price,
                "closing_price": None,
                "quantity": quantities[symbol],
                "profit": 0.0,
                "timestamp": time.time()
            })

        elif fast_sma < slow_sma and current_pos != -1:
            # Flip to SHORT
            order = client.order_market_sell(symbol=symbol, quantity=quantities[symbol])
            price = float(order['fills'][0]['price'])
            positions[symbol] = -1
            print(f"✅ FLIP to SELL {symbol} at {price:.2f} (Fast SMA {fast_sma:.2f} < Slow SMA {slow_sma:.2f})")
            trade_collection.insert_one({
                "symbol": symbol,
                "side": "SELL",
                "opening_price": price,
                "closing_price": None,
                "quantity": quantities[symbol],
                "profit": 0.0,
                "timestamp": time.time()
            })
        else:
            print(f"⏸ HOLD {symbol} (Fast SMA {fast_sma:.2f}, Slow SMA {slow_sma:.2f})")

    time.sleep(1)