from binance.client import Client
import os
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

# Initialize Binance client
client = Client(API_KEY, API_SECRET)

# Check server status
try:
    price = client.get_symbol_ticker(symbol="BTCUSDT")
    print(f"✅ API works! BTCUSDT Price: ${price['price']}")
except Exception as e:
    print(f"❌ API Error: {e}")
