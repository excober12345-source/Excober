from dotenv import load_dotenv
import os

load_dotenv()

EXCHANGE = os.getenv("EXCHANGE", "okx")
SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAME = os.getenv("TIMEFRAME", "1h")
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", 1000))
MODE = os.getenv("MODE", "backtest")

# OKX KEYS
OKX_API_KEY = os.getenv("OKX_API_KEY")
OKX_API_SECRET = os.getenv("OKX_API_SECRET")
OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE")

# BINANCE KEYS
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
