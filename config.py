from dotenv import load_dotenv
import os

load_dotenv()

# Single exchange (binance, okx, bybit...)
EXCHANGE = os.getenv("EXCHANGE", "binance")

# Single trading pair
SYMBOL = os.getenv("SYMBOL", "BTC/USDT")

# Timeframe (1m, 5m, 15m, 1h, 4h)
TIMEFRAME = os.getenv("TIMEFRAME", "1h")

# How many candles to load for training/backtest
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", 1000))

# Mode: backtest or live
MODE = os.getenv("MODE", "backtest")
