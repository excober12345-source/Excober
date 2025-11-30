from dotenv import load_dotenv
import os
load_dotenv()

# ------------------------------
# Exchange settings
# ------------------------------
# Accepts a list of exchange as a comma-separated string:
# Example in .env: EXCHANGE=binance,okx,bybit
EXCHANGE = os.getenv("EXCHANGE", "binance").split(",")

# ------------------------------
# Symbol settings
# ------------------------------
# Accepts a list of symbols:
# Example in .env: SYMBOLS=BTC/USDT,ETH/USDT,EUR/USD,TSLA/USD
SYMBOLS = os.getenv("SYMBOLS", "BTC/USDT").split(",")

TIMEFRAME = os.getenv("TIMEFRAME", "1h")
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "2000"))

# ------------------------------
# Execution settings
# ------------------------------
PAPER = os.getenv("PAPER", "true").lower() == "true"

API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")