from dotenv import load_dotenv
import os

load_dotenv()

EXCHANGE = os.getenv("EXCHANGE", "okx")
MARKET_TYPE = os.getenv("MARKET_TYPE", "spot")
SYMBOLS = os.getenv("SYMBOLS", "BTC/USDT").split(",")
TIMEFRAME = os.getenv("TIMEFRAME", "1h")
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", 1000))
MODE = os.getenv("MODE", "backtest")
PAPER_MODE = os.getenv("PAPER_MODE", "true").lower() == "true"

# Risk & safety
RISK_PERCENT = float(os.getenv("RISK_PERCENT", 1))
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", 50))
ORDER_DELAY = int(os.getenv("ORDER_DELAY", 2))

# TP / SL / ATR / Multi-TF
ATR_PERIOD = int(os.getenv("ATR_PERIOD", 14))
TP_PERCENT = float(os.getenv("TP_PERCENT", 2))
SL_PERCENT = float(os.getenv("SL_PERCENT", 1))
MULTI_TIMEFRAMES = os.getenv("MULTI_TIMEFRAMES", TIMEFRAME).split(",")

# OKX keys
OKX_API_KEY = os.getenv("OKX_API_KEY")
OKX_API_SECRET = os.getenv("OKX_API_SECRET")
OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE")

# Binance keys
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


# --------- Compatibility helper for main.py ----------
class Config:
    def __init__(self):
        # single-symbol convenience: pick first from SYMBOLS
        self.exchange = EXCHANGE
        self.market_type = MARKET_TYPE
        self.symbol = SYMBOLS[0].strip() if SYMBOLS else "BTC/USDT"
        self.symbols = [s.strip() for s in SYMBOLS]
        self.timeframe = TIMEFRAME
        self.limit = HISTORY_LIMIT
        self.paper = PAPER_MODE
        self.sleep = ORDER_DELAY
        self.mode = MODE

def load_config():
    """Return a lightweight config object expected by main.py / Strategy."""
    return Config()
