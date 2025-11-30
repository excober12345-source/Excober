import ccxt
from config import (
    EXCHANGE, MARKET_TYPE,
    OKX_API_KEY, OKX_API_SECRET, OKX_PASSPHRASE,
    BINANCE_API_KEY, BINANCE_API_SECRET
)

def create_exchange():
    ex = EXCHANGE.lower()
    market = MARKET_TYPE.lower()

    if ex == "okx":
        exchange = ccxt.okx({
            "apiKey": OKX_API_KEY,
            "secret": OKX_API_SECRET,
            "password": OKX_PASSPHRASE
        })
        exchange.options = {"defaultType": "swap"} if market == "futures" else {"defaultType": "spot"}
        return exchange

    elif ex == "binance":
        if market == "futures":
            return ccxt.binanceusdm({
                "apiKey": BINANCE_API_KEY,
                "secret": BINANCE_API_SECRET,
                "options": {"defaultType": "future"}
            })
        else:
            return ccxt.binance({
                "apiKey": BINANCE_API_KEY,
                "secret": BINANCE_API_SECRET,
                "options": {"defaultType": "spot"}
            })
    else:
        raise ValueError(f"Exchange {EXCHANGE} not supported.")
