import ccxt
from config import (
    EXCHANGE,
    OKX_API_KEY, OKX_API_SECRET, OKX_PASSPHRASE,
    BINANCE_API_KEY, BINANCE_API_SECRET
)

def create_exchange():
    if EXCHANGE.lower() == "okx":
        return ccxt.okx({
            "apiKey": OKX_API_KEY,
            "secret": OKX_API_SECRET,
            "password": OKX_PASSPHRASE,   # OKX requires password field
        })

    elif EXCHANGE.lower() == "binance":
        return ccxt.binance({
            "apiKey": BINANCE_API_KEY,
            "secret": BINANCE_API_SECRET,
        })

    else:
        raise ValueError(f"Exchange {EXCHANGE} is not supported")
