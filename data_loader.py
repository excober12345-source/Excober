import ccxt
import pandas as pd
from config import EXCHANGE, SYMBOL, TIMEFRAME, HISTORY_LIMIT
import time

def fetch_ohlcv():
    exchange_cls = getattr(ccxt, EXCHANGE)  # e.g., ccxt.binance()
    ex = exchange_cls({'enableRateLimit': True})
    # public data; if private, pass api keys in config
    limit = HISTORY_LIMIT
    ohlcv = ex.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    return df

if __name__ == "__main__":
    df = fetch_ohlcv()
    print(df.tail())