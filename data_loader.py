def fetch_ohlcv(exchange, symbol, timeframe, limit):
    # Example using ccxt
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    import pandas as pd
    df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df
