from config import EXCHANGE, MARKET_TYPE, SYMBOL, TIMEFRAME, MODE, HISTORY_LIMIT
from exchange import create_exchange
from data_loader import fetch_ohlcv  # adjust if your function name is different
from strategy import run_signal  # adjust if your strategy function is named differently

import pandas as pd

def run_strategy_once():
    """
    Run the bot for ONE candle on the active exchange + market
    """
    print(f"\n=== RUNNING {EXCHANGE.upper()} - {MARKET_TYPE.upper()} ===\n")
    
    # Connect to exchange
    exchange = create_exchange()
    
    # Fetch OHLCV data for one candle
    df = fetch_ohlcv(exchange, SYMBOL, TIMEFRAME, HISTORY_LIMIT)
    
    # Run your trading signals / strategy
    run_signal(exchange, df, SYMBOL, MARKET_TYPE)

    print(f"=== FINISHED {EXCHANGE.upper()} - {MARKET_TYPE.upper()} ===\n")
