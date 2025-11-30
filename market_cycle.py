import time
from main_train_and_backtest import run_strategy_once
from config import TIMEFRAME

def timeframe_to_seconds(tf):
    tf = tf.lower().strip()
    if tf.endswith('m'):
        return int(tf[:-1]) * 60
    if tf.endswith('h'):
        return int(tf[:-1]) * 3600
    if tf.endswith('d'):
        return int(tf[:-1]) * 86400
    return 3600

TIMEFRAME_SECONDS = timeframe_to_seconds(TIMEFRAME)

markets = [
    {"exchange": "okx", "market_type": "spot"},
    {"exchange": "okx", "market_type": "futures"},
    {"exchange": "binance", "market_type": "spot"},
    {"exchange": "binance", "market_type": "futures"},
]

def update_env(exchange, market_type):
    lines = []
    with open(".env", "r") as f:
        for line in f:
            if line.startswith("EXCHANGE="):
                lines.append(f"EXCHANGE={exchange}\n")
            elif line.startswith("MARKET_TYPE="):
                lines.append(f"MARKET_TYPE={market_type}\n")
            else:
                lines.append(line)
    with open(".env", "w") as f:
        f.writelines(lines)

def cycle_markets():
    while True:
        for m in markets:
            update_env(m["exchange"], m["market_type"])
            run_strategy_once()
            time.sleep(TIMEFRAME_SECONDS)

if __name__ == "__main__":
    cycle_markets()
