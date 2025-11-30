import time
from main_train_and_backtest import run_strategy  # adjust if your function name is different

markets = [
    {"exchange": "okx", "market_type": "spot"},
    {"exchange": "okx", "market_type": "futures"},
    {"exchange": "binance", "market_type": "spot"},
    {"exchange": "binance", "market_type": "futures"},
]

def cycle_markets():
    while True:
        for m in markets:
            print(f"\n===== RUNNING {m['exchange'].upper()} - {m['market_type'].upper()} =====\n")

            # Set active exchange dynamically
            with open(".env", "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                if line.startswith("EXCHANGE="):
                    new_lines.append(f"EXCHANGE={m['exchange']}\n")
                elif line.startswith("MARKET_TYPE="):
                    new_lines.append(f"MARKET_TYPE={m['market_type']}\n")
                else:
                    new_lines.append(line)

            # Rewrite .env with new settings
            with open(".env", "w") as f:
                f.writelines(new_lines)

            # Run your strategy for this market
            run_strategy()

            # Wait 10 seconds before switching
            time.sleep(10)
