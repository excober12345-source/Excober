from exchange import create_exchange
from strategy import Strategy
from config import load_config
from notifications import send_notification
import time

def main():
    cfg = load_config()
    exchange = create_exchange(cfg)
    strategy = Strategy(cfg)

    print("🚀 Live trading started...")

    while True:
        try:
            df = exchange.fetch_ohlcv(cfg.symbol, cfg.timeframe, cfg.limit)

            if df is None:
                continue

            signal = strategy.generate_signal(df)

            if signal == "BUY":
                exchange.open_position("buy")
                send_notification("BUY executed")

            elif signal == "SELL":
                exchange.open_position("sell")
                send_notification("SELL executed")

            time.sleep(cfg.sleep)

        except Exception as e:
            print("Error:", e)
            time.sleep(5)

if __name__ == "__main__":
    main()
