from exchange import create_exchange
from strategy import Strategy
from config import load_config
from notifications import send_telegram
from data_loader import fetch_ohlcv
import time

def main():
    cfg = load_config()
    exchange = create_exchange()
    strategy = Strategy(cfg)

    print("🚀 Live trading started... (press CTRL+C to stop)")

    while True:
        try:
            for symbol in cfg.symbols:
                try:
                    df = fetch_ohlcv(exchange, symbol, cfg.timeframe, cfg.limit)
                    if df is None or df.empty:
                        continue

                    trades = strategy.run(exchange, df, symbol, cfg.market_type, paper=cfg.paper)
                    if trades:
                        # keep using same log + notifications as backtest file
                        # but main.py doesn't handle logging; we notify only
                        for t in trades:
                            msg = (f"{t['exchange'].upper()} | {t['market_type'].upper()} | "
                                   f"{t['side'].upper()} {t['symbol']} @ {t['price']} x {t['amount']} | "
                                   f"TP: {t.get('tp')} SL: {t.get('sl')} | Status: {t['status']}")
                            send_telegram(msg)
                except Exception as e:
                    print(f"[{symbol} Error] {e}")
                    send_telegram(f"Error on live {symbol}: {e}")
            time.sleep(cfg.sleep)
        except KeyboardInterrupt:
            print("Stopping live loop.")
            break
        except Exception as e:
            print("Fatal loop error:", e)
            send_telegram(f"Fatal error in main loop: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
