import pandas as pd
import os
import time
from config import EXCHANGE, MARKET_TYPE, SYMBOLS, TIMEFRAME, HISTORY_LIMIT, PAPER_MODE, MAX_DAILY_LOSS, ORDER_DELAY
from exchange import create_exchange
from data_loader import fetch_ohlcv
from strategy import run_signal
from notifications import send_telegram

LOG_FILE = "trades_log.csv"

def log_trades(trades):
    df = pd.DataFrame(trades)
    mode = 'a' if os.path.exists(LOG_FILE) else 'w'
    header = not os.path.exists(LOG_FILE)
    df.to_csv(LOG_FILE, mode=mode, header=header, index=False)

def check_daily_loss():
    if not os.path.exists(LOG_FILE):
        return False
    df = pd.read_csv(LOG_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    today = pd.Timestamp.now().date()
    today_trades = df[df['timestamp'].dt.date == today]
    loss = 0
    for _, row in today_trades.iterrows():
        if row['side'].lower() == 'sell' and row.get('status', '') != 'paper':
            loss += row['amount'] * row['price']
    if loss >= MAX_DAILY_LOSS:
        send_telegram(f"Daily loss limit reached: ${loss}. Bot paused.")
        return True
    return False

def run_strategy_once():
    if check_daily_loss():
        return

    print(f"\n=== {EXCHANGE.upper()} | {MARKET_TYPE.upper()} ===\n")
    try:
        exchange = create_exchange()
        for symbol in SYMBOLS:
            symbol = symbol.strip()
            try:
                df = fetch_ohlcv(exchange, symbol, TIMEFRAME, HISTORY_LIMIT)
                trades = run_signal(exchange, df, symbol, MARKET_TYPE, paper=PAPER_MODE)
                if trades:
                    log_trades(trades)
                    for t in trades:
                        msg = (f"{t['exchange'].upper()} | {t['market_type'].upper()} | "
                               f"{t['side'].upper()} {t['symbol']} @ {t['price']} x {t['amount']} | "
                               f"TP: {t.get('tp')} SL: {t.get('sl')} | Status: {t['status']}")
                        send_telegram(msg)
                time.sleep(ORDER_DELAY)
            except Exception as e:
                print(f"[{symbol} Error] {e}")
                send_telegram(f"Error on {EXCHANGE.upper()} {symbol}: {e}")
    except Exception as e:
        print(f"[{EXCHANGE.upper()} Error] {e}")
        send_telegram(f"Fatal error on {EXCHANGE.upper()}: {e}")
