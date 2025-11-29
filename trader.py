import ccxt, time
from config import EXCHANGE, SYMBOL, API_KEY, API_SECRET, PAPER
from features import add_features
from model import load_model
import pandas as pd

def get_exchange():
    excls = getattr(ccxt, EXCHANGE)
    params = {'enableRateLimit': True}
    # If using testnet / paper, check exchange docs. Example: for binance testnet
    if API_KEY:
        params.update({'apiKey': API_KEY, 'secret': API_SECRET})
    ex = excls(params)
    return ex

def run_live():
    ex = get_exchange()
    model = load_model()
    while True:
        try:
            ohlcv = ex.fetch_ohlcv(SYMBOL, timeframe='1h', limit=100)
            df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
            df.set_index('ts', inplace=True)
            df = add_features(df)
            X = df[['r_1','r_2','sma_diff','vol_ema']].values[-1].reshape(1,-1)
            pred = model.predict(X)[0]
            print("Pred:", pred, "Latest close:", df['close'].iloc[-1])
            # Example action: if pred==1, place a market buy for small size
            # IMPORTANT: only enable trading after robust testing and in PAPER mode
            if pred == 1:
                print("Would place buy order (paper mode).")
                # ex.create_order(SYMBOL, 'market', 'buy', amount)
            else:
                print("No action (paper mode).")
        except Exception as e:
            print("Error:", e)
        time.sleep(60*30)  # sleep 30 minutes for 1h timeframe loop

if __name__ == "__main__":
    run_live()