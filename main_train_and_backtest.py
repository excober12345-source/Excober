from data_loader import fetch_ohlcv
from features import add_features
from model import train_and_save, load_model
from backtest import simple_backtest

if __name__ == "__main__":
    df = fetch_ohlcv()
    df = add_features(df)
    clf = train_and_save(df)
    res = simple_backtest(df, clf)
    print(res.equity.tail())