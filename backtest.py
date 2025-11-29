import pandas as pd
import numpy as np
from model import load_model, prepare_X_y
import joblib

def simple_backtest(df, model, initial_capital=10000, position_size=0.1):
    df = df.copy().iloc[:-1]  # align with model target
    X, y = prepare_X_y(df)
    preds = model.predict(X)
    # simulate a simple strategy: long when pred==1 else flat
    capital = initial_capital
    position = 0
    cash = capital
    equity_curve = []
    sizes = []
    for i, pred in enumerate(preds):
        price = df['close'].iloc[i+1]  # using next row price as execution approx
        if pred == 1 and position == 0:
            # buy
            qty = (capital * position_size) / price
            position = qty
            cash -= qty * price
        elif pred == 0 and position > 0:
            # sell all
            cash += position * price
            position = 0
        total = cash + position * price
        equity_curve.append(total)
        sizes.append(position)
    result = pd.DataFrame({
        'equity': equity_curve,
        'position': sizes
    }, index=df.index[1:1+len(preds)])
    return result

if __name__ == "__main__":
    import data_loader, features, model
    df = data_loader.fetch_ohlcv()
    df = features.add_features(df)
    clf = model.train_and_save(df)
    res = simple_backtest(df, clf)
    print(res.tail())