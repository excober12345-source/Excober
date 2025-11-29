import pandas as pd
import numpy as np
import ta  # pip install ta

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['r_1'] = df['return'].shift(1)
    df['r_2'] = df['return'].shift(2)
    df['sma10'] = ta.trend.sma_indicator(df['close'], window=10)
    df['sma50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['sma_diff'] = df['sma10'] - df['sma50']
    df['vol_ema'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df = df.dropna()
    return df