import pandas as pd
import numpy as np
from config import PAPER_MODE, RISK_PERCENT, ATR_PERIOD, TP_PERCENT, SL_PERCENT

class Strategy:
    def __init__(self, cfg):
        self.cfg = cfg

    def calculate_atr(self, df, period=14):
        df = df.copy()
        df['H-L'] = df['high'] - df['low']
        df['H-PC'] = abs(df['high'] - df['close'].shift(1))
        df['L-PC'] = abs(df['low'] - df['close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR'] = df['TR'].rolling(period).mean()
        return df['ATR'].iloc[-1]

    def generate_signal(self, df):
        """
        Returns: 'BUY', 'SELL', or None
        """
        if df['close'].iloc[-1] > df['close'].iloc[-2]:
            return "BUY"
        else:
            return "SELL"

    def run(self, exchange, df, symbol, market_type, paper=False):
        trades = []

        atr = self.calculate_atr(df, ATR_PERIOD)
        signal = self.generate_signal(df).lower()

        balance = self.get_balance(exchange, market_type, paper)

        risk_amount = balance * (RISK_PERCENT / 100)
        unit_risk = atr if atr > 0 else df['close'].iloc[-1]
        amount = round(risk_amount / unit_risk, 6)

        price = df['close'].iloc[-1]
        tp = price * (1 + TP_PERCENT/100) if signal == "buy" else price * (1 - TP_PERCENT/100)
        sl = price * (1 - SL_PERCENT/100) if signal == "buy" else price * (1 + SL_PERCENT/100)

        if paper:
            trades.append({
                "timestamp": pd.Timestamp.now(),
                "exchange": exchange.name,
                "market_type": market_type,
                "symbol": symbol,
                "side": signal,
                "price": price,
                "amount": amount,
                "status": "paper",
                "tp": tp,
                "sl": sl
            })
        else:
            order = self.place_order(exchange, symbol, signal, amount, market_type)
            order.update({"tp": tp, "sl": sl})
            trades.append(order)

        return trades

    def get_balance(self, exchange, market_type, paper):
        if paper:
            return 1000
        if market_type == "spot":
            return exchange.fetch_balance()['total'].get('USDT', 0)
        return exchange.fetch_balance({'type': 'future'})['total'].get('USDT', 0)

    def place_order(self, exchange, symbol, side, amount, market_type):
        if market_type == "spot":
            order = exchange.create_order(symbol, 'market', side, amount)
        else:
            order = exchange.create_order(symbol, 'market', side, amount, params={"reduceOnly": False})

        return {
            "timestamp": pd.Timestamp.now(),
            "exchange": exchange.name,
            "market_type": market_type,
            "symbol": symbol,
            "side": side,
            "price": order.get('price', 0),
            "amount": amount,
            "status": order.get('status', '')
        }
